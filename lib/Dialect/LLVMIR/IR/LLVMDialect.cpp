//===- LLVMDialect.cpp - LLVM IR Ops and Dialect registration -------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file defines the types and operation details for the LLVM IR dialect in
// MLIR, and the LLVM IR dialect.  It also registers the dialect.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::LLVM;

#include "mlir/Dialect/LLVMIR/LLVMOpsEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::CmpOp.
//===----------------------------------------------------------------------===//
static void printICmpOp(OpAsmPrinter &p, ICmpOp &op) {
  p << op.getOperationName() << " \"" << stringifyICmpPredicate(op.predicate())
    << "\" " << *op.getOperand(0) << ", " << *op.getOperand(1);
  p.printOptionalAttrDict(op.getAttrs(), {"predicate"});
  p << " : " << op.lhs()->getType();
}

static void printFCmpOp(OpAsmPrinter &p, FCmpOp &op) {
  p << op.getOperationName() << " \"" << stringifyFCmpPredicate(op.predicate())
    << "\" " << *op.getOperand(0) << ", " << *op.getOperand(1);
  p.printOptionalAttrDict(op.getAttrs(), {"predicate"});
  p << " : " << op.lhs()->getType();
}

// <operation> ::= `llvm.icmp` string-literal ssa-use `,` ssa-use
//                 attribute-dict? `:` type
// <operation> ::= `llvm.fcmp` string-literal ssa-use `,` ssa-use
//                 attribute-dict? `:` type
template <typename CmpPredicateType>
static ParseResult parseCmpOp(OpAsmParser &parser, OperationState &result) {
  Builder &builder = parser.getBuilder();

  Attribute predicate;
  SmallVector<NamedAttribute, 4> attrs;
  OpAsmParser::OperandType lhs, rhs;
  Type type;
  llvm::SMLoc predicateLoc, trailingTypeLoc;
  if (parser.getCurrentLocation(&predicateLoc) ||
      parser.parseAttribute(predicate, "predicate", attrs) ||
      parser.parseOperand(lhs) || parser.parseComma() ||
      parser.parseOperand(rhs) || parser.parseOptionalAttributeDict(attrs) ||
      parser.parseColon() || parser.getCurrentLocation(&trailingTypeLoc) ||
      parser.parseType(type) ||
      parser.resolveOperand(lhs, type, result.operands) ||
      parser.resolveOperand(rhs, type, result.operands))
    return failure();

  // Replace the string attribute `predicate` with an integer attribute.
  auto predicateStr = predicate.dyn_cast<StringAttr>();
  if (!predicateStr)
    return parser.emitError(predicateLoc,
                            "expected 'predicate' attribute of string type");

  int64_t predicateValue = 0;
  if (std::is_same<CmpPredicateType, ICmpPredicate>()) {
    Optional<ICmpPredicate> predicate =
        symbolizeICmpPredicate(predicateStr.getValue());
    if (!predicate)
      return parser.emitError(predicateLoc)
             << "'" << predicateStr.getValue()
             << "' is an incorrect value of the 'predicate' attribute";
    predicateValue = static_cast<int64_t>(predicate.getValue());
  } else {
    Optional<FCmpPredicate> predicate =
        symbolizeFCmpPredicate(predicateStr.getValue());
    if (!predicate)
      return parser.emitError(predicateLoc)
             << "'" << predicateStr.getValue()
             << "' is an incorrect value of the 'predicate' attribute";
    predicateValue = static_cast<int64_t>(predicate.getValue());
  }

  attrs[0].second = parser.getBuilder().getI64IntegerAttr(predicateValue);

  // The result type is either i1 or a vector type <? x i1> if the inputs are
  // vectors.
  auto *dialect = builder.getContext()->getRegisteredDialect<LLVMDialect>();
  auto resultType = LLVMType::getInt1Ty(dialect);
  auto argType = type.dyn_cast<LLVM::LLVMType>();
  if (!argType)
    return parser.emitError(trailingTypeLoc, "expected LLVM IR dialect type");
  if (argType.getUnderlyingType()->isVectorTy())
    resultType = LLVMType::getVectorTy(
        resultType, argType.getUnderlyingType()->getVectorNumElements());

  result.attributes = attrs;
  result.addTypes({resultType});
  return success();
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::AllocaOp.
//===----------------------------------------------------------------------===//

static void printAllocaOp(OpAsmPrinter &p, AllocaOp &op) {
  auto elemTy = op.getType().cast<LLVM::LLVMType>().getPointerElementTy();

  auto funcTy = FunctionType::get({op.arraySize()->getType()}, {op.getType()},
                                  op.getContext());

  p << op.getOperationName() << ' ' << *op.arraySize() << " x " << elemTy;
  if (op.alignment().hasValue() && op.alignment()->getSExtValue() != 0)
    p.printOptionalAttrDict(op.getAttrs());
  else
    p.printOptionalAttrDict(op.getAttrs(), {"alignment"});
  p << " : " << funcTy;
}

// <operation> ::= `llvm.alloca` ssa-use `x` type attribute-dict?
//                 `:` type `,` type
static ParseResult parseAllocaOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<NamedAttribute, 4> attrs;
  OpAsmParser::OperandType arraySize;
  Type type, elemType;
  llvm::SMLoc trailingTypeLoc;
  if (parser.parseOperand(arraySize) || parser.parseKeyword("x") ||
      parser.parseType(elemType) || parser.parseOptionalAttributeDict(attrs) ||
      parser.parseColon() || parser.getCurrentLocation(&trailingTypeLoc) ||
      parser.parseType(type))
    return failure();

  // Extract the result type from the trailing function type.
  auto funcType = type.dyn_cast<FunctionType>();
  if (!funcType || funcType.getNumInputs() != 1 ||
      funcType.getNumResults() != 1)
    return parser.emitError(
        trailingTypeLoc,
        "expected trailing function type with one argument and one result");

  if (parser.resolveOperand(arraySize, funcType.getInput(0), result.operands))
    return failure();

  result.attributes = attrs;
  result.addTypes({funcType.getResult(0)});
  return success();
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::GEPOp.
//===----------------------------------------------------------------------===//

static void printGEPOp(OpAsmPrinter &p, GEPOp &op) {
  SmallVector<Type, 8> types(op.getOperandTypes());
  auto funcTy = FunctionType::get(types, op.getType(), op.getContext());

  p << op.getOperationName() << ' ' << *op.base() << '[';
  p.printOperands(std::next(op.operand_begin()), op.operand_end());
  p << ']';
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << funcTy;
}

// <operation> ::= `llvm.getelementptr` ssa-use `[` ssa-use-list `]`
//                 attribute-dict? `:` type
static ParseResult parseGEPOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<NamedAttribute, 4> attrs;
  OpAsmParser::OperandType base;
  SmallVector<OpAsmParser::OperandType, 8> indices;
  Type type;
  llvm::SMLoc trailingTypeLoc;
  if (parser.parseOperand(base) ||
      parser.parseOperandList(indices, OpAsmParser::Delimiter::Square) ||
      parser.parseOptionalAttributeDict(attrs) || parser.parseColon() ||
      parser.getCurrentLocation(&trailingTypeLoc) || parser.parseType(type))
    return failure();

  // Deconstruct the trailing function type to extract the types of the base
  // pointer and result (same type) and the types of the indices.
  auto funcType = type.dyn_cast<FunctionType>();
  if (!funcType || funcType.getNumResults() != 1 ||
      funcType.getNumInputs() == 0)
    return parser.emitError(trailingTypeLoc,
                            "expected trailing function type with at least "
                            "one argument and one result");

  if (parser.resolveOperand(base, funcType.getInput(0), result.operands) ||
      parser.resolveOperands(indices, funcType.getInputs().drop_front(),
                             parser.getNameLoc(), result.operands))
    return failure();

  result.attributes = attrs;
  result.addTypes(funcType.getResults());
  return success();
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::LoadOp.
//===----------------------------------------------------------------------===//

static void printLoadOp(OpAsmPrinter &p, LoadOp &op) {
  p << op.getOperationName() << ' ' << *op.addr();
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.addr()->getType();
}

// Extract the pointee type from the LLVM pointer type wrapped in MLIR.  Return
// the resulting type wrapped in MLIR, or nullptr on error.
static Type getLoadStoreElementType(OpAsmParser &parser, Type type,
                                    llvm::SMLoc trailingTypeLoc) {
  auto llvmTy = type.dyn_cast<LLVM::LLVMType>();
  if (!llvmTy)
    return parser.emitError(trailingTypeLoc, "expected LLVM IR dialect type"),
           nullptr;
  if (!llvmTy.getUnderlyingType()->isPointerTy())
    return parser.emitError(trailingTypeLoc, "expected LLVM pointer type"),
           nullptr;
  return llvmTy.getPointerElementTy();
}

// <operation> ::= `llvm.load` ssa-use attribute-dict? `:` type
static ParseResult parseLoadOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<NamedAttribute, 4> attrs;
  OpAsmParser::OperandType addr;
  Type type;
  llvm::SMLoc trailingTypeLoc;

  if (parser.parseOperand(addr) || parser.parseOptionalAttributeDict(attrs) ||
      parser.parseColon() || parser.getCurrentLocation(&trailingTypeLoc) ||
      parser.parseType(type) ||
      parser.resolveOperand(addr, type, result.operands))
    return failure();

  Type elemTy = getLoadStoreElementType(parser, type, trailingTypeLoc);

  result.attributes = attrs;
  result.addTypes(elemTy);
  return success();
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::StoreOp.
//===----------------------------------------------------------------------===//

static void printStoreOp(OpAsmPrinter &p, StoreOp &op) {
  p << op.getOperationName() << ' ' << *op.value() << ", " << *op.addr();
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.addr()->getType();
}

// <operation> ::= `llvm.store` ssa-use `,` ssa-use attribute-dict? `:` type
static ParseResult parseStoreOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<NamedAttribute, 4> attrs;
  OpAsmParser::OperandType addr, value;
  Type type;
  llvm::SMLoc trailingTypeLoc;

  if (parser.parseOperand(value) || parser.parseComma() ||
      parser.parseOperand(addr) || parser.parseOptionalAttributeDict(attrs) ||
      parser.parseColon() || parser.getCurrentLocation(&trailingTypeLoc) ||
      parser.parseType(type))
    return failure();

  Type elemTy = getLoadStoreElementType(parser, type, trailingTypeLoc);
  if (!elemTy)
    return failure();

  if (parser.resolveOperand(value, elemTy, result.operands) ||
      parser.resolveOperand(addr, type, result.operands))
    return failure();

  result.attributes = attrs;
  return success();
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::CallOp.
//===----------------------------------------------------------------------===//

static void printCallOp(OpAsmPrinter &p, CallOp &op) {
  auto callee = op.callee();
  bool isDirect = callee.hasValue();

  // Print the direct callee if present as a function attribute, or an indirect
  // callee (first operand) otherwise.
  p << op.getOperationName() << ' ';
  if (isDirect)
    p.printSymbolName(callee.getValue());
  else
    p << *op.getOperand(0);

  p << '(';
  p.printOperands(llvm::drop_begin(op.getOperands(), isDirect ? 0 : 1));
  p << ')';

  p.printOptionalAttrDict(op.getAttrs(), {"callee"});

  // Reconstruct the function MLIR function type from operand and result types.
  SmallVector<Type, 1> resultTypes(op.getResultTypes());
  SmallVector<Type, 8> argTypes(
      llvm::drop_begin(op.getOperandTypes(), isDirect ? 0 : 1));

  p << " : " << FunctionType::get(argTypes, resultTypes, op.getContext());
}

// <operation> ::= `llvm.call` (function-id | ssa-use) `(` ssa-use-list `)`
//                 attribute-dict? `:` function-type
static ParseResult parseCallOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<NamedAttribute, 4> attrs;
  SmallVector<OpAsmParser::OperandType, 8> operands;
  Type type;
  SymbolRefAttr funcAttr;
  llvm::SMLoc trailingTypeLoc;

  // Parse an operand list that will, in practice, contain 0 or 1 operand.  In
  // case of an indirect call, there will be 1 operand before `(`.  In case of a
  // direct call, there will be no operands and the parser will stop at the
  // function identifier without complaining.
  if (parser.parseOperandList(operands))
    return failure();
  bool isDirect = operands.empty();

  // Optionally parse a function identifier.
  if (isDirect)
    if (parser.parseAttribute(funcAttr, "callee", attrs))
      return failure();

  if (parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttributeDict(attrs) || parser.parseColon() ||
      parser.getCurrentLocation(&trailingTypeLoc) || parser.parseType(type))
    return failure();

  auto funcType = type.dyn_cast<FunctionType>();
  if (!funcType)
    return parser.emitError(trailingTypeLoc, "expected function type");
  if (isDirect) {
    // Make sure types match.
    if (parser.resolveOperands(operands, funcType.getInputs(),
                               parser.getNameLoc(), result.operands))
      return failure();
    result.addTypes(funcType.getResults());
  } else {
    // Construct the LLVM IR Dialect function type that the first operand
    // should match.
    if (funcType.getNumResults() > 1)
      return parser.emitError(trailingTypeLoc,
                              "expected function with 0 or 1 result");

    Builder &builder = parser.getBuilder();
    auto *llvmDialect =
        builder.getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
    LLVM::LLVMType llvmResultType;
    if (funcType.getNumResults() == 0) {
      llvmResultType = LLVM::LLVMType::getVoidTy(llvmDialect);
    } else {
      llvmResultType = funcType.getResult(0).dyn_cast<LLVM::LLVMType>();
      if (!llvmResultType)
        return parser.emitError(trailingTypeLoc,
                                "expected result to have LLVM type");
    }

    SmallVector<LLVM::LLVMType, 8> argTypes;
    argTypes.reserve(funcType.getNumInputs());
    for (int i = 0, e = funcType.getNumInputs(); i < e; ++i) {
      auto argType = funcType.getInput(i).dyn_cast<LLVM::LLVMType>();
      if (!argType)
        return parser.emitError(trailingTypeLoc,
                                "expected LLVM types as inputs");
      argTypes.push_back(argType);
    }
    auto llvmFuncType = LLVM::LLVMType::getFunctionTy(llvmResultType, argTypes,
                                                      /*isVarArg=*/false);
    auto wrappedFuncType = llvmFuncType.getPointerTo();

    auto funcArguments =
        ArrayRef<OpAsmParser::OperandType>(operands).drop_front();

    // Make sure that the first operand (indirect callee) matches the wrapped
    // LLVM IR function type, and that the types of the other call operands
    // match the types of the function arguments.
    if (parser.resolveOperand(operands[0], wrappedFuncType, result.operands) ||
        parser.resolveOperands(funcArguments, funcType.getInputs(),
                               parser.getNameLoc(), result.operands))
      return failure();

    result.addTypes(llvmResultType);
  }

  result.attributes = attrs;
  return success();
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::ExtractElementOp.
//===----------------------------------------------------------------------===//
// Expects vector to be of wrapped LLVM vector type and position to be of
// wrapped LLVM i32 type.
void LLVM::ExtractElementOp::build(Builder *b, OperationState &result,
                                   Value *vector, Value *position,
                                   ArrayRef<NamedAttribute> attrs) {
  auto wrappedVectorType = vector->getType().cast<LLVM::LLVMType>();
  auto llvmType = wrappedVectorType.getVectorElementType();
  build(b, result, llvmType, vector, position);
  result.addAttributes(attrs);
}

static void printExtractElementOp(OpAsmPrinter &p, ExtractElementOp &op) {
  p << op.getOperationName() << ' ' << *op.vector() << ", " << *op.position();
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.vector()->getType();
}

// <operation> ::= `llvm.extractelement` ssa-use `, ` ssa-use
//                 attribute-dict? `:` type
static ParseResult parseExtractElementOp(OpAsmParser &parser,
                                         OperationState &result) {
  llvm::SMLoc loc;
  OpAsmParser::OperandType vector, position;
  auto *llvmDialect = parser.getBuilder()
                          .getContext()
                          ->getRegisteredDialect<LLVM::LLVMDialect>();
  Type type, i32Type = LLVMType::getInt32Ty(llvmDialect);
  if (parser.getCurrentLocation(&loc) || parser.parseOperand(vector) ||
      parser.parseComma() || parser.parseOperand(position) ||
      parser.parseOptionalAttributeDict(result.attributes) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(vector, type, result.operands) ||
      parser.resolveOperand(position, i32Type, result.operands))
    return failure();
  auto wrappedVectorType = type.dyn_cast<LLVM::LLVMType>();
  if (!wrappedVectorType ||
      !wrappedVectorType.getUnderlyingType()->isVectorTy())
    return parser.emitError(
        loc, "expected LLVM IR dialect vector type for operand #1");
  result.addTypes(wrappedVectorType.getVectorElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::ExtractValueOp.
//===----------------------------------------------------------------------===//

static void printExtractValueOp(OpAsmPrinter &p, ExtractValueOp &op) {
  p << op.getOperationName() << ' ' << *op.container() << op.position();
  p.printOptionalAttrDict(op.getAttrs(), {"position"});
  p << " : " << op.container()->getType();
}

// Extract the type at `position` in the wrapped LLVM IR aggregate type
// `containerType`.  Position is an integer array attribute where each value
// is a zero-based position of the element in the aggregate type.  Return the
// resulting type wrapped in MLIR, or nullptr on error.
static LLVM::LLVMType getInsertExtractValueElementType(OpAsmParser &parser,
                                                       Type containerType,
                                                       Attribute positionAttr,
                                                       llvm::SMLoc attributeLoc,
                                                       llvm::SMLoc typeLoc) {
  auto wrappedContainerType = containerType.dyn_cast<LLVM::LLVMType>();
  if (!wrappedContainerType)
    return parser.emitError(typeLoc, "expected LLVM IR Dialect type"), nullptr;

  auto positionArrayAttr = positionAttr.dyn_cast<ArrayAttr>();
  if (!positionArrayAttr)
    return parser.emitError(attributeLoc, "expected an array attribute"),
           nullptr;

  // Infer the element type from the structure type: iteratively step inside the
  // type by taking the element type, indexed by the position attribute for
  // stuctures.  Check the position index before accessing, it is supposed to be
  // in bounds.
  for (Attribute subAttr : positionArrayAttr) {
    auto positionElementAttr = subAttr.dyn_cast<IntegerAttr>();
    if (!positionElementAttr)
      return parser.emitError(attributeLoc,
                              "expected an array of integer literals"),
             nullptr;
    int position = positionElementAttr.getInt();
    auto *llvmContainerType = wrappedContainerType.getUnderlyingType();
    if (llvmContainerType->isArrayTy()) {
      if (position < 0 || static_cast<unsigned>(position) >=
                              llvmContainerType->getArrayNumElements())
        return parser.emitError(attributeLoc, "position out of bounds"),
               nullptr;
      wrappedContainerType = wrappedContainerType.getArrayElementType();
    } else if (llvmContainerType->isStructTy()) {
      if (position < 0 || static_cast<unsigned>(position) >=
                              llvmContainerType->getStructNumElements())
        return parser.emitError(attributeLoc, "position out of bounds"),
               nullptr;
      wrappedContainerType =
          wrappedContainerType.getStructElementType(position);
    } else {
      return parser.emitError(typeLoc,
                              "expected wrapped LLVM IR structure/array type"),
             nullptr;
    }
  }
  return wrappedContainerType;
}

// <operation> ::= `llvm.extractvalue` ssa-use
//                 `[` integer-literal (`,` integer-literal)* `]`
//                 attribute-dict? `:` type
static ParseResult parseExtractValueOp(OpAsmParser &parser,
                                       OperationState &result) {
  SmallVector<NamedAttribute, 4> attrs;
  OpAsmParser::OperandType container;
  Type containerType;
  Attribute positionAttr;
  llvm::SMLoc attributeLoc, trailingTypeLoc;

  if (parser.parseOperand(container) ||
      parser.getCurrentLocation(&attributeLoc) ||
      parser.parseAttribute(positionAttr, "position", attrs) ||
      parser.parseOptionalAttributeDict(attrs) || parser.parseColon() ||
      parser.getCurrentLocation(&trailingTypeLoc) ||
      parser.parseType(containerType) ||
      parser.resolveOperand(container, containerType, result.operands))
    return failure();

  auto elementType = getInsertExtractValueElementType(
      parser, containerType, positionAttr, attributeLoc, trailingTypeLoc);
  if (!elementType)
    return failure();

  result.attributes = attrs;
  result.addTypes(elementType);
  return success();
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::InsertElementOp.
//===----------------------------------------------------------------------===//

static void printInsertElementOp(OpAsmPrinter &p, InsertElementOp &op) {
  p << op.getOperationName() << ' ' << *op.vector() << ", " << *op.value()
    << ", " << *op.position();
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.vector()->getType();
}

// <operation> ::= `llvm.insertelement` ssa-use `,` ssa-use `,` ssa-use
//                 attribute-dict? `:` type
static ParseResult parseInsertElementOp(OpAsmParser &parser,
                                        OperationState &result) {
  llvm::SMLoc loc;
  OpAsmParser::OperandType vector, value, position;
  auto *llvmDialect = parser.getBuilder()
                          .getContext()
                          ->getRegisteredDialect<LLVM::LLVMDialect>();
  Type vectorType, i32Type = LLVMType::getInt32Ty(llvmDialect);
  if (parser.getCurrentLocation(&loc) || parser.parseOperand(vector) ||
      parser.parseComma() || parser.parseOperand(value) ||
      parser.parseComma() || parser.parseOperand(position) ||
      parser.parseOptionalAttributeDict(result.attributes) ||
      parser.parseColonType(vectorType))
    return failure();

  auto wrappedVectorType = vectorType.dyn_cast<LLVM::LLVMType>();
  if (!wrappedVectorType ||
      !wrappedVectorType.getUnderlyingType()->isVectorTy())
    return parser.emitError(
        loc, "expected LLVM IR dialect vector type for operand #1");
  auto valueType = wrappedVectorType.getVectorElementType();
  if (!valueType)
    return failure();

  if (parser.resolveOperand(vector, vectorType, result.operands) ||
      parser.resolveOperand(value, valueType, result.operands) ||
      parser.resolveOperand(position, i32Type, result.operands))
    return failure();

  result.addTypes(vectorType);
  return success();
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::InsertValueOp.
//===----------------------------------------------------------------------===//

static void printInsertValueOp(OpAsmPrinter &p, InsertValueOp &op) {
  p << op.getOperationName() << ' ' << *op.value() << ", " << *op.container()
    << op.position();
  p.printOptionalAttrDict(op.getAttrs(), {"position"});
  p << " : " << op.container()->getType();
}

// <operation> ::= `llvm.insertvaluevalue` ssa-use `,` ssa-use
//                 `[` integer-literal (`,` integer-literal)* `]`
//                 attribute-dict? `:` type
static ParseResult parseInsertValueOp(OpAsmParser &parser,
                                      OperationState &result) {
  OpAsmParser::OperandType container, value;
  Type containerType;
  Attribute positionAttr;
  llvm::SMLoc attributeLoc, trailingTypeLoc;

  if (parser.parseOperand(value) || parser.parseComma() ||
      parser.parseOperand(container) ||
      parser.getCurrentLocation(&attributeLoc) ||
      parser.parseAttribute(positionAttr, "position", result.attributes) ||
      parser.parseOptionalAttributeDict(result.attributes) ||
      parser.parseColon() || parser.getCurrentLocation(&trailingTypeLoc) ||
      parser.parseType(containerType))
    return failure();

  auto valueType = getInsertExtractValueElementType(
      parser, containerType, positionAttr, attributeLoc, trailingTypeLoc);
  if (!valueType)
    return failure();

  if (parser.resolveOperand(container, containerType, result.operands) ||
      parser.resolveOperand(value, valueType, result.operands))
    return failure();

  result.addTypes(containerType);
  return success();
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::SelectOp.
//===----------------------------------------------------------------------===//

static void printSelectOp(OpAsmPrinter &p, SelectOp &op) {
  p << op.getOperationName() << ' ' << *op.condition() << ", "
    << *op.trueValue() << ", " << *op.falseValue();
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.condition()->getType() << ", " << op.trueValue()->getType();
}

// <operation> ::= `llvm.select` ssa-use `,` ssa-use `,` ssa-use
//                 attribute-dict? `:` type, type
static ParseResult parseSelectOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType condition, trueValue, falseValue;
  Type conditionType, argType;

  if (parser.parseOperand(condition) || parser.parseComma() ||
      parser.parseOperand(trueValue) || parser.parseComma() ||
      parser.parseOperand(falseValue) ||
      parser.parseOptionalAttributeDict(result.attributes) ||
      parser.parseColonType(conditionType) || parser.parseComma() ||
      parser.parseType(argType))
    return failure();

  if (parser.resolveOperand(condition, conditionType, result.operands) ||
      parser.resolveOperand(trueValue, argType, result.operands) ||
      parser.resolveOperand(falseValue, argType, result.operands))
    return failure();

  result.addTypes(argType);
  return success();
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::BrOp.
//===----------------------------------------------------------------------===//

static void printBrOp(OpAsmPrinter &p, BrOp &op) {
  p << op.getOperationName() << ' ';
  p.printSuccessorAndUseList(op.getOperation(), 0);
  p.printOptionalAttrDict(op.getAttrs());
}

// <operation> ::= `llvm.br` bb-id (`[` ssa-use-and-type-list `]`)?
// attribute-dict?
static ParseResult parseBrOp(OpAsmParser &parser, OperationState &result) {
  Block *dest;
  SmallVector<Value *, 4> operands;
  if (parser.parseSuccessorAndUseList(dest, operands) ||
      parser.parseOptionalAttributeDict(result.attributes))
    return failure();

  result.addSuccessor(dest, operands);
  return success();
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::CondBrOp.
//===----------------------------------------------------------------------===//

static void printCondBrOp(OpAsmPrinter &p, CondBrOp &op) {
  p << op.getOperationName() << ' ' << *op.getOperand(0) << ", ";
  p.printSuccessorAndUseList(op.getOperation(), 0);
  p << ", ";
  p.printSuccessorAndUseList(op.getOperation(), 1);
  p.printOptionalAttrDict(op.getAttrs());
}

// <operation> ::= `llvm.cond_br` ssa-use `,`
//                  bb-id (`[` ssa-use-and-type-list `]`)? `,`
//                  bb-id (`[` ssa-use-and-type-list `]`)? attribute-dict?
static ParseResult parseCondBrOp(OpAsmParser &parser, OperationState &result) {
  Block *trueDest;
  Block *falseDest;
  SmallVector<Value *, 4> trueOperands;
  SmallVector<Value *, 4> falseOperands;
  OpAsmParser::OperandType condition;

  Builder &builder = parser.getBuilder();
  auto *llvmDialect =
      builder.getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
  auto i1Type = LLVM::LLVMType::getInt1Ty(llvmDialect);

  if (parser.parseOperand(condition) || parser.parseComma() ||
      parser.parseSuccessorAndUseList(trueDest, trueOperands) ||
      parser.parseComma() ||
      parser.parseSuccessorAndUseList(falseDest, falseOperands) ||
      parser.parseOptionalAttributeDict(result.attributes) ||
      parser.resolveOperand(condition, i1Type, result.operands))
    return failure();

  result.addSuccessor(trueDest, trueOperands);
  result.addSuccessor(falseDest, falseOperands);
  return success();
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::ReturnOp.
//===----------------------------------------------------------------------===//

static void printReturnOp(OpAsmPrinter &p, ReturnOp &op) {
  p << op.getOperationName();
  p.printOptionalAttrDict(op.getAttrs());
  assert(op.getNumOperands() <= 1);

  if (op.getNumOperands() == 0)
    return;

  p << ' ' << *op.getOperand(0) << " : " << op.getOperand(0)->getType();
}

// <operation> ::= `llvm.return` ssa-use-list attribute-dict? `:`
//                 type-list-no-parens
static ParseResult parseReturnOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 1> operands;
  Type type;

  if (parser.parseOperandList(operands) ||
      parser.parseOptionalAttributeDict(result.attributes))
    return failure();
  if (operands.empty())
    return success();

  if (parser.parseColonType(type) ||
      parser.resolveOperand(operands[0], type, result.operands))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::UndefOp.
//===----------------------------------------------------------------------===//

static void printUndefOp(OpAsmPrinter &p, UndefOp &op) {
  p << op.getOperationName();
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.res()->getType();
}

// <operation> ::= `llvm.mlir.undef` attribute-dict? : type
static ParseResult parseUndefOp(OpAsmParser &parser, OperationState &result) {
  Type type;

  if (parser.parseOptionalAttributeDict(result.attributes) ||
      parser.parseColonType(type))
    return failure();

  result.addTypes(type);
  return success();
}

//===----------------------------------------------------------------------===//
// Printer, parser and verifier for LLVM::AddressOfOp.
//===----------------------------------------------------------------------===//

GlobalOp AddressOfOp::getGlobal() {
  auto module = getParentOfType<ModuleOp>();
  assert(module && "unexpected operation outside of a module");
  return module.lookupSymbol<LLVM::GlobalOp>(global_name());
}

static void printAddressOfOp(OpAsmPrinter &p, AddressOfOp op) {
  p << op.getOperationName() << " @" << op.global_name();
  p.printOptionalAttrDict(op.getAttrs(), {"global_name"});
  p << " : " << op.getResult()->getType();
}

static ParseResult parseAddressOfOp(OpAsmParser &parser,
                                    OperationState &result) {
  Attribute symRef;
  Type type;
  if (parser.parseAttribute(symRef, "global_name", result.attributes) ||
      parser.parseOptionalAttributeDict(result.attributes) ||
      parser.parseColonType(type) || parser.addTypeToList(type, result.types))
    return failure();

  if (!symRef.isa<SymbolRefAttr>())
    return parser.emitError(parser.getNameLoc(), "expected symbol reference");
  return success();
}

static LogicalResult verify(AddressOfOp op) {
  auto global = op.getGlobal();
  if (!global)
    return op.emitOpError(
        "must reference a global defined by 'llvm.mlir.global'");

  if (global.getType().getPointerTo() != op.getResult()->getType())
    return op.emitOpError(
        "the type must be a pointer to the type of the referred global");

  return success();
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::ConstantOp.
//===----------------------------------------------------------------------===//

static void printConstantOp(OpAsmPrinter &p, ConstantOp &op) {
  p << op.getOperationName() << '(' << op.value() << ')';
  p.printOptionalAttrDict(op.getAttrs(), {"value"});
  p << " : " << op.res()->getType();
}

// <operation> ::= `llvm.mlir.constant` `(` attribute `)` attribute-list? : type
static ParseResult parseConstantOp(OpAsmParser &parser,
                                   OperationState &result) {
  Attribute valueAttr;
  Type type;

  if (parser.parseLParen() ||
      parser.parseAttribute(valueAttr, "value", result.attributes) ||
      parser.parseRParen() ||
      parser.parseOptionalAttributeDict(result.attributes) ||
      parser.parseColonType(type))
    return failure();

  result.addTypes(type);
  return success();
}

//===----------------------------------------------------------------------===//
// Builder, printer and verifier for LLVM::GlobalOp.
//===----------------------------------------------------------------------===//

void GlobalOp::build(Builder *builder, OperationState &result, LLVMType type,
                     bool isConstant, StringRef name, Attribute value,
                     ArrayRef<NamedAttribute> attrs) {
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder->getStringAttr(name));
  result.addAttribute("type", builder->getTypeAttr(type));
  if (isConstant)
    result.addAttribute("constant", builder->getUnitAttr());
  if (value)
    result.addAttribute("value", value);
  result.attributes.append(attrs.begin(), attrs.end());
}

static void printGlobalOp(OpAsmPrinter &p, GlobalOp op) {
  p << op.getOperationName() << ' ';
  if (op.constant())
    p << "constant ";
  p.printSymbolName(op.sym_name());
  p << '(';
  if (auto value = op.getValueOrNull())
    p.printAttribute(value);
  p << ')';
  p.printOptionalAttrDict(op.getAttrs(), {SymbolTable::getSymbolAttrName(),
                                          "type", "constant", "value"});

  // Print the trailing type unless it's a string global.
  if (op.getValueOrNull().dyn_cast_or_null<StringAttr>())
    return;
  p << " : ";
  p.printType(op.type());
}

// <operation> ::= `llvm.mlir.global` `constant`? `@` identifier
//                 `(` attribute? `)` attribute-list? (`:` type)?
//
// The type can be omitted for string attributes, in which case it will be
// inferred from the value of the string as [strlen(value) x i8].
static ParseResult parseGlobalOp(OpAsmParser &parser, OperationState &result) {
  if (succeeded(parser.parseOptionalKeyword("constant")))
    result.addAttribute("constant", parser.getBuilder().getUnitAttr());

  StringAttr name;
  if (parser.parseSymbolName(name, SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      parser.parseLParen())
    return failure();

  Attribute value;
  if (parser.parseOptionalRParen()) {
    if (parser.parseAttribute(value, "value", result.attributes) ||
        parser.parseRParen())
      return failure();
  }

  SmallVector<Type, 1> types;
  if (parser.parseOptionalAttributeDict(result.attributes) ||
      parser.parseOptionalColonTypeList(types))
    return failure();

  if (types.size() > 1)
    return parser.emitError(parser.getNameLoc(), "expected zero or one type");

  if (types.empty()) {
    if (auto strAttr = value.dyn_cast_or_null<StringAttr>()) {
      MLIRContext *context = parser.getBuilder().getContext();
      auto *dialect = context->getRegisteredDialect<LLVMDialect>();
      auto arrayType = LLVM::LLVMType::getArrayTy(
          LLVM::LLVMType::getInt8Ty(dialect), strAttr.getValue().size());
      types.push_back(arrayType);
    } else {
      return parser.emitError(parser.getNameLoc(),
                              "type can only be omitted for string globals");
    }
  }

  result.addAttribute("type", parser.getBuilder().getTypeAttr(types[0]));
  return success();
}

static LogicalResult verify(GlobalOp op) {
  if (!llvm::PointerType::isValidElementType(op.getType().getUnderlyingType()))
    return op.emitOpError(
        "expects type to be a valid element type for an LLVM pointer");
  if (op.getParentOp() && !isa<ModuleOp>(op.getParentOp()))
    return op.emitOpError("must appear at the module level");

  if (auto strAttr = op.getValueOrNull().dyn_cast_or_null<StringAttr>()) {
    auto type = op.getType();
    if (!type.getUnderlyingType()->isArrayTy() ||
        !type.getArrayElementType().getUnderlyingType()->isIntegerTy(8) ||
        type.getArrayNumElements() != strAttr.getValue().size())
      return op.emitOpError(
          "requires an i8 array type of the length equal to that of the string "
          "attribute");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Printing/parsing for LLVM::ShuffleVectorOp.
//===----------------------------------------------------------------------===//
// Expects vector to be of wrapped LLVM vector type and position to be of
// wrapped LLVM i32 type.
void LLVM::ShuffleVectorOp::build(Builder *b, OperationState &result, Value *v1,
                                  Value *v2, ArrayAttr mask,
                                  ArrayRef<NamedAttribute> attrs) {
  auto wrappedContainerType1 = v1->getType().cast<LLVM::LLVMType>();
  auto vType = LLVMType::getVectorTy(
      wrappedContainerType1.getVectorElementType(), mask.size());
  build(b, result, vType, v1, v2, mask);
  result.addAttributes(attrs);
}

static void printShuffleVectorOp(OpAsmPrinter &p, ShuffleVectorOp &op) {
  p << op.getOperationName() << ' ' << *op.v1() << ", " << *op.v2() << " "
    << op.mask();
  p.printOptionalAttrDict(op.getAttrs(), {"mask"});
  p << " : " << op.v1()->getType() << ", " << op.v2()->getType();
}

// <operation> ::= `llvm.shufflevector` ssa-use `, ` ssa-use
//                 `[` integer-literal (`,` integer-literal)* `]`
//                 attribute-dict? `:` type
static ParseResult parseShuffleVectorOp(OpAsmParser &parser,
                                        OperationState &result) {
  llvm::SMLoc loc;
  SmallVector<NamedAttribute, 4> attrs;
  OpAsmParser::OperandType v1, v2;
  Attribute maskAttr;
  Type typeV1, typeV2;
  if (parser.getCurrentLocation(&loc) || parser.parseOperand(v1) ||
      parser.parseComma() || parser.parseOperand(v2) ||
      parser.parseAttribute(maskAttr, "mask", attrs) ||
      parser.parseOptionalAttributeDict(attrs) ||
      parser.parseColonType(typeV1) || parser.parseComma() ||
      parser.parseType(typeV2) ||
      parser.resolveOperand(v1, typeV1, result.operands) ||
      parser.resolveOperand(v2, typeV2, result.operands))
    return failure();
  auto wrappedContainerType1 = typeV1.dyn_cast<LLVM::LLVMType>();
  if (!wrappedContainerType1 ||
      !wrappedContainerType1.getUnderlyingType()->isVectorTy())
    return parser.emitError(
        loc, "expected LLVM IR dialect vector type for operand #1");
  auto vType =
      LLVMType::getVectorTy(wrappedContainerType1.getVectorElementType(),
                            maskAttr.cast<ArrayAttr>().size());
  result.attributes = attrs;
  result.addTypes(vType);
  return success();
}

//===----------------------------------------------------------------------===//
// Builder, printer and verifier for LLVM::LLVMFuncOp.
//===----------------------------------------------------------------------===//

void LLVMFuncOp::build(Builder *builder, OperationState &result, StringRef name,
                       LLVMType type, ArrayRef<NamedAttribute> attrs,
                       ArrayRef<NamedAttributeList> argAttrs) {
  result.addRegion();
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder->getStringAttr(name));
  result.addAttribute("type", builder->getTypeAttr(type));
  result.attributes.append(attrs.begin(), attrs.end());
  if (argAttrs.empty())
    return;

  unsigned numInputs = type.getUnderlyingType()->getFunctionNumParams();
  assert(numInputs == argAttrs.size() &&
         "expected as many argument attribute lists as arguments");
  SmallString<8> argAttrName;
  for (unsigned i = 0; i < numInputs; ++i)
    if (auto argDict = argAttrs[i].getDictionary())
      result.addAttribute(getArgAttrName(i, argAttrName), argDict);
}

// Build an LLVM function type from the given lists of input and output types.
// Returns a null type if any of the types provided are non-LLVM types, or if
// there is more than one output type.
static Type buildLLVMFunctionType(Builder &b, ArrayRef<Type> inputs,
                                  ArrayRef<Type> outputs,
                                  impl::VariadicFlag variadicFlag,
                                  std::string &errorMessage) {
  if (outputs.size() > 1) {
    errorMessage = "expected zero or one function result";
    return {};
  }

  // Convert inputs to LLVM types, exit early on error.
  SmallVector<LLVMType, 4> llvmInputs;
  for (auto t : inputs) {
    auto llvmTy = t.dyn_cast<LLVMType>();
    if (!llvmTy) {
      errorMessage = "expected LLVM type for function arguments";
      return {};
    }
    llvmInputs.push_back(llvmTy);
  }

  // Get the dialect from the input type, if any exist.  Look it up in the
  // context otherwise.
  LLVMDialect *dialect =
      llvmInputs.empty() ? b.getContext()->getRegisteredDialect<LLVMDialect>()
                         : &llvmInputs.front().getDialect();

  // No output is denoted as "void" in LLVM type system.
  LLVMType llvmOutput = outputs.empty() ? LLVMType::getVoidTy(dialect)
                                        : outputs.front().dyn_cast<LLVMType>();
  if (!llvmOutput) {
    errorMessage = "expected LLVM type for function results";
    return {};
  }
  return LLVMType::getFunctionTy(llvmOutput, llvmInputs,
                                 variadicFlag.isVariadic());
}

// Print the LLVMFuncOp.  Collects argument and result types and passes them
// to the trait printer.  Drops "void" result since it cannot be parsed back.
static void printLLVMFuncOp(OpAsmPrinter &p, LLVMFuncOp op) {
  LLVMType fnType = op.getType();
  SmallVector<Type, 8> argTypes;
  SmallVector<Type, 1> resTypes;
  argTypes.reserve(fnType.getFunctionNumParams());
  for (unsigned i = 0, e = fnType.getFunctionNumParams(); i < e; ++i)
    argTypes.push_back(fnType.getFunctionParamType(i));

  LLVMType returnType = fnType.getFunctionResultType();
  if (!returnType.getUnderlyingType()->isVoidTy())
    resTypes.push_back(returnType);

  impl::printFunctionLikeOp(p, op, argTypes, op.isVarArg(), resTypes);
}

// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
// attribute is present.  This can check for preconditions of the
// getNumArguments hook not failing.
LogicalResult LLVMFuncOp::verifyType() {
  auto llvmType = getTypeAttr().getValue().dyn_cast_or_null<LLVMType>();
  if (!llvmType || !llvmType.getUnderlyingType()->isFunctionTy())
    return emitOpError("requires '" + getTypeAttrName() +
                       "' attribute of wrapped LLVM function type");

  return success();
}

// Hook for OpTrait::FunctionLike, returns the number of function arguments.
// Depends on the type attribute being correct as checked by verifyType
unsigned LLVMFuncOp::getNumFuncArguments() {
  return getType().getUnderlyingType()->getFunctionNumParams();
}

static LogicalResult verify(LLVMFuncOp op) {
  if (op.isExternal())
    return success();

  if (op.isVarArg())
    return op.emitOpError("only external functions can be variadic");

  auto *funcType = cast<llvm::FunctionType>(op.getType().getUnderlyingType());
  unsigned numArguments = funcType->getNumParams();
  Block &entryBlock = op.front();
  for (unsigned i = 0; i < numArguments; ++i) {
    Type argType = entryBlock.getArgument(i)->getType();
    auto argLLVMType = argType.dyn_cast<LLVMType>();
    if (!argLLVMType)
      return op.emitOpError("entry block argument #")
             << i << " is not of LLVM type";
    if (funcType->getParamType(i) != argLLVMType.getUnderlyingType())
      return op.emitOpError("the type of entry block argument #")
             << i << " does not match the function signature";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Printing, parsing and verification for LLVM::NullOp.
//===----------------------------------------------------------------------===//

static void printNullOp(OpAsmPrinter &p, LLVM::NullOp op) {
  p << NullOp::getOperationName();
  p.printOptionalAttrDict(op.getAttrs());
  p << " : ";
  p.printType(op.getType());
}

// <operation> = `llvm.mlir.null` : type
static ParseResult parseNullOp(OpAsmParser &parser, OperationState &result) {
  Type type;
  return failure(parser.parseOptionalAttributeDict(result.attributes) ||
                 parser.parseColonType(type) ||
                 parser.addTypeToList(type, result.types));
}

// Only LLVM pointer types are supported.
static LogicalResult verify(LLVM::NullOp op) {
  auto llvmType = op.getType().dyn_cast<LLVM::LLVMType>();
  if (!llvmType || !llvmType.isPointerTy())
    return op.emitOpError("expected LLVM IR pointer type");
  return success();
}

//===----------------------------------------------------------------------===//
// LLVMDialect initialization, type parsing, and registration.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace LLVM {
namespace detail {
struct LLVMDialectImpl {
  LLVMDialectImpl() : module("LLVMDialectModule", llvmContext) {}

  llvm::LLVMContext llvmContext;
  llvm::Module module;

  /// A set of LLVMTypes that are cached on construction to avoid any lookups or
  /// locking.
  LLVMType int1Ty, int8Ty, int16Ty, int32Ty, int64Ty, int128Ty;
  LLVMType doubleTy, floatTy, halfTy, fp128Ty, x86_fp80Ty;
  LLVMType voidTy;

  /// A smart mutex to lock access to the llvm context. Unlike MLIR, LLVM is not
  /// multi-threaded and requires locked access to prevent race conditions.
  llvm::sys::SmartMutex<true> mutex;
};
} // end namespace detail
} // end namespace LLVM
} // end namespace mlir

LLVMDialect::LLVMDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context),
      impl(new detail::LLVMDialectImpl()) {
  addTypes<LLVMType>();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LLVMIR/LLVMOps.cpp.inc"
      >();

  // Support unknown operations because not all LLVM operations are registered.
  allowUnknownOperations();

  // Cache some of the common LLVM types to avoid the need for lookups/locking.
  auto &llvmContext = impl->llvmContext;
  /// Integer Types.
  impl->int1Ty = LLVMType::get(context, llvm::Type::getInt1Ty(llvmContext));
  impl->int8Ty = LLVMType::get(context, llvm::Type::getInt8Ty(llvmContext));
  impl->int16Ty = LLVMType::get(context, llvm::Type::getInt16Ty(llvmContext));
  impl->int32Ty = LLVMType::get(context, llvm::Type::getInt32Ty(llvmContext));
  impl->int64Ty = LLVMType::get(context, llvm::Type::getInt64Ty(llvmContext));
  impl->int128Ty = LLVMType::get(context, llvm::Type::getInt128Ty(llvmContext));
  /// Float Types.
  impl->doubleTy = LLVMType::get(context, llvm::Type::getDoubleTy(llvmContext));
  impl->floatTy = LLVMType::get(context, llvm::Type::getFloatTy(llvmContext));
  impl->halfTy = LLVMType::get(context, llvm::Type::getHalfTy(llvmContext));
  impl->fp128Ty = LLVMType::get(context, llvm::Type::getFP128Ty(llvmContext));
  impl->x86_fp80Ty = LLVMType::get(context, llvm::Type::getX86_FP80Ty(llvmContext));
  /// Other Types.
  impl->voidTy = LLVMType::get(context, llvm::Type::getVoidTy(llvmContext));
}

LLVMDialect::~LLVMDialect() {}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMOps.cpp.inc"

llvm::LLVMContext &LLVMDialect::getLLVMContext() { return impl->llvmContext; }
llvm::Module &LLVMDialect::getLLVMModule() { return impl->module; }

/// Parse a type registered to this dialect.
Type LLVMDialect::parseType(StringRef tyData, Location loc) const {
  // LLVM is not thread-safe, so lock access to it.
  llvm::sys::SmartScopedLock<true> lock(impl->mutex);

  llvm::SMDiagnostic errorMessage;
  llvm::Type *type = llvm::parseType(tyData, errorMessage, impl->module);
  if (!type)
    return (emitError(loc, errorMessage.getMessage()), nullptr);
  return LLVMType::get(getContext(), type);
}

/// Print a type registered to this dialect.
void LLVMDialect::printType(Type type, raw_ostream &os) const {
  auto llvmType = type.dyn_cast<LLVMType>();
  assert(llvmType && "printing wrong type");
  assert(llvmType.getUnderlyingType() && "no underlying LLVM type");
  llvmType.getUnderlyingType()->print(os);
}

/// Verify LLVMIR function argument attributes.
LogicalResult LLVMDialect::verifyRegionArgAttribute(Operation *op,
                                                    unsigned regionIdx,
                                                    unsigned argIdx,
                                                    NamedAttribute argAttr) {
  // Check that llvm.noalias is a boolean attribute.
  if (argAttr.first == "llvm.noalias" && !argAttr.second.isa<BoolAttr>())
    return op->emitError()
           << "llvm.noalias argument attribute of non boolean type";
  return success();
}

static DialectRegistration<LLVMDialect> llvmDialect;

//===----------------------------------------------------------------------===//
// LLVMType.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace LLVM {
namespace detail {
struct LLVMTypeStorage : public ::mlir::TypeStorage {
  LLVMTypeStorage(llvm::Type *ty) : underlyingType(ty) {}

  // LLVM types are pointer-unique.
  using KeyTy = llvm::Type *;
  bool operator==(const KeyTy &key) const { return key == underlyingType; }

  static LLVMTypeStorage *construct(TypeStorageAllocator &allocator,
                                    llvm::Type *ty) {
    return new (allocator.allocate<LLVMTypeStorage>()) LLVMTypeStorage(ty);
  }

  llvm::Type *underlyingType;
};
} // end namespace detail
} // end namespace LLVM
} // end namespace mlir

LLVMType LLVMType::get(MLIRContext *context, llvm::Type *llvmType) {
  return Base::get(context, FIRST_LLVM_TYPE, llvmType);
}

/// Get an LLVMType with an llvm type that may cause changes to the underlying
/// llvm context when constructed.
LLVMType LLVMType::getLocked(LLVMDialect *dialect,
                             llvm::function_ref<llvm::Type *()> typeBuilder) {
  // Lock access to the llvm context and build the type.
  llvm::sys::SmartScopedLock<true> lock(dialect->impl->mutex);
  return get(dialect->getContext(), typeBuilder());
}

LLVMDialect &LLVMType::getDialect() {
  return static_cast<LLVMDialect &>(Type::getDialect());
}

llvm::Type *LLVMType::getUnderlyingType() const {
  return getImpl()->underlyingType;
}

/// Array type utilities.
LLVMType LLVMType::getArrayElementType() {
  return get(getContext(), getUnderlyingType()->getArrayElementType());
}
unsigned LLVMType::getArrayNumElements() {
  return getUnderlyingType()->getArrayNumElements();
}
bool LLVMType::isArrayTy() { return getUnderlyingType()->isArrayTy(); }

/// Vector type utilities.
LLVMType LLVMType::getVectorElementType() {
  return get(getContext(), getUnderlyingType()->getVectorElementType());
}
bool LLVMType::isVectorTy() { return getUnderlyingType()->isVectorTy(); }

/// Function type utilities.
LLVMType LLVMType::getFunctionParamType(unsigned argIdx) {
  return get(getContext(), getUnderlyingType()->getFunctionParamType(argIdx));
}
unsigned LLVMType::getFunctionNumParams() {
  return getUnderlyingType()->getFunctionNumParams();
}
LLVMType LLVMType::getFunctionResultType() {
  return get(
      getContext(),
      llvm::cast<llvm::FunctionType>(getUnderlyingType())->getReturnType());
}
bool LLVMType::isFunctionTy() { return getUnderlyingType()->isFunctionTy(); }

/// Pointer type utilities.
LLVMType LLVMType::getPointerTo(unsigned addrSpace) {
  // Lock access to the dialect as this may modify the LLVM context.
  return getLocked(&getDialect(), [=] {
    return getUnderlyingType()->getPointerTo(addrSpace);
  });
}
LLVMType LLVMType::getPointerElementTy() {
  return get(getContext(), getUnderlyingType()->getPointerElementType());
}
bool LLVMType::isPointerTy() { return getUnderlyingType()->isPointerTy(); }

/// Struct type utilities.
LLVMType LLVMType::getStructElementType(unsigned i) {
  return get(getContext(), getUnderlyingType()->getStructElementType(i));
}
bool LLVMType::isStructTy() { return getUnderlyingType()->isStructTy(); }

/// Utilities used to generate floating point types.
LLVMType LLVMType::getDoubleTy(LLVMDialect *dialect) {
  return dialect->impl->doubleTy;
}
LLVMType LLVMType::getFloatTy(LLVMDialect *dialect) {
  return dialect->impl->floatTy;
}
LLVMType LLVMType::getHalfTy(LLVMDialect *dialect) {
  return dialect->impl->halfTy;
}
LLVMType LLVMType::getFP128Ty(LLVMDialect *dialect) {
  return dialect->impl->fp128Ty;
}
LLVMType LLVMType::getX86_FP80Ty(LLVMDialect *dialect) {
  return dialect->impl->x86_fp80Ty;
}

/// Utilities used to generate integer types.
LLVMType LLVMType::getIntNTy(LLVMDialect *dialect, unsigned numBits) {
  switch (numBits) {
  case 1:
    return dialect->impl->int1Ty;
  case 8:
    return dialect->impl->int8Ty;
  case 16:
    return dialect->impl->int16Ty;
  case 32:
    return dialect->impl->int32Ty;
  case 64:
    return dialect->impl->int64Ty;
  case 128:
    return dialect->impl->int128Ty;
  default:
    break;
  }

  // Lock access to the dialect as this may modify the LLVM context.
  return getLocked(dialect, [=] {
    return llvm::Type::getIntNTy(dialect->getLLVMContext(), numBits);
  });
}

/// Utilities used to generate other miscellaneous types.
LLVMType LLVMType::getArrayTy(LLVMType elementType, uint64_t numElements) {
  // Lock access to the dialect as this may modify the LLVM context.
  return getLocked(&elementType.getDialect(), [=] {
    return llvm::ArrayType::get(elementType.getUnderlyingType(), numElements);
  });
}
LLVMType LLVMType::getFunctionTy(LLVMType result, ArrayRef<LLVMType> params,
                                 bool isVarArg) {
  SmallVector<llvm::Type *, 8> llvmParams;
  for (auto param : params)
    llvmParams.push_back(param.getUnderlyingType());

  // Lock access to the dialect as this may modify the LLVM context.
  return getLocked(&result.getDialect(), [=] {
    return llvm::FunctionType::get(result.getUnderlyingType(), llvmParams,
                                   isVarArg);
  });
}
LLVMType LLVMType::getStructTy(LLVMDialect *dialect,
                               ArrayRef<LLVMType> elements, bool isPacked) {
  SmallVector<llvm::Type *, 8> llvmElements;
  for (auto elt : elements)
    llvmElements.push_back(elt.getUnderlyingType());

  // Lock access to the dialect as this may modify the LLVM context.
  return getLocked(dialect, [=] {
    return llvm::StructType::get(dialect->getLLVMContext(), llvmElements,
                                 isPacked);
  });
}
LLVMType LLVMType::getVectorTy(LLVMType elementType, unsigned numElements) {
  // Lock access to the dialect as this may modify the LLVM context.
  return getLocked(&elementType.getDialect(), [=] {
    return llvm::VectorType::get(elementType.getUnderlyingType(), numElements);
  });
}
LLVMType LLVMType::getVoidTy(LLVMDialect *dialect) {
  return dialect->impl->voidTy;
}

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

Value *mlir::LLVM::createGlobalString(Location loc, OpBuilder &builder,
                                      StringRef name, StringRef value,
                                      LLVM::LLVMDialect *llvmDialect) {
  assert(builder.getInsertionBlock() &&
         builder.getInsertionBlock()->getParentOp() &&
         "expected builder to point to a block constained in an op");
  auto module =
      builder.getInsertionBlock()->getParentOp()->getParentOfType<ModuleOp>();
  assert(module && "builder points to an op outside of a module");

  // Create the global at the entry of the module.
  OpBuilder moduleBuilder(module.getBodyRegion());
  auto type = LLVM::LLVMType::getArrayTy(LLVM::LLVMType::getInt8Ty(llvmDialect),
                                         value.size());
  auto global = moduleBuilder.create<LLVM::GlobalOp>(
      loc, type, /*isConstant=*/true, name, builder.getStringAttr(value));

  // Get the pointer to the first character in the global string.
  Value *globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
  Value *cst0 = builder.create<LLVM::ConstantOp>(
      loc, LLVM::LLVMType::getInt64Ty(llvmDialect),
      builder.getIntegerAttr(builder.getIndexType(), 0));
  return builder.create<LLVM::GEPOp>(
      loc, LLVM::LLVMType::getInt8PtrTy(llvmDialect), globalPtr,
      ArrayRef<Value *>({cst0, cst0}));
}
