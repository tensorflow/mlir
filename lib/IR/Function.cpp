//===- Function.cpp - MLIR Function Classes -------------------------------===//
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

#include "mlir/IR/Function.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Function Operation.
//===----------------------------------------------------------------------===//

FuncOp FuncOp::create(Location location, StringRef name, FunctionType type,
                      ArrayRef<NamedAttribute> attrs) {
  OperationState state(location, "func");
  Builder builder(location->getContext());
  FuncOp::build(&builder, &state, name, type, attrs);
  return llvm::cast<FuncOp>(Operation::create(state));
}
FuncOp FuncOp::create(Location location, StringRef name, FunctionType type,
                      llvm::iterator_range<dialect_attr_iterator> attrs) {
  SmallVector<NamedAttribute, 8> attrRef(attrs);
  return create(location, name, type, llvm::makeArrayRef(attrRef));
}
FuncOp FuncOp::create(Location location, StringRef name, FunctionType type,
                      ArrayRef<NamedAttribute> attrs,
                      ArrayRef<NamedAttributeList> argAttrs) {
  FuncOp func = create(location, name, type, attrs);
  func.setAllArgAttrs(argAttrs);
  return func;
}

void FuncOp::build(Builder *builder, OperationState *result, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs) {
  result->addAttribute(SymbolTable::getSymbolAttrName(),
                       builder->getStringAttr(name));
  result->addAttribute(getTypeAttrName(), builder->getTypeAttr(type));
  result->attributes.append(attrs.begin(), attrs.end());
  result->addRegion();
}

void FuncOp::build(Builder *builder, OperationState *result, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs,
                   ArrayRef<NamedAttributeList> argAttrs) {
  build(builder, result, name, type, attrs);
  assert(type.getNumInputs() == argAttrs.size());
  SmallString<8> argAttrName;
  for (unsigned i = 0, e = type.getNumInputs(); i != e; ++i)
    if (auto argDict = argAttrs[i].getDictionary())
      result->addAttribute(getArgAttrName(i, argAttrName), argDict);
}

/// Parsing/Printing methods.

ParseResult FuncOp::parse(OpAsmParser *parser, OperationState *result) {
  return impl::parseFunctionLikeOp(
      parser, result,
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results) {
        return builder.getFunctionType(argTypes, results);
      });
}

void FuncOp::print(OpAsmPrinter *p) {
  FunctionType fnType = getType();
  impl::printFunctionLikeOp(p, *this, fnType.getInputs(), fnType.getResults());
}

LogicalResult FuncOp::verify() {
  // If this function is external there is nothing to do.
  if (isExternal())
    return success();

  // Verify that the argument list of the function and the arg list of the entry
  // block line up.  The trait already verified that the number of arguments is
  // the same between the signature and the block.
  auto fnInputTypes = getType().getInputs();
  Block &entryBlock = front();
  for (unsigned i = 0, e = entryBlock.getNumArguments(); i != e; ++i)
    if (fnInputTypes[i] != entryBlock.getArgument(i)->getType())
      return emitOpError("type of entry block argument #")
             << i << '(' << entryBlock.getArgument(i)->getType()
             << ") must match the type of the corresponding argument in "
             << "function signature(" << fnInputTypes[i] << ')';

  return success();
}

/// Add an entry block to an empty function, and set up the block arguments
/// to match the signature of the function.
void FuncOp::addEntryBlock() {
  assert(empty() && "function already has an entry block");
  auto *entry = new Block();
  push_back(entry);
  entry->addArguments(getType().getInputs());
}

/// Clone the internal blocks from this function into dest and all attributes
/// from this function to dest.
void FuncOp::cloneInto(FuncOp dest, BlockAndValueMapping &mapper) {
  // Add the attributes of this function to dest.
  llvm::MapVector<Identifier, Attribute> newAttrs;
  for (auto &attr : dest.getAttrs())
    newAttrs.insert(attr);
  for (auto &attr : getAttrs())
    newAttrs.insert(attr);
  dest.getOperation()->setAttrs(
      DictionaryAttr::get(newAttrs.takeVector(), getContext()));

  // Clone the body.
  getBody().cloneInto(&dest.getBody(), mapper);
}

/// Create a deep copy of this function and all of its blocks, remapping
/// any operands that use values outside of the function using the map that is
/// provided (leaving them alone if no entry is present). Replaces references
/// to cloned sub-values with the corresponding value that is copied, and adds
/// those mappings to the mapper.
FuncOp FuncOp::clone(BlockAndValueMapping &mapper) {
  FunctionType newType = getType();

  // If the function has a body, then the user might be deleting arguments to
  // the function by specifying them in the mapper. If so, we don't add the
  // argument to the input type vector.
  bool isExternalFn = isExternal();
  if (!isExternalFn) {
    SmallVector<Type, 4> inputTypes;
    inputTypes.reserve(newType.getNumInputs());
    for (unsigned i = 0, e = getNumArguments(); i != e; ++i)
      if (!mapper.contains(getArgument(i)))
        inputTypes.push_back(newType.getInput(i));
    newType = FunctionType::get(inputTypes, newType.getResults(), getContext());
  }

  // Create the new function.
  FuncOp newFunc = llvm::cast<FuncOp>(getOperation()->cloneWithoutRegions());
  newFunc.setType(newType);

  /// Set the argument attributes for arguments that aren't being replaced.
  for (unsigned i = 0, e = getNumArguments(), destI = 0; i != e; ++i)
    if (isExternalFn || !mapper.contains(getArgument(i)))
      newFunc.setArgAttrs(destI++, getArgAttrs(i));

  /// Clone the current function into the new one and return it.
  cloneInto(newFunc, mapper);
  return newFunc;
}
FuncOp FuncOp::clone() {
  BlockAndValueMapping mapper;
  return clone(mapper);
}
