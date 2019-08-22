//===- Builders.cpp - Helpers for constructing MLIR Classes ---------------===//
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

#include "mlir/IR/Builders.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/Functional.h"
using namespace mlir;

Builder::Builder(ModuleOp module) : context(module.getContext()) {}

Identifier Builder::getIdentifier(StringRef str) {
  return Identifier::get(str, context);
}

//===----------------------------------------------------------------------===//
// Locations.
//===----------------------------------------------------------------------===//

Location Builder::getUnknownLoc() { return UnknownLoc::get(context); }

Location Builder::getFileLineColLoc(Identifier filename, unsigned line,
                                    unsigned column) {
  return FileLineColLoc::get(filename, line, column, context);
}

Location Builder::getFusedLoc(ArrayRef<Location> locs, Attribute metadata) {
  return FusedLoc::get(locs, metadata, context);
}

//===----------------------------------------------------------------------===//
// Types.
//===----------------------------------------------------------------------===//

FloatType Builder::getBF16Type() { return FloatType::getBF16(context); }

FloatType Builder::getF16Type() { return FloatType::getF16(context); }

FloatType Builder::getF32Type() { return FloatType::getF32(context); }

FloatType Builder::getF64Type() { return FloatType::getF64(context); }

IndexType Builder::getIndexType() { return IndexType::get(context); }

IntegerType Builder::getI1Type() { return IntegerType::get(1, context); }

IntegerType Builder::getIntegerType(unsigned width) {
  return IntegerType::get(width, context);
}

FunctionType Builder::getFunctionType(ArrayRef<Type> inputs,
                                      ArrayRef<Type> results) {
  return FunctionType::get(inputs, results, context);
}

MemRefType Builder::getMemRefType(ArrayRef<int64_t> shape, Type elementType,
                                  ArrayRef<AffineMap> affineMapComposition,
                                  unsigned memorySpace) {
  return MemRefType::get(shape, elementType, affineMapComposition, memorySpace);
}

VectorType Builder::getVectorType(ArrayRef<int64_t> shape, Type elementType) {
  return VectorType::get(shape, elementType);
}

RankedTensorType Builder::getTensorType(ArrayRef<int64_t> shape,
                                        Type elementType) {
  return RankedTensorType::get(shape, elementType);
}

UnrankedTensorType Builder::getTensorType(Type elementType) {
  return UnrankedTensorType::get(elementType);
}

TupleType Builder::getTupleType(ArrayRef<Type> elementTypes) {
  return TupleType::get(elementTypes, context);
}

NoneType Builder::getNoneType() { return NoneType::get(context); }

//===----------------------------------------------------------------------===//
// Attributes.
//===----------------------------------------------------------------------===//

NamedAttribute Builder::getNamedAttr(StringRef name, Attribute val) {
  return NamedAttribute(getIdentifier(name), val);
}

UnitAttr Builder::getUnitAttr() { return UnitAttr::get(context); }

BoolAttr Builder::getBoolAttr(bool value) {
  return BoolAttr::get(value, context);
}

DictionaryAttr Builder::getDictionaryAttr(ArrayRef<NamedAttribute> value) {
  return DictionaryAttr::get(value, context);
}

IntegerAttr Builder::getI64IntegerAttr(int64_t value) {
  return IntegerAttr::get(getIntegerType(64), APInt(64, value));
}

IntegerAttr Builder::getI32IntegerAttr(int32_t value) {
  return IntegerAttr::get(getIntegerType(32), APInt(32, value));
}

IntegerAttr Builder::getIntegerAttr(Type type, int64_t value) {
  if (type.isIndex())
    return IntegerAttr::get(type, APInt(64, value));
  return IntegerAttr::get(type, APInt(type.getIntOrFloatBitWidth(), value));
}

IntegerAttr Builder::getIntegerAttr(Type type, const APInt &value) {
  return IntegerAttr::get(type, value);
}

FloatAttr Builder::getF64FloatAttr(double value) {
  return FloatAttr::get(getF64Type(), APFloat(value));
}

FloatAttr Builder::getF32FloatAttr(float value) {
  return FloatAttr::get(getF32Type(), APFloat(value));
}

FloatAttr Builder::getF16FloatAttr(float value) {
  return FloatAttr::get(getF16Type(), value);
}

FloatAttr Builder::getFloatAttr(Type type, double value) {
  return FloatAttr::get(type, value);
}

FloatAttr Builder::getFloatAttr(Type type, const APFloat &value) {
  return FloatAttr::get(type, value);
}

StringAttr Builder::getStringAttr(StringRef bytes) {
  return StringAttr::get(bytes, context);
}

StringAttr Builder::getStringAttr(StringRef bytes, Type type) {
  return StringAttr::get(bytes, type);
}

ArrayAttr Builder::getArrayAttr(ArrayRef<Attribute> value) {
  return ArrayAttr::get(value, context);
}

AffineMapAttr Builder::getAffineMapAttr(AffineMap map) {
  return AffineMapAttr::get(map);
}

IntegerSetAttr Builder::getIntegerSetAttr(IntegerSet set) {
  return IntegerSetAttr::get(set);
}

TypeAttr Builder::getTypeAttr(Type type) { return TypeAttr::get(type); }

SymbolRefAttr Builder::getSymbolRefAttr(Operation *value) {
  auto symName =
      value->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
  assert(symName && "value does not have a valid symbol name");
  return getSymbolRefAttr(symName.getValue());
}
SymbolRefAttr Builder::getSymbolRefAttr(StringRef value) {
  return SymbolRefAttr::get(value, getContext());
}

ElementsAttr Builder::getDenseElementsAttr(ShapedType type,
                                           ArrayRef<Attribute> values) {
  return DenseElementsAttr::get(type, values);
}

ElementsAttr Builder::getDenseIntElementsAttr(ShapedType type,
                                              ArrayRef<int64_t> values) {
  return DenseIntElementsAttr::get(type, values);
}

ElementsAttr Builder::getSparseElementsAttr(ShapedType type,
                                            DenseIntElementsAttr indices,
                                            DenseElementsAttr values) {
  return SparseElementsAttr::get(type, indices, values);
}

ElementsAttr Builder::getOpaqueElementsAttr(Dialect *dialect, ShapedType type,
                                            StringRef bytes) {
  return OpaqueElementsAttr::get(dialect, type, bytes);
}

ArrayAttr Builder::getI32ArrayAttr(ArrayRef<int32_t> values) {
  auto attrs = functional::map(
      [this](int32_t v) -> Attribute { return getI32IntegerAttr(v); }, values);
  return getArrayAttr(attrs);
}

ArrayAttr Builder::getI64ArrayAttr(ArrayRef<int64_t> values) {
  auto attrs = functional::map(
      [this](int64_t v) -> Attribute { return getI64IntegerAttr(v); }, values);
  return getArrayAttr(attrs);
}

ArrayAttr Builder::getIndexArrayAttr(ArrayRef<int64_t> values) {
  auto attrs = functional::map(
      [this](int64_t v) -> Attribute {
        return getIntegerAttr(IndexType::get(getContext()), v);
      },
      values);
  return getArrayAttr(attrs);
}

ArrayAttr Builder::getF32ArrayAttr(ArrayRef<float> values) {
  auto attrs = functional::map(
      [this](float v) -> Attribute { return getF32FloatAttr(v); }, values);
  return getArrayAttr(attrs);
}

ArrayAttr Builder::getF64ArrayAttr(ArrayRef<double> values) {
  auto attrs = functional::map(
      [this](double v) -> Attribute { return getF64FloatAttr(v); }, values);
  return getArrayAttr(attrs);
}

ArrayAttr Builder::getStrArrayAttr(ArrayRef<StringRef> values) {
  auto attrs = functional::map(
      [this](StringRef v) -> Attribute { return getStringAttr(v); }, values);
  return getArrayAttr(attrs);
}

ArrayAttr Builder::getAffineMapArrayAttr(ArrayRef<AffineMap> values) {
  auto attrs = functional::map(
      [this](AffineMap v) -> Attribute { return getAffineMapAttr(v); }, values);
  return getArrayAttr(attrs);
}

Attribute Builder::getZeroAttr(Type type) {
  switch (type.getKind()) {
  case StandardTypes::F16:
    return getF16FloatAttr(0);
  case StandardTypes::F32:
    return getF32FloatAttr(0);
  case StandardTypes::F64:
    return getF64FloatAttr(0);
  case StandardTypes::Integer: {
    auto width = type.cast<IntegerType>().getWidth();
    if (width == 1)
      return getBoolAttr(false);
    return getIntegerAttr(type, APInt(width, 0));
  }
  case StandardTypes::Vector:
  case StandardTypes::RankedTensor: {
    auto vtType = type.cast<ShapedType>();
    auto element = getZeroAttr(vtType.getElementType());
    if (!element)
      return {};
    return getDenseElementsAttr(vtType, element);
  }
  default:
    break;
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Affine Expressions, Affine Maps, and Integet Sets.
//===----------------------------------------------------------------------===//

AffineMap Builder::getAffineMap(unsigned dimCount, unsigned symbolCount,
                                ArrayRef<AffineExpr> results) {
  return AffineMap::get(dimCount, symbolCount, results);
}

AffineExpr Builder::getAffineDimExpr(unsigned position) {
  return mlir::getAffineDimExpr(position, context);
}

AffineExpr Builder::getAffineSymbolExpr(unsigned position) {
  return mlir::getAffineSymbolExpr(position, context);
}

AffineExpr Builder::getAffineConstantExpr(int64_t constant) {
  return mlir::getAffineConstantExpr(constant, context);
}

IntegerSet Builder::getIntegerSet(unsigned dimCount, unsigned symbolCount,
                                  ArrayRef<AffineExpr> constraints,
                                  ArrayRef<bool> isEq) {
  return IntegerSet::get(dimCount, symbolCount, constraints, isEq);
}

AffineMap Builder::getEmptyAffineMap() { return AffineMap::get(context); }

AffineMap Builder::getConstantAffineMap(int64_t val) {
  return AffineMap::get(/*dimCount=*/0, /*symbolCount=*/0,
                        {getAffineConstantExpr(val)});
}

AffineMap Builder::getDimIdentityMap() {
  return AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                        {getAffineDimExpr(0)});
}

AffineMap Builder::getMultiDimIdentityMap(unsigned rank) {
  SmallVector<AffineExpr, 4> dimExprs;
  dimExprs.reserve(rank);
  for (unsigned i = 0; i < rank; ++i)
    dimExprs.push_back(getAffineDimExpr(i));
  return AffineMap::get(/*dimCount=*/rank, /*symbolCount=*/0, dimExprs);
}

AffineMap Builder::getSymbolIdentityMap() {
  return AffineMap::get(/*dimCount=*/0, /*symbolCount=*/1,
                        {getAffineSymbolExpr(0)});
}

AffineMap Builder::getSingleDimShiftAffineMap(int64_t shift) {
  // expr = d0 + shift.
  auto expr = getAffineDimExpr(0) + shift;
  return AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0, {expr});
}

AffineMap Builder::getShiftedAffineMap(AffineMap map, int64_t shift) {
  SmallVector<AffineExpr, 4> shiftedResults;
  shiftedResults.reserve(map.getNumResults());
  for (auto resultExpr : map.getResults()) {
    shiftedResults.push_back(resultExpr + shift);
  }
  return AffineMap::get(map.getNumDims(), map.getNumSymbols(), shiftedResults);
}

//===----------------------------------------------------------------------===//
// OpBuilder.
//===----------------------------------------------------------------------===//

OpBuilder::~OpBuilder() {}

/// Add new block and set the insertion point to the end of it. The block is
/// inserted at the provided insertion point of 'parent'.
Block *OpBuilder::createBlock(Region *parent, Region::iterator insertPt) {
  assert(parent && "expected valid parent region");
  if (insertPt == Region::iterator())
    insertPt = parent->end();

  Block *b = new Block();
  parent->getBlocks().insert(insertPt, b);
  setInsertionPointToEnd(b);
  return b;
}

/// Add new block and set the insertion point to the end of it.  The block is
/// placed before 'insertBefore'.
Block *OpBuilder::createBlock(Block *insertBefore) {
  assert(insertBefore && "expected valid insertion block");
  return createBlock(insertBefore->getParent(), Region::iterator(insertBefore));
}

/// Create an operation given the fields represented as an OperationState.
Operation *OpBuilder::createOperation(const OperationState &state) {
  assert(block && "createOperation() called without setting builder's block");
  auto *op = Operation::create(state);
  insert(op);
  return op;
}

/// Attempts to fold the given operation and places new results within
/// 'results'.
void OpBuilder::tryFold(Operation *op, SmallVectorImpl<Value *> &results) {
  results.reserve(op->getNumResults());
  SmallVector<OpFoldResult, 4> foldResults;

  // Returns if the given fold result corresponds to a valid existing value.
  auto isValidValue = [](OpFoldResult result) {
    return result.dyn_cast<Value *>();
  };

  // Check if the fold failed, or did not result in only existing values.
  SmallVector<Attribute, 4> constOperands(op->getNumOperands());
  if (failed(op->fold(constOperands, foldResults)) || foldResults.empty() ||
      !llvm::all_of(foldResults, isValidValue)) {
    // Simply return the existing operation results.
    results.assign(op->result_begin(), op->result_end());
    return;
  }

  // Populate the results with the folded results and remove the original op.
  llvm::transform(foldResults, std::back_inserter(results),
                  [](OpFoldResult result) { return result.get<Value *>(); });
  op->erase();
}

/// Insert the given operation at the current insertion point.
void OpBuilder::insert(Operation *op) {
  if (block)
    block->getOperations().insert(insertPoint, op);
}
