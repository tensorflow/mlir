//===- LinalgOps.cpp - Implementation of the linalg operations ------------===//
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
// This file implements a the Linalg operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/EDSC/Helpers.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/Transforms/FoldUtils.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;

namespace {
/// Fold constant dimensions into an alloc operation.
struct SimplifyDimOp : public OpRewritePattern<linalg::DimOp> {
  using OpRewritePattern<linalg::DimOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(linalg::DimOp dimOp,
                                     PatternRewriter &rewriter) const override;
};
} // end namespace

PatternMatchResult
SimplifyDimOp::matchAndRewrite(linalg::DimOp dimOp,
                               PatternRewriter &rewriter) const {
  auto *viewProducingOp = dimOp.view()->getDefiningOp();
  auto subView = dyn_cast_or_null<SubViewOp>(viewProducingOp);
  auto slice = dyn_cast_or_null<SliceOp>(viewProducingOp);
  auto view = dyn_cast_or_null<ViewOp>(viewProducingOp);
  assert(subView || slice || view);

  unsigned dim = dimOp.getIndex();
  Value *min, *max, *step;
  if (view) {
    // Cannot traverse block arguments, fail.
    if (isa<BlockArgument>(view.getRange(dim)))
      return matchFailure();
    // Record min, max, step for further processing.
    auto range = cast<RangeOp>(view.getRange(dim)->getDefiningOp());
    std::tie(min, max, step) =
        std::make_tuple(range.min(), range.max(), range.step());
  } else if (subView) {
    // Record min, max, step for further processing.
    auto range = subView.getRange(dim);
    std::tie(min, max, step) =
        std::make_tuple(range.min, range.max, range.step);
  } else {
    // Taking the dim of a slice must take a range (since other dims have been
    // rank-reduced).
    auto *rangeValue = slice.getRanges()[dim];
    // Cannot traverse block arguments, fail.
    if (isa<BlockArgument>(rangeValue))
      return matchFailure();
    auto range = cast<RangeOp>(rangeValue->getDefiningOp());
    // Record min, max, step for further processing.
    std::tie(min, max, step) =
        std::make_tuple(range.min(), range.max(), range.step());
  }

  // Only support constant steps of 1 atm.
  auto constant = dyn_cast_or_null<ConstantIndexOp>(step->getDefiningOp());
  if (!constant || constant.getValue() != 1)
    return matchFailure();

  // Circumvent affine constraints:
  //   emit an affine_apply when possible, otherwise emit a `subi`.
  bool validAffineMin = isValidDim(min) || isValidSymbol(min) ||
                        isa_and_nonnull<ConstantIndexOp>(min->getDefiningOp());
  bool validAffineMax = isValidDim(max) || isValidSymbol(max) ||
                        isa_and_nonnull<ConstantIndexOp>(max->getDefiningOp());

  OpBuilder b(dimOp);
  ScopedContext scope(b, dimOp.getLoc());
  // Emit `subi`.
  if (!validAffineMin || !validAffineMax) {
    rewriter.replaceOp(dimOp, {subi(max, min)}, {dimOp.view()});
    return matchSuccess();
  }

  // Emit affine_apply.
  using edsc::op::operator-;
  rewriter.replaceOp(dimOp, {ValueHandle(max) - ValueHandle(min)},
                     {dimOp.view()});
  return matchSuccess();
}

///////////////////// Operations defined with Tablegen /////////////////////////
// For such operations that do not correspond to library calls (i.e. defined in
// LinalgOps.td), we define an overloaded `print` function and a
// parse`className` function.

//===----------------------------------------------------------------------===//
// BufferAllocOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter *p, BufferAllocOp op) {
  *p << op.getOperationName() << " ";
  if (!llvm::empty(op.size()))
    *p << *op.getOperand(0);
  if (op.alignment().hasValue() && op.alignment()->getSExtValue() != 0)
    p->printOptionalAttrDict(op.getAttrs());
  else
    p->printOptionalAttrDict(op.getAttrs(),
                             BufferAllocOp::getAlignmentAttrName());
  *p << " : " << op.getBufferType();
}

static ParseResult parseBufferAllocOp(OpAsmParser *parser,
                                      OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 1> sizeInfo;
  BufferType bufferType;
  auto indexTy = parser->getBuilder().getIndexType();
  if (parser->parseOperandList(sizeInfo) ||
      parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColonType(bufferType))
    return failure();
  if (sizeInfo.empty())
    return parser->addTypeToList(bufferType, result->types);
  return failure(parser->resolveOperands(sizeInfo, indexTy, result->operands) ||
                 parser->addTypeToList(bufferType, result->types));
}

static LogicalResult verify(BufferAllocOp op) {
  if (!op.getBufferType().hasConstantSize()) {
    if (llvm::size(op.size()) != 1)
      return op.emitOpError("expected one index operand");
  } else { // op.getBufferType().hasConstantSize()
    if (!llvm::empty(op.size()))
      return op.emitOpError("expected zero operand");
    if (op.getBufferType().getBufferSize().getValue() <= 0)
      return op.emitOpError("expected nonnegative static buffer size");
  }
  if (op.alignment().hasValue()) {
    auto align = op.alignment().getValue();
    if (align.getSExtValue() < 0)
      return op.emitOpError("expected positive alignment");
    if (!llvm::isPowerOf2_64(align.getZExtValue()))
      return op.emitOpError("expected power of 2 alignment");
  }
  if (!TensorType::isValidElementType(op.getElementType()))
    return op.emitOpError("expected valid buffer element type");
  return success();
}

//===----------------------------------------------------------------------===//
// BufferDeallocOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter *p, BufferDeallocOp op) {
  *p << op.getOperationName() << " " << *op.buffer();
  p->printOptionalAttrDict(op.getAttrs());
  *p << " : " << op.getBufferType();
}

static ParseResult parseBufferDeallocOp(OpAsmParser *parser,
                                        OperationState *result) {
  OpAsmParser::OperandType bufferInfo;
  BufferType bufferType;
  if (parser->parseOperand(bufferInfo) ||
      parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColonType(bufferType))
    return failure();
  return parser->resolveOperands(bufferInfo, bufferType, result->operands);
}

//===----------------------------------------------------------------------===//
// BufferSizeOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter *p, BufferSizeOp op) {
  *p << op.getOperationName() << " " << *op.buffer();
  p->printOptionalAttrDict(op.getAttrs());
  *p << " : " << op.buffer()->getType();
}

static ParseResult parseBufferSizeOp(OpAsmParser *parser,
                                     OperationState *result) {
  OpAsmParser::OperandType op;
  Type type;
  return failure(parser->parseOperand(op) ||
                 parser->parseOptionalAttributeDict(result->attributes) ||
                 parser->parseColonType(type) ||
                 parser->resolveOperand(op, type, result->operands) ||
                 parser->addTypeToList(parser->getBuilder().getIndexType(),
                                       result->types));
}

//===----------------------------------------------------------------------===//
// DimOp
//===----------------------------------------------------------------------===//
void mlir::linalg::DimOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SimplifyDimOp>(context);
}

static void print(OpAsmPrinter *p, linalg::DimOp op) {
  *p << op.getOperationName() << " " << *op.getOperand() << ", "
     << op.getIndex();
  p->printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"index"});
  *p << " : " << op.getOperand()->getType();
}

static ParseResult parseDimOp(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType operandInfo;
  IntegerAttr indexAttr;
  Type type;
  Type indexType = parser->getBuilder().getIndexType();
  return failure(parser->parseOperand(operandInfo) || parser->parseComma() ||
                 parser->parseAttribute(indexAttr, indexType, "index",
                                        result->attributes) ||
                 parser->parseOptionalAttributeDict(result->attributes) ||
                 parser->parseColonType(type) ||
                 parser->resolveOperand(operandInfo, type, result->operands) ||
                 parser->addTypeToList(indexType, result->types));
}

//===----------------------------------------------------------------------===//
// GenericOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter *p, GenericOp op) {
  auto attrNames = op.linalgTraitAttrNames();
  llvm::StringSet<> linalgTraitAttrsSet;
  linalgTraitAttrsSet.insert(attrNames.begin(), attrNames.end());
  SmallVector<NamedAttribute, 8> attrs;
  for (auto attr : op.getAttrs()) {
    if (linalgTraitAttrsSet.count(attr.first.strref()) > 0)
      attrs.push_back(attr);
  }
  auto dictAttr = DictionaryAttr::get(attrs, op.getContext());
  *p << op.getOperationName() << " " << dictAttr << " ";
  p->printOperands(op.getOperands());
  if (!op.region().empty())
    p->printRegion(op.region());
  p->printOptionalAttrDict(op.getAttrs(), attrNames);
  *p << ": ";
  interleaveComma(op.getOperandTypes(), *p);
}

static ParseResult parseGenericOp(OpAsmParser *parser, OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 8> operandsInfo, regionOperandsInfo;
  DictionaryAttr dictAttr;
  // Parse the core linalg traits that must check into a dictAttr.
  // The name is unimportant as we will overwrite result->attributes.
  // The core linalg traits must contain the information necessary to pass the
  // verifier.
  if (parser->parseAttribute(dictAttr, "_", result->attributes) ||
      parser->parseOperandList(operandsInfo))
    return failure();
  result->attributes.assign(dictAttr.getValue().begin(),
                            dictAttr.getValue().end());

  Region &region = *result->addRegion();
  SmallVector<Type, 8> operandTypes, regionTypes;
  // Optional attributes may be added.
  // Either Optional "fun" attribute or region must be specified.
  if (!dictAttr.get("fun") &&
      parser->parseOptionalRegion(region, regionOperandsInfo, regionTypes))
    return failure();
  if (parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColonTypeList(operandTypes))
    return failure();
  return parser->resolveOperands(operandsInfo, operandTypes,
                                 parser->getCurrentLocation(),
                                 result->operands);
}

static LogicalResult verify(GenericOp op) {
  auto nInputViews = op.getNumInputs();
  auto nViews = op.getNumInputsAndOutputs();
  if (nViews != llvm::size(op.views()))
    return op.emitError("op expected exactly ") << nViews << " view operands";

  auto &region = op.region();
  auto funOp = op.getFunction();
  auto funType = funOp ? funOp.getType() : FunctionType();
  if (!region.empty()) {
    if (region.getBlocks().size() != 1)
      return op.emitError("op expected region with 1 block");

    auto &block = region.getBlocks().front();
    if (block.getNumArguments() != nViews)
      return op.emitError(
          "op expected number of block arguments to match number of views");

    for (unsigned i = 0; i < nViews; ++i) {
      auto viewType = op.getViewType(i);
      if (viewType.getElementType() != block.getArgument(i)->getType())
        return op.emitError("op expected block argument ")
               << i << " of the same type as elemental type of "
               << ((i < nInputViews) ? "input " : "output ")
               << "view: " << viewType;
    }
  } else {
    if (!funOp || !funOp.getType())
      return op.emitError(
          "op expected fun attribute to refer to a defined symbol");
    if (funType.getNumInputs() != nViews)
      return op.emitError("op expected fun arguments to match number of views");
    if (funType.getNumResults() != op.getNumOutputs())
      return op.emitError(
          "op expected fun results to match number of output views");
  }

  auto nLoops = op.getNumLoops();
  SmallVector<AffineMap, 4> indexingMaps;
  indexingMaps.reserve(op.indexing_maps().size());
  for (auto en : llvm::enumerate(op.indexing_maps())) {
    auto idx = en.index();
    auto m = en.value().cast<AffineMapAttr>().getValue();
    indexingMaps.push_back(m); // Save reference to map for further checks.
    auto view = (idx < nInputViews) ? op.getInputViewType(idx)
                                    : op.getOutputViewType(idx - nInputViews);

    if (m.getNumSymbols() != 0)
      return op.emitError("op expected indexing_map #")
             << idx << " to have no symbols";

    if (m.getNumDims() != nLoops)
      return op.emitError("op expected indexing_map #")
             << idx << " to have " << nLoops
             << " dim(s) to match the number of loops";

    if (m.getNumResults() == 1 && view.getRank() == 0) {
      auto cst = m.getResult(0).dyn_cast<AffineConstantExpr>();
      if (!cst || cst.getValue() != 0)
        return op.emitError("op expected indexing_map #")
               << idx << " to be 0 to match 0-D view: " << view;
    }

    if (m.getNumResults() != view.getRank())
      return op.emitError("op expected indexing_map #")
             << idx << " results to match view rank: " << view;

    if (funType) {
      if (funType.getInput(idx) != view.getElementType())
        return op.emitError("op expected fun argument ")
               << idx
               << " to match view element type: " << view.getElementType();

      if (idx >= nInputViews)
        if (funType.getResult(idx - nInputViews) != view.getElementType())
          return op.emitError("op expected fun result ")
                 << idx << " to match output view element type: "
                 << view.getElementType();
    }
  }

  auto concatMap = concatAffineMaps(indexingMaps);
  auto aggregateMap = inversePermutation(concatMap);
  if (!aggregateMap)
    return op.emitError("op expected the concatenation of maps in indexing_map "
                        "to be invertible");

  return success();
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter *p, linalg::LoadOp op) {
  *p << op.getOperationName() << " " << *op.view() << '[';
  p->printOperands(op.indices());
  *p << ']';
  p->printOptionalAttrDict(op.getAttrs());
  *p << " : " << op.getViewType();
}

static ParseResult parseLoadOp(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType viewInfo;
  SmallVector<OpAsmParser::OperandType, 4> indexInfo;
  ViewType type;

  auto affineIntTy = parser->getBuilder().getIndexType();
  return failure(
      parser->parseOperand(viewInfo) ||
      parser->parseOperandList(indexInfo, OpAsmParser::Delimiter::Square) ||
      parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColonType(type) ||
      parser->resolveOperand(viewInfo, type, result->operands) ||
      parser->resolveOperands(indexInfo, affineIntTy, result->operands) ||
      parser->addTypeToList(type.getElementType(), result->types));
}

static LogicalResult verify(linalg::LoadOp op) {
  if (op.getRank() != llvm::size(op.indices()))
    return op.emitOpError("expected ")
           << op.getRank() << " indices, got " << llvm::size(op.indices());
  return success();
}

//===----------------------------------------------------------------------===//
// RangeOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter *p, RangeOp op) {
  *p << op.getOperationName() << " " << *op.min() << ":" << *op.max() << ":"
     << *op.step();
  p->printOptionalAttrDict(op.getAttrs());
  *p << " : " << op.getResult()->getType();
}

static ParseResult parseRangeOp(OpAsmParser *parser, OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 3> rangeInfo(3);
  RangeType type;
  auto affineIntTy = parser->getBuilder().getIndexType();
  return failure(
      parser->parseOperand(rangeInfo[0]) || parser->parseColon() ||
      parser->parseOperand(rangeInfo[1]) || parser->parseColon() ||
      parser->parseOperand(rangeInfo[2]) ||
      parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColonType(type) ||
      parser->resolveOperands(rangeInfo, affineIntTy, result->operands) ||
      parser->addTypeToList(type, result->types));
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

void mlir::linalg::SliceOp::build(Builder *b, OperationState *result,
                                  Value *base, ArrayRef<Value *> indexings) {
  result->addOperands(base);
  result->addOperands(indexings);

  ViewType viewType = base->getType().cast<ViewType>();
  unsigned rank = viewType.getRank();
  for (auto *i : indexings)
    if (!i->getType().isa<RangeType>())
      rank--;
  Type elementType = viewType.getElementType();
  result->addTypes({ViewType::get(b->getContext(), elementType, rank)});
}

static void print(OpAsmPrinter *p, SliceOp op) {
  *p << SliceOp::getOperationName() << " " << *op.view() << "[";
  p->printOperands(op.indexings());
  *p << "] ";
  p->printOptionalAttrDict(op.getAttrs());
  *p << " : " << op.getBaseViewType();
  for (auto indexing : op.indexings()) {
    *p << ", " << indexing->getType();
  }
  *p << ", " << op.getType();
}

static ParseResult parseSliceOp(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType baseInfo;
  SmallVector<OpAsmParser::OperandType, 8> operands;
  SmallVector<Type, 8> types;
  if (parser->parseOperand(baseInfo) ||
      parser->parseOperandList(operands, OpAsmParser::Delimiter::Square) ||
      parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColonTypeList(types))
    return failure();

  if (types.size() < 2)
    return parser->emitError(parser->getCurrentLocation(),
                             "expected at least input and result view types");

  ArrayRef<Type> indexingTypes = ArrayRef<Type>(types).drop_front().drop_back();
  return failure(
      parser->resolveOperand(baseInfo, types.front(), result->operands) ||
      (!operands.empty() &&
       parser->resolveOperands(operands, indexingTypes,
                               operands.front().location, result->operands)) ||
      parser->addTypeToList(types.back(), result->types));
}

static LogicalResult verify(SliceOp op) {
  unsigned rank = op.getBaseViewRank();
  if (rank != llvm::size(op.indexings()))
    return op.emitOpError("expected ")
           << op.getRank() << " indexings, got " << llvm::size(op.indexings());
  unsigned index = 0;
  for (auto indexing : op.indexings()) {
    if (indexing->getType().isa<IndexType>())
      --rank;
    ++index;
  }
  if (op.getRank() != rank)
    return op.emitOpError() << "expected rank of the view(" << op.getRank()
                            << ") to be the number of ranges(" << rank << ")";
  return success();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter *p, linalg::StoreOp op) {
  *p << op.getOperationName() << " " << *op.value();
  *p << ", " << *op.view() << '[';
  p->printOperands(op.indices());
  *p << ']';
  p->printOptionalAttrDict(op.getAttrs());
  *p << " : " << op.getViewType();
}

static ParseResult parseStoreOp(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType storeValueInfo;
  OpAsmParser::OperandType viewInfo;
  SmallVector<OpAsmParser::OperandType, 4> indexInfo;
  ViewType viewType;

  auto affineIntTy = parser->getBuilder().getIndexType();
  return failure(
      parser->parseOperand(storeValueInfo) || parser->parseComma() ||
      parser->parseOperand(viewInfo) ||
      parser->parseOperandList(indexInfo, OpAsmParser::Delimiter::Square) ||
      parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColonType(viewType) ||
      parser->resolveOperand(storeValueInfo, viewType.getElementType(),
                             result->operands) ||
      parser->resolveOperand(viewInfo, viewType, result->operands) ||
      parser->resolveOperands(indexInfo, affineIntTy, result->operands));
}

static LogicalResult verify(linalg::StoreOp op) {
  if (op.value()->getType() != op.getViewType().getElementType())
    return op.emitOpError("expected value type to match view element type");
  if (op.getRank() != llvm::size(op.indices()))
    return op.emitOpError("expected ")
           << op.getRank() << " indices, got " << llvm::size(op.indices());
  return success();
}

//===----------------------------------------------------------------------===//
// SubViewOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter *p, SubViewOp op) {
  *p << op.getOperationName() << " " << *op.getOperand(0) << "[";
  auto ranges = op.getRanges();
  interleaveComma(ranges, *p, [&p](const SubViewOp::Range &i) {
    *p << *i.min << ", " << *i.max << ", " << *i.step;
  });
  *p << "]";
  p->printOptionalAttrDict(op.getAttrs());
  *p << " : " << op.getViewType();
}

static ParseResult parseSubViewOp(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType inputView, resultView;
  Type viewType;
  if (parser->parseOperand(inputView))
    return failure();

  SmallVector<OpAsmParser::OperandType, 12> ops;
  // TODO(ntv) evolve parsing from
  //    linalg.subview %0[%1, %2, %3, %4, %5, %6]
  // to something resembling
  //    linalg.subview %0[%1:%2:%3][%4:%5:%6]
  if (parser->parseOperandList(ops, OpAsmParser::Delimiter::Square) ||
      parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColonType(viewType))
    return failure();

  auto indexTy = parser->getBuilder().getIndexType();
  return failure(
      parser->resolveOperand(inputView, viewType, result->operands) ||
      parser->resolveOperands(ops, indexTy, result->operands) ||
      parser->addTypeToList(viewType, result->types));
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//
void mlir::linalg::TransposeOp::build(Builder *b, OperationState *result,
                                      Value *view, AffineMapAttr permutation,
                                      ArrayRef<NamedAttribute> attrs) {
  // TODO(ntv): once views have static dimensions, compute the permuted type.
  build(b, result, view->getType(), view, attrs);
  result->addAttribute(TransposeOp::getPermutationAttrName(), permutation);
}

static void print(OpAsmPrinter *p, TransposeOp op) {
  *p << op.getOperationName() << " " << *op.view() << " " << op.permutation();
  p->printOptionalAttrDict(op.getAttrs(),
                           {TransposeOp::getPermutationAttrName()});
  *p << " : " << op.view()->getType();
}

static ParseResult parseTransposeOp(OpAsmParser *parser,
                                    OperationState *result) {
  OpAsmParser::OperandType view;
  AffineMapAttr permutation;
  Type type;
  return failure(parser->parseOperand(view) ||
                 parser->parseAttribute(permutation,
                                        TransposeOp::getPermutationAttrName(),
                                        result->attributes) ||
                 parser->parseOptionalAttributeDict(result->attributes) ||
                 parser->parseColonType(type) ||
                 parser->resolveOperand(view, type, result->operands) ||
                 parser->addTypeToList(type, result->types));
}

//===----------------------------------------------------------------------===//
// ViewOp
//===----------------------------------------------------------------------===//
void mlir::linalg::ViewOp::build(Builder *b, OperationState *result,
                                 Value *buffer, ArrayRef<Value *> ranges,
                                 Type resultType,
                                 ArrayRef<NamedAttribute> attrs) {
  if (!resultType) {
    Type elementType = buffer->getType().cast<BufferType>().getElementType();
    resultType = ViewType::get(b->getContext(), elementType, ranges.size());
  }
  build(b, result, resultType, buffer, ranges);
  result->addAttributes(attrs);
}

static void print(OpAsmPrinter *p, ViewOp op) {
  *p << op.getOperationName() << " " << *op.buffer() << "[";
  interleaveComma(op.ranges(), *p, [&](Value *v) { *p << *v; });
  *p << "] ";
  p->printOptionalAttrDict(op.getAttrs());
  *p << " : " << op.buffer()->getType() << " -> " << op.getType();
}

static ParseResult parseViewOp(OpAsmParser *parser, OperationState *result) {
  OpAsmParser::OperandType bufferInfo;
  SmallVector<OpAsmParser::OperandType, 8> rangesInfo;
  Type bType, vType;
  if (parser->parseOperand(bufferInfo) ||
      parser->parseOperandList(rangesInfo, OpAsmParser::Delimiter::Square) ||
      parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColon() || parser->parseType(bType) ||
      parser->parseArrow() || parser->parseType(vType)) {
    return failure();
  }

  ViewType viewType = vType.dyn_cast<ViewType>();
  if (!viewType)
    return parser->emitError(parser->getNameLoc(), "expected view type");
  if (viewType.getRank() != rangesInfo.size())
    return parser->emitError(parser->getNameLoc(), "expected ")
           << viewType.getRank() << " ranges";
  return failure(
      parser->resolveOperand(bufferInfo, bType, result->operands) ||
      (!rangesInfo.empty() &&
       parser->resolveOperands(rangesInfo, RangeType::get(vType.getContext()),
                               result->operands)) ||
      parser->addTypeToList(viewType, result->types));
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter *p, YieldOp op) {
  *p << op.getOperationName();
  if (op.getNumOperands() > 0) {
    *p << ' ';
    p->printOperands(op.operand_begin(), op.operand_end());
  }
  p->printOptionalAttrDict(op.getAttrs());
  if (op.getNumOperands() > 0) {
    *p << " : ";
    interleaveComma(op.getOperands(), *p,
                    [&](Value *e) { p->printType(e->getType()); });
  }
}

static ParseResult parseYieldOp(OpAsmParser *parser, OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 2> opInfo;
  SmallVector<Type, 2> types;
  llvm::SMLoc loc = parser->getCurrentLocation();
  return failure(parser->parseOperandList(opInfo) ||
                 parser->parseOptionalAttributeDict(result->attributes) ||
                 (!opInfo.empty() && parser->parseColonTypeList(types)) ||
                 parser->resolveOperands(opInfo, types, loc, result->operands));
}

static LogicalResult verify(YieldOp op) {
  auto *parentOp = op.getParentOp();
  if (parentOp->getNumRegions() != 1 || parentOp->getRegion(0).empty())
    return op.emitOpError("op expected single non-empty parent region");

  auto genericOp = dyn_cast<GenericOp>(parentOp);
  if (!genericOp)
    return op.emitOpError("op expected '")
           << GenericOp::getOperationName() << "' parent op";

  // The operand number and types must match the view element types.
  auto nOutputViews = genericOp.getNumOutputs();
  if (op.getNumOperands() != nOutputViews)
    return op.emitOpError("op expected ")
           << nOutputViews << " operand to match enclosing linalg.generic op";

  for (unsigned i = 0; i != nOutputViews; ++i) {
    auto elementType = genericOp.getOutputViewType(i).getElementType();
    if (op.getOperand(i)->getType() != elementType)
      return op.emitError("type of return operand ")
             << i << " (" << op.getOperand(i)->getType()
             << ") doesn't match view element type (" << elementType << ")";
  }
  return success();
}

/////// Operations corresponding to library calls defined with Tablegen ////////
// For such operations correspond to library calls (i.e. defined in
// LinalgLibraryOps.td), we define an overloaded `print` function and a
// parse`className` function.

// A LinalgLibraryOp prints as:
//
// ```{.mlir}
//   concrete_op_name (ssa-inputs, ssa-outputs) : view-types
// ```
//
// for example:
//
// ```
//   linalg.matmul(%0, %1, %2) :
//     !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
// ```
//
// Where %0, %1 and %2 are ssa-values of type ViewType.
static void printLinalgLibraryOp(OpAsmPrinter *p, Operation *op) {
  assert(op->getAbstractOperation() && "unregistered operation");
  *p << op->getName().getStringRef() << "(";
  interleave(
      op->getOperands().begin(), op->getOperands().end(),
      [&](Value *v) { *p << *v; }, [&]() { *p << ", "; });
  *p << ")";
  p->printOptionalAttrDict(op->getAttrs());
  *p << " : ";
  interleave(
      op->getOperands().begin(), op->getOperands().end(),
      [&](Value *v) { *p << v->getType(); }, [&]() { *p << ", "; });
}

static ParseResult parseLinalgLibraryOp(OpAsmParser *parser,
                                        OperationState *result) {
  SmallVector<OpAsmParser::OperandType, 3> ops;
  SmallVector<Type, 3> types;
  return failure(parser->parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
                 parser->parseOptionalAttributeDict(result->attributes) ||
                 parser->parseColonTypeList(types) ||
                 parser->resolveOperands(ops, types, parser->getNameLoc(),
                                         result->operands));
}

static LogicalResult verify(FillOp op) {
  auto viewType = op.getOutputViewType(0);
  auto fillType = op.getValue()->getType();
  if (viewType.getElementType() != fillType)
    return op.emitOpError("expects fill type to match view elemental type");
  return success();
}

static LogicalResult verify(CopyOp op) {
  auto outputViewType = op.getOutputViewType(0);
  auto inputViewType = op.getInputViewType(0);
  if (inputViewType.getElementType() != outputViewType.getElementType())
    return op.emitOpError("expects views of the same type");
  if (inputViewType.getRank() != outputViewType.getRank())
    return op.emitOpError("expects views of the same rank");
  auto rank = op.getNumParallelLoops();
  auto inputPermutationMap = op.inputPermutation();
  if (inputPermutationMap) {
    if (inputPermutationMap->getNumInputs() != rank)
      return op.emitOpError("expects optional input_permutation map of rank ")
             << rank;
    if (!inputPermutationMap->isPermutation())
      return op.emitOpError(
          "expects optional input_permutation map to be a permutation");
  }
  auto outputPermutationMap = op.outputPermutation();
  if (outputPermutationMap) {
    if (outputPermutationMap->getNumInputs() != rank)
      return op.emitOpError("expects optional output_permutation map of rank ")
             << rank;
    if (!outputPermutationMap->isPermutation())
      return op.emitOpError(
          "expects optional output_permutation map to be a permutation");
  }
  if (rank == 0 && inputPermutationMap)
    return op.emitOpError("expected no input permutation when rank == 0");
  if (rank == 0 && outputPermutationMap)
    return op.emitOpError("expected no output permutation when rank == 0");
  return success();
}

static LogicalResult
verifyStrideOrDilation(ConvOp op, ArrayRef<Attribute> attrs, bool isStride) {
  auto strideOrDilation = isStride ? "stride" : "dilation";
  if (attrs.size() != op.getNumWindowLoops())
    return op.emitOpError("expects num ")
           << strideOrDilation
           << "s equal to number of window dimensions: " << attrs.size()
           << " vs " << op.getNumWindowLoops();
  return success();
}

static LogicalResult verify(ConvOp op) {
  auto oType = op.output()->getType().cast<ViewType>();
  auto fType = op.filter()->getType().cast<ViewType>();
  auto iType = op.input()->getType().cast<ViewType>();
  if (oType.getElementType() != iType.getElementType() ||
      oType.getElementType() != fType.getElementType())
    return op.emitOpError("expects view elemental types to match");
  if (oType.getRank() != iType.getRank() || oType.getRank() != fType.getRank())
    return op.emitOpError("expects view ranks to match");
  if (auto strides = op.strides()) {
    if (failed(
            verifyStrideOrDilation(op, strides->getValue(), /*isStride=*/true)))
      return failure();
  }
  if (auto dilations = op.dilations()) {
    if (failed(verifyStrideOrDilation(op, dilations->getValue(),
                                      /*isStride=*/false)))
      return failure();
  }
  return success();
}

llvm::raw_ostream &mlir::linalg::operator<<(llvm::raw_ostream &os,
                                            SubViewOp::Range &range) {
  return os << "range " << *range.min << ":" << *range.max << ":"
            << *range.step;
}

namespace mlir {
namespace linalg {

#include "mlir/Dialect/Linalg/IR/LinalgLibraryOpInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgOps.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgLibraryOps.cpp.inc"

} // namespace linalg
} // namespace mlir

static AffineMap extractOrIdentityMap(llvm::Optional<AffineMap> maybeMap,
                                      unsigned rank, MLIRContext *context) {
  if (maybeMap)
    return maybeMap.getValue();
  if (rank == 0)
    return AffineMap();
  return AffineMap::getMultiDimIdentityMap(rank, context);
}

// Returns `num` AffineDimExpr dimensions at positions [curIdx, curIdx + num)
// and increments `curIdx` to `curIdx + num`.
static SmallVector<AffineExpr, 4>
makeAffineDimExprs(unsigned num, unsigned &curIdx, MLIRContext *context) {
  SmallVector<AffineExpr, 4> res;
  res.reserve(num);
  for (unsigned i = 0; i < num; ++i)
    res.push_back(getAffineDimExpr(curIdx++, context));
  return res;
}

static SmallVector<AffineExpr, 4>
weightedConvInputIndex(ConvOp op, ArrayRef<AffineExpr> a,
                       ArrayRef<AffineExpr> b) {
  assert(a.size() == b.size());
  SmallVector<AffineExpr, 4> res;
  res.reserve(a.size());
  for (unsigned i = 0, e = a.size(); i < e; ++i) {
    res.push_back(op.getStride(i) * a[i] + op.getDilation(i) * b[i]);
  }
  return res;
}

static SmallVector<AffineExpr, 4> concat(ArrayRef<AffineExpr> a,
                                         ArrayRef<AffineExpr> b) {
  SmallVector<AffineExpr, 4> res;
  res.reserve(a.size() + b.size());
  res.assign(a.begin(), a.end());
  res.append(b.begin(), b.end());
  return res;
}

// Note: both functions below would completely disappear with a simple tensor
// kernel language.
//
// Ideally this should all be Tablegen'd but there is no good story for
// AffineMap for now.
SmallVector<AffineMap, 4> mlir::linalg::loopToOperandRangesMaps(Operation *op) {
  MLIRContext *context = op->getContext();
  if (auto copyOp = dyn_cast<CopyOp>(op)) {
    // I(input_perm(ivs)) -> O(output_perm(ivs))
    auto maybeInputMap = copyOp.inputPermutation();
    auto maybeOutputMap = copyOp.outputPermutation();
    unsigned inputRank = copyOp.getInputViewType(0).getRank();
    unsigned outputRank = copyOp.getOutputViewType(0).getRank();
    return SmallVector<AffineMap, 4>{
        extractOrIdentityMap(maybeInputMap, inputRank, context),
        extractOrIdentityMap(maybeOutputMap, outputRank, context)};
  }
  if (auto fillOp = dyn_cast<FillOp>(op)) {
    // filling_value -> O(ivs)
    unsigned rank = fillOp.getNumParallelLoops();
    return SmallVector<AffineMap, 4>{
        extractOrIdentityMap(llvm::None, rank, context)};
  }
  auto i = getAffineDimExpr(0, context);
  auto j = getAffineDimExpr(1, context);
  auto k = getAffineDimExpr(2, context);
  if (isa<DotOp>(op))
    // A(r_i) * B(r_i) -> C()
    return SmallVector<AffineMap, 4>{AffineMap::get(1, 0, {i}),
                                     AffineMap::get(1, 0, {i}), AffineMap()};
  if (isa<MatvecOp>(op))
    //   A(i, r_j) * B(r_j) -> C(i)
    return SmallVector<AffineMap, 4>{AffineMap::get(2, 0, {i, j}),
                                     AffineMap::get(2, 0, {j}),
                                     AffineMap::get(2, 0, {i})};
  if (isa<MatmulOp>(op))
    //   A(i, r_k) * B(r_k, j) -> C(i, j)
    return SmallVector<AffineMap, 4>{AffineMap::get(3, 0, {i, k}),
                                     AffineMap::get(3, 0, {k, j}),
                                     AffineMap::get(3, 0, {i, j})};
  if (auto convOp = dyn_cast<ConvOp>(op)) {
    //   F(z0, ..., zN-1, q, k) * I(b, x0 + z0, ..., xN-1 + zN-1, q) ->
    //     O(b, x0, ..., xN-1, k)
    // for N equal to `nWindow`.
    auto nWin = convOp.getNumWindowLoops();
    assert(nWin > 0 && "expected at least one window dimension");
    unsigned idx = 0;
    // In the following, AffineDimExprs are indexed in loop order:
    //   [ b, xs, k,           q,                     zs]
    //    parallels     non-window reductions     windows
    //
    // Parallel dims are exactly the dimensions indexing `output`:
    //     output[b, x[0], ..., x[N-1], k]; i.e.
    //  * batch dimensions (bs with #bs = 1 for now)
    //  * "image" dimensions (xs with #xs = #zs = output_rank - #bs - #ks)
    //  * output filter dimensions (ks with #ks = 1 for now)
    auto bs = makeAffineDimExprs(convOp.getNumBatchDimensions(), idx, context);
    auto xs = makeAffineDimExprs(nWin, idx, context);
    auto ks = makeAffineDimExprs(convOp.getNumOutputFeatureDimensions(), idx,
                                 context);
    // Non-window reduction dim: sum_{z[0], ..., z[N-1], q}
    auto qs =
        makeAffineDimExprs(convOp.getNumInputFeatureDimensions(), idx, context);
    // Window reduction dims: sum_{z[0], ..., z[N-1], q}
    auto zs = makeAffineDimExprs(nWin, idx, context);
    // Construct the weighedSum expression.
    auto ws = weightedConvInputIndex(convOp, xs, zs);
    return SmallVector<AffineMap, 4>{
        // filter[z[0], ..., z[N-1], q, k]
        AffineMap::get(idx, 0, concat(concat(zs, qs), ks)),
        // input[b,
        //       x[0]*s[0] + d[0]*z[0], ..., x[N-1]*s[N-1] + d[N-1]*z[N-1],
        //       q]
        AffineMap::get(idx, 0, concat(concat(bs, ws), qs)),
        // output[b, x[0], ..., x[N-1], k]
        AffineMap::get(idx, 0, concat(concat(bs, xs), ks))};
  } else if (auto genericOp = dyn_cast<GenericOp>(op)) {
    SmallVector<AffineMap, 4> res;
    unsigned nViews = genericOp.getNumInputsAndOutputs();
    res.reserve(nViews);
    for (unsigned i = 0, e = nViews; i < e; ++i) {
      res.push_back(genericOp.getIndexingMap(i));
    }
    return res;
  }
  llvm_unreachable("Missing loopToOperandRangesMaps for op");
}

static void appendMangledType(llvm::raw_string_ostream &ss, Type t) {
  if (auto view = t.dyn_cast<ViewType>()) {
    ss << "view";
    for (unsigned i = 0, e = view.getRank(); i < e; ++i)
      ss << "x";
    appendMangledType(ss, view.getElementType());
  } else if (auto vec = t.dyn_cast<VectorType>()) {
    ss << "vector";
    interleave(
        vec.getShape(), [&](int64_t i) { ss << i; }, [&]() { ss << "x"; });
    appendMangledType(ss, vec.getElementType());
  } else if (t.isIntOrIndexOrFloat()) {
    ss << t;
  } else {
    llvm_unreachable("Invalid type for linalg library name mangling");
  }
}

std::string mlir::linalg::generateLibraryCallName(Operation *op) {
  assert(isa<LinalgOp>(op));
  std::string name(op->getName().getStringRef().str());
  name.reserve(128);
  std::replace(name.begin(), name.end(), '.', '_');
  llvm::raw_string_ostream ss(name);
  ss << "_";
  auto types = op->getOperandTypes();
  interleave(
      types.begin(), types.end(), [&](Type t) { appendMangledType(ss, t); },
      [&]() { ss << "_"; });
  return ss.str();
}
