//===- Dialect.cpp - Toy IR Dialect registration in MLIR ------------------===//
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
// This file implements the dialect for the Toy IR: custom type parsing and
// operation verification.
//
//===----------------------------------------------------------------------===//

#include "toy/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/InliningUtils.h"
#include "include/ShapeInferencePass.h"

using namespace mlir;
using namespace mlir::toy;

/// This class defines the interface for handling inlining with Toy
/// operations.
struct ToyInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  bool shouldAnalyzeRecursively(Operation *op) const final { return true; }

  /// All operations within toy can be inlined.
  bool isLegalToInline(Operation *, Region *,
                       BlockAndValueMapping &) const final {
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(Operation *op,
                        ArrayRef<Value *> valuesToRepl) const final {
    // Only "toy.return" needs to be handled here.
    auto returnOp = cast<ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()]->replaceAllUsesWith(it.value());
  }
};

/// Override the shape inference interface to add support for toy operations.
struct ToyShapeInferenceInterface : public DialectShapeInferenceInterface {
  using DialectShapeInferenceInterface::DialectShapeInferenceInterface;

  virtual bool requiresShapeInference() const final {
      return true;
  }

  virtual bool returnsGenericArray(Operation *op) const final {
      if (op->getNumResults() == 1) {
        auto arrayTy = op->getResult(0)->getType().cast<RankedTensorType>();
        return arrayTy.getShape().empty();
      }
      return false;
  }

  void inferShape(Operation *op) {
      // The add operation is trivial: propagate the input type as is.
      if (auto addOp = llvm::dyn_cast<AddOp>(op)) {
        op->getResult(0)->setType(op->getOperand(0)->getType());
        return;
      }

      // Transpose is easy: just invert the dimensions.
      if (op->getName().getStringRef() == "toy.transpose") {
        SmallVector<int64_t, 2> dims;
        auto arrayTy = op->getOperand(0)->getType().cast<RankedTensorType>();
        dims.insert(dims.end(), arrayTy.getShape().begin(),
                    arrayTy.getShape().end());
        if (dims.size() == 2)
          std::swap(dims[0], dims[1]);
        op->getResult(0)->setType(RankedTensorType::get(dims, arrayTy.getElementType()));
        return;
      }

      // Multiplication is a bit trickier, handle rank 1 as dot product and rank
      // 2 as matrix multiplications.
      // We need to be careful about rank mismatch here: the verifier could
      // catch it but shape inference earlier in the pass could generate an
      // invalid IR (from an invalid Toy input of course) and we wouldn't want
      // to crash here.
      if (auto mulOp = llvm::dyn_cast<MulOp>(op)) {
        auto lhs = mulOp.getOperand(0)->getType().cast<RankedTensorType>();
        auto rhs = mulOp.getOperand(1)->getType().cast<RankedTensorType>();
        auto lhsRank = lhs.getShape().size();
        auto rhsRank = rhs.getShape().size();
        if (lhsRank != rhsRank) {
          op->emitError("Shape mismatch: LHS and RHS must have the same "
                               "rank for multiplication, got ")
                 << lhsRank << " vs  " << lhsRank;
	  return;
        }
        SmallVector<int64_t, 2> dims;
        if (lhsRank == 1) {
          // dot product, result shape is <1>
          dims.push_back(1);
        } else {
          if (lhsRank != 2) {
            op->emitError("Shape mismatch: expect rank 1 or 2 for mul "
                                 "operands, got ")
                   << lhsRank;
	    return;
          }
          dims.push_back(lhs.getShape()[0]);
          dims.push_back(rhs.getShape()[1]);
        }
        op->getResult(0)->setType(RankedTensorType::get(dims, lhs.getElementType()));
        return;
      }
  }
};

//===----------------------------------------------------------------------===//
// ToyDialect
//===----------------------------------------------------------------------===//

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
ToyDialect::ToyDialect(mlir::MLIRContext *ctx) : mlir::Dialect("toy", ctx) {
  addOperations<
#define GET_OP_LIST
#include "toy/Ops.cpp.inc"
      >();
  addInterfaces<ToyInlinerInterface>();
  addInterfaces<ToyShapeInferenceInterface>();
}

//===----------------------------------------------------------------------===//
// Toy Operations
//===----------------------------------------------------------------------===//

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
static void buildConstantOp(mlir::Builder *builder, mlir::OperationState *state,
                            double value) {
  auto dataType = builder->getTensorType({}, builder->getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  ConstantOp::build(builder, state, dataType, dataAttribute);
}

/// Verifier for constant operation.
static mlir::LogicalResult verify(ConstantOp op) {
  // If the return type of the constant is not an unranked tensor, the shape
  // must match the shape of the attribute holding the data.
  auto resultType = op.getResult()->getType().cast<RankedTensorType>();
  if (!resultType)
    return success();

  auto attrType = op.value().getType().cast<mlir::TensorType>();
  if (attrType.getRank() != resultType.getRank()) {
    return op.emitOpError(
               "return type must match the one of the attached value "
               "attribute: ")
           << attrType.getRank() << " != " << resultType.getRank();
  }
  for (int dim = 0; dim < attrType.getRank(); ++dim) {
    if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
      return op.emitOpError(
                 "return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim]
             << " != " << resultType.getShape()[dim];
    }
  }
  return mlir::success();
}

static void buildAddOp(mlir::Builder *builder, mlir::OperationState *state,
                       mlir::Value *lhs, mlir::Value *rhs) {
  state->addTypes(builder->getTensorType(builder->getF64Type()));
  state->addOperands({lhs, rhs});
}

static void buildGenericCallOp(mlir::Builder *builder,
                               mlir::OperationState *state, StringRef callee,
                               ArrayRef<mlir::Value *> arguments) {
  // Generic call always returns an unranked Tensor initially.
  state->addTypes(builder->getTensorType(builder->getF64Type()));
  state->addOperands(arguments);
  state->addAttribute("callee", builder->getSymbolRefAttr(callee));
}

static void buildMulOp(mlir::Builder *builder, mlir::OperationState *state,
                       mlir::Value *lhs, mlir::Value *rhs) {
  state->addTypes(builder->getTensorType(builder->getF64Type()));
  state->addOperands({lhs, rhs});
}

static mlir::LogicalResult verify(ReturnOp op) {
  // We know that the parent operation is a function, because of the 'HasParent'
  // trait attached to the operation definition.
  auto function = cast<FuncOp>(op.getParentOp());

  /// ReturnOps can only have a single optional operand.
  if (op.getNumOperands() > 1)
    return op.emitOpError() << "expects at most 1 return operand";

  // The operand number and types must match the function signature.
  const auto &results = function.getType().getResults();
  if (op.getNumOperands() != results.size())
    return op.emitOpError()
           << "does not return the same number of values ("
           << op.getNumOperands() << ") as the enclosing function ("
           << results.size() << ")";

  // If the operation does not have an input, we are done.
  if (!op.hasOperand())
    return mlir::success();

  auto inputType = *op.operand_type_begin();
  auto resultType = results.front();

  // Check that the result type of the function matches the operand type.
  if (inputType == resultType || inputType.isa<mlir::UnrankedTensorType>() ||
      resultType.isa<mlir::UnrankedTensorType>())
    return mlir::success();

  return op.emitError() << "type of return operand ("
                        << *op.operand_type_begin()
                        << ") doesn't match function result type ("
                        << results.front() << ")";
}

static void buildTransposeOp(mlir::Builder *builder,
                             mlir::OperationState *state, mlir::Value *value) {
  state->addTypes(builder->getTensorType(builder->getF64Type()));
  state->addOperands(value);
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "toy/Ops.cpp.inc"
