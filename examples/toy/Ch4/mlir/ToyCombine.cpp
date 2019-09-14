//===- ToyCombine.cpp - Toy High Level Optimizer --------------------------===//
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
// This file implements a simple combiner for optimizing pattern in the Toy
// dialect.
//
//===----------------------------------------------------------------------===//

#include "toy/Dialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include <numeric>
using namespace mlir;
using namespace toy;

// Helper function to fold reshape(constant) in place
Value *reshapeConstant(Builder &builder, Value* arg) {
    ReshapeOp reshape = llvm::dyn_cast_or_null<ReshapeOp>(arg->getDefiningOp());
    mlir::OpBuilder builder2(reshape.getOperation());
    ConstantOp constantOp = llvm::dyn_cast_or_null<ConstantOp>(
        reshape.getOperand()->getDefiningOp());
    auto reshapeType = reshape.getType().cast<TensorType>();
    if (auto valueAttr =
            constantOp.getAttrOfType<mlir::DenseElementsAttr>("value")) {
      // FIXME Check matching of element count!
      //      auto oldType = constantOp.getType();
      auto newType = builder.getTensorType(
          reshapeType.getShape(), valueAttr.getType().getElementType());
      auto newAttr = valueAttr.reshape(newType);
      return builder2.create<ConstantOp>(reshape.getLoc(), newType, newAttr);
    } else if (auto valueAttr =
                   constantOp.getAttrOfType<mlir::FloatAttr>("value")) {
      // Broadcast
      auto dataSize = std::accumulate(reshapeType.getShape().begin(),
                                      reshapeType.getShape().end(), 1,
                                      std::multiplies<int>());
      std::vector<mlir::Attribute> data(dataSize, valueAttr);
      auto tensorTy = builder.getTensorType(reshapeType.getShape(),
                                             reshapeType.getElementType());
      auto newAttr = mlir::DenseElementsAttr::get(tensorTy, data);
      return builder2.create<ConstantOp>(reshape.getLoc(), tensorTy, newAttr);
    } else {
      llvm_unreachable("Unsupported Constant format");
    }
    return reshape;
}

#include "ToyCombine.inc"

namespace {
struct ReshapeOpt : public FunctionPass<ReshapeOpt> {
  void runOnFunction() override {
    mlir::OwningRewritePatternList patterns;
    populateWithGenerated(&getContext(), &patterns);

    patterns.insert<ReshapeReshapeOptPattern>(&getContext());
    patterns.insert<RedundantReshapeOptPattern>(&getContext());
    patterns.insert<FoldConstantReshapeOptPattern>(&getContext());

    applyPatternsGreedily(getFunction(), patterns);
  }
};
struct TransposeOpt : public FunctionPass<TransposeOpt> {
  void runOnFunction() override {
    mlir::OwningRewritePatternList patterns;
    populateWithGenerated(&getContext(), &patterns);

    patterns.insert<TransposeOptPattern>(&getContext());

    applyPatternsGreedily(getFunction(), patterns);
  }
};
} // end anonymous namespace
namespace toy{
std::unique_ptr<mlir::Pass> createTransposeOptPass() {
  return std::make_unique<TransposeOpt>();
}
std::unique_ptr<mlir::Pass> createReshapeOptPass() {
  return std::make_unique<ReshapeOpt>();
}
}
