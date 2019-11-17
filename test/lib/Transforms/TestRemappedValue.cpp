//===- TestInlining.cpp - Pass to inline calls in the test dialect --------===//
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
// Pass to test ConversionPatternRewriter::getRemappedValue. This method is used
// to get the remapped value of a original value that was replaced using
// ConversionPatternRewriter.
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

// Converter that replaces a one-result one-operand OneVResOneVOperandOp1 with
// a one-operand two-result OneVResOneVOperandOp1 by replicating its original
// operand twice.
//
// Example:
//   %1 = test.one_variadic_out_one_variadic_in1"(%0)
// is replaced with:
//   %1 = test.one_variadic_out_one_variadic_in1"(%0, %0)
struct OneVResOneVOperandOp1Converter
    : public OpConversionPattern<OneVResOneVOperandOp1> {
  using OpConversionPattern<OneVResOneVOperandOp1>::OpConversionPattern;

  PatternMatchResult
  matchAndRewrite(OneVResOneVOperandOp1 op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto origOps = op.getOperands();
    assert(std::distance(origOps.begin(), origOps.end()) == 1 &&
           "One operand expected");
    Value *origOp = *origOps.begin();
    SmallVector<Value *, 2> remappedOperands;
    // Replicate the remapped original operand twice. Note that we don't used
    // the remapped 'operand' since the goal is testing 'getRemappedValue'.
    remappedOperands.push_back(rewriter.getRemappedValue(origOp));
    remappedOperands.push_back(rewriter.getRemappedValue(origOp));

    SmallVector<Type, 1> resultTypes(op.getResultTypes());
    rewriter.replaceOpWithNewOp<OneVResOneVOperandOp1>(op, resultTypes,
                                                       remappedOperands);
    return matchSuccess();
  }
};

struct TestRemappedValue : public mlir::FunctionPass<TestRemappedValue> {
  void runOnFunction() override {
    mlir::OwningRewritePatternList patterns;
    patterns.insert<OneVResOneVOperandOp1Converter>(&getContext());

    mlir::ConversionTarget target(getContext());
    target.addLegalOp<ModuleOp, ModuleTerminatorOp, FuncOp, TestReturnOp>();
    // We make OneVResOneVOperandOp1 legal only when it has more that one
    // operand. This will trigger the conversion that will replace one-operand
    // OneVResOneVOperandOp1 with two-operand OneVResOneVOperandOp1.
    target.addDynamicallyLegalOp<OneVResOneVOperandOp1>(
        [](Operation *op) -> bool {
          return std::distance(op->operand_begin(), op->operand_end()) > 1;
        });

    if (failed(mlir::applyFullConversion(getFunction(), target, patterns))) {
      signalPassFailure();
    }
  }
};

} // end anonymous namespace

static PassRegistration<TestRemappedValue>
    pass("test-remapped-value",
         "Test public remapped value mechanism in ConversionPatternRewriter");
