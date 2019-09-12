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

namespace {
#include "ToyCombine.inc"
} // end anonymous namespace

namespace {
struct TransposeOpt : public FunctionPass<TransposeOpt> {
  void runOnFunction() override {
    mlir::OwningRewritePatternList patterns;
    populateWithGenerated(&getContext(), &patterns);

    patterns.insert<TransposeOptPattern>(&getContext());

    applyPatternsGreedily(getFunction(), patterns);
  }
};
} // end anonymous namespace

namespace toy {
std::unique_ptr<mlir::Pass> createTransposeOptPass() {
  return std::make_unique<TransposeOpt>();
}
} // namespace toy

