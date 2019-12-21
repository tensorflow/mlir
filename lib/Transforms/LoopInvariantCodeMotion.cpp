//===- LoopInvariantCodeMotion.cpp - Code to perform loop fusion-----------===//
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
// This file implements loop invariant code motion.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopLikeInterface.h"
#include "mlir/Transforms/SideEffectsInterface.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "licm"

using namespace mlir;

namespace {

using SideEffecting = SideEffectsInterface::SideEffecting;

/// Loop invariant code motion (LICM) pass.
struct LoopInvariantCodeMotion : public OperationPass<LoopInvariantCodeMotion> {
public:
  void runOnOperation() override;
};

// Checks whether the given op can be hoisted by checking that
// - the op and any of its contained operations do not depend on SSA values
//   defined inside of the loop (by means of calling definedOutside).
// - the op has no side-effects. If sideEffecting is Never, sideeffects of this
//   op and its nested ops are ignored.
static bool canBeHoisted(Operation *op,
                         function_ref<bool(Value *)> definedOutside,
                         SideEffecting sideEffecting,
                         SideEffectsInterface &interface) {
  // Check that dependencies are defined outside of loop.
  if (!llvm::all_of(op->getOperands(), definedOutside))
    return false;
  // Check whether this op is side-effect free. If we already know that there
  // can be no side-effects because the surrounding op has claimed so, we can
  // (and have to) skip this step.
  auto thisOpIsSideEffecting = sideEffecting;
  if (thisOpIsSideEffecting != SideEffecting::Never) {
    thisOpIsSideEffecting = interface.isSideEffecting(op);
    // If the op always has side effects, we cannot hoist.
    if (thisOpIsSideEffecting == SideEffecting::Always)
      return false;
  }
  // Recurse into the regions for this op and check whether the contained ops
  // can be hoisted.
  for (auto &region : op->getRegions()) {
    for (auto &block : region.getBlocks()) {
      for (auto &innerOp : block.without_terminator()) {
        if (!canBeHoisted(&innerOp, definedOutside, thisOpIsSideEffecting,
                          interface))
          return false;
      }
    }
  }
  return true;
}

static LogicalResult moveLoopInvariantCode(LoopLikeOpInterface looplike,
                                           SideEffectsInterface &interface) {
  auto &loopBody = looplike.getLoopBody();

  // We use two collections here as we need to preserve the order for insertion
  // and this is easiest.
  SmallPtrSet<Operation *, 8> willBeMovedSet;
  SmallVector<Operation *, 8> opsToMove;

  // Helper to check whether an operation is loop invariant wrt. SSA properties.
  auto isDefinedOutsideOfBody = [&](Value *value) {
    auto definingOp = value->getDefiningOp();
    return (definingOp && !!willBeMovedSet.count(definingOp)) ||
           looplike.isDefinedOutsideOfLoop(value);
  };

  // Do not use walk here, as we do not want to go into nested regions and hoist
  // operations from there. These regions might have semantics unknown to this
  // rewriting. If the nested regions are loops, they will have been processed.
  for (auto &block : loopBody) {
    for (auto &op : block.without_terminator()) {
      if (canBeHoisted(&op, isDefinedOutsideOfBody,
                       mlir::SideEffectsDialectInterface::Recursive,
                       interface)) {
        opsToMove.push_back(&op);
        willBeMovedSet.insert(&op);
      }
    }
  }

  // For all operations that we found to be invariant, move outside of the
  // loop.
  auto result = looplike.moveOutOfLoop(opsToMove);
  LLVM_DEBUG(looplike.print(llvm::dbgs() << "Modified loop\n"));
  return result;
}

} // end anonymous namespace

void LoopInvariantCodeMotion::runOnOperation() {
  SideEffectsInterface interface(&getContext());
  // Walk through all loops in a function in innermost-loop-first order. This
  // way, we first LICM from the inner loop, and place the ops in
  // the outer loop, which in turn can be further LICM'ed.
  getOperation()->walk([&](LoopLikeOpInterface loopLikeOp) {
    // Skip zero trip count loops. For unknown trip counts, we still move
    // invariant code since it is side-effect free, and in general profitable.
    // TODO: when necessary, we could only move when the trip count is
    // guaranteed to be at least one.
    if (loopLikeOp.getConstantTripCount() == uint64_t(0))
      return;
    LLVM_DEBUG(loopLikeOp.print(llvm::dbgs() << "\nOriginal loop\n"));
    if (failed(moveLoopInvariantCode(loopLikeOp, interface)))
      signalPassFailure();
  });
}

// Include the generated code for the loop-like interface here, as it otherwise
// has no compilation unit. This works as loop-invariant code motion is the
// only user of that interface.
#include "mlir/Transforms/LoopLikeInterface.cpp.inc"

std::unique_ptr<Pass> mlir::createLoopInvariantCodeMotionPass() {
  return std::make_unique<LoopInvariantCodeMotion>();
}

static PassRegistration<LoopInvariantCodeMotion>
    pass("loop-invariant-code-motion",
         "Hoist loop invariant operations outside of the loop");
