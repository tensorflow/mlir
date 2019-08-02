//===- DependenceAnalysis.cpp - Dependence analysis on SSA views ----------===//
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
// This file implements view-based alias and dependence analyses.
//
//===----------------------------------------------------------------------===//

#include "mlir/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Linalg/IR/LinalgOps.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-dependence-analysis"

using namespace mlir;
using namespace mlir::linalg;

using llvm::dbgs;

Value *Aliases::find(Value *v) {
  if (isa<BlockArgument>(v))
    return v;

  auto it = aliases.find(v);
  if (it != aliases.end()) {
    assert(((isa<BlockArgument>(it->getSecond()) &&
             it->getSecond()->getType().isa<ViewType>()) ||
            it->getSecond()->getType().isa<BufferType>()) &&
           "Buffer or block argument expected");
    return it->getSecond();
  }

  while (true) {
    if (isa<BlockArgument>(v))
      return v;
    if (auto slice = dyn_cast_or_null<SliceOp>(v->getDefiningOp())) {
      auto it = aliases.insert(std::make_pair(v, find(slice.getBaseView())));
      return it.first->second;
    }
    if (auto view = dyn_cast_or_null<ViewOp>(v->getDefiningOp())) {
      auto it = aliases.insert(std::make_pair(v, view.getSupportingBuffer()));
      return it.first->second;
    }
    if (auto view = dyn_cast_or_null<SubViewOp>(v->getDefiningOp())) {
      v = view.getView();
      continue;
    }
    llvm::errs() << "View alias analysis reduces to: " << *v << "\n";
    llvm_unreachable("unsupported view alias case");
  }
}

LinalgDependenceGraph::LinalgDependenceGraph(Aliases &aliases,
                                             ArrayRef<Operation *> ops)
    : aliases(aliases), linalgOps(ops.begin(), ops.end()) {
  for (auto en : llvm::enumerate(linalgOps)) {
    assert(isa<LinalgOp>(en.value()) && "Expected value for LinalgOp");
    linalgOpPositions.insert(std::make_pair(en.value(), en.index()));
  }
  for (unsigned i = 0, e = ops.size(); i < e; ++i) {
    for (unsigned j = i + 1; j < e; ++j) {
      addDependencesBetween(cast<LinalgOp>(ops[i]), cast<LinalgOp>(ops[j]));
    }
  }
}

void LinalgDependenceGraph::addDependenceElem(DependenceType dt,
                                              LinalgOpView indexingOpView,
                                              LinalgOpView dependentOpView) {
  LLVM_DEBUG(dbgs() << "\nAdd dep type " << dt << ":\t" << *indexingOpView.op
                    << " -> " << *dependentOpView.op);
  dependencesFromGraphs[dt][indexingOpView.op].push_back(
      LinalgDependenceGraphElem{dependentOpView, indexingOpView.view});
  dependencesIntoGraphs[dt][dependentOpView.op].push_back(
      LinalgDependenceGraphElem{indexingOpView, dependentOpView.view});
}

LinalgDependenceGraph::dependence_range
LinalgDependenceGraph::getDependencesFrom(
    LinalgOp src, LinalgDependenceGraph::DependenceType dt) {
  return getDependencesFrom(src.getOperation(), dt);
}

LinalgDependenceGraph::dependence_range
LinalgDependenceGraph::getDependencesFrom(
    Operation *src, LinalgDependenceGraph::DependenceType dt) {
  auto &vec = dependencesFromGraphs[dt][src];
  return llvm::make_range(vec.begin(), vec.end());
}

LinalgDependenceGraph::dependence_range
LinalgDependenceGraph::getDependencesInto(
    LinalgOp dst, LinalgDependenceGraph::DependenceType dt) {
  return getDependencesInto(dst.getOperation(), dt);
}

LinalgDependenceGraph::dependence_range
LinalgDependenceGraph::getDependencesInto(
    Operation *dst, LinalgDependenceGraph::DependenceType dt) {
  auto &vec = dependencesIntoGraphs[dt][dst];
  return llvm::make_range(vec.begin(), vec.end());
}

void LinalgDependenceGraph::addDependencesBetween(LinalgOp src, LinalgOp dst) {
  for (auto *srcView : src.getOutputs()) { // W
    // RAW graph
    for (auto *dstView : dst.getInputs()) {  // R
      if (aliases.alias(srcView, dstView)) { // if alias, fill RAW
        addDependenceElem(DependenceType::RAW,
                          LinalgOpView{src.getOperation(), srcView},
                          LinalgOpView{dst.getOperation(), dstView});
      }
    }
    // WAW graph
    for (auto *dstView : dst.getOutputs()) { // W
      if (aliases.alias(srcView, dstView)) { // if alias, fill WAW
        addDependenceElem(DependenceType::WAW,
                          LinalgOpView{src.getOperation(), srcView},
                          LinalgOpView{dst.getOperation(), dstView});
      }
    }
  }
  for (auto *srcView : src.getInputs()) { // R
    // RAR graph
    for (auto *dstView : dst.getInputs()) {  // R
      if (aliases.alias(srcView, dstView)) { // if alias, fill RAR
        addDependenceElem(DependenceType::RAR,
                          LinalgOpView{src.getOperation(), srcView},
                          LinalgOpView{dst.getOperation(), dstView});
      }
    }
    // WAR graph
    for (auto *dstView : dst.getOutputs()) { // W
      if (aliases.alias(srcView, dstView)) { // if alias, fill WAR
        addDependenceElem(DependenceType::WAR,
                          LinalgOpView{src.getOperation(), srcView},
                          LinalgOpView{dst.getOperation(), dstView});
      }
    }
  }
}

SmallVector<Operation *, 8>
LinalgDependenceGraph::findCoveringDependences(LinalgOp srcLinalgOp,
                                               LinalgOp dstLinalgOp) {
  return findOperationsWithCoveringDependences(
      srcLinalgOp, dstLinalgOp, nullptr,
      {DependenceType::WAW, DependenceType::WAR, DependenceType::RAW});
}

SmallVector<Operation *, 8>
LinalgDependenceGraph::findCoveringWrites(LinalgOp srcLinalgOp,
                                          LinalgOp dstLinalgOp, Value *view) {
  return findOperationsWithCoveringDependences(
      srcLinalgOp, dstLinalgOp, view,
      {DependenceType::WAW, DependenceType::WAR});
}

SmallVector<Operation *, 8>
LinalgDependenceGraph::findCoveringReads(LinalgOp srcLinalgOp,
                                         LinalgOp dstLinalgOp, Value *view) {
  return findOperationsWithCoveringDependences(
      srcLinalgOp, dstLinalgOp, view,
      {DependenceType::RAR, DependenceType::RAW});
}

SmallVector<Operation *, 8>
LinalgDependenceGraph::findOperationsWithCoveringDependences(
    LinalgOp srcLinalgOp, LinalgOp dstLinalgOp, Value *view,
    ArrayRef<DependenceType> types) {
  auto *src = srcLinalgOp.getOperation();
  auto *dst = dstLinalgOp.getOperation();
  auto srcPos = linalgOpPositions[src];
  auto dstPos = linalgOpPositions[dst];
  assert(srcPos < dstPos && "expected dst after src in IR traversal order");

  SmallVector<Operation *, 8> res;
  // Consider an intermediate interleaved `interim` op, look for any dependence
  // to an aliasing view on a src -> op -> dst path.
  // TODO(ntv) we are not considering paths yet, just interleaved positions.
  for (auto dt : types) {
    for (auto dependence : getDependencesFrom(src, dt)) {
      auto interimPos = linalgOpPositions[dependence.dependentOpView.op];
      // Skip if not interleaved.
      if (interimPos >= dstPos || interimPos <= srcPos)
        continue;
      if (view && !aliases.alias(view, dependence.indexingView))
        continue;
      auto *op = dependence.dependentOpView.op;
      LLVM_DEBUG(dbgs() << "\n***Found covering dependence of type " << dt
                        << ": " << *src << " -> " << *op << " on "
                        << *dependence.indexingView);
      res.push_back(op);
    }
  }
  return res;
}
