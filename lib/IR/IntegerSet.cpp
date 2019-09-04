//===- IntegerSet.cpp - MLIR Integer Set class ----------------------------===//
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

#include "mlir/IR/IntegerSet.h"
#include "IntegerSetDetail.h"
#include "mlir/IR/AffineExpr.h"

using namespace mlir;
using namespace mlir::detail;

unsigned IntegerSet::getNumDims() const { return set->dimCount; }
unsigned IntegerSet::getNumSymbols() const { return set->symbolCount; }
unsigned IntegerSet::getNumInputs() const {
  return set->dimCount + set->symbolCount;
}

unsigned IntegerSet::getNumConstraints() const {
  return set->constraints.size();
}

unsigned IntegerSet::getNumEqualities() const {
  unsigned numEqualities = 0;
  for (unsigned i = 0, e = getNumConstraints(); i < e; i++)
    if (isEq(i))
      ++numEqualities;
  return numEqualities;
}

unsigned IntegerSet::getNumInequalities() const {
  return getNumConstraints() - getNumEqualities();
}

bool IntegerSet::isEmptyIntegerSet() const {
  // This will only work if uniqui'ing is on.
  static_assert(kUniquingThreshold >= 1,
                "uniquing threshold should be at least one");
  return *this == getEmptySet(set->dimCount, set->symbolCount, getContext());
}

ArrayRef<AffineExpr> IntegerSet::getConstraints() const {
  return set->constraints;
}

AffineExpr IntegerSet::getConstraint(unsigned idx) const {
  return getConstraints()[idx];
}

/// Returns the equality bits, which specify whether each of the constraints
/// is an equality or inequality.
ArrayRef<bool> IntegerSet::getEqFlags() const { return set->eqFlags; }

/// Returns true if the idx^th constraint is an equality, false if it is an
/// inequality.
bool IntegerSet::isEq(unsigned idx) const { return getEqFlags()[idx]; }

MLIRContext *IntegerSet::getContext() const {
  return getConstraint(0).getContext();
}

/// Walk all of the AffineExpr's in this set. Each node in an expression
/// tree is visited in postorder.
void IntegerSet::walkExprs(
    llvm::function_ref<void(AffineExpr)> callback) const {
  for (auto expr : getConstraints())
    expr.walk(callback);
}

IntegerSet IntegerSet::replaceDimsAndSymbols(
    ArrayRef<AffineExpr> dimReplacements, ArrayRef<AffineExpr> symReplacements,
    unsigned numResultDims, unsigned numResultSyms) {
  SmallVector<AffineExpr, 8> constraints;
  constraints.reserve(getNumConstraints());
  for (auto cst : getConstraints())
    constraints.push_back(
        cst.replaceDimsAndSymbols(dimReplacements, symReplacements));

  return get(numResultDims, numResultSyms, constraints, getEqFlags());
}
