//===- SDBMTest.cpp - SDBM expression unit tests --------------------------===//
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

#include "mlir/Dialect/SDBM/SDBM.h"
#include "mlir/Dialect/SDBM/SDBMDialect.h"
#include "mlir/Dialect/SDBM/SDBMExpr.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

#include "llvm/ADT/DenseSet.h"

using namespace mlir;

static MLIRContext *ctx() {
  static thread_local MLIRContext context;
  return &context;
}

static SDBMDialect *dialect() {
  static thread_local SDBMDialect *d = nullptr;
  if (!d) {
    d = ctx()->getRegisteredDialect<SDBMDialect>();
  }
  return d;
}

static SDBMExpr dim(unsigned pos) { return SDBMDimExpr::get(dialect(), pos); }

static SDBMExpr symb(unsigned pos) {
  return SDBMSymbolExpr::get(dialect(), pos);
}

namespace {

using namespace mlir::ops_assertions;

TEST(SDBMOperators, Add) {
  auto expr = dim(0) + 42;
  auto sumExpr = expr.dyn_cast<SDBMSumExpr>();
  ASSERT_TRUE(sumExpr);
  EXPECT_EQ(sumExpr.getLHS(), dim(0));
  EXPECT_EQ(sumExpr.getRHS().getValue(), 42);
}

TEST(SDBMOperators, AddFolding) {
  auto constant = SDBMConstantExpr::get(dialect(), 2) + 42;
  auto constantExpr = constant.dyn_cast<SDBMConstantExpr>();
  ASSERT_TRUE(constantExpr);
  EXPECT_EQ(constantExpr.getValue(), 44);

  auto expr = (dim(0) + 10) + 32;
  auto sumExpr = expr.dyn_cast<SDBMSumExpr>();
  ASSERT_TRUE(sumExpr);
  EXPECT_EQ(sumExpr.getRHS().getValue(), 42);

  expr = dim(0) + SDBMNegExpr::get(SDBMDimExpr::get(dialect(), 1));
  auto diffExpr = expr.dyn_cast<SDBMDiffExpr>();
  ASSERT_TRUE(diffExpr);
  EXPECT_EQ(diffExpr.getLHS(), dim(0));
  EXPECT_EQ(diffExpr.getRHS(), dim(1));

  auto inverted = SDBMNegExpr::get(SDBMDimExpr::get(dialect(), 1)) + dim(0);
  EXPECT_EQ(inverted, expr);
}

TEST(SDBMOperators, Diff) {
  auto expr = dim(0) - dim(1);
  auto diffExpr = expr.dyn_cast<SDBMDiffExpr>();
  ASSERT_TRUE(diffExpr);
  EXPECT_EQ(diffExpr.getLHS(), dim(0));
  EXPECT_EQ(diffExpr.getRHS(), dim(1));
}

TEST(SDBMOperators, DiffFolding) {
  auto constant = SDBMConstantExpr::get(dialect(), 10) - 3;
  auto constantExpr = constant.dyn_cast<SDBMConstantExpr>();
  ASSERT_TRUE(constantExpr);
  EXPECT_EQ(constantExpr.getValue(), 7);

  auto expr = dim(0) - 3;
  auto sumExpr = expr.dyn_cast<SDBMSumExpr>();
  ASSERT_TRUE(sumExpr);
  EXPECT_EQ(sumExpr.getRHS().getValue(), -3);

  auto zero = dim(0) - dim(0);
  constantExpr = zero.dyn_cast<SDBMConstantExpr>();
  ASSERT_TRUE(constantExpr);
  EXPECT_EQ(constantExpr.getValue(), 0);
}

TEST(SDBMOperators, Stripe) {
  auto expr = stripe(dim(0), 3);
  auto stripeExpr = expr.dyn_cast<SDBMStripeExpr>();
  ASSERT_TRUE(stripeExpr);
  EXPECT_EQ(stripeExpr.getVar(), dim(0));
  EXPECT_EQ(stripeExpr.getStripeFactor().getValue(), 3);
}

TEST(SDBM, RoundTripEqs) {
  // Build and SDBM defined by
  //
  //   d0 = s0 # 3 # 5
  //   s0 # 3 # 5 - d1 + 42 = 0
  //
  // and perform a double round-trip between the "list of equalities" and SDBM
  // representation.  After the first round-trip, the equalities may be
  // different due to simplification or equivalent substitutions (e.g., the
  // second equality may become d0 - d1 + 42 = 0).  However, there should not
  // be any further simplification after the second round-trip,

  // Build the SDBM from a pair of equalities and extract back the lists of
  // inequalities and equalities.  Check that all equalities are properly
  // detected and none of them decayed into inequalities.
  auto s = stripe(stripe(symb(0), 3), 5);
  auto sdbm = SDBM::get(llvm::None, {s - dim(0), s - dim(1) + 42});
  SmallVector<SDBMExpr, 4> eqs, ineqs;
  sdbm.getSDBMExpressions(dialect(), ineqs, eqs);
  ASSERT_TRUE(ineqs.empty());

  // Do the second round-trip.
  auto sdbm2 = SDBM::get(llvm::None, eqs);
  SmallVector<SDBMExpr, 4> eqs2, ineqs2;
  sdbm2.getSDBMExpressions(dialect(), ineqs2, eqs2);
  ASSERT_EQ(eqs.size(), eqs2.size());

  // Convert that the sets of equalities are equal, their order is not relevant.
  llvm::DenseSet<SDBMExpr> eqSet, eq2Set;
  eqSet.insert(eqs.begin(), eqs.end());
  eq2Set.insert(eqs2.begin(), eqs2.end());
  EXPECT_EQ(eqSet, eq2Set);
}

TEST(SDBMExpr, Constant) {
  // We can create consants and query them.
  auto expr = SDBMConstantExpr::get(dialect(), 42);
  EXPECT_EQ(expr.getValue(), 42);

  // Two separately created constants with identical values are trivially equal.
  auto expr2 = SDBMConstantExpr::get(dialect(), 42);
  EXPECT_EQ(expr, expr2);

  // Hierarchy is okay.
  auto generic = static_cast<SDBMExpr>(expr);
  EXPECT_TRUE(generic.isa<SDBMConstantExpr>());
}

TEST(SDBMExpr, Dim) {
  // We can create dimension expressions and query them.
  auto expr = SDBMDimExpr::get(dialect(), 0);
  EXPECT_EQ(expr.getPosition(), 0u);

  // Two separately created dimensions with the same position are trivially
  // equal.
  auto expr2 = SDBMDimExpr::get(dialect(), 0);
  EXPECT_EQ(expr, expr2);

  // Hierarchy is okay.
  auto generic = static_cast<SDBMExpr>(expr);
  EXPECT_TRUE(generic.isa<SDBMDimExpr>());
  EXPECT_TRUE(generic.isa<SDBMInputExpr>());
  EXPECT_TRUE(generic.isa<SDBMPositiveExpr>());
  EXPECT_TRUE(generic.isa<SDBMVaryingExpr>());

  // Dimensions are not Symbols.
  auto symbol = SDBMSymbolExpr::get(dialect(), 0);
  EXPECT_NE(expr, symbol);
  EXPECT_FALSE(expr.isa<SDBMSymbolExpr>());
}

TEST(SDBMExpr, Symbol) {
  // We can create symbol expressions and query them.
  auto expr = SDBMSymbolExpr::get(dialect(), 0);
  EXPECT_EQ(expr.getPosition(), 0u);

  // Two separately created symbols with the same position are trivially equal.
  auto expr2 = SDBMSymbolExpr::get(dialect(), 0);
  EXPECT_EQ(expr, expr2);

  // Hierarchy is okay.
  auto generic = static_cast<SDBMExpr>(expr);
  EXPECT_TRUE(generic.isa<SDBMSymbolExpr>());
  EXPECT_TRUE(generic.isa<SDBMInputExpr>());
  EXPECT_TRUE(generic.isa<SDBMPositiveExpr>());
  EXPECT_TRUE(generic.isa<SDBMVaryingExpr>());

  // Dimensions are not Symbols.
  auto symbol = SDBMDimExpr::get(dialect(), 0);
  EXPECT_NE(expr, symbol);
  EXPECT_FALSE(expr.isa<SDBMDimExpr>());
}

TEST(SDBMExpr, Stripe) {
  auto cst2 = SDBMConstantExpr::get(dialect(), 2);
  auto cst0 = SDBMConstantExpr::get(dialect(), 0);
  auto var = SDBMSymbolExpr::get(dialect(), 0);

  // We can create stripe expressions and query them.
  auto expr = SDBMStripeExpr::get(var, cst2);
  EXPECT_EQ(expr.getVar(), var);
  EXPECT_EQ(expr.getStripeFactor(), cst2);

  // Two separately created stripe expressions with the same LHS and RHS are
  // trivially equal.
  auto expr2 = SDBMStripeExpr::get(SDBMSymbolExpr::get(dialect(), 0), cst2);
  EXPECT_EQ(expr, expr2);

  // Stripes can be nested.
  SDBMStripeExpr::get(expr, SDBMConstantExpr::get(dialect(), 4));

  // Non-positive stripe factors are not allowed.
  EXPECT_DEATH(SDBMStripeExpr::get(var, cst0), "non-positive");

  // Hierarchy is okay.
  auto generic = static_cast<SDBMExpr>(expr);
  EXPECT_TRUE(generic.isa<SDBMStripeExpr>());
  EXPECT_TRUE(generic.isa<SDBMPositiveExpr>());
  EXPECT_TRUE(generic.isa<SDBMVaryingExpr>());
}

TEST(SDBMExpr, Neg) {
  auto cst2 = SDBMConstantExpr::get(dialect(), 2);
  auto var = SDBMSymbolExpr::get(dialect(), 0);
  auto stripe = SDBMStripeExpr::get(var, cst2);

  // We can create negation expressions and query them.
  auto expr = SDBMNegExpr::get(var);
  EXPECT_EQ(expr.getVar(), var);
  auto expr2 = SDBMNegExpr::get(stripe);
  EXPECT_EQ(expr2.getVar(), stripe);

  // Neg expressions are trivially comparable.
  EXPECT_EQ(expr, SDBMNegExpr::get(var));

  // Hierarchy is okay.
  auto generic = static_cast<SDBMExpr>(expr);
  EXPECT_TRUE(generic.isa<SDBMNegExpr>());
  EXPECT_TRUE(generic.isa<SDBMVaryingExpr>());
}

TEST(SDBMExpr, Sum) {
  auto cst2 = SDBMConstantExpr::get(dialect(), 2);
  auto var = SDBMSymbolExpr::get(dialect(), 0);
  auto stripe = SDBMStripeExpr::get(var, cst2);

  // We can create sum expressions and query them.
  auto expr = SDBMSumExpr::get(var, cst2);
  EXPECT_EQ(expr.getLHS(), var);
  EXPECT_EQ(expr.getRHS(), cst2);
  auto expr2 = SDBMSumExpr::get(stripe, cst2);
  EXPECT_EQ(expr2.getLHS(), stripe);
  EXPECT_EQ(expr2.getRHS(), cst2);

  // Sum expressions are trivially comparable.
  EXPECT_EQ(expr, SDBMSumExpr::get(var, cst2));

  // Hierarchy is okay.
  auto generic = static_cast<SDBMExpr>(expr);
  EXPECT_TRUE(generic.isa<SDBMSumExpr>());
  EXPECT_TRUE(generic.isa<SDBMVaryingExpr>());
}

TEST(SDBMExpr, Diff) {
  auto cst2 = SDBMConstantExpr::get(dialect(), 2);
  auto var = SDBMSymbolExpr::get(dialect(), 0);
  auto stripe = SDBMStripeExpr::get(var, cst2);

  // We can create sum expressions and query them.
  auto expr = SDBMDiffExpr::get(var, stripe);
  EXPECT_EQ(expr.getLHS(), var);
  EXPECT_EQ(expr.getRHS(), stripe);
  auto expr2 = SDBMDiffExpr::get(stripe, var);
  EXPECT_EQ(expr2.getLHS(), stripe);
  EXPECT_EQ(expr2.getRHS(), var);

  // Sum expressions are trivially comparable.
  EXPECT_EQ(expr, SDBMDiffExpr::get(var, stripe));

  // Hierarchy is okay.
  auto generic = static_cast<SDBMExpr>(expr);
  EXPECT_TRUE(generic.isa<SDBMDiffExpr>());
  EXPECT_TRUE(generic.isa<SDBMVaryingExpr>());
}

TEST(SDBMExpr, AffineRoundTrip) {
  // Build an expression (s0 - s0 # 2)
  auto cst2 = SDBMConstantExpr::get(dialect(), 2);
  auto var = SDBMSymbolExpr::get(dialect(), 0);
  auto stripe = SDBMStripeExpr::get(var, cst2);
  auto expr = SDBMDiffExpr::get(var, stripe);

  // Check that it can be converted to AffineExpr and back, i.e. stripe
  // detection works correctly.
  Optional<SDBMExpr> roundtripped =
      SDBMExpr::tryConvertAffineExpr(expr.getAsAffineExpr());
  ASSERT_TRUE(roundtripped.hasValue());
  EXPECT_EQ(roundtripped, static_cast<SDBMExpr>(expr));

  // Check that (s0 # 2 # 5) can be converted to AffineExpr, i.e. stripe
  // detection supports nested expressions.
  auto cst5 = SDBMConstantExpr::get(dialect(), 5);
  auto outerStripe = SDBMStripeExpr::get(stripe, cst5);
  roundtripped = SDBMExpr::tryConvertAffineExpr(outerStripe.getAsAffineExpr());
  ASSERT_TRUE(roundtripped.hasValue());
  EXPECT_EQ(roundtripped, static_cast<SDBMExpr>(outerStripe));

  // Check that (s0 # 2 # 5 - s0 # 2) + 2 can be converted as an example of a
  // deeper expression tree.
  auto diff = SDBMDiffExpr::get(outerStripe, stripe);
  auto sum = SDBMSumExpr::get(diff, cst2);
  roundtripped = SDBMExpr::tryConvertAffineExpr(sum.getAsAffineExpr());
  ASSERT_TRUE(roundtripped.hasValue());
  EXPECT_EQ(roundtripped, static_cast<SDBMExpr>(sum));
}

TEST(SDBMExpr, MatchStripeMulPattern) {
  // Make sure conversion from AffineExpr recognizes multiplicative stripe
  // pattern (x floordiv B) * B == x # B.
  auto cst = getAffineConstantExpr(42, ctx());
  auto dim = getAffineDimExpr(0, ctx());
  auto floor = dim.floorDiv(cst);
  auto mul = cst * floor;
  Optional<SDBMExpr> converted = SDBMStripeExpr::tryConvertAffineExpr(mul);
  ASSERT_TRUE(converted.hasValue());
  EXPECT_TRUE(converted->isa<SDBMStripeExpr>());
}

TEST(SDBMExpr, NonSDBM) {
  auto d0 = getAffineDimExpr(0, ctx());
  auto d1 = getAffineDimExpr(1, ctx());
  auto sum = d0 + d1;
  auto c2 = getAffineConstantExpr(2, ctx());
  auto prod = d0 * c2;
  auto ceildiv = d1.ceilDiv(c2);

  // The following are not valid SDBM expressions:
  // - a sum of two variables
  EXPECT_FALSE(SDBMExpr::tryConvertAffineExpr(sum).hasValue());
  // - a variable with coefficient other than 1 or -1
  EXPECT_FALSE(SDBMExpr::tryConvertAffineExpr(prod).hasValue());
  // - a ceildiv expression
  EXPECT_FALSE(SDBMExpr::tryConvertAffineExpr(ceildiv).hasValue());
}

} // end namespace
