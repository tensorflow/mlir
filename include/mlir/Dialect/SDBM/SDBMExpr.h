//===- SDBMExpr.h - MLIR SDBM Expression ------------------------*- C++ -*-===//
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
// A striped difference-bound matrix (SDBM) expression is a constant expression,
// an identifier, a binary expression with constant RHS and +, stripe operators
// or a difference expression between two identifiers.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SDBM_SDBMEXPR_H
#define MLIR_DIALECT_SDBM_SDBMEXPR_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMapInfo.h"

namespace mlir {

class AffineExpr;
class MLIRContext;

enum class SDBMExprKind { Add, Stripe, Diff, Constant, DimId, SymbolId, Neg };

namespace detail {
struct SDBMExprStorage;
struct SDBMBinaryExprStorage;
struct SDBMDiffExprStorage;
struct SDBMPositiveExprStorage;
struct SDBMConstantExprStorage;
struct SDBMNegExprStorage;
} // namespace detail

class SDBMConstantExpr;
class SDBMDialect;
class SDBMDimExpr;
class SDBMSymbolExpr;

/// Striped Difference-Bounded Matrix (SDBM) expression is a base left-hand side
/// expression for the SDBM framework.  SDBM expressions are a subset of affine
/// expressions supporting low-complexity algorithms for the operations used in
/// loop transformations.  In particular, are supported:
///   - constant expressions;
///   - single variables (dimensions and symbols) with +1 or -1 coefficient;
///   - stripe expressions: "x # C", where "x" is a single variable or another
///     stripe expression, "#" is the stripe operator, and "C" is a constant
///     expression; "#" is defined as x - x mod C.
///   - sum expressions between single variable/stripe expressions and constant
///     expressions;
///   - difference expressions between single variable/stripe expressions.
/// `SDBMExpr` class hierarchy provides a type-safe interface to constructing
/// and operating on SDBM expressions.  For example, it requires the LHS of a
/// sum expression to be a single variable or a stripe expression.  These
/// restrictions are intended to force the caller to perform the necessary
/// simplifications to stay within the SDBM domain, because SDBM expressions do
/// not combine in more cases than they do.  This choice may be reconsidered in
/// the future.
///
/// `SDBMExpr` and derived classes are thin wrappers around a pointer owned by
/// an MLIRContext, and should be used by-value.  They are uniqued in the
/// MLIRContext and immortal.
class SDBMExpr {
public:
  using ImplType = detail::SDBMExprStorage;
  SDBMExpr() : impl(nullptr) {}
  /* implicit */ SDBMExpr(ImplType *expr) : impl(expr) {}

  /// SDBM expressions are thin wrappers around a unique'ed immutable pointer,
  /// which makes them trivially assignable and trivially copyable.
  SDBMExpr(const SDBMExpr &) = default;
  SDBMExpr &operator=(const SDBMExpr &) = default;

  /// SDBM expressions can be compared straight-forwardly.
  bool operator==(const SDBMExpr &other) const { return impl == other.impl; }
  bool operator!=(const SDBMExpr &other) const { return !(*this == other); }

  /// SDBM expressions are convertible to `bool`: null expressions are converted
  /// to false, non-null expressions are converted to true.
  explicit operator bool() const { return impl != nullptr; }
  bool operator!() const { return !static_cast<bool>(*this); }

  /// Negate the given SDBM expression.
  SDBMExpr operator-();

  /// Prints the SDBM expression.
  void print(raw_ostream &os) const;
  void dump() const;

  /// LLVM-style casts.
  template <typename U> bool isa() const { return U::isClassFor(*this); }
  template <typename U> U dyn_cast() const {
    if (!isa<U>())
      return {};
    return U(const_cast<SDBMExpr *>(this)->impl);
  }
  template <typename U> U cast() const {
    assert(isa<U>() && "cast to incorrect subtype");
    return U(const_cast<SDBMExpr *>(this)->impl);
  }

  /// Support for LLVM hashing.
  ::llvm::hash_code hash_value() const { return ::llvm::hash_value(impl); }

  /// Returns the kind of the SDBM expression.
  SDBMExprKind getKind() const;

  /// Returns the MLIR context in which this expression lives.
  MLIRContext *getContext() const;

  /// Returns the SDBM dialect instance.
  SDBMDialect *getDialect() const;

  /// Convert the SDBM expression into an Affine expression.  This always
  /// succeeds because SDBM are a subset of affine.
  AffineExpr getAsAffineExpr() const;

  /// Try constructing an SDBM expression from the given affine expression.
  /// This may fail if the affine expression is not representable as SDBM, in
  /// which case llvm::None is returned.  The conversion procedure recognizes
  /// (nested) multiplicative ((x floordiv B) * B) and additive (x - x mod B)
  /// patterns for the stripe expression.
  static Optional<SDBMExpr> tryConvertAffineExpr(AffineExpr affine);

protected:
  ImplType *impl;
};

/// SDBM constant expression, wraps a 64-bit integer.
class SDBMConstantExpr : public SDBMExpr {
public:
  using ImplType = detail::SDBMConstantExprStorage;

  using SDBMExpr::SDBMExpr;

  /// Obtain or create a constant expression unique'ed in the given dialect
  /// (which belongs to a context).
  static SDBMConstantExpr get(SDBMDialect *dialect, int64_t value);

  static bool isClassFor(const SDBMExpr &expr) {
    return expr.getKind() == SDBMExprKind::Constant;
  }

  int64_t getValue() const;
};

/// SDBM varying expression can be one of:
///   - input variable expression;
///   - stripe expression;
///   - negation (product with -1) of either of the above.
///   - sum of a varying and a constant expression
///   - difference between varying expressions
class SDBMVaryingExpr : public SDBMExpr {
public:
  using ImplType = detail::SDBMExprStorage;
  using SDBMExpr::SDBMExpr;

  static bool isClassFor(const SDBMExpr &expr) {
    return expr.getKind() == SDBMExprKind::DimId ||
           expr.getKind() == SDBMExprKind::SymbolId ||
           expr.getKind() == SDBMExprKind::Neg ||
           expr.getKind() == SDBMExprKind::Stripe ||
           expr.getKind() == SDBMExprKind::Add ||
           expr.getKind() == SDBMExprKind::Diff;
  }
};

/// SDBM positive variable expression can be one of:
///  - single variable expression;
///  - stripe expression.
class SDBMPositiveExpr : public SDBMVaryingExpr {
public:
  using SDBMVaryingExpr::SDBMVaryingExpr;

  static bool isClassFor(const SDBMExpr &expr) {
    return expr.getKind() == SDBMExprKind::DimId ||
           expr.getKind() == SDBMExprKind::SymbolId ||
           expr.getKind() == SDBMExprKind::Stripe;
  }
};

/// SDBM sum expression.  LHS is a varying expression and RHS is always a
/// constant expression.
class SDBMSumExpr : public SDBMVaryingExpr {
public:
  using ImplType = detail::SDBMBinaryExprStorage;
  using SDBMVaryingExpr::SDBMVaryingExpr;

  /// Obtain or create a sum expression unique'ed in the given context.
  static SDBMSumExpr get(SDBMVaryingExpr lhs, SDBMConstantExpr rhs);

  static bool isClassFor(const SDBMExpr &expr) {
    SDBMExprKind kind = expr.getKind();
    return kind == SDBMExprKind::Add;
  }

  SDBMVaryingExpr getLHS() const;
  SDBMConstantExpr getRHS() const;
};

/// SDBM difference expression.  Both LHS and RHS are positive variable
/// expressions.
class SDBMDiffExpr : public SDBMVaryingExpr {
public:
  using ImplType = detail::SDBMDiffExprStorage;
  using SDBMVaryingExpr::SDBMVaryingExpr;

  /// Obtain or create a difference expression unique'ed in the given context.
  static SDBMDiffExpr get(SDBMPositiveExpr lhs, SDBMPositiveExpr rhs);

  static bool isClassFor(const SDBMExpr &expr) {
    return expr.getKind() == SDBMExprKind::Diff;
  }

  SDBMPositiveExpr getLHS() const;
  SDBMPositiveExpr getRHS() const;
};

/// SDBM stripe expression "x # C" where "x" is a positive variable expression,
/// "C" is a constant expression and "#" is the stripe operator defined as:
///   x # C = x - x mod C.
class SDBMStripeExpr : public SDBMPositiveExpr {
public:
  using ImplType = detail::SDBMBinaryExprStorage;
  using SDBMPositiveExpr::SDBMPositiveExpr;

  static bool isClassFor(const SDBMExpr &expr) {
    return expr.getKind() == SDBMExprKind::Stripe;
  }

  static SDBMStripeExpr get(SDBMPositiveExpr var,
                            SDBMConstantExpr stripeFactor);

  SDBMPositiveExpr getVar() const;
  SDBMConstantExpr getStripeFactor() const;
};

/// SDBM "input" variable expression can be either a dimension identifier or
/// a symbol identifier.  When used to define SDBM functions, dimensions are
/// interpreted as function arguments while symbols are treated as unknown but
/// constant values, hence the name.
class SDBMInputExpr : public SDBMPositiveExpr {
public:
  using ImplType = detail::SDBMPositiveExprStorage;
  using SDBMPositiveExpr::SDBMPositiveExpr;

  static bool isClassFor(const SDBMExpr &expr) {
    return expr.getKind() == SDBMExprKind::DimId ||
           expr.getKind() == SDBMExprKind::SymbolId;
  }

  unsigned getPosition() const;
};

/// SDBM dimension expression.  Dimensions correspond to function arguments
/// when defining functions using SDBM expressions.
class SDBMDimExpr : public SDBMInputExpr {
public:
  using ImplType = detail::SDBMPositiveExprStorage;
  using SDBMInputExpr::SDBMInputExpr;

  /// Obtain or create a dimension expression unique'ed in the given dialect
  /// (which belongs to a context).
  static SDBMDimExpr get(SDBMDialect *dialect, unsigned position);

  static bool isClassFor(const SDBMExpr &expr) {
    return expr.getKind() == SDBMExprKind::DimId;
  }
};

/// SDBM symbol expression.  Symbols correspond to symbolic constants when
/// defining functions using SDBM expressions.
class SDBMSymbolExpr : public SDBMInputExpr {
public:
  using ImplType = detail::SDBMPositiveExprStorage;
  using SDBMInputExpr::SDBMInputExpr;

  /// Obtain or create a symbol expression unique'ed in the given dialect (which
  /// belongs to a context).
  static SDBMSymbolExpr get(SDBMDialect *dialect, unsigned position);

  static bool isClassFor(const SDBMExpr &expr) {
    return expr.getKind() == SDBMExprKind::SymbolId;
  }
};

/// Negation of an SDBM variable expression.  Equivalent to multiplying the
/// expression with -1 (SDBM does not support other coefficients that 1 and -1).
class SDBMNegExpr : public SDBMVaryingExpr {
public:
  using ImplType = detail::SDBMNegExprStorage;
  using SDBMVaryingExpr::SDBMVaryingExpr;

  /// Obtain or create a negation expression unique'ed in the given context.
  static SDBMNegExpr get(SDBMPositiveExpr var);

  static bool isClassFor(const SDBMExpr &expr) {
    return expr.getKind() == SDBMExprKind::Neg;
  }

  SDBMPositiveExpr getVar() const;
};

/// A visitor class for SDBM expressions.  Calls the kind-specific function
/// depending on the kind of expression it visits.
template <typename Derived, typename Result = void> class SDBMVisitor {
public:
  /// Visit the given SDBM expression, dispatching to kind-specific functions.
  Result visit(SDBMExpr expr) {
    auto *derived = static_cast<Derived *>(this);
    switch (expr.getKind()) {
    case SDBMExprKind::Add:
    case SDBMExprKind::Diff:
    case SDBMExprKind::DimId:
    case SDBMExprKind::SymbolId:
    case SDBMExprKind::Neg:
    case SDBMExprKind::Stripe:
      return derived->visitVarying(expr.cast<SDBMVaryingExpr>());
    case SDBMExprKind::Constant:
      return derived->visitConstant(expr.cast<SDBMConstantExpr>());
    }

    llvm_unreachable("unsupported SDBM expression kind");
  }

  /// Traverse the SDBM expression tree calling `visit` on each node
  /// in depth-first preorder.
  void walkPreorder(SDBMExpr expr) { return walk</*isPreorder=*/true>(expr); }

  /// Traverse the SDBM expression tree calling `visit` on each node in
  /// depth-first postorder.
  void walkPostorder(SDBMExpr expr) { return walk</*isPreorder=*/false>(expr); }

protected:
  /// Default visitors do nothing.
  void visitSum(SDBMSumExpr) {}
  void visitDiff(SDBMDiffExpr) {}
  void visitStripe(SDBMStripeExpr) {}
  void visitDim(SDBMDimExpr) {}
  void visitSymbol(SDBMSymbolExpr) {}
  void visitNeg(SDBMNegExpr) {}
  void visitConstant(SDBMConstantExpr) {}

  /// Default implementation of visitPositive dispatches to the special
  /// functions for stripes and other variables.  Concrete visitors can override
  /// it.
  Result visitPositive(SDBMPositiveExpr expr) {
    auto *derived = static_cast<Derived *>(this);
    if (expr.getKind() == SDBMExprKind::Stripe)
      return derived->visitStripe(expr.cast<SDBMStripeExpr>());
    else
      return derived->visitInput(expr.cast<SDBMInputExpr>());
  }

  /// Default implementation of visitInput dispatches to the special
  /// functions for dimensions or symbols.  Concrete visitors can override it to
  /// visit all variables instead.
  Result visitInput(SDBMInputExpr expr) {
    auto *derived = static_cast<Derived *>(this);
    if (expr.getKind() == SDBMExprKind::DimId)
      return derived->visitDim(expr.cast<SDBMDimExpr>());
    else
      return derived->visitSymbol(expr.cast<SDBMSymbolExpr>());
  }

  /// Default implementation of visitVarying dispatches to the special
  /// functions for variables and negations thereof.  Concerete visitors can
  /// override it to visit all variables and negations instead.
  Result visitVarying(SDBMVaryingExpr expr) {
    auto *derived = static_cast<Derived *>(this);
    if (auto var = expr.dyn_cast<SDBMPositiveExpr>())
      return derived->visitPositive(var);
    else if (auto neg = expr.dyn_cast<SDBMNegExpr>())
      return derived->visitNeg(neg);
    else if (auto sum = expr.dyn_cast<SDBMSumExpr>())
      return derived->visitSum(sum);
    else if (auto diff = expr.dyn_cast<SDBMDiffExpr>())
      return derived->visitDiff(diff);

    llvm_unreachable("unhandled subtype of varying SDBM expression");
  }

  template <bool isPreorder> void walk(SDBMExpr expr) {
    if (isPreorder)
      visit(expr);
    if (auto sumExpr = expr.dyn_cast<SDBMSumExpr>()) {
      walk<isPreorder>(sumExpr.getLHS());
      walk<isPreorder>(sumExpr.getRHS());
    } else if (auto diffExpr = expr.dyn_cast<SDBMDiffExpr>()) {
      walk<isPreorder>(diffExpr.getLHS());
      walk<isPreorder>(diffExpr.getRHS());
    } else if (auto stripeExpr = expr.dyn_cast<SDBMStripeExpr>()) {
      walk<isPreorder>(stripeExpr.getVar());
      walk<isPreorder>(stripeExpr.getStripeFactor());
    } else if (auto negExpr = expr.dyn_cast<SDBMNegExpr>()) {
      walk<isPreorder>(negExpr.getVar());
    }
    if (!isPreorder)
      visit(expr);
  }
};

/// Overloaded arithmetic operators for SDBM expressions asserting that their
/// arguments have the proper SDBM expression subtype.  Perform canonicalization
/// and constant folding on these expressions.
namespace ops_assertions {

/// Add two SDBM expressions.  At least one of the expressions must be a
/// constant or a negation, but both expressions cannot be negations
/// simultaneously.
SDBMExpr operator+(SDBMExpr lhs, SDBMExpr rhs);
inline SDBMExpr operator+(SDBMExpr lhs, int64_t rhs) {
  return lhs + SDBMConstantExpr::get(lhs.getDialect(), rhs);
}
inline SDBMExpr operator+(int64_t lhs, SDBMExpr rhs) {
  return SDBMConstantExpr::get(rhs.getDialect(), lhs) + rhs;
}

/// Subtract an SDBM expression from another SDBM expression.  Both expressions
/// must not be difference expressions.
SDBMExpr operator-(SDBMExpr lhs, SDBMExpr rhs);
inline SDBMExpr operator-(SDBMExpr lhs, int64_t rhs) {
  return lhs - SDBMConstantExpr::get(lhs.getDialect(), rhs);
}
inline SDBMExpr operator-(int64_t lhs, SDBMExpr rhs) {
  return SDBMConstantExpr::get(rhs.getDialect(), lhs) - rhs;
}

/// Construct a stripe expression from a positive expression and a positive
/// constant stripe factor.
SDBMExpr stripe(SDBMExpr expr, SDBMExpr factor);
inline SDBMExpr stripe(SDBMExpr expr, int64_t factor) {
  return stripe(expr, SDBMConstantExpr::get(expr.getDialect(), factor));
}
} // namespace ops_assertions

} // end namespace mlir

namespace llvm {
// SDBMExpr hash just like pointers.
template <> struct DenseMapInfo<mlir::SDBMExpr> {
  static mlir::SDBMExpr getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::SDBMExpr(static_cast<mlir::SDBMExpr::ImplType *>(pointer));
  }
  static mlir::SDBMExpr getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::SDBMExpr(static_cast<mlir::SDBMExpr::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::SDBMExpr expr) {
    return expr.hash_value();
  }
  static bool isEqual(mlir::SDBMExpr lhs, mlir::SDBMExpr rhs) {
    return lhs == rhs;
  }
};

// SDBMVaryingExpr hash just like pointers.
template <> struct DenseMapInfo<mlir::SDBMVaryingExpr> {
  static mlir::SDBMVaryingExpr getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::SDBMVaryingExpr(
        static_cast<mlir::SDBMExpr::ImplType *>(pointer));
  }
  static mlir::SDBMVaryingExpr getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::SDBMVaryingExpr(
        static_cast<mlir::SDBMExpr::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::SDBMVaryingExpr expr) {
    return expr.hash_value();
  }
  static bool isEqual(mlir::SDBMVaryingExpr lhs, mlir::SDBMVaryingExpr rhs) {
    return lhs == rhs;
  }
};

// SDBMPositiveExpr hash just like pointers.
template <> struct DenseMapInfo<mlir::SDBMPositiveExpr> {
  static mlir::SDBMPositiveExpr getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::SDBMPositiveExpr(
        static_cast<mlir::SDBMExpr::ImplType *>(pointer));
  }
  static mlir::SDBMPositiveExpr getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::SDBMPositiveExpr(
        static_cast<mlir::SDBMExpr::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::SDBMPositiveExpr expr) {
    return expr.hash_value();
  }
  static bool isEqual(mlir::SDBMPositiveExpr lhs, mlir::SDBMPositiveExpr rhs) {
    return lhs == rhs;
  }
};

// SDBMConstantExpr hash just like pointers.
template <> struct DenseMapInfo<mlir::SDBMConstantExpr> {
  static mlir::SDBMConstantExpr getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::SDBMConstantExpr(
        static_cast<mlir::SDBMExpr::ImplType *>(pointer));
  }
  static mlir::SDBMConstantExpr getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::SDBMConstantExpr(
        static_cast<mlir::SDBMExpr::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::SDBMConstantExpr expr) {
    return expr.hash_value();
  }
  static bool isEqual(mlir::SDBMConstantExpr lhs, mlir::SDBMConstantExpr rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

#endif // MLIR_DIALECT_SDBM_SDBMEXPR_H
