//===- AffineMapDetail.h - MLIR Affine Map details Class --------*- C++ -*-===//
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
// This holds implementation details of AffineMap.
//
//===----------------------------------------------------------------------===//

#ifndef AFFINEMAPDETAIL_H_
#define AFFINEMAPDETAIL_H_

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace detail {

struct AffineMapStorage {
  unsigned numDims;
  unsigned numSymbols;

  /// The affine expressions for this (multi-dimensional) map.
  /// TODO: use trailing objects for this.
  ArrayRef<AffineExpr> results;

  MLIRContext *context;
};

} // end namespace detail
} // end namespace mlir

#endif // AFFINEMAPDETAIL_H_
