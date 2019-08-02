//===- TypeUtilities.h - Helper function for type queries -------*- C++ -*-===//
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
// This file defines generic type utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_TYPEUTILITIES_H
#define MLIR_SUPPORT_TYPEUTILITIES_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {

class Attribute;
class TupleType;
class Type;
class Value;

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

/// Return the element type or return the type itself.
Type getElementTypeOrSelf(Type type);

/// Return the element type or return the type itself.
Type getElementTypeOrSelf(Attribute attr);
Type getElementTypeOrSelf(Value *val);
Type getElementTypeOrSelf(Value &val);

/// Get the types within a nested Tuple. A helper for the class method that
/// handles storage concerns, which is tricky to do in tablegen.
SmallVector<Type, 10> getFlattenedTypes(TupleType t);

//===----------------------------------------------------------------------===//
// Utility Iterators
//===----------------------------------------------------------------------===//

// An iterator for the element types of an op's operands of shaped types.
class OperandElementTypeIterator final
    : public llvm::mapped_iterator<OperandIterator, Type (*)(Value *)> {
public:
  using reference = Type;

  /// Initializes the result element type iterator to the specified operand
  /// iterator.
  explicit OperandElementTypeIterator(OperandIterator it);

private:
  static Type unwrap(Value *value);
};

using OperandElementTypeRange =
    llvm::iterator_range<OperandElementTypeIterator>;

// An iterator for the tensor element types of an op's results of shaped types.
class ResultElementTypeIterator final
    : public llvm::mapped_iterator<ResultIterator, Type (*)(Value *)> {
public:
  using reference = Type;

  /// Initializes the result element type iterator to the specified result
  /// iterator.
  explicit ResultElementTypeIterator(ResultIterator it);

private:
  static Type unwrap(Value *value);
};

using ResultElementTypeRange = llvm::iterator_range<ResultElementTypeIterator>;

} // end namespace mlir

#endif // MLIR_SUPPORT_TYPEUTILITIES_H
