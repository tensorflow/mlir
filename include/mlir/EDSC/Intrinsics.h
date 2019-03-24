//===- Intrinsics.h - MLIR Operations for Declarative Builders ---*- C++-*-===//
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
// Provides intuitive composable intrinsics for building snippets of MLIR
// declaratively
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EDSC_INTRINSICS_H_
#define MLIR_EDSC_INTRINSICS_H_

#include "mlir/EDSC/Builders.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

class MemRefType;
class Type;

namespace edsc {

/// An IndexHandle is a simple wrapper around a ValueHandle.
/// IndexHandles are ubiquitous enough to justify a new type to allow simple
/// declarations without boilerplate such as:
///
/// ```c++
///    IndexHandle i, j, k;
/// ```
struct IndexHandle : public ValueHandle {
  explicit IndexHandle()
      : ValueHandle(ScopedContext::getBuilder()->getIndexType()) {}
  explicit IndexHandle(index_t v) : ValueHandle(v) {}
  explicit IndexHandle(Value *v) : ValueHandle(v) {
    assert(v->getType() == ScopedContext::getBuilder()->getIndexType() &&
           "Expected index type");
  }
  explicit IndexHandle(ValueHandle v) : ValueHandle(v) {
    assert(v.getType() == ScopedContext::getBuilder()->getIndexType() &&
           "Expected index type");
  }
  IndexHandle &operator=(const ValueHandle &v) {
    assert(v.getType() == ScopedContext::getBuilder()->getIndexType() &&
           "Expected index type");
    /// Creating a new IndexHandle(v) and then std::swap rightly complains the
    /// binding has already occurred and that we should use another name.
    this->t = v.getType();
    this->v = v.getValue();
    return *this;
  }
  static SmallVector<IndexHandle, 8> makeIndexHandles(unsigned rank) {
    return SmallVector<IndexHandle, 8>(rank);
  }
  static SmallVector<ValueHandle *, 8>
  makeIndexHandlePointers(SmallVectorImpl<IndexHandle> &ivs) {
    SmallVector<ValueHandle *, 8> pivs;
    pivs.reserve(ivs.size());
    for (auto &iv : ivs) {
      pivs.push_back(&iv);
    }
    return pivs;
  }
};

/// Provides a set of first class intrinsics.
/// In the future, most of intrinsics reated to Instruction that don't contain
/// other instructions should be Tablegen'd.
namespace intrinsics {
namespace detail {
/// Helper structure to be used with ValueBuilder / InstructionBuilder.
/// It serves the purpose of removing boilerplate specialization for the sole
/// purpose of implicitly converting ArrayRef<ValueHandle> -> ArrayRef<Value*>.
class ValueHandleArray {
public:
  ValueHandleArray(ArrayRef<ValueHandle> vals) {
    values.append(vals.begin(), vals.end());
  }
  ValueHandleArray(ArrayRef<IndexHandle> vals) {
    values.append(vals.begin(), vals.end());
  }
  ValueHandleArray(ArrayRef<index_t> vals) {
    llvm::SmallVector<IndexHandle, 8> tmp(vals.begin(), vals.end());
    values.append(tmp.begin(), tmp.end());
  }
  operator ArrayRef<Value *>() { return values; }

private:
  ValueHandleArray() = default;
  llvm::SmallVector<Value *, 8> values;
};

template <typename T> inline T unpack(T value) { return value; }

inline detail::ValueHandleArray unpack(ArrayRef<ValueHandle> values) {
  return detail::ValueHandleArray(values);
}

} // namespace detail

/// Helper variadic abstraction to allow extending to any MLIR op without
/// boilerplate or Tablegen.
/// Arguably a builder is not a ValueHandle but in practice it is only used as
/// an alias to a notional ValueHandle<Op>.
/// Implementing it as a subclass allows it to compose all the way to Value*.
/// Without subclassing, implicit conversion to Value* would fail when composing
/// in patterns such as: `select(a, b, select(c, d, e))`.
template <typename Op> struct ValueBuilder : public ValueHandle {
  template <typename... Args>
  ValueBuilder(Args... args)
      : ValueHandle(ValueHandle::create<Op>(detail::unpack(args)...)) {}
  ValueBuilder(ArrayRef<ValueHandle> vs)
      : ValueBuilder(ValueBuilder::create<Op>(detail::unpack(vs))) {}
  template <typename... Args>
  ValueBuilder(ArrayRef<ValueHandle> vs, Args... args)
      : ValueHandle(ValueHandle::create<Op>(detail::unpack(vs),
                                            detail::unpack(args)...)) {}
  template <typename T, typename... Args>
  ValueBuilder(T t, ArrayRef<ValueHandle> vs, Args... args)
      : ValueHandle(ValueHandle::create<Op>(
            detail::unpack(t), detail::unpack(vs), detail::unpack(args)...)) {}
  template <typename T1, typename T2, typename... Args>
  ValueBuilder(T1 t1, T2 t2, ArrayRef<ValueHandle> vs, Args... args)
      : ValueHandle(ValueHandle::create<Op>(
            detail::unpack(t1), detail::unpack(t2), detail::unpack(vs),
            detail::unpack(args)...)) {}
  ValueBuilder() : ValueHandle(ValueHandle::create<Op>()) {}
};

template <typename Op> struct InstructionBuilder : public InstructionHandle {
  template <typename... Args>
  InstructionBuilder(Args... args)
      : InstructionHandle(
            InstructionHandle::create<Op>(detail::unpack(args)...)) {}
  InstructionBuilder(ArrayRef<ValueHandle> vs)
      : InstructionHandle(InstructionHandle::create<Op>(detail::unpack(vs))) {}
  template <typename... Args>
  InstructionBuilder(ArrayRef<ValueHandle> vs, Args... args)
      : InstructionHandle(InstructionHandle::create<Op>(
            detail::unpack(vs), detail::unpack(args)...)) {}
  template <typename T, typename... Args>
  InstructionBuilder(T t, ArrayRef<ValueHandle> vs, Args... args)
      : InstructionHandle(InstructionHandle::create<Op>(
            detail::unpack(t), detail::unpack(vs), detail::unpack(args)...)) {}
  template <typename T1, typename T2, typename... Args>
  InstructionBuilder(T1 t1, T2 t2, ArrayRef<ValueHandle> vs, Args... args)
      : InstructionHandle(InstructionHandle::create<Op>(
            detail::unpack(t1), detail::unpack(t2), detail::unpack(vs),
            detail::unpack(args)...)) {}
  InstructionBuilder() : InstructionHandle(InstructionHandle::create<Op>()) {}
};

using alloc = ValueBuilder<AllocOp>;
using constant_float = ValueBuilder<ConstantFloatOp>;
using constant_index = ValueBuilder<ConstantIndexOp>;
using constant_int = ValueBuilder<ConstantIntOp>;
using dealloc = InstructionBuilder<DeallocOp>;
using load = ValueBuilder<LoadOp>;
using ret = InstructionBuilder<ReturnOp>;
using select = ValueBuilder<SelectOp>;
using store = InstructionBuilder<StoreOp>;
using vector_type_cast = ValueBuilder<VectorTypeCastOp>;

/// Branches into the mlir::Block* captured by BlockHandle `b` with `operands`.
///
/// Prerequisites:
///   All Handles have already captured previously constructed IR objects.
InstructionHandle br(BlockHandle bh, ArrayRef<ValueHandle> operands);

/// Creates a new mlir::Block* and branches to it from the current block.
/// Argument types are specified by `operands`.
/// Captures the new block in `bh` and the actual `operands` in `captures`. To
/// insert the new mlir::Block*, a local ScopedContext is constructed and
/// released to the current block. The branch instruction is then added to the
/// new block.
///
/// Prerequisites:
///   `b` has not yet captured an mlir::Block*.
///   No `captures` have captured any mlir::Value*.
///   All `operands` have already captured an mlir::Value*
///   captures.size() == operands.size()
///   captures and operands are pairwise of the same type.
InstructionHandle br(BlockHandle *bh, ArrayRef<ValueHandle *> captures,
                     ArrayRef<ValueHandle> operands);

/// Branches into the mlir::Block* captured by BlockHandle `trueBranch` with
/// `trueOperands` if `cond` evaluates to `true` (resp. `falseBranch` and
/// `falseOperand` if `cond` evaluates to `false`).
///
/// Prerequisites:
///   All Handles have captured previouly constructed IR objects.
InstructionHandle cond_br(ValueHandle cond, BlockHandle trueBranch,
                          ArrayRef<ValueHandle> trueOperands,
                          BlockHandle falseBranch,
                          ArrayRef<ValueHandle> falseOperands);

/// Eagerly creates new mlir::Block* with argument types specified by
/// `trueOperands`/`falseOperands`.
/// Captures the new blocks in `trueBranch`/`falseBranch` and the arguments in
/// `trueCaptures/falseCaptures`.
/// To insert the new mlir::Block*, a local ScopedContext is constructed and
/// released. The branch instruction is then added in the original location and
/// targeting the eagerly constructed blocks.
///
/// Prerequisites:
///   `trueBranch`/`falseBranch` has not yet captured an mlir::Block*.
///   No `trueCaptures`/`falseCaptures` have captured any mlir::Value*.
///   All `trueOperands`/`trueOperands` have already captured an mlir::Value*
///   `trueCaptures`.size() == `trueOperands`.size()
///   `falseCaptures`.size() == `falseOperands`.size()
///   `trueCaptures` and `trueOperands` are pairwise of the same type
///   `falseCaptures` and `falseOperands` are pairwise of the same type.
InstructionHandle cond_br(ValueHandle cond, BlockHandle *trueBranch,
                          ArrayRef<ValueHandle *> trueCaptures,
                          ArrayRef<ValueHandle> trueOperands,
                          BlockHandle *falseBranch,
                          ArrayRef<ValueHandle *> falseCaptures,
                          ArrayRef<ValueHandle> falseOperands);
} // namespace intrinsics
} // namespace edsc
} // namespace mlir

#endif // MLIR_EDSC_INTRINSICS_H_
