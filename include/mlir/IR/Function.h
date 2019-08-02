//===- Function.h - MLIR Function Class -------------------------*- C++ -*-===//
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
// Functions are the basic unit of composition in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_FUNCTION_H
#define MLIR_IR_FUNCTION_H

#include "mlir/IR/Block.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
//===--------------------------------------------------------------------===//
// Function Operation.
//===--------------------------------------------------------------------===//

/// FuncOp represents a function, or an operation containing one region that
/// forms a CFG(Control Flow Graph). The region of a function is not allowed to
/// implicitly capture global values, and all external references must use
/// Function arguments or attributes that establish a symbolic connection(e.g.
/// symbols referenced by name via a string attribute).
class FuncOp : public Op<FuncOp, OpTrait::ZeroOperands, OpTrait::ZeroResult,
                         OpTrait::IsIsolatedFromAbove, OpTrait::FunctionLike> {
public:
  using Op::Op;
  using Op::print;

  static StringRef getOperationName() { return "func"; }

  static FuncOp create(Location location, StringRef name, FunctionType type,
                       ArrayRef<NamedAttribute> attrs = {});
  static FuncOp create(Location location, StringRef name, FunctionType type,
                       llvm::iterator_range<dialect_attr_iterator> attrs);
  static FuncOp create(Location location, StringRef name, FunctionType type,
                       ArrayRef<NamedAttribute> attrs,
                       ArrayRef<NamedAttributeList> argAttrs);

  static void build(Builder *builder, OperationState *result, StringRef name,
                    FunctionType type, ArrayRef<NamedAttribute> attrs);
  static void build(Builder *builder, OperationState *result, StringRef name,
                    FunctionType type, ArrayRef<NamedAttribute> attrs,
                    ArrayRef<NamedAttributeList> argAttrs);

  /// Operation hooks.
  static ParseResult parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p);
  LogicalResult verify();

  /// Returns the type of this function.
  FunctionType getType() {
    return getAttrOfType<TypeAttr>(getTypeAttrName())
        .getValue()
        .cast<FunctionType>();
  }

  /// Change the type of this function in place. This is an extremely dangerous
  /// operation and it is up to the caller to ensure that this is legal for this
  /// function, and to restore invariants:
  ///  - the entry block args must be updated to match the function params.
  ///  - the arguments attributes may need an update: if the new type has less
  ///    parameters we drop the extra attributes, if there are more parameters
  ///    they won't have any attributes.
  void setType(FunctionType newType) {
    setAttr(getTypeAttrName(), TypeAttr::get(newType));
  }

  /// Create a deep copy of this function and all of its blocks, remapping
  /// any operands that use values outside of the function using the map that is
  /// provided (leaving them alone if no entry is present). If the mapper
  /// contains entries for function arguments, these arguments are not included
  /// in the new function. Replaces references to cloned sub-values with the
  /// corresponding value that is copied, and adds those mappings to the mapper.
  FuncOp clone(BlockAndValueMapping &mapper);
  FuncOp clone();

  /// Clone the internal blocks and attributes from this function into dest. Any
  /// cloned blocks are appended to the back of dest. This function asserts that
  /// the attributes of the current function and dest are compatible.
  void cloneInto(FuncOp dest, BlockAndValueMapping &mapper);

  //===--------------------------------------------------------------------===//
  // Body Handling
  //===--------------------------------------------------------------------===//

  /// Add an entry block to an empty function, and set up the block arguments
  /// to match the signature of the function.
  void addEntryBlock();

private:
  // This trait needs access to `getNumFuncArguments` and `verifyType` hooks
  // defined below.
  friend class OpTrait::FunctionLike<FuncOp>;

  /// Returns the number of arguments. This is a hook for OpTrait::FunctionLike.
  unsigned getNumFuncArguments() { return getType().getInputs().size(); }

  /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
  /// attribute is present and checks if it holds a function type.  Ensures
  /// getType and getNumFuncArguments can be called safely.
  LogicalResult verifyType() {
    auto type = getTypeAttr().getValue();
    if (!type.isa<FunctionType>())
      return emitOpError("requires '" + getTypeAttrName() +
                         "' attribute of function type");
    return success();
  }
};
} // end namespace mlir

namespace llvm {

// Functions hash just like pointers.
template <> struct DenseMapInfo<mlir::FuncOp> {
  static mlir::FuncOp getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::FuncOp::getFromOpaquePointer(pointer);
  }
  static mlir::FuncOp getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::FuncOp::getFromOpaquePointer(pointer);
  }
  static unsigned getHashValue(mlir::FuncOp val) {
    return hash_value(val.getAsOpaquePointer());
  }
  static bool isEqual(mlir::FuncOp LHS, mlir::FuncOp RHS) { return LHS == RHS; }
};

/// Allow stealing the low bits of FuncOp.
template <> struct PointerLikeTypeTraits<mlir::FuncOp> {
public:
  static inline void *getAsVoidPointer(mlir::FuncOp I) {
    return const_cast<void *>(I.getAsOpaquePointer());
  }
  static inline mlir::FuncOp getFromVoidPointer(void *P) {
    return mlir::FuncOp::getFromOpaquePointer(P);
  }
  enum { NumLowBitsAvailable = 3 };
};

} // namespace llvm

#endif // MLIR_IR_FUNCTION_H
