//===- LLVMDialect.h - MLIR LLVM IR dialect ---------------------*- C++ -*-===//
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
// This file defines the LLVM IR dialect in MLIR, containing LLVM operations and
// LLVM type system.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVMDIALECT_H_
#define MLIR_TARGET_LLVMDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

#include "mlir/LLVMIR/LLVMOpsEnums.h.inc"

namespace llvm {
class Type;
class LLVMContext;
} // end namespace llvm

namespace mlir {
namespace LLVM {
class LLVMDialect;

namespace detail {
struct LLVMTypeStorage;
struct LLVMDialectImpl;
} // namespace detail

class LLVMType : public mlir::Type::TypeBase<LLVMType, mlir::Type,
                                             detail::LLVMTypeStorage> {
public:
  enum Kind {
    LLVM_TYPE = FIRST_LLVM_TYPE,
  };

  using Base::Base;

  static bool kindof(unsigned kind) { return kind == LLVM_TYPE; }

  LLVMDialect &getDialect();
  llvm::Type *getUnderlyingType() const;

  /// Array type utilities.
  LLVMType getArrayElementType();

  /// Pointer type utilities.
  LLVMType getPointerTo(unsigned addrSpace = 0);
  LLVMType getPointerElementTy();

  /// Struct type utilities.
  LLVMType getStructElementType(unsigned i);

  /// Utilities used to generate floating point types.
  static LLVMType getDoubleTy(LLVMDialect *dialect);
  static LLVMType getFloatTy(LLVMDialect *dialect);
  static LLVMType getHalfTy(LLVMDialect *dialect);

  /// Utilities used to generate integer types.
  static LLVMType getIntNTy(LLVMDialect *dialect, unsigned numBits);
  static LLVMType getInt1Ty(LLVMDialect *dialect) {
    return getIntNTy(dialect, /*numBits=*/1);
  }
  static LLVMType getInt8Ty(LLVMDialect *dialect) {
    return getIntNTy(dialect, /*numBits=*/8);
  }
  static LLVMType getInt8PtrTy(LLVMDialect *dialect) {
    return getInt8Ty(dialect).getPointerTo();
  }
  static LLVMType getInt16Ty(LLVMDialect *dialect) {
    return getIntNTy(dialect, /*numBits=*/16);
  }
  static LLVMType getInt32Ty(LLVMDialect *dialect) {
    return getIntNTy(dialect, /*numBits=*/32);
  }
  static LLVMType getInt64Ty(LLVMDialect *dialect) {
    return getIntNTy(dialect, /*numBits=*/64);
  }

  /// Utilities used to generate other miscellaneous types.
  static LLVMType getArrayTy(LLVMType elementType, uint64_t numElements);
  static LLVMType getFunctionTy(LLVMType result, ArrayRef<LLVMType> params,
                                bool isVarArg);
  static LLVMType getFunctionTy(LLVMType result, bool isVarArg) {
    return getFunctionTy(result, llvm::None, isVarArg);
  }
  static LLVMType getStructTy(LLVMDialect *dialect, ArrayRef<LLVMType> elements,
                              bool isPacked = false);
  static LLVMType getStructTy(LLVMDialect *dialect, bool isPacked = false) {
    return getStructTy(dialect, llvm::None, isPacked);
  }
  template <typename... Args>
  static typename std::enable_if<llvm::are_base_of<LLVMType, Args...>::value,
                                 LLVMType>::type
  getStructTy(LLVMType elt1, Args... elts) {
    SmallVector<LLVMType, 8> fields({elt1, elts...});
    return getStructTy(&elt1.getDialect(), fields);
  }
  static LLVMType getVectorTy(LLVMType elementType, unsigned numElements);
  static LLVMType getVoidTy(LLVMDialect *dialect);

private:
  friend LLVMDialect;

  /// Get an LLVMType with a pre-existing llvm type.
  static LLVMType get(MLIRContext *context, llvm::Type *llvmType);

  /// Get an LLVMType with an llvm type that may cause changes to the underlying
  /// llvm context when constructed.
  static LLVMType getLocked(LLVMDialect *dialect,
                            llvm::function_ref<llvm::Type *()> typeBuilder);
};

///// Ops /////
#define GET_OP_CLASSES
#include "mlir/LLVMIR/LLVMOps.h.inc"

class LLVMDialect : public Dialect {
public:
  explicit LLVMDialect(MLIRContext *context);
  ~LLVMDialect();
  static StringRef getDialectNamespace() { return "llvm"; }

  llvm::LLVMContext &getLLVMContext();
  llvm::Module &getLLVMModule();

  /// Parse a type registered to this dialect.
  Type parseType(StringRef tyData, Location loc) const override;

  /// Print a type registered to this dialect.
  void printType(Type type, raw_ostream &os) const override;

  /// Verify a region argument attribute registered to this dialect.
  /// Returns failure if the verification failed, success otherwise.
  LogicalResult verifyRegionArgAttribute(Operation *op, unsigned regionIdx,
                                         unsigned argIdx,
                                         NamedAttribute argAttr) override;

private:
  friend LLVMType;

  std::unique_ptr<detail::LLVMDialectImpl> impl;
};

} // end namespace LLVM
} // end namespace mlir

#endif // MLIR_TARGET_LLVMDIALECT_H_
