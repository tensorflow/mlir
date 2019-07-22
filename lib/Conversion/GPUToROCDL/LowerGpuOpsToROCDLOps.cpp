//===- LowerGpuOpsToROCDLOps.cpp - MLIR GPU to ROCDL lowering passes --------===//
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
// This file implements a pass to generate ROCDLIR operations for higher-level
// GPU operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/GPU/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/LLVMIR/LLVMDialect.h"
#include "mlir/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/StringSwitch.h"

namespace mlir {
namespace {

// A pass that replaces all occurences of GPU operations with their
// corresponding ROCDL equivalent.
//
// This pass does not handle launching of kernels. Instead, it is meant to be
// used on the body region of a launch or the body region of a kernel
// function.
class LowerGpuOpsToROCDLOpsPass : public FunctionPass<LowerGpuOpsToROCDLOpsPass> {
private:
  enum dimension { X = 0, Y = 1, Z = 2, invalid };

  template <typename T> dimension dimensionToIndex(T op) {
    return llvm::StringSwitch<dimension>(op.dimension())
        .Case("x", X)
        .Case("y", Y)
        .Case("z", Z)
        .Default(invalid);
  }

  // Helper that replaces Op with XOp, YOp, or ZOp dependeing on the dimension
  // that Op operates on.  Op is assumed to return an `std.index` value and
  // XOp, YOp and ZOp are assumed to return an `llvm.i32` value.  Depending on
  // `indexBitwidth`, sign-extend or truncate the resulting value to match the
  // bitwidth expected by the consumers of the value.
  template <typename XOp, typename YOp, typename ZOp, class Op>
  void replaceWithIntrinsicOrDeviceFunction(Op operation, LLVM::LLVMDialect *dialect,
                            unsigned indexBitwidth) {
    assert(operation.getType().isIndex() &&
           "expected an operation returning index");
    OpBuilder builder(operation);
    auto loc = operation.getLoc();
    Value *newOp;
    switch (dimensionToIndex(operation)) {
    case X:
      newOp = builder.create<XOp>(loc, LLVM::LLVMType::getInt32Ty(dialect));
      break;
    case Y:
      newOp = builder.create<YOp>(loc, LLVM::LLVMType::getInt32Ty(dialect));
      break;
    case Z:
      newOp = builder.create<ZOp>(loc, LLVM::LLVMType::getInt32Ty(dialect));
      break;
    default:
      operation.emitError("Illegal dimension: " + operation.dimension());
      signalPassFailure();
      return;
    }

    if (indexBitwidth > 32) {
      newOp = builder.create<LLVM::SExtOp>(
          loc, LLVM::LLVMType::getIntNTy(dialect, indexBitwidth), newOp);
    } else if (indexBitwidth < 32) {
      newOp = builder.create<LLVM::TruncOp>(
          loc, LLVM::LLVMType::getIntNTy(dialect, indexBitwidth), newOp);
    }
    operation.replaceAllUsesWith(newOp);
    operation.erase();
  }

public:
  void runOnFunction() {
    LLVM::LLVMDialect *llvmDialect =
        getContext().getRegisteredDialect<LLVM::LLVMDialect>();
    unsigned indexBitwidth =
        llvmDialect->getLLVMModule().getDataLayout().getPointerSizeInBits();
    getFunction().walk([&](Operation *opInst) {
      if (auto threadId = dyn_cast<gpu::ThreadId>(opInst)) {
        replaceWithIntrinsicOrDeviceFunction<ROCDL::ThreadIdXOp, ROCDL::ThreadIdYOp,
                             ROCDL::ThreadIdZOp>(threadId, llvmDialect,
                                                indexBitwidth);
        return;
      }
      if (auto blockId = dyn_cast<gpu::BlockId>(opInst)) {
        replaceWithIntrinsicOrDeviceFunction<ROCDL::BlockIdXOp, ROCDL::BlockIdYOp,
                             ROCDL::BlockIdZOp>(blockId, llvmDialect,
                                               indexBitwidth);
        return;
      }

      // BlockDimX/Y/Z are implemented as device functions on ROCDL.
      if (auto blockDim = dyn_cast<gpu::BlockDim>(opInst)) {
        replaceWithIntrinsicOrDeviceFunction<
            ROCDL::BlockDimXOp, ROCDL::BlockDimYOp, ROCDL::BlockDimZOp>(
            blockDim, llvmDialect, indexBitwidth);
        return;
      }
      // GridDimX/Y/Z are implemented as device functions on ROCDL.
      if (auto gridDim = dyn_cast<gpu::GridDim>(opInst)) {
        replaceWithIntrinsicOrDeviceFunction<
            ROCDL::GridDimXOp, ROCDL::GridDimYOp, ROCDL::GridDimZOp>(
            gridDim, llvmDialect, indexBitwidth);
        return;
      }
    });
  }
};

} // anonymous namespace

FunctionPassBase *createLowerGpuOpsToROCDLOpsPass() {
  return new LowerGpuOpsToROCDLOpsPass();
}

static PassRegistration<LowerGpuOpsToROCDLOpsPass>
    pass("lower-gpu-ops-to-rocdl-ops",
         "Generate ROCDL operations for gpu operations");

} // namespace mlir
