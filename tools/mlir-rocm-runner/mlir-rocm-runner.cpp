//===- mlir-rocm-runner.cpp - MLIR ROCm Execution
// Driver---------------------===//
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
// This is a command line utility that executes an MLIR file on the GPU by
// translating MLIR to ROCm-Device-Libs/LLVM IR before JIT-compiling and
// executing
// the latter.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"

#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/GPUToROCM/GPUToROCMPass.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/JitRunner.h"
#include "mlir/Transforms/DialectConversion.h"

#include "hip/hip_runtime.h"

using namespace mlir;

static LogicalResult runMLIRPasses(ModuleOp m) {
  PassManager pm(m.getContext());
  applyPassManagerCLOptions(pm);

  pm.addPass(createGpuKernelOutliningPass());
  auto &kernelPm = pm.nest<ModuleOp>();
  kernelPm.addPass(createLowerGpuOpsToROCDLOpsPass());
  kernelPm.addPass(
      createConvertGPUKernelToHSACOPass(rocm::HSACOGeneratorConfig(false)));
  pm.addPass(createLowerToLLVMPass());
  pm.addPass(createGenerateHSACOAccessorPass());
  pm.addPass(createConvertGpuLaunchFuncToHIPCallsPass());

  return pm.run(m);
}

int main(int argc, char **argv) {
  registerPassManagerCLOptions();
  return mlir::JitRunnerMain(argc, argv, &runMLIRPasses);
}
