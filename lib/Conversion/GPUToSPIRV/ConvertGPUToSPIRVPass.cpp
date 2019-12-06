//===- ConvertGPUToSPIRVPass.cpp - GPU to SPIR-V dialect lowering passes --===//
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
// This file implements a pass to convert a kernel function in the GPU Dialect
// into a spv.module operation
//
//===----------------------------------------------------------------------===//
#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.h"
#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRV.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRV.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVLowering.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace {
/// Pass to lower GPU Dialect to SPIR-V. The pass only converts those functions
/// that have the "gpu.kernel" attribute, i.e. those functions that are
/// referenced in gpu::LaunchKernelOp operations. For each such function
///
/// 1) Create a spirv::ModuleOp, and clone the function into spirv::ModuleOp
/// (the original function is still needed by the gpu::LaunchKernelOp, so cannot
/// replace it).
///
/// 2) Lower the body of the spirv::ModuleOp.
class GPUToSPIRVPass : public ModulePass<GPUToSPIRVPass> {
public:
  GPUToSPIRVPass(ArrayRef<int64_t> workGroupSize)
      : workGroupSize(workGroupSize.begin(), workGroupSize.end()) {}
  void runOnModule() override;

private:
  SmallVector<int64_t, 3> workGroupSize;
};

/// Command line option to specify the workgroup size.
struct GPUToSPIRVPassOptions : public PassOptions<GPUToSPIRVPassOptions> {
  List<unsigned> workGroupSize{
      *this, "workgroup-size",
      llvm::cl::desc(
          "Workgroup Sizes in the SPIR-V module for x, followed by y, followed "
          "by z dimension of the dispatch (others will be ignored)"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated};
};
} // namespace

void GPUToSPIRVPass::runOnModule() {
  auto context = &getContext();
  auto module = getModule();

  SmallVector<Operation *, 4> conversionTargets;
  module.walk([&module, &conversionTargets](ModuleOp moduleOp) {
                if (!moduleOp.getAttrOfType<UnitAttr>(gpu::GPUDialect::getKernelModuleAttrName())) {
                  return;
                }
                // Create a new spir-vkernel module and clone the gpu.kernel_module
                // operations into this module. Still need to keep the original
                // module around cause the gpu.launch needs the launch function
                // to be still present.

    OpBuilder builder(moduleOp.getOperation());
    conversionTargets.push_back(builder.clone(*moduleOp.getOperation()));
  });

  /// Dialect conversion to lower the functions with the spirv::ModuleOps.
  SPIRVTypeConverter typeConverter;
  OwningRewritePatternList patterns;
  populateGPUToSPIRVPatterns(context, typeConverter, patterns, workGroupSize);
  populateStandardToSPIRVPatterns(context, typeConverter, patterns);

  ConversionTarget target(*context);
  target.addLegalDialect<spirv::SPIRVDialect>();
  target.addDynamicallyLegalOp<FuncOp>(
      [&](FuncOp op) { return typeConverter.isSignatureLegal(op.getType()); });

  if (failed(applyFullConversion(conversionTargets, target, patterns,
                                 &typeConverter))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OpPassBase<ModuleOp>>
mlir::createConvertGPUToSPIRVPass(ArrayRef<int64_t> workGroupSize) {
  return std::make_unique<GPUToSPIRVPass>(workGroupSize);
}

static PassRegistration<GPUToSPIRVPass, GPUToSPIRVPassOptions>
    pass("convert-gpu-to-spirv", "Convert GPU dialect to SPIR-V dialect",
         [](const GPUToSPIRVPassOptions &passOptions) {
           SmallVector<int64_t, 3> workGroupSize;
           workGroupSize.assign(passOptions.workGroupSize.begin(),
                                passOptions.workGroupSize.end());
           return std::make_unique<GPUToSPIRVPass>(workGroupSize);
         });
