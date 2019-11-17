//===- GPUToROCmPass.h - MLIR ROCm runtime support --------------*- C++ -*-===//
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
#ifndef MLIR_CONVERSION_GPUTOROCM_GPUTOROCMPASS_H_
#define MLIR_CONVERSION_GPUTOROCM_GPUTOROCMPASS_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "mlir/Conversion/GPUToROCM/ROCMConfig.h"

namespace mlir {

namespace rocm {

/// string constants used by the ROCM backend
static constexpr const char *kHSACOAnnotation = "amdgpu.hsaco";
static constexpr const char *kHSACOGetterAnnotation = "amdgpu.hsacogetter";
static constexpr const char *kHSACOGetterSuffix = "_hsaco";
static constexpr const char *kHSACOStorageSuffix = "_hsaco_cst";

/// enum to represent the AMD GPU versions supported by the ROCM backend
enum class AMDGPUVersion { GFX900 };

/// enum to represent the HSA Code Object versions supported by the ROCM backend
enum class HSACOVersion { V3 };

/// Configurable parameters for generating the HSACO blobs from GPU Kernels
struct HSACOGeneratorConfig {

  /// Constructor - sets the default values for the configurable parameters
  HSACOGeneratorConfig(bool isTestMode)
      : testMode(isTestMode), amdgpuVersion(AMDGPUVersion::GFX900),
        hsacoVersion(HSACOVersion::V3), rocdlDir(ROCM_DEVICE_LIB_DIR),
        linkerPath(ROCM_HCC_LINKER) {}

  /// testMode == true will result in skipping the HASCO generation process, and
  /// simply return the string "HSACO" as the HSACO blob
  bool testMode;

  /// the AMDGPU version for which to generate the HSACO
  AMDGPUVersion amdgpuVersion;

  /// the code object version for the generated HSACO
  HSACOVersion hsacoVersion;

  /// the directory containing the ROCDL bitcode libraries
  std::string rocdlDir;

  /// the path the ld.lld linker to use when generating the HSACO
  std::string linkerPath;
};

} // namespace rocm

// unique pointer to the HSA Code Object (which is stored as char vector)
using OwnedHSACO = std::unique_ptr<std::vector<char>>;

class ModuleOp;
template <typename T>
class OpPassBase;

/// Creates a pass to convert kernel functions into HSA Code Object blobs.
///
/// This transformation takes the body of each function that is annotated with
/// the amdgpu_kernel calling convention, copies it to a new LLVM module,
/// compiles the module with help of the AMDGPU backend to GCN ISA, and then
/// invokes lld to produce a binary blob in HSA Code Object format. Such blob
/// is then attached as a string attribute named 'amdgpu.hsaco' to the kernel
/// function.  After the transformation, the body of the kernel function is
/// removed (i.e., it is turned into a declaration).
std::unique_ptr<OpPassBase<ModuleOp>> createConvertGPUKernelToHSACOPass(
    rocm::HSACOGeneratorConfig hsacoGeneratorConfig);

} // namespace mlir

#endif // MLIR_CONVERSION_GPUTOROCM_GPUTOROCMPASS_H_
