//===- RuntimeTest.cpp - SPIR-V Runtime Tests -----------------------------===//
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
// This file is for testing the Vulkan runtime API which takes in a spirv::ModuleOp,
// a bunch of resourses and number of work groups.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/SPIRVBinaryUtils.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "gmock/gmock.h"
#include <string>
#include <random>

using namespace mlir;
using namespace llvm;

using DescriptorSetIndex = uint32_t;
using BindingIndex = uint32_t;

// Struct containing information regarding to a host memory buffer.
struct VulkanHostMemoryBuffer {
  void *ptr{nullptr};
  uint64_t size{0};
};

// Struct containing the number of local workgroups to dispatch for each
// dimension.
struct NumWorkGroups {
  uint32_t x{1};
  uint32_t y{1};
  uint32_t z{1};
};

// This is a temporary function and will be removed in the future.
// See the full description in tools/mlir-vulkan-runner/VulkanRutime.cpp
extern mlir::LogicalResult runOnVulkan(
    mlir::ModuleOp,
    llvm::DenseMap<DescriptorSetIndex,
                   llvm::DenseMap<BindingIndex, VulkanHostMemoryBuffer>> &,
    const NumWorkGroups &);

class RuntimeTest : public ::testing::Test {
protected:
  LogicalResult parseAndRunModule(llvm::StringRef sourceFile,
                                  NumWorkGroups numWorkGroups) {
    std::string errorMessage;
    auto inputFile = llvm::MemoryBuffer::getMemBuffer(sourceFile);
    if (!inputFile) {
      llvm::errs() << errorMessage << "\n";
      return failure();
    }

    SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(inputFile), SMLoc());

    MLIRContext context;
    OwningModuleRef moduleRef(parseSourceFile(sourceMgr, &context));
    if (!moduleRef) {
      llvm::errs() << "\ncannot parse the file as a MLIR module" << '\n';
      return failure();
    }

    if (failed(runOnVulkan(moduleRef.get(), vars, numWorkGroups))) {
      return failure();
    }

    return success();
  }

  void createResourceVarFloat(uint32_t descriptorSet, uint32_t binding,
                              uint32_t elementCount) {
    float *ptr = new float[elementCount];
    std::mt19937 gen(randomDevice());
    std::uniform_real_distribution<> distribution(0.0, 10.0);
    for (uint32_t i = 0; i < elementCount; ++i) {
      ptr[i] = static_cast<float>(distribution(gen));
    }
    VulkanHostMemoryBuffer hostMemoryBuffer;
    hostMemoryBuffer.ptr = ptr;
    hostMemoryBuffer.size = sizeof(float) * elementCount;
    vars[descriptorSet][binding] = hostMemoryBuffer;
  }

  void destroyResourceVarFloat(VulkanHostMemoryBuffer &hostMemoryBuffer) {
    float *ptr = static_cast<float *>(hostMemoryBuffer.ptr);
    delete ptr;
  }

  VulkanHostMemoryBuffer FMul(VulkanHostMemoryBuffer &var1,
                              VulkanHostMemoryBuffer &var2) {
    VulkanHostMemoryBuffer resultHostMemoryBuffer;
    uint32_t size = var1.size / sizeof(float);
    float *result = new float[size];
    const float *rhs = reinterpret_cast<float *>(var1.ptr);
    const float *lhs = reinterpret_cast<float *>(var2.ptr);

    for (uint32_t i = 0; i < size; ++i) {
      result[i] = lhs[i] * rhs[i];
    }
    resultHostMemoryBuffer.ptr = static_cast<void *>(result);
    resultHostMemoryBuffer.size = size * sizeof(float);
    return resultHostMemoryBuffer;
  }

  VulkanHostMemoryBuffer FAdd(VulkanHostMemoryBuffer &var1,
                              VulkanHostMemoryBuffer &var2) {
    VulkanHostMemoryBuffer resultHostMemoryBuffer;
    uint32_t size = var1.size / sizeof(float);
    float *result = new float[size];
    const float *rhs = reinterpret_cast<float *>(var1.ptr);
    const float *lhs = reinterpret_cast<float *>(var2.ptr);

    for (uint32_t i = 0; i < size; ++i) {
      result[i] = lhs[i] + rhs[i];
    }
    resultHostMemoryBuffer.ptr = static_cast<void *>(result);
    resultHostMemoryBuffer.size = size * sizeof(float);
    return resultHostMemoryBuffer;
  }

  bool isEqualFloat(const VulkanHostMemoryBuffer &hostMemoryBuffer1,
                    const VulkanHostMemoryBuffer &hostMemoryBuffer2) {
    if (hostMemoryBuffer1.size != hostMemoryBuffer2.size)
      return false;

    uint32_t size = hostMemoryBuffer1.size / sizeof(float);

    const float *lhs = static_cast<float *>(hostMemoryBuffer1.ptr);
    const float *rhs = static_cast<float *>(hostMemoryBuffer2.ptr);
    const float epsilon = 0.0001f;
    for (uint32_t i = 0; i < size; ++i) {
      if (fabs(lhs[i] - rhs[i]) > epsilon)
        return false;
    }
    return true;
  }

protected:
  llvm::DenseMap<DescriptorSetIndex,
                 llvm::DenseMap<BindingIndex, VulkanHostMemoryBuffer>>
      vars;
  std::random_device randomDevice;
};

TEST_F(RuntimeTest, SimpleTest) {
  // SPIRV module embedded into the string.
  // This module contains 4 resource variables devided into 2 sets.
  std::string spirvModuleSource =
"spv.module \"Logical\" \"GLSL450\" {\n"
    "spv.globalVariable @var3 bind(1, 1) : !spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>\n"
    "spv.globalVariable @var2 bind(1, 0) : !spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>\n"
    "spv.globalVariable @var1 bind(0, 1) : !spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>\n"
    "spv.globalVariable @var0 bind(0, 0) : !spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>\n"
    "spv.globalVariable @globalInvocationID built_in(\"GlobalInvocationId\"): !spv.ptr<vector<3xi32>, Input>\n"
    "func @kernel() -> () {\n"
      "%c0 = spv.constant 0 : i32\n"

      "%0 = spv._address_of @var0 : !spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>\n"
      "%1 = spv._address_of @var1 : !spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>\n"
      "%2 = spv._address_of @var2 : !spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>\n"
      "%3 = spv._address_of @var3 : !spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>\n"

      "%ptr_id = spv._address_of @globalInvocationID: !spv.ptr<vector<3xi32>, Input>\n"

      "%id = spv.AccessChain %ptr_id[%c0] : !spv.ptr<vector<3xi32>, Input>\n"
      "%index = spv.Load \"Input\" %id: i32\n"

      "%4 = spv.AccessChain %0[%c0, %index] : !spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>\n"
      "%5 = spv.AccessChain %1[%c0, %index] : !spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>\n"
      "%6 = spv.AccessChain %2[%c0, %index] : !spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>\n"
      "%7 = spv.AccessChain %3[%c0, %index] : !spv.ptr<!spv.struct<!spv.array<1024 x f32 [4]> [0]>, StorageBuffer>\n"

      "%8 = spv.Load \"StorageBuffer\" %4 : f32\n"
      "%9 = spv.Load \"StorageBuffer\" %5 : f32\n"
      "%10 = spv.Load \"StorageBuffer\" %6 : f32\n"

      "%11 = spv.FMul %8, %9 : f32\n"
      "%12 = spv.FAdd %11, %10 : f32\n"

      "spv.Store \"StorageBuffer\" %7, %12 : f32\n"
      "spv.Return\n"
    "}\n"
    "spv.EntryPoint \"GLCompute\" @kernel, @globalInvocationID\n"
    "spv.ExecutionMode @kernel \"LocalSize\", 1, 1, 1\n"
"} attributes {\n"
  "capabilities = [\"Shader\"],\n"
  "extensions = [\"SPV_KHR_storage_buffer_storage_class\"]\n"
"}\n";

  createResourceVarFloat(0, 0, 1024);
  createResourceVarFloat(0, 1, 1024);
  createResourceVarFloat(1, 0, 1024);
  createResourceVarFloat(1, 1, 1024);

  auto fmulResult = FMul(vars[0][0], vars[0][1]);
  auto expected = FAdd(vars[1][0], fmulResult);

  NumWorkGroups numWorkGroups;
  numWorkGroups.x = 1024;
  ASSERT_TRUE(succeeded(parseAndRunModule(spirvModuleSource, numWorkGroups)));
  ASSERT_TRUE(isEqualFloat(expected, vars[1][1]));

  destroyResourceVarFloat(vars[0][0]);
  destroyResourceVarFloat(vars[0][1]);
  destroyResourceVarFloat(vars[1][0]);
  destroyResourceVarFloat(vars[1][1]);
  destroyResourceVarFloat(fmulResult);
  destroyResourceVarFloat(expected);
}
