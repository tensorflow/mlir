//===- hip-runtime-wrappers.cpp - MLIR ROCm runner wrapper library -------===//
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
// Implements C wrappers around the HIP library for easy linking in ORC jit.
// Also adds some debugging helpers that are helpful when writing MLIR code to
// run on GPUs.
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <iostream>
#include <memory.h>

#include "llvm/Support/raw_ostream.h"

#include "hip/hip_runtime.h"

namespace {
int32_t reportErrorIfAny(hipError_t result, const char *where) {
  if (result != hipSuccess) {
    llvm::errs() << "HIP failed with " << result << " in " << where << "\n";
  }
  return result;
}
} // anonymous namespace

extern "C" int32_t mhipModuleLoad(void **module, void *data) {
  int32_t err = reportErrorIfAny(
      hipModuleLoadData(reinterpret_cast<hipModule_t *>(module), data),
      "ModuleLoad");
  return err;
}

extern "C" int32_t mhipModuleGetFunction(void **function, void *module,
                                         const char *name) {
  return reportErrorIfAny(
      hipModuleGetFunction(reinterpret_cast<hipFunction_t *>(function),
                           reinterpret_cast<hipModule_t>(module), name),
      "GetFunction");
}

// The wrapper uses intptr_t instead of CUDA's unsigned int to match
// the type of MLIR's index type. This avoids the need for casts in the
// generated MLIR code.
extern "C" int32_t mhipLaunchKernel(void *function, intptr_t gridX,
                                    intptr_t gridY, intptr_t gridZ,
                                    intptr_t blockX, intptr_t blockY,
                                    intptr_t blockZ, int32_t smem, void *stream,
                                    void **params, void **extra) {
  return reportErrorIfAny(
      hipModuleLaunchKernel(reinterpret_cast<hipFunction_t>(function), gridX,
                            gridY, gridZ, blockX, blockY, blockZ, smem,
                            reinterpret_cast<hipStream_t>(stream), params,
                            extra),
      "LaunchKernel");
}

extern "C" void *mhipGetStreamHelper() {
  hipStream_t stream;
  reportErrorIfAny(hipStreamCreate(&stream), "StreamCreate");
  return stream;
}

extern "C" int32_t mhipStreamSynchronize(void *stream) {
  return reportErrorIfAny(
      hipStreamSynchronize(reinterpret_cast<hipStream_t>(stream)),
      "StreamSync");
}

/// Helper functions for writing mlir example code

// A struct that corresponds to how MLIR represents memrefs.
template <typename T, int N>
struct MemRefType {
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

// Allows to register a pointer with the HIP runtime.
// Helpful until we have transfer functions implemented.
extern "C" void mhipHostRegisterMemRef(MemRefType<float, 1> *arg,
                                       int32_t flags) {

  reportErrorIfAny(
      hipHostRegister(arg->data, arg->sizes[0] * sizeof(float), flags),
      "hipHostRegister");

  return;
}

extern "C" MemRefType<float, 1>
mhipHostGetDevicePointerMemRef(MemRefType<float, 1> *arg, int32_t flags) {

  MemRefType<float, 1> result(*arg);

  reportErrorIfAny(hipSetDevice(0), "hipSetDevice");

  reportErrorIfAny(
      hipHostGetDevicePointer((void **)&result.data, arg->data, flags),
      "hipHostGetDevicePointer");

  return result;
}

// Allows to register a pointer with the HIP runtime.
// Helpful until we have transfer functions implemented.
extern "C" void mhipHostRegisterPointer(void *arg, int32_t flags) {
  reportErrorIfAny(hipHostRegister(arg, sizeof(void*), flags),
                   "hipHostRegister");
}

// Get the device pointer corresponding to the given registered pointer
extern "C" void *mhipHostGetDevicePointer(void *arg, int32_t flags) {

  reportErrorIfAny(hipSetDevice(0), "hipSetDevice");

  void *result = nullptr;
  reportErrorIfAny(hipHostGetDevicePointer((void **)&result, arg, flags),
                   "hipHostGetDevicePointer");

  return result;
}

/// Prints the given memref
extern "C" void mhipPrintMemRef(const MemRefType<float, 1> *arg) {
  if (arg->sizes[0] == 0) {
    llvm::outs() << "[]\n";
    return;
  }
  llvm::outs() << "[" << arg->data[0];
  for (int pos = 1; pos < arg->sizes[0]; pos++) {
    llvm::outs() << ", " << arg->data[pos];
  }
  llvm::outs() << "]\n";
}
