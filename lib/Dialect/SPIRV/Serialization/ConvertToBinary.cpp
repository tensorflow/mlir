//===- ConvertToBinary.cpp - MLIR SPIR-V module to binary conversion ------===//
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
// This file implements a translation from MLIR SPIR-V ModuleOp to SPIR-V
// binary module.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

LogicalResult serializeModule(ModuleOp module, StringRef outputFilename) {
  if (!module)
    return failure();

  SmallVector<uint32_t, 0> binary;
  bool done = false;
  auto result = failure();

  for (auto spirvModule : module.getOps<spirv::ModuleOp>()) {
    if (done)
      return spirvModule.emitError("found more than one 'spv.module' op");

    done = true;
    result = spirv::serialize(spirvModule, binary);
  }

  if (failed(result))
    return failure();

  auto file = openOutputFile(outputFilename);
  if (!file)
    return failure();

  file->os().write(reinterpret_cast<char *>(binary.data()),
                   binary.size() * sizeof(uint32_t));
  file->keep();

  return mlir::success();
}

static TranslateFromMLIRRegistration
    registration("serialize-spirv",
                 [](ModuleOp module, StringRef outputFilename) {
                   return serializeModule(module, outputFilename);
                 });
