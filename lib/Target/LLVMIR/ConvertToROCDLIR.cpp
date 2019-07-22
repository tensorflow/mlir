//===- ConvertToROCDLIR.cpp - MLIR to LLVM IR conversion -------------------===//
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
// This file implements a translation between the MLIR LLVM + ROCDL dialects and
// LLVM IR with ROCDL intrinsics and metadata.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/ROCDLIR.h"

#include "mlir/GPU/GPUDialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/LLVMIR/ROCDLDialect.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Translation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

namespace {
static llvm::Value *createIntrinsicCall(llvm::IRBuilder<> &builder,
                                        llvm::Intrinsic::ID intrinsic) {
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  llvm::Function *fn = llvm::Intrinsic::getDeclaration(module, intrinsic, {});
  return builder.CreateCall(fn);
}

// ROCM TODO: review interface
static llvm::Value *createDeviceFunctionCall(llvm::IRBuilder<> &builder,
                                             StringRef fn_name, int parameter) {
  llvm::Module *module = builder.GetInsertBlock()->getModule();
  llvm::FunctionType *fn_type = llvm::FunctionType::get(
      llvm::Type::getInt32Ty(module->getContext()), // return type.
      llvm::Type::getInt32Ty(module->getContext()), // parameter type.
      false);                                       // no variadic arguments.
  llvm::Function *fn = llvm::dyn_cast<llvm::Function>(
      module->getOrInsertFunction(fn_name, fn_type).getCallee());
  llvm::ArrayRef<llvm::Value *> operands(llvm::ConstantInt::get(
      llvm::Type::getInt32Ty(module->getContext()), parameter));
  return builder.CreateCall(fn, operands);
}

class ModuleTranslation : public LLVM::ModuleTranslation {

public:
  explicit ModuleTranslation(ModuleOp module)
      : LLVM::ModuleTranslation(module) {}
  ~ModuleTranslation() override {}

protected:
  bool convertOperation(Operation &opInst,
                        llvm::IRBuilder<> &builder) override {

#include "mlir/LLVMIR/ROCDLConversions.inc"

    return LLVM::ModuleTranslation::convertOperation(opInst, builder);
  }
};
} // namespace

std::unique_ptr<llvm::Module> mlir::translateModuleToROCDLIR(ModuleOp m) {
  ModuleTranslation translation(m);
  auto llvmModule =
      LLVM::ModuleTranslation::translateModule<ModuleTranslation>(m);

  // Insert AMDGPU_KERNEL calling convention.
  // Insert amdgpu-flat-workgroup-size(1, 1024) attribute.
  for (FuncOp func : m.getOps<FuncOp>()) {
    if (!func.getAttrOfType<UnitAttr>(gpu::GPUDialect::getKernelFuncAttrName()))
      continue;

    auto *llvmFunc = llvmModule->getFunction(func.getName());

    llvmFunc->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
    llvmFunc->addFnAttr("amdgpu-flat-work-group-size", "1, 1024");
  }

  return llvmModule;
}

static TranslateFromMLIRRegistration
    registration("mlir-to-rocdlir",
                 [](ModuleOp module, llvm::StringRef outputFilename) {
                   if (!module)
                     return failure();

                   auto llvmModule = mlir::translateModuleToROCDLIR(module);
                   if (!llvmModule)
                     return failure();

                   auto file = openOutputFile(outputFilename);
                   if (!file)
                     return failure();

                   llvmModule->print(file->os(), nullptr);
                   file->keep();
                   return success();
                 });
