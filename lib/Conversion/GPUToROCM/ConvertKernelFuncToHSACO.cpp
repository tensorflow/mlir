//===- ConvertKernelFuncToHSACO.cpp - MLIR GPU lowering passes ------------===//
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
// This file implements a pass to convert gpu kernel functions into a
// corresponding binary blob that can be executed on a AMD GPU. Currently
// only translates the function itself but no dependencies.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToROCM/GPUToROCMPass.h"

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/ROCDLIR.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/Internalize.h"

#include <fstream>
#include <iostream>
#include <string>

using namespace mlir;

namespace {

/// A pass converting tagged kernel functions to HSA Code Object blobs.
///
/// If tagged as a kernel module, each contained function is translated to ROCDL
/// IR, which is then compiled using the llvm AMDGPU backend to generate the GPU
/// binary code (i.e. the HSACO file). The HSACO binary blob is attached as an
/// attribute to the function and the function body is erased.
class GpuKernelToHSACOPass : public ModulePass<GpuKernelToHSACOPass> {
public:
  GpuKernelToHSACOPass(rocm::HSACOGeneratorConfig hsacoGeneratorConfig =
                           rocm::HSACOGeneratorConfig(/*isTestMode=*/true))
      : config(hsacoGeneratorConfig) {}

  // Run the dialect converter on the module.
  void runOnModule() override {

    // Nothing to do if this module does not contain the "gpu.kernel_module"
    // attribute,
    // which is used to mark the (nested) modules created to house the GPU
    // kernel functions
    if (!getModule().getAttrOfType<UnitAttr>(
            gpu::GPUDialect::getKernelModuleAttrName()))
      return;

    // This is a module containing a GPU kernel function, we have work to do!

    // Make sure the AMDGPU target is initialized.
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUAsmPrinter();

    auto llvmModule = translateModuleToROCDLIR(getModule());
    if (!llvmModule)
      return signalPassFailure();

    for (auto function : getModule().getOps<FuncOp>()) {
      if (!gpu::GPUDialect::isKernel(function))
        continue;

      if (failed(translateGpuKernelToHSACOAnnotation(*llvmModule, function)))
        signalPassFailure();
    }
  }

private:
  LogicalResult translateGpuKernelToHSACOAnnotation(llvm::Module &llvmModule,
                                                    FuncOp &function);

  OwnedHSACO convertModuleToHSACO(llvm::Module &llvmModule, FuncOp &function);

  OwnedHSACO emitModuleToHSACO(llvm::Module &llvmModule,
                               llvm::TargetMachine &targetMachine);

  OwnedHSACO emitModuleToHSACOForTesting(llvm::Module &llvmModule,
                                         FuncOp &function);

  rocm::HSACOGeneratorConfig config;
};

} // anonymous namespace

// get the "-mcpu" option string corresponding to the given AMDGPU version enum
static std::string getMcpuOptionString(rocm::AMDGPUVersion v) {
  switch (v) {
  case rocm::AMDGPUVersion::GFX900:
    return "gfx900";
  }
  return "<invalid AMDGPU version>";
}

// get filename for file containing the AMDGPU version specific bitcodes
static std::string getBitcodeFilename(rocm::AMDGPUVersion v) {
  switch (v) {
  case rocm::AMDGPUVersion::GFX900:
    return "oclc_isa_version_900.amdgcn.bc";
  }
  return "<invalid AMDGPU version>";
}

// get the option string corresponding to the given HSACO version enum
static std::string getCodeObjectOptionString(rocm::HSACOVersion v) {
  switch (v) {
  case rocm::HSACOVersion::V3:
    return "-code-object-v3";
  }
  return "invalid HSACO version";
}

// Gets the ROCm-Device-Libs filenames for a particular AMDGPU version.
static std::vector<std::string>
getROCDLPaths(const rocm::AMDGPUVersion amdgpuVersion,
              const std::string &rocdlDir) {

  // AMDGPU version-neutral bitcodes.
  static constexpr StringLiteral rocdlFilenames[] = {
      "hc.amdgcn.bc",
      "opencl.amdgcn.bc",
      "ocml.amdgcn.bc",
      "ockl.amdgcn.bc",
      "oclc_finite_only_off.amdgcn.bc",
      "oclc_daz_opt_off.amdgcn.bc",
      "oclc_correctly_rounded_sqrt_on.amdgcn.bc",
      "oclc_unsafe_math_off.amdgcn.bc",
      "oclc_wavefrontsize64_on.amdgcn.bc"};

  // Construct full path to ROCDL bitcode libraries.
  std::vector<std::string> result;
  for (auto filename : rocdlFilenames) {
    llvm::SmallString<128> appendedPath;
    llvm::sys::path::append(appendedPath, rocdlDir, filename);
    result.push_back(appendedPath.c_str());
  }

  // Add AMDGPU version-specific bitcodes.
  llvm::SmallString<128> appendedPath;
  llvm::sys::path::append(appendedPath, rocdlDir,
                          getBitcodeFilename(amdgpuVersion));
  result.push_back(appendedPath.c_str());

  return std::move(result);
}

static std::unique_ptr<llvm::Module>
loadBitcodeModule(const std::string &filename, llvm::LLVMContext &llvmContext) {

  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> bitcodeModule(
      llvm::parseIRFile(llvm::StringRef(filename.data(), filename.size()),
                        diagnostic, llvmContext));

  if (bitcodeModule == nullptr) {
    llvm::errs() << diagnostic.getFilename().str() << ":"
                 << diagnostic.getLineNo() << ":" << diagnostic.getColumnNo()
                 << ": " << diagnostic.getMessage().str();
  }

  return bitcodeModule;
}

// Links the given llvm module with the given bitcode modules.
static void
linkWithBitcodeModules(llvm::Module &llvmModule,
                       const std::vector<std::string> &bitcodeModulePaths) {
  llvm::Linker linker(llvmModule);

  for (auto &filename : bitcodeModulePaths) {
    if (!llvm::sys::fs::exists(filename)) {
      llvm::errs() << "ROCDL bitcode module, required by this MLIR module, was "
                      "not found at "
                   << filename << "\n";
      continue;
    }

    std::unique_ptr<llvm::Module> bitcodeModule =
        loadBitcodeModule(filename, llvmModule.getContext());

    if (linker.linkInModule(
            std::move(bitcodeModule), llvm::Linker::Flags::LinkOnlyNeeded,
            [](llvm::Module &M, const llvm::StringSet<> &GVS) {
              internalizeModule(M, [&M, &GVS](const llvm::GlobalValue &GV) {
                return !GV.hasName() || (GVS.count(GV.getName()) == 0);
              });
            })) {
      llvm::errs() << "Error linking bitcode module from " << filename << "\n";
      return;
    }
  }
}

// Returns whether the module uses any ROCDL bitcode functions. This function
// may have false positives
static bool couldNeedDeviceBitcode(const llvm::Module &llvmModule) {
  for (const llvm::Function &llvmFunction : llvmModule.functions()) {
    // This is a conservative approximation
    //  - not all such functions are in ROCm-Device-Libs.
    if (!llvmFunction.isIntrinsic() && llvmFunction.isDeclaration())
      return true;
  }
  return false;
}

// Links ROCm-Device-Libs into the given module if the module needs it.
static void linkROCDLIfNecessary(llvm::Module &llvmModule,
                                 rocm::AMDGPUVersion amdgpuVersion,
                                 const std::string &rocdlDir) {

  if (!couldNeedDeviceBitcode(llvmModule))
    return;

  linkWithBitcodeModules(llvmModule, getROCDLPaths(amdgpuVersion, rocdlDir));
}

// Emits the given module to HSA Code Object. targetMachine is an initialized
// TargetMachine for the AMDGPU target.
OwnedHSACO
GpuKernelToHSACOPass::emitModuleToHSACO(llvm::Module &llvmModule,
                                        llvm::TargetMachine &targetMachine) {
  llvm::SmallString<128> tempdirName;
  if (llvm::sys::fs::createUniqueDirectory("/tmp/amdgpu_mlir", tempdirName)) {
    llvm::errs() << "Failed to create tempdir for generating HSACO\n";
    return std::make_unique<std::vector<char>>();
  }

  // prepare filenames for all stages of compilation:
  // IR, ISA, binary ISA, and HSACO
  llvm::Twine irFilename =
      llvm::Twine(llvmModule.getModuleIdentifier()) + ".ll";
  llvm::SmallString<128> irPath;
  llvm::sys::path::append(irPath, tempdirName, irFilename);

  llvm::Twine isabinFilename =
      llvm::Twine(llvmModule.getModuleIdentifier()) + ".o";
  llvm::SmallString<128> isabinPath;
  llvm::sys::path::append(isabinPath, tempdirName, isabinFilename);

  llvm::Twine hsacoFilename =
      llvm::Twine(llvmModule.getModuleIdentifier()) + ".hsaco";
  llvm::SmallString<128> hsacoPath;
  llvm::sys::path::append(hsacoPath, tempdirName, hsacoFilename);

  std::error_code ec;

  // dump LLVM IR
  llvm::raw_fd_ostream irFileStream(irPath, ec, llvm::sys::fs::F_None);
  llvmModule.print(irFileStream, nullptr);
  irFileStream.flush();

  //// emit GCN ISA binary
  llvm::legacy::PassManager codegenPasses;
  llvm::SmallVector<char, 0> stream;
  llvm::raw_svector_ostream pstream(stream);
  llvm::raw_fd_ostream isabinFileStream(isabinPath, ec, llvm::sys::fs::F_Text);
  llvmModule.setDataLayout(targetMachine.createDataLayout());
  targetMachine.addPassesToEmitFile(codegenPasses, isabinFileStream, nullptr,
                                    llvm::TargetMachine::CGFT_ObjectFile);
  codegenPasses.run(llvmModule);
  isabinFileStream.flush();

  llvm::StringRef lldProgram(config.linkerPath);
  std::vector<llvm::StringRef> lldArgs{
      llvm::StringRef("ld.lld"),     llvm::StringRef("-flavor"),
      llvm::StringRef("gnu"),        llvm::StringRef("-shared"),
      llvm::StringRef("isabinPath"), llvm::StringRef("-o"),
      llvm::StringRef("hsacoPath"),
  };
  lldArgs[4] = llvm::StringRef(isabinPath.c_str());
  lldArgs[6] = llvm::StringRef(hsacoPath.c_str());

  std::string errorMessage;
  int lldResult = llvm::sys::ExecuteAndWait(
      lldProgram, llvm::ArrayRef<llvm::StringRef>(lldArgs), llvm::None, {}, 0,
      0, &errorMessage);
  if (lldResult) {
    llvm::errs() << "ld.lld execute fail: " << errorMessage;
  }

  // read HSACO
  std::ifstream hsacoFile(hsacoPath.c_str(), std::ios::binary | std::ios::ate);
  std::ifstream::pos_type hsacoFileSize = hsacoFile.tellg();

  std::vector<char> hsaco(hsacoFileSize);
  hsacoFile.seekg(0, std::ios::beg);
  hsacoFile.read(reinterpret_cast<char *>(&hsaco[0]), hsacoFileSize);
  return std::make_unique<std::vector<char>>(hsaco);
}

OwnedHSACO
GpuKernelToHSACOPass::emitModuleToHSACOForTesting(llvm::Module &llvmModule,
                                                  FuncOp &function) {
  const char data[] = "HSACO";
  return std::make_unique<std::vector<char>>(data, data + sizeof(data) - 1);
}

OwnedHSACO GpuKernelToHSACOPass::convertModuleToHSACO(llvm::Module &llvmModule,
                                                      FuncOp &function) {

  if (config.testMode) {
    return emitModuleToHSACOForTesting(llvmModule, function);
  }

  // Construct LLVM TargetMachine for AMDGPU target.
  std::unique_ptr<llvm::TargetMachine> targetMachine;
  {
    std::string error;
    llvm::Triple triple("amdgcn--amdhsa-amdgiz");
    const llvm::Target *target =
        llvm::TargetRegistry::lookupTarget("", triple, error);
    if (target == nullptr) {
      function.emitError("Cannot initialize target triple");
      return {};
    }
    std::string mcpuStr = getMcpuOptionString(config.amdgpuVersion);
    std::string codeObjectStr = getCodeObjectOptionString(config.hsacoVersion);
    targetMachine.reset(target->createTargetMachine(triple.str(), mcpuStr,
                                                    codeObjectStr, {}, {}));
  }

  // Set the data layout of the llvm module to match what the target needs.
  llvmModule.setDataLayout(targetMachine->createDataLayout());

  linkROCDLIfNecessary(llvmModule, config.amdgpuVersion, config.rocdlDir);

  // Lower LLVM module to HSA code object
  return emitModuleToHSACO(llvmModule, *targetMachine);
}

LogicalResult GpuKernelToHSACOPass::translateGpuKernelToHSACOAnnotation(
    llvm::Module &llvmModule, FuncOp &function) {

  auto hsaco = convertModuleToHSACO(llvmModule, function);
  if (!hsaco) {
    return function.emitError("translation to HSA Code Object failed.");
  }

  Builder builder(function.getContext());
  function.setAttr(rocm::kHSACOAnnotation,
                   builder.getStringAttr({hsaco->data(), hsaco->size()}));

  // Remove the body of the kernel function now that it has been translated.
  // The main reason to do this is so that the resulting module no longer
  // contains kernel instructions, and hence can be compiled into host code by
  // a separate pass.
  function.eraseBody();

  return success();
}

std::unique_ptr<OpPassBase<ModuleOp>> mlir::createConvertGPUKernelToHSACOPass(
    rocm::HSACOGeneratorConfig hsacoGeneratorConfig) {
  return std::make_unique<GpuKernelToHSACOPass>(hsacoGeneratorConfig);
}

static PassRegistration<GpuKernelToHSACOPass>
    pass("test-kernel-to-hsaco",
         "Convert all kernel functions to HSA code object blobs");
