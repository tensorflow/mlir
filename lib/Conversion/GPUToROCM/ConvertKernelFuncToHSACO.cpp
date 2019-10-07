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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
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

#define DEBUG_TYPE "gpu-to-rocm-conversion"

static llvm::cl::OptionCategory clOptionsCategory(DEBUG_TYPE " options");

static llvm::cl::opt<bool>
    clDumpLLVMIR("rocm-dump-lllvm-ir",
                 llvm::cl::desc("Dump the LLVM IR when generating HSACO"),
                 llvm::cl::cat(clOptionsCategory));
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
    ModuleOp module = getModule();

    // Nothing to do if this module does not contain the "gpu.kernel_module"
    // attribute, which is used to mark the (nested) modules created to house
    // the GPU kernel functions
    if (!module.getAttrOfType<UnitAttr>(
            gpu::GPUDialect::getKernelModuleAttrName()) ||
        !module.getName())
      return;

    // This is a module containing a GPU kernel function, we have work to do!

    // Make sure the AMDGPU target is initialized.
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUAsmPrinter();

    auto llvmModule = translateModuleToROCDLIR(module);
    if (!llvmModule)
      return signalPassFailure();

    if (StringAttr hsacoAttr =
            translateGpuModuleToHSACOAnnotation(*llvmModule, module))
      module.setAttr(rocm::kHSACOAnnotation, hsacoAttr);
    else
      signalPassFailure();
  }

private:
  /// Translates llvmModule to cubin and returns the result as attribute.
  StringAttr translateGpuModuleToHSACOAnnotation(llvm::Module &llvmModule,
                                                 ModuleOp module);

  OwnedHSACO convertModuleToHSACO(llvm::Module &llvmModule, ModuleOp module);

  OwnedHSACO emitModuleToHSACO(llvm::Module &llvmModule, ModuleOp module,
                               llvm::TargetMachine &targetMachine);

  OwnedHSACO emitModuleToHSACOForTesting(llvm::Module &llvmModule,
                                         ModuleOp module);

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
              llvm::StringRef rocdlDir) {

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
    result.push_back(appendedPath.str());
  }

  // Add AMDGPU version-specific bitcodes.
  llvm::SmallString<128> appendedPath;
  llvm::sys::path::append(appendedPath, rocdlDir,
                          getBitcodeFilename(amdgpuVersion));
  result.push_back(appendedPath.str());

  return result;
}

// Links the given llvm module with the given bitcode modules.
static LogicalResult
linkWithBitcodeModules(llvm::Module &llvmModule, ModuleOp module,
                       llvm::ArrayRef<std::string> bitcodeModulePaths) {
  llvm::Linker linker(llvmModule);

  for (auto &filename : bitcodeModulePaths) {
    if (!llvm::sys::fs::exists(filename)) {
      module.emitWarning("ROCDL bitcode module was not found at " + filename);
      // TODO(rocm)
      // The list currently returned by "getROCDLPaths" routine is a superset
      // and some files in that list may not be available on older ROCM
      // releases. So commenting out the call to propagate error status.
      // Error propagation should be restored once the list returned by
      // "getROCDLPaths" is stable/accurate.
      // return failure();
      continue;
    }

    llvm::SMDiagnostic diagnostic;
    std::unique_ptr<llvm::Module> bitcodeModule(
        llvm::parseIRFile(llvm::StringRef(filename.data(), filename.size()),
                          diagnostic, llvmModule.getContext()));

    if (bitcodeModule == nullptr) {
      MLIRContext *mlirContext = module.getContext();
      auto parseErrorLocation = mlir::FileLineColLoc::get(
          diagnostic.getFilename().str(), diagnostic.getLineNo(),
          diagnostic.getColumnNo(), mlirContext);
      mlir::emitError(parseErrorLocation, diagnostic.getMessage().str());
      module.emitError("Error parsing ROCDL bitcode module from " + filename);
      return failure();
    }

    if (linker.linkInModule(
            std::move(bitcodeModule), llvm::Linker::Flags::LinkOnlyNeeded,
            [](llvm::Module &M, const llvm::StringSet<> &GVS) {
              internalizeModule(M, [&M, &GVS](const llvm::GlobalValue &GV) {
                return !GV.hasName() || (GVS.count(GV.getName()) == 0);
              });
            })) {
      module.emitError("Error linking ROCDL bitcode module from " + filename);
      return failure();
    }
  }

  return success();
}

// Returns whether the module uses any ROCDL bitcode functions.
// This function may have false positives
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
static LogicalResult linkROCDLIfNecessary(llvm::Module &llvmModule,
                                          ModuleOp module,
                                          rocm::AMDGPUVersion amdgpuVersion,
                                          llvm::StringRef rocdlDir) {

  if (!couldNeedDeviceBitcode(llvmModule))
    return success();

  return linkWithBitcodeModules(llvmModule, module,
                                getROCDLPaths(amdgpuVersion, rocdlDir));
}

// Emits the given module to HSA Code Object. targetMachine is an initialized
// TargetMachine for the AMDGPU target.
OwnedHSACO
GpuKernelToHSACOPass::emitModuleToHSACO(llvm::Module &llvmModule,
                                        ModuleOp module,
                                        llvm::TargetMachine &targetMachine) {
  llvm::SmallString<128> tempdirName;
  if (llvm::sys::fs::createUniqueDirectory("/tmp/amdgpu_mlir", tempdirName)) {
    module.emitError("Failed to create tempdir for generating HSACO\n");
    return {};
  }

  std::error_code ec;
  if (clDumpLLVMIR) {
    // dump the LLVM IR to file...this is just for debugging purposes
    llvm::Twine irFilename =
        llvm::Twine(llvmModule.getModuleIdentifier()) + ".ll";
    llvm::SmallString<128> irPath;
    llvm::sys::path::append(irPath, tempdirName, irFilename);

    llvm::raw_fd_ostream irFileStream(irPath, ec, llvm::sys::fs::F_None);
    llvmModule.print(irFileStream, nullptr);
    irFileStream.flush();
  }

  // dump the GCN ISA binary file
  llvm::Twine isabinFilename =
      llvm::Twine(llvmModule.getModuleIdentifier()) + ".o";
  llvm::SmallString<128> isabinPath;
  llvm::sys::path::append(isabinPath, tempdirName, isabinFilename);

  llvm::legacy::PassManager codegenPasses;
  llvm::SmallVector<char, 0> stream;
  llvm::raw_svector_ostream pstream(stream);
  llvm::raw_fd_ostream isabinFileStream(isabinPath, ec, llvm::sys::fs::F_Text);
  llvmModule.setDataLayout(targetMachine.createDataLayout());
  targetMachine.addPassesToEmitFile(codegenPasses, isabinFileStream, nullptr,
                                    llvm::TargetMachine::CGFT_ObjectFile);
  codegenPasses.run(llvmModule);
  isabinFileStream.flush();

  // generate the hsaco binary
  // TODO(rocm):
  // Currently we invoke lld.ld as a separate process to generate the hsaco
  // file. Ideally we would like invoke it (ld.lld) via an API call to do the
  // same. That will require building the "lld" project (which apparently is
  // at the same level as "llvm") and figuring out how to call it from within
  // this "mlir" project.
  llvm::Twine hsacoFilename =
      llvm::Twine(llvmModule.getModuleIdentifier()) + ".hsaco";
  llvm::SmallString<128> hsacoPath;
  llvm::sys::path::append(hsacoPath, tempdirName, hsacoFilename);

  llvm::StringRef lldProgram(config.linkerPath);
  std::vector<llvm::StringRef> lldArgs{
      llvm::StringRef("ld.lld"),     llvm::StringRef("-flavor"),
      llvm::StringRef("gnu"),        llvm::StringRef("-shared"),
      llvm::StringRef("isabinPath"), llvm::StringRef("-o"),
      llvm::StringRef("hsacoPath"),
  };
  lldArgs[4] = llvm::StringRef(isabinPath);
  lldArgs[6] = llvm::StringRef(hsacoPath);

  std::string errorMessage;
  int lldResult = llvm::sys::ExecuteAndWait(
      lldProgram, llvm::ArrayRef<llvm::StringRef>(lldArgs), llvm::None, {}, 0,
      0, &errorMessage);
  if (lldResult) {
    module.emitError("ld.lld execution failed : " + errorMessage);
    return {};
  }
  // read HSACO
  auto hsacoFileOrError = llvm::MemoryBuffer::getFileAsStream(hsacoPath);
  if ((ec = hsacoFileOrError.getError()))
    return {};

  std::unique_ptr<llvm::MemoryBuffer> hsacoFile =
      std::move(hsacoFileOrError.get());

  return std::make_unique<std::vector<char>>(hsacoFile->getBufferStart(),
                                             hsacoFile->getBufferEnd());
}

OwnedHSACO
GpuKernelToHSACOPass::emitModuleToHSACOForTesting(llvm::Module &llvmModule,
                                                  ModuleOp module) {
  const char data[] = "HSACO";
  return std::make_unique<std::vector<char>>(data, data + sizeof(data) - 1);
}

OwnedHSACO GpuKernelToHSACOPass::convertModuleToHSACO(llvm::Module &llvmModule,
                                                      ModuleOp module) {
  if (config.testMode)
    return emitModuleToHSACOForTesting(llvmModule, module);

  // Construct LLVM TargetMachine for AMDGPU target.
  std::unique_ptr<llvm::TargetMachine> targetMachine;
  {
    std::string error;
    llvm::Triple triple("amdgcn--amdhsa-amdgiz");
    const llvm::Target *target =
        llvm::TargetRegistry::lookupTarget("", triple, error);
    if (target == nullptr) {
      module.emitError("Cannot initialize target triple");
      return {};
    }
    std::string mcpuStr = getMcpuOptionString(config.amdgpuVersion);
    std::string codeObjectStr = getCodeObjectOptionString(config.hsacoVersion);
    targetMachine.reset(target->createTargetMachine(triple.str(), mcpuStr,
                                                    codeObjectStr, {}, {}));
  }

  // Set the data layout of the llvm module to match what the target needs.
  llvmModule.setDataLayout(targetMachine->createDataLayout());

  if (failed(linkROCDLIfNecessary(llvmModule, module, config.amdgpuVersion,
                                  config.rocdlDir)))
    return {};

  // Lower LLVM module to HSA code object
  return emitModuleToHSACO(llvmModule, module, *targetMachine);
}

StringAttr GpuKernelToHSACOPass::translateGpuModuleToHSACOAnnotation(
    llvm::Module &llvmModule, ModuleOp module) {

  OwnedHSACO hsaco = convertModuleToHSACO(llvmModule, module);
  if (!hsaco)
    return {};

  return StringAttr::get({hsaco->data(), hsaco->size()}, module.getContext());
}

std::unique_ptr<OpPassBase<ModuleOp>> mlir::createConvertGPUKernelToHSACOPass(
    rocm::HSACOGeneratorConfig hsacoGeneratorConfig) {
  return std::make_unique<GpuKernelToHSACOPass>(hsacoGeneratorConfig);
}

static PassRegistration<GpuKernelToHSACOPass>
    pass("test-kernel-to-hsaco",
         "Convert all kernel functions to HSA code object blobs");
