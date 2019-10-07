//===- ConvertLaunchFuncToHIPCalls.cpp - MLIR HIP lowering passes -------===//
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
// This file implements a pass to convert gpu.launch_func op into a sequence of
// HIP runtime calls. As the HIP runtime does not have a stable published ABI,
// this pass uses a slim runtime layer that builds on top of the public API from
// the HIP headers.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToROCM/GPUToROCMPass.h"

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

#include <iostream>

using namespace mlir;

// To avoid name mangling, these are defined in the mini-runtime file.
static constexpr const char *hipModuleLoadName = "mhipModuleLoad";
static constexpr const char *hipModuleGetFunctionName = "mhipModuleGetFunction";
static constexpr const char *hipGetStreamHelperName = "mhipGetStreamHelper";
static constexpr const char *hipLaunchKernelName = "mhipLaunchKernel";
static constexpr const char *hipStreamSynchronizeName = "mhipStreamSynchronize";

static constexpr const char *hipHostRegisterPointerName =
    "mhipHostRegisterPointer";
static constexpr const char *hipHostGetDevicePointerName =
    "mhipHostGetDevicePointer";

namespace {

/// A pass to convert gpu.launch_func operations into a sequence of HIP
/// runtime calls.
///
/// In essence, a gpu.launch_func operations gets compiled into the following
/// sequence of runtime calls:
///
/// * mhipModuleLoad        -- loads the module given the HSACO data
/// * mhipModuleGetFunction -- gets a handle to the actual kernel function
/// * mhipGetStreamHelper   -- initializes a new HIP stream
/// * mhipLaunchKernelName  -- launches the kernel on a stream
/// * mhipStreamSynchronize -- waits for operations on the stream to finish
///
/// Intermediate data structures are allocated on the stack.
class GpuLaunchFuncToHIPCallsPass
    : public ModulePass<GpuLaunchFuncToHIPCallsPass> {
private:
  LLVM::LLVMDialect *getLLVMDialect() { return llvmDialect; }

  llvm::LLVMContext &getLLVMContext() {
    return getLLVMDialect()->getLLVMContext();
  }

  void initializeCachedTypes() {
    const llvm::Module &module = llvmDialect->getLLVMModule();
    llvmPointerType = LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    llvmPointerPointerType = llvmPointerType.getPointerTo();
    llvmInt8Type = LLVM::LLVMType::getInt8Ty(llvmDialect);
    llvmInt32Type = LLVM::LLVMType::getInt32Ty(llvmDialect);
    llvmInt64Type = LLVM::LLVMType::getInt64Ty(llvmDialect);
    llvmIntPtrType = LLVM::LLVMType::getIntNTy(
        llvmDialect, module.getDataLayout().getPointerSizeInBits());
  }

  LLVM::LLVMType getPointerType() { return llvmPointerType; }

  LLVM::LLVMType getPointerPointerType() { return llvmPointerPointerType; }

  LLVM::LLVMType getInt8Type() { return llvmInt8Type; }

  LLVM::LLVMType getInt32Type() { return llvmInt32Type; }

  LLVM::LLVMType getInt64Type() { return llvmInt64Type; }

  LLVM::LLVMType getIntPtrType() {
    const llvm::Module &module = getLLVMDialect()->getLLVMModule();
    return LLVM::LLVMType::getIntNTy(
        getLLVMDialect(), module.getDataLayout().getPointerSizeInBits());
  }

  LLVM::LLVMType getHIPResultType() {
    // This is declared as an enum in HIP but helpers use i32.
    return getInt32Type();
  }

  // Allocate a void pointer on the stack.
  Value *allocatePointer(OpBuilder &builder, Location loc) {
    // %18 = llvm.mlir.constant(1 : i32) : !llvm.i32
    // %19 = llvm.alloca %18 x !llvm<"i8*"> : (!llvm.i32) -> !llvm<"i8**">
    auto one = builder.create<LLVM::ConstantOp>(loc, getInt32Type(),
                                                builder.getI32IntegerAttr(1));
    return builder.create<LLVM::AllocaOp>(loc, getPointerPointerType(), one,
                                          /*alignment*/ 0);
  }

  void declareHIPFunctions(Location loc);
  Value *setupParamsArray(gpu::LaunchFuncOp launchOp, OpBuilder &builder);
  Value *generateKernelNameConstant(FuncOp kernelFunction, Location &loc,
                                    OpBuilder &builder);
  void translateGpuLaunchCalls(mlir::gpu::LaunchFuncOp launchOp);

public:
  // Run the dialect converter on the module.
  void runOnModule() override {

    // Cache the LLVMDialect for the current module.
    llvmDialect = getContext().getRegisteredDialect<LLVM::LLVMDialect>();
    // Cache the used LLVM types.
    initializeCachedTypes();

    for (auto func : getModule().getOps<FuncOp>()) {
      func.walk(
          [this](mlir::gpu::LaunchFuncOp op) { translateGpuLaunchCalls(op); });
    }
  }

private:
  LLVM::LLVMDialect *llvmDialect;
  LLVM::LLVMType llvmPointerType;
  LLVM::LLVMType llvmPointerPointerType;
  LLVM::LLVMType llvmInt8Type;
  LLVM::LLVMType llvmInt32Type;
  LLVM::LLVMType llvmInt64Type;
  LLVM::LLVMType llvmIntPtrType;
};

} // anonymous namespace

// Adds declarations for the needed helper functions from the HIP wrapper.
// The types in comments give the actual types expected/returned but the API
// uses void pointers. This is fine as they have the same linkage in C.
void GpuLaunchFuncToHIPCallsPass::declareHIPFunctions(Location loc) {
  ModuleOp module = getModule();
  Builder builder(module);
  if (!module.lookupSymbol<FuncOp>(hipModuleLoadName)) {
    module.push_back(FuncOp::create(
        loc, hipModuleLoadName,
        builder.getFunctionType(
            {
                getPointerPointerType(), /* hipModule_t *module */
                getPointerType()         /* void *HSACO */
            },
            getHIPResultType())));
  }
  if (!module.lookupSymbol<FuncOp>(hipModuleGetFunctionName)) {
    // The helper uses void* instead of HIP's opaque hipModule_t and
    // hipFunction_t.
    module.push_back(FuncOp::create(
        loc, hipModuleGetFunctionName,
        builder.getFunctionType(
            {
                getPointerPointerType(), /* hipFunction_t *function */
                getPointerType(),        /* hipModule_t module */
                getPointerType()         /* char *name */
            },
            getHIPResultType())));
  }
  if (!module.lookupSymbol<FuncOp>(hipLaunchKernelName)) {
    // Other than the HIP api, the wrappers use uintptr_t to match the
    // LLVM type if MLIR's index type, which the GPU dialect uses.
    // Furthermore, they use void* instead of HIP's opaque hipFunction_t and
    // hipStream_t.
    module.push_back(FuncOp::create(
        loc, hipLaunchKernelName,
        builder.getFunctionType(
            {
                getPointerType(),        /* hipFunction_t f */
                getIntPtrType(),         /* intptr_t gridXDim */
                getIntPtrType(),         /* intptr_t gridyDim */
                getIntPtrType(),         /* intptr_t gridZDim */
                getIntPtrType(),         /* intptr_t blockXDim */
                getIntPtrType(),         /* intptr_t blockYDim */
                getIntPtrType(),         /* intptr_t blockZDim */
                getInt32Type(),          /* unsigned int sharedMemBytes */
                getPointerType(),        /* hipStream_t stream */
                getPointerPointerType(), /* void **kernelParams */
                getPointerPointerType()  /* void **extra */
            },
            getHIPResultType())));
  }
  if (!module.lookupSymbol<FuncOp>(hipGetStreamHelperName)) {
    // Helper function to get the current HIP stream. Uses void* instead of
    // HIPs opaque hipStream_t.
    module.push_back(FuncOp::create(
        loc, hipGetStreamHelperName,
        builder.getFunctionType({}, getPointerType() /* hipStream_t */)));
  }
  if (!module.lookupSymbol<FuncOp>(hipStreamSynchronizeName)) {
    module.push_back(
        FuncOp::create(loc, hipStreamSynchronizeName,
                       builder.getFunctionType(
                           {
                               getPointerType() /* hipStream_t stream */
                           },
                           getHIPResultType())));
  }
  if (!module.lookupSymbol<FuncOp>(hipHostRegisterPointerName)) {
    module.push_back(FuncOp::create(loc, hipHostRegisterPointerName,
                                    builder.getFunctionType(
                                        {
                                            getPointerType(), /* void *ptr */
                                            getInt32Type()    /* int32 flags*/
                                        },
                                        {})));
  }
  if (!module.lookupSymbol<FuncOp>(hipHostGetDevicePointerName)) {
    module.push_back(FuncOp::create(loc, hipHostGetDevicePointerName,
                                    builder.getFunctionType(
                                        {
                                            getPointerType(), /* void *ptr */
                                            getInt32Type()    /* int32 flags*/
                                        },
                                        getPointerType())));
  }
}

// Generates a parameters array to be used with a HIP kernel launch call. The
// arguments are extracted from the launchOp.
// The generated code is essentially as follows:
//
// %array = alloca(numparams * sizeof(void *))
// for (i : [0, NumKernelOperands))
//   %array[i] = cast<void*>(KernelOperand[i])
// return %array
Value *GpuLaunchFuncToHIPCallsPass::setupParamsArray(gpu::LaunchFuncOp launchOp,
                                                     OpBuilder &builder) {
  auto numKernelOperands = launchOp.getNumKernelOperands();
  Location loc = launchOp.getLoc();
  auto one = builder.create<LLVM::ConstantOp>(loc, getInt32Type(),
                                              builder.getI32IntegerAttr(1));
  auto arraySize = builder.create<LLVM::ConstantOp>(
      loc, getInt32Type(),
      builder.getI32IntegerAttr(launchOp.getNumKernelOperands()));
  auto array = builder.create<LLVM::AllocaOp>(loc, getPointerPointerType(),
                                              arraySize, /*alignment=*/0);
  for (unsigned int idx = 0; idx < numKernelOperands; ++idx) {
    auto operand = launchOp.getKernelOperand(idx);
    auto llvmType = operand->getType().cast<LLVM::LLVMType>();
    Value *memLocation = builder.create<LLVM::AllocaOp>(
        loc, llvmType.getPointerTo(), one, /*alignment=*/0);
    builder.create<LLVM::StoreOp>(loc, operand, memLocation);
    auto casted =
        builder.create<LLVM::BitcastOp>(loc, getPointerType(), memLocation);

    // Assume all struct arguments come from MemRef. If this assumption does not
    // hold anymore then we `launchOp` to lower from MemRefType and not after
    // LLVMConversion has taken place and the MemRef information is lost.
    // Extra level of indirection in the `array`:
    //   the descriptor pointer is registered via @mhipHostRegister
    //   and translated to a device pointer via @mhipHostGetDevicePointer
    if (llvmType.isStructTy()) {
      auto registerFunc =
          getModule().lookupSymbol<FuncOp>(hipHostRegisterPointerName);
      auto zero = builder.create<LLVM::ConstantOp>(
          loc, getInt32Type(), builder.getI32IntegerAttr(0));
      builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{},
                                   builder.getSymbolRefAttr(registerFunc),
                                   ArrayRef<Value *>{casted, zero});

      auto getDevicePtrFunc =
          getModule().lookupSymbol<FuncOp>(hipHostGetDevicePointerName);

      auto devicePtr = builder.create<LLVM::CallOp>(
          loc, ArrayRef<Type>{getPointerType()},
          builder.getSymbolRefAttr(getDevicePtrFunc),
          ArrayRef<Value *>{casted, zero});

      Value *memLocation = builder.create<LLVM::AllocaOp>(
          loc, getPointerPointerType(), one, /*alignment=*/0);
      builder.create<LLVM::StoreOp>(loc, devicePtr.getResult(0), memLocation);
      casted =
          builder.create<LLVM::BitcastOp>(loc, getPointerType(), memLocation);
    }

    auto index = builder.create<LLVM::ConstantOp>(
        loc, getInt32Type(), builder.getI32IntegerAttr(idx));
    auto gep = builder.create<LLVM::GEPOp>(loc, getPointerPointerType(), array,
                                           ArrayRef<Value *>{index});
    builder.create<LLVM::StoreOp>(loc, casted, gep);
  }
  return array;
}

// Generates an LLVM IR dialect global that contains the name of the given
// kernel function as a C string, and returns a pointer to its beginning.
// The code is essentially:
//
// llvm.global constant @kernel_name("function_name\00")
// func(...) {
//   %0 = llvm.addressof @kernel_name
//   %1 = llvm.constant (0 : index)
//   %2 = llvm.getelementptr %0[%1, %1] : !llvm<"i8*">
// }
Value *GpuLaunchFuncToHIPCallsPass::generateKernelNameConstant(
    FuncOp kernelFunction, Location &loc, OpBuilder &builder) {
  // Make sure the trailing zero is included in the constant.
  std::vector<char> kernelName(kernelFunction.getName().begin(),
                               kernelFunction.getName().end());
  kernelName.push_back('\0');

  std::string globalName =
      llvm::formatv("{0}_kernel_name", kernelFunction.getName());
  return LLVM::createGlobalString(
      loc, builder, globalName, StringRef(kernelName.data(), kernelName.size()),
      llvmDialect);
}

// Emits LLVM IR to launch a kernel function. Expects the module that contains
// the compiled kernel function as a HSACO in the 'amdgpu.HSACO' attribute of
// the kernel function in the IR. While MLIR has no global constants, also
// expects a HSACO getter function in an 'amdgpu.hsacogetter' attribute. Such
// function is expected to return a pointer to the HSACO blob when invoked. With
// these given, the generated code in essence is
//
//
// %hsaco_blob = call %hsacogetter
// %module_handle_addr = alloca sizeof(void*)
// call %mhipModuleLoad(%module_handle_addr, %hsaco_blob)
// %module_handle = load %module_handle_addr
// %kernel_name = <see generateKernelNameConstant>
// %function_handle_addr = alloca sizeof(void*)
// call %mhipModuleGetFunction(%module_handle, %function_handle_addr,
// %kernel_name)
// %function_handle = load %function_handle_addr
// %stream_handle = call %mhipGetStreamHelper()
// %params_array = <see setupParamsArray>
// call %mhipLaunchKernel(%function_handle,
//                        <launchOp operands 0..5>,
//                        0,
//                        %stream_handle,
//                        %params_array,
//                        nullptr)
// call %mhipStreamSynchronize(%stream_handle)
//
void GpuLaunchFuncToHIPCallsPass::translateGpuLaunchCalls(
    mlir::gpu::LaunchFuncOp launchOp) {
  OpBuilder builder(launchOp);
  Location loc = launchOp.getLoc();
  declareHIPFunctions(loc);

  auto zero = builder.create<LLVM::ConstantOp>(loc, getInt32Type(),
                                               builder.getI32IntegerAttr(0));
  // Emit a call to the HSACO getter to retrieve a pointer to the data that
  // represents the HSACO at runtime.
  // TODO(herhut): This should rather be a static global once supported.
  auto kernelFunction = getModule().lookupSymbol<FuncOp>(launchOp.kernel());
  auto hsacoGetter =
      kernelFunction.getAttrOfType<SymbolRefAttr>(rocm::kHSACOGetterAnnotation);
  if (!hsacoGetter) {
    kernelFunction.emitError("Missing ") << rocm::kHSACOGetterAnnotation
                                         << " attribute.";
    return signalPassFailure();
  }
  auto data = builder.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{getPointerType()}, hsacoGetter, ArrayRef<Value *>{});
  // Emit the load module call to load the module data. Error checking is done
  // in the called helper function.
  auto hipModule = allocatePointer(builder, loc);
  FuncOp hipModuleLoad = getModule().lookupSymbol<FuncOp>(hipModuleLoadName);
  builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{getHIPResultType()},
                               builder.getSymbolRefAttr(hipModuleLoad),
                               ArrayRef<Value *>{hipModule, data.getResult(0)});
  // Get the function from the module. The name corresponds to the name of
  // the kernel function.
  auto hipOwningModuleRef =
      builder.create<LLVM::LoadOp>(loc, getPointerType(), hipModule);
  auto kernelName = generateKernelNameConstant(kernelFunction, loc, builder);
  auto hipFunction = allocatePointer(builder, loc);
  FuncOp hipModuleGetFunction =
      getModule().lookupSymbol<FuncOp>(hipModuleGetFunctionName);
  builder.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{getHIPResultType()},
      builder.getSymbolRefAttr(hipModuleGetFunction),
      ArrayRef<Value *>{hipFunction, hipOwningModuleRef, kernelName});
  // Grab the global stream needed for execution.
  FuncOp hipGetStreamHelper =
      getModule().lookupSymbol<FuncOp>(hipGetStreamHelperName);
  auto hipStream = builder.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{getPointerType()},
      builder.getSymbolRefAttr(hipGetStreamHelper), ArrayRef<Value *>{});
  // Invoke the function with required arguments.
  auto hipLaunchKernel = getModule().lookupSymbol<FuncOp>(hipLaunchKernelName);
  auto hipFunctionRef =
      builder.create<LLVM::LoadOp>(loc, getPointerType(), hipFunction);
  auto paramsArray = setupParamsArray(launchOp, builder);
  auto nullpointer =
      builder.create<LLVM::IntToPtrOp>(loc, getPointerPointerType(), zero);
  builder.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{getHIPResultType()},
      builder.getSymbolRefAttr(hipLaunchKernel),
      ArrayRef<Value *>{hipFunctionRef, launchOp.getOperand(0),
                        launchOp.getOperand(1), launchOp.getOperand(2),
                        launchOp.getOperand(3), launchOp.getOperand(4),
                        launchOp.getOperand(5), zero, /* sharedMemBytes */
                        hipStream.getResult(0),       /* stream */
                        paramsArray,                  /* kernel params */
                        nullpointer /* extra */});
  // Sync on the stream to make it synchronous.
  auto hipStreamSync =
      getModule().lookupSymbol<FuncOp>(hipStreamSynchronizeName);
  builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{getHIPResultType()},
                               builder.getSymbolRefAttr(hipStreamSync),
                               ArrayRef<Value *>(hipStream.getResult(0)));
  launchOp.erase();
}

std::unique_ptr<mlir::OpPassBase<mlir::ModuleOp>>
mlir::createConvertGpuLaunchFuncToHIPCallsPass() {
  return std::make_unique<GpuLaunchFuncToHIPCallsPass>();
}

static PassRegistration<GpuLaunchFuncToHIPCallsPass>
    pass("launch-func-to-hip",
         "Convert all launch_func ops to HIP runtime calls");
