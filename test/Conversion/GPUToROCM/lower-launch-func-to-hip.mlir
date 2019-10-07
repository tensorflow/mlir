// RUN: mlir-opt %s --lower-to-llvm --launch-func-to-hip | FileCheck %s

// CHECK: llvm.mlir.global constant @[[kernel_name:.*]]("kernel\00")

func @hsaco_getter() -> !llvm<"i8*">

func @kernel(!llvm.float, !llvm<"float*">, memref<?xf32>)
    attributes { gpu.kernel, amdgpu.hsacogetter = @hsaco_getter }


func @foo() {
  %0 = "op"() : () -> (!llvm.float)
  %1 = "op"() : () -> (!llvm<"float*">)
  %arg2 = alloc() : memref<5xf32>
  %2 = memref_cast %arg2 : memref<5xf32> to memref<?xf32>

  %cst = constant 8 : index

  // CHECK: [[module_ptr:%.*]] = llvm.alloca {{.*}} x !llvm<"i8*"> : (!llvm.i32) -> !llvm<"i8**">
  // CHECK: llvm.call @mhipModuleLoad([[module_ptr]], {{.*}}) : (!llvm<"i8**">, !llvm<"i8*">) -> !llvm.i32
  // CHECK: [[func_ptr:%.*]] = llvm.alloca {{.*}} x !llvm<"i8*"> : (!llvm.i32) -> !llvm<"i8**">
  // CHECK: llvm.call @mhipModuleGetFunction([[func_ptr]], {{.*}}, {{.*}}) : (!llvm<"i8**">, !llvm<"i8*">, !llvm<"i8*">) -> !llvm.i32
  // CHECK: llvm.call @mhipGetStreamHelper
  // CHECK: llvm.call @mhipHostRegisterPointer
  // CHECK: llvm.call @mhipHostGetDevicePointer
  // CHECK: llvm.call @mhipLaunchKernel
  // CHECK: llvm.call @mhipStreamSynchronize

  "gpu.launch_func"(%cst, %cst, %cst, %cst, %cst, %cst, %0, %1, %2) { kernel = @kernel }
      : (index, index, index, index, index, index, !llvm.float, !llvm<"float*">, memref<?xf32>) -> ()

  return
}
