// RUN: mlir-opt %s --test-kernel-to-hsaco -split-input-file | FileCheck %s

// CHECK: attributes  {amdgpu.hsaco = "HSACO", gpu.kernel_module}
module @gpu_kernels attributes  { gpu.kernel_module } {
  // CHECK-LABEL: @kernel_A
  llvm.func @kernel_A(%arg0 : !llvm.float, %arg1 : !llvm<"float*">)
    // CHECK: attributes  {gpu.kernel}
    attributes  { gpu.kernel } {
    llvm.return
  }
}

// -----

// CHECK: attributes  {amdgpu.hsaco = "HSACO", gpu.kernel_module}
module @gpu_kernels attributes  { gpu.kernel_module } {
  // CHECK-LABEL: @kernel_A
  llvm.func @kernel_A(%arg0 : !llvm.float, %arg1 : !llvm<"float*">)
    // CHECK: attributes  {gpu.kernel}
    attributes  { gpu.kernel } {
    llvm.return
  }
  
  // CHECK-LABEL: @kernel_B
  llvm.func @kernel_B(%arg0 : !llvm.float, %arg1 : !llvm<"float*">)
    // CHECK: attributes  {gpu.kernel}
    attributes  { gpu.kernel } {
    llvm.return
  }
}

