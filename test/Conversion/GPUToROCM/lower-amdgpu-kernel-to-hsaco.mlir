// RUN: mlir-opt %s --test-kernel-to-hsaco -split-input-file | FileCheck %s
module attributes  { gpu.kernel_module } {
  func @kernel(%arg0 : !llvm.float, %arg1 : !llvm<"float*">)
  // CHECK: attributes  {amdgpu.hsaco = "HSACO", gpu.kernel}
    attributes  { gpu.kernel } {
  // CHECK-NOT: llvm.return
    llvm.return
  }
}

// -----

module attributes  { gpu.kernel_module } {
  func @kernel_A(%arg0 : !llvm.float, %arg1 : !llvm<"float*">)
    // CHECK: attributes  {amdgpu.hsaco = "HSACO", gpu.kernel}
    attributes  { gpu.kernel } {
    // CHECK-NOT: llvm.return
    llvm.return
  }
  
  func @kernel_B(%arg0 : !llvm.float, %arg1 : !llvm<"float*">)
    // CHECK: attributes  {amdgpu.hsaco = "HSACO", gpu.kernel}
    attributes  { gpu.kernel } {
    // CHECK-NOT: llvm.return
    llvm.return
  }
}

