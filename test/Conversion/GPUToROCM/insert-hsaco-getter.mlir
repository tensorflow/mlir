// RUN: mlir-opt %s --generate-hsaco-accessors | FileCheck %s

// CHECK: llvm.mlir.global constant @[[global:.*]]("HSACO")
// CHECK-NOT: module
module attributes {gpu.kernel_module} {
  func @kernel(!llvm.float, !llvm<"float*">)
  attributes  {amdgpu.hsaco = "HSACO"}
}
func @kernel(!llvm.float, !llvm<"float*">)
// CHECK: attributes  {amdgpu.hsacogetter = @[[getter:.*]], gpu.kernel}
  attributes  {gpu.kernel}

// CHECK: func @[[getter]]() -> !llvm<"i8*">
// CHECK: %[[addressof:.*]] = llvm.mlir.addressof @[[global]]
// CHECK: %[[c0:.*]] = llvm.mlir.constant(0 : index)
// CHECK: %[[gep:.*]] = llvm.getelementptr %[[addressof]][%[[c0]], %[[c0]]]
// CHECK-SAME: -> !llvm<"i8*">
// CHECK: llvm.return %[[gep]] : !llvm<"i8*">
