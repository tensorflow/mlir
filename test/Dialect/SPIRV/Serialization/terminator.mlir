// RUN: mlir-translate -serialize-spirv %s | mlir-translate -deserialize-spirv | FileCheck %s

spv.module "Logical" "GLSL450" {
  // CHECK-LABEL: @ret
  func @ret() -> () {
    // CHECK: spv.Return
    spv.Return
  }

  // CHECK-LABEL: @ret_val
  func @ret_val() -> (i32) {
    %0 = spv.Variable : !spv.ptr<i32, Function>
    %1 = spv.Load "Function" %0 : i32
    // CHECK: spv.ReturnValue {{.*}} : i32
    spv.ReturnValue %1 : i32
  }
}
