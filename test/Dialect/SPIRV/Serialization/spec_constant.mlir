// RUN: mlir-translate -serialize-spirv %s | mlir-translate -deserialize-spirv | FileCheck %s

spv.module "Logical" "GLSL450" {
  // CHECK: spv.specConstant @sc_true = true
  spv.specConstant @sc_true = true
  // CHECK: spv.specConstant @sc_false = false
  spv.specConstant @sc_false = false

  // CHECK: spv.specConstant @sc_int = -5 : i32
  spv.specConstant @sc_int = -5 : i32

  // CHECK: spv.specConstant @sc_float = 1.000000e+00 : f32
  spv.specConstant @sc_float = 1. : f32

  // CHECK-LABEL: @use
  func @use() -> (i32) {
    // We materialize a `spv._reference_of` op at every use of a
    // specialization constant in the deserializer. So two ops here.
    // CHECK: %[[USE1:.*]] = spv._reference_of @sc_int : i32
    // CHECK: %[[USE2:.*]] = spv._reference_of @sc_int : i32
    // CHECK: spv.IAdd %[[USE1]], %[[USE2]]

    %0 = spv._reference_of @sc_int : i32
    %1 = spv.IAdd %0, %0 : i32
    spv.ReturnValue %1 : i32
  }
}
