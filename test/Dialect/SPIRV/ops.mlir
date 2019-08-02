// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.AccessChain
//===----------------------------------------------------------------------===//

func @access_chain_struct() -> () {
  %0 = spv.constant 1: i32
  %1 = spv.Variable : !spv.ptr<!spv.struct<f32, !spv.array<4xf32>>, Function>
  // CHECK: {{.*}} = spv.AccessChain {{.*}}[{{.*}}, {{.*}}] : !spv.ptr<!spv.struct<f32, !spv.array<4 x f32>>, Function>
  %2 = spv.AccessChain %1[%0, %0] : !spv.ptr<!spv.struct<f32, !spv.array<4xf32>>, Function>
  return
}

// -----

func @access_chain_1D_array(%arg0 : i32) -> () {
  %0 = spv.Variable : !spv.ptr<!spv.array<4xf32>, Function>
  // CHECK: {{.*}} = spv.AccessChain {{.*}}[{{.*}}] : !spv.ptr<!spv.array<4 x f32>, Function>
  %1 = spv.AccessChain %0[%arg0] : !spv.ptr<!spv.array<4xf32>, Function>
  return
}

// -----

func @access_chain_2D_array_1(%arg0 : i32) -> () {
  %0 = spv.Variable : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
  // CHECK: {{.*}} = spv.AccessChain {{.*}}[{{.*}}, {{.*}}] : !spv.ptr<!spv.array<4 x !spv.array<4 x f32>>, Function>
  %1 = spv.AccessChain %0[%arg0, %arg0] : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
  %2 = spv.Load "Function" %1 ["Volatile"] : f32
  return
}

// -----

func @access_chain_2D_array_2(%arg0 : i32) -> () {
  %0 = spv.Variable : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
  // CHECK: {{.*}} = spv.AccessChain {{.*}}[{{.*}}] : !spv.ptr<!spv.array<4 x !spv.array<4 x f32>>, Function>
  %1 = spv.AccessChain %0[%arg0] : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
  %2 = spv.Load "Function" %1 ["Volatile"] : !spv.array<4xf32>
  return
}

// -----

func @access_chain_non_composite() -> () {
  %0 = spv.constant 1: i32
  // expected-error @+1 {{cannot extract from non-composite type 'f32' with index 0}}
  %1 = spv.Variable : !spv.ptr<f32, Function>
  %2 = spv.AccessChain %1[%0] : !spv.ptr<f32, Function>
  return
}

// -----

func @access_chain_no_indices(%index0 : i32) -> () {
  // expected-error @+1 {{expected at least one index}}
  %0 = spv.Variable : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
  %1 = spv.AccessChain %0[] : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
  return
}

// -----

func @access_chain_invalid_type(%index0 : i32) -> () {
  %0 = spv.Variable : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
  // expected-error @+1 {{expected a pointer to composite type, but provided '!spv.array<4 x !spv.array<4 x f32>>'}}
  %1 = spv.Load "Function" %0 ["Volatile"] : !spv.array<4x!spv.array<4xf32>>
  %2 = spv.AccessChain %1[%index0] : !spv.array<4x!spv.array<4xf32>>
  return
}

// -----

func @access_chain_invalid_index_1(%index0 : i32) -> () {
   %0 = spv.Variable : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
  // expected-error @+1 {{expected SSA operand}}
  %1 = spv.AccessChain %0[%index, 4] : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
  return
}

// -----

func @access_chain_invalid_index_2(%index0 : i32) -> () {
  // expected-error @+1 {{index must be an integer spv.constant to access element of spv.struct}}
  %0 = spv.Variable : !spv.ptr<!spv.struct<f32, !spv.array<4xf32>>, Function>
  %1 = spv.AccessChain %0[%index0, %index0] : !spv.ptr<!spv.struct<f32, !spv.array<4xf32>>, Function>
  return
}

// -----

func @access_chain_invalid_constant_type_1() -> () {
  %0 = std.constant 1: i32
  // expected-error @+1 {{index must be an integer spv.constant to access element of spv.struct, but provided std.constant}}
  %1 = spv.Variable : !spv.ptr<!spv.struct<f32, !spv.array<4xf32>>, Function>
  %2 = spv.AccessChain %1[%0, %0] : !spv.ptr<!spv.struct<f32, !spv.array<4xf32>>, Function>
  return
}

// -----

func @access_chain_out_of_bounds() -> () {
  %index0 = "spv.constant"() { value = 12: i32} : () -> i32
  // expected-error @+1 {{'spv.AccessChain' op index 12 out of bounds for '!spv.struct<f32, !spv.array<4 x f32>>'}}
  %0 = spv.Variable : !spv.ptr<!spv.struct<f32, !spv.array<4xf32>>, Function>
  %1 = spv.AccessChain %0[%index0, %index0] : !spv.ptr<!spv.struct<f32, !spv.array<4xf32>>, Function>
  return
}

// -----

func @access_chain_invalid_accessing_type(%index0 : i32) -> () {
  // expected-error @+1 {{cannot extract from non-composite type 'f32' with index 0}}
  %0 = spv.Variable : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
  %1 = spv.AccessChain %0[%index, %index0, %index0] : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.CompositeExtractOp
//===----------------------------------------------------------------------===//

func @composite_extract_f32_from_1D_array(%arg0: !spv.array<4xf32>) -> f32 {
  // CHECK: %0 = spv.CompositeExtract %arg0[1 : i32] : !spv.array<4 x f32>
  %0 = spv.CompositeExtract %arg0[1 : i32] : !spv.array<4xf32>
  return %0: f32
}

// -----

func @composite_extract_f32_from_2D_array(%arg0: !spv.array<4x!spv.array<4xf32>>) -> f32 {
  // CHECK: %0 = spv.CompositeExtract %arg0[1 : i32, 2 : i32] : !spv.array<4 x !spv.array<4 x f32>>
  %0 = spv.CompositeExtract %arg0[1 : i32, 2 : i32] : !spv.array<4x!spv.array<4xf32>>
  return %0: f32
}

// -----

func @composite_extract_1D_array_from_2D_array(%arg0: !spv.array<4x!spv.array<4xf32>>) -> !spv.array<4xf32> {
  // CHECK: %0 = spv.CompositeExtract %arg0[1 : i32] : !spv.array<4 x !spv.array<4 x f32>>
  %0 = spv.CompositeExtract %arg0[1 : i32] : !spv.array<4x!spv.array<4xf32>>
  return %0 : !spv.array<4xf32>
}

// -----

func @composite_extract_struct(%arg0 : !spv.struct<f32, !spv.array<4xf32>>) -> f32 {
  // CHECK: %0 = spv.CompositeExtract %arg0[1 : i32, 2 : i32] : !spv.struct<f32, !spv.array<4 x f32>>
  %0 = spv.CompositeExtract %arg0[1 : i32, 2 : i32] : !spv.struct<f32, !spv.array<4xf32>>
  return %0 : f32
}

// -----

func @composite_extract_vector(%arg0 : vector<4xf32>) -> f32 {
  // CHECK: %0 = spv.CompositeExtract %arg0[1 : i32] : vector<4xf32>
  %0 = spv.CompositeExtract %arg0[1 : i32] : vector<4xf32>
  return %0 : f32
}

// -----

func @composite_extract_no_ssa_operand() -> () {
  // expected-error @+1 {{expected SSA operand}}
  %0 = spv.CompositeExtract [4 : i32, 1 : i32] : !spv.array<4x!spv.array<4xf32>>
  return
}

// -----

func @composite_extract_invalid_index_type_1() -> () {
  %0 = spv.constant 10 : i32
  %1 = spv.Variable : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
  %2 = spv.Load "Function" %1 ["Volatile"] : !spv.array<4x!spv.array<4xf32>>
  // expected-error @+1 {{expected non-function type}}
  %3 = spv.CompositeExtract %2[%0] : !spv.array<4x!spv.array<4xf32>>
  return
}

// -----

func @composite_extract_invalid_index_type_2(%arg0 : !spv.array<4x!spv.array<4xf32>>) -> () {
  // expected-error @+1 {{op attribute 'indices' failed to satisfy constraint: 32-bit integer array attribute}}
  %0 = spv.CompositeExtract %arg0[1] : !spv.array<4x!spv.array<4xf32>>
  return
}

// -----

func @composite_extract_invalid_index_identifier(%arg0 : !spv.array<4x!spv.array<4xf32>>) -> () {
  // expected-error @+1 {{expected bare identifier}}
  %0 = spv.CompositeExtract %arg0(1 : i32) : !spv.array<4x!spv.array<4xf32>>
  return
}

// -----

func @composite_extract_2D_array_out_of_bounds_access_1(%arg0: !spv.array<4x!spv.array<4xf32>>) -> () {
  // expected-error @+1 {{index 4 out of bounds for '!spv.array<4 x !spv.array<4 x f32>>'}}
  %0 = spv.CompositeExtract %arg0[4 : i32, 1 : i32] : !spv.array<4x!spv.array<4xf32>>
  return
}

// -----

func @composite_extract_2D_array_out_of_bounds_access_2(%arg0: !spv.array<4x!spv.array<4xf32>>
) -> () {
  // expected-error @+1 {{index 4 out of bounds for '!spv.array<4 x f32>'}}
  %0 = spv.CompositeExtract %arg0[1 : i32, 4 : i32] : !spv.array<4x!spv.array<4xf32>>
  return
}

// -----

func @composite_extract_struct_element_out_of_bounds_access(%arg0 : !spv.struct<f32, !spv.array<4xf32>>) -> () {
  // expected-error @+1 {{index 2 out of bounds for '!spv.struct<f32, !spv.array<4 x f32>>'}}
  %0 = spv.CompositeExtract %arg0[2 : i32, 0 : i32] : !spv.struct<f32, !spv.array<4xf32>>
  return
}

// -----

func @composite_extract_vector_out_of_bounds_access(%arg0: vector<4xf32>) -> () {
  // expected-error @+1 {{index 4 out of bounds for 'vector<4xf32>'}}
  %0 = spv.CompositeExtract %arg0[4 : i32] : vector<4xf32>
  return
}

// -----

func @composite_extract_invalid_types_1(%arg0: !spv.array<4x!spv.array<4xf32>>) -> () {
  // expected-error @+1 {{cannot extract from non-composite type 'f32' with index 3}}
  %0 = spv.CompositeExtract %arg0[1 : i32, 2 : i32, 3 : i32] : !spv.array<4x!spv.array<4xf32>>
  return
}

// -----

func @composite_extract_invalid_types_2(%arg0: f32) -> () {
  // expected-error @+1 {{cannot extract from non-composite type 'f32' with index 1}}
  %0 = spv.CompositeExtract %arg0[1 : i32] : f32
  return
}

// -----

func @composite_extract_invalid_extracted_type(%arg0: !spv.array<4x!spv.array<4xf32>>) -> () {
  // expected-error @+1 {{expected at least one index for spv.CompositeExtract}}
  %0 = spv.CompositeExtract %arg0[] : !spv.array<4x!spv.array<4xf32>>
  return
}

// -----

func @composite_extract_result_type_mismatch(%arg0: !spv.array<4xf32>) -> i32 {
  // expected-error @+1 {{invalid result type: expected 'f32' but provided 'i32'}}
  %0 = "spv.CompositeExtract"(%arg0) {indices = [2: i32]} : (!spv.array<4xf32>) -> (i32)
  return %0: i32
}

// -----

//===----------------------------------------------------------------------===//
// spv.EntryPoint
//===----------------------------------------------------------------------===//

spv.module "Logical" "VulkanKHR" {
   func @do_nothing() -> () {
     spv.Return
   }
   // CHECK: spv.EntryPoint "GLCompute" @do_nothing
   spv.EntryPoint "GLCompute" @do_nothing
}

spv.module "Logical" "VulkanKHR" {
   %2 = spv.Variable : !spv.ptr<f32, Input>
   %3 = spv.Variable : !spv.ptr<f32, Output>
   func @do_something(%arg0 : !spv.ptr<f32, Input>, %arg1 : !spv.ptr<f32, Output>) -> () {
     %1 = spv.Load "Input" %arg0 : f32
     spv.Store "Output" %arg1, %1 : f32
     spv.Return
   }
   // CHECK: spv.EntryPoint "GLCompute" @do_something, {{%.*}}, {{%.*}} : !spv.ptr<f32, Input>, !spv.ptr<f32, Output>
   spv.EntryPoint "GLCompute" @do_something, %2, %3 : !spv.ptr<f32, Input>, !spv.ptr<f32, Output>
}

// -----

spv.module "Logical" "VulkanKHR" {
   func @do_nothing() -> () {
     spv.Return
   }
   // expected-error @+1 {{custom op 'spv.EntryPoint' expected symbol reference attribute}}
   spv.EntryPoint "GLCompute" "do_nothing"
}

// -----

spv.module "Logical" "VulkanKHR" {
   func @do_nothing() -> () {
     spv.Return
   }
   // expected-error @+1 {{function 'do_something' not found in 'spv.module'}}
   spv.EntryPoint "GLCompute" @do_something
}

/// TODO(ravishankarm) : Add a test that verifies an error is thrown
/// when interface entries of EntryPointOp are not
/// spv.Variables. There is currently no other op that has a spv.ptr
/// return type

// -----

spv.module "Logical" "VulkanKHR" {
   func @do_nothing() -> () {
     // expected-error @+1 {{'spv.EntryPoint' op failed to verify that op can only be used in a 'spv.module' block}}
     spv.EntryPoint "GLCompute" @do_something
   }
}

// -----

spv.module "Logical" "VulkanKHR" {
   func @do_nothing() -> () {
     spv.Return
   }
   spv.EntryPoint "GLCompute" @do_nothing
   // expected-error @+1 {{duplicate of a previous EntryPointOp}}
   spv.EntryPoint "GLCompute" @do_nothing
}

// -----

spv.module "Logical" "VulkanKHR" {
   func @do_nothing() -> () {
     spv.Return
   }
   spv.EntryPoint "GLCompute" @do_nothing
   // expected-error @+1 {{custom op 'spv.EntryPoint' invalid execution_model attribute specification: "ContractionOff"}}
   spv.EntryPoint "ContractionOff" @do_nothing
}

// -----

//===----------------------------------------------------------------------===//
// spv.ExecutionMode
//===----------------------------------------------------------------------===//

spv.module "Logical" "VulkanKHR" {
   func @do_nothing() -> () {
     spv.Return
   }
   spv.EntryPoint "GLCompute" @do_nothing
   // CHECK: spv.ExecutionMode {{@.*}} "ContractionOff"
   spv.ExecutionMode @do_nothing "ContractionOff"
}

spv.module "Logical" "VulkanKHR" {
   func @do_nothing() -> () {
     spv.Return
   }
   spv.EntryPoint "GLCompute" @do_nothing
   // CHECK: spv.ExecutionMode {{@.*}} "LocalSizeHint", 3, 4, 5
   spv.ExecutionMode @do_nothing "LocalSizeHint", 3, 4, 5
}

// -----

spv.module "Logical" "VulkanKHR" {
   func @do_nothing() -> () {
     spv.Return
   }
   spv.EntryPoint "GLCompute" @do_nothing
   // expected-error @+1 {{custom op 'spv.ExecutionMode' invalid execution_mode attribute specification: "GLCompute"}}
   spv.ExecutionMode @do_nothing "GLCompute", 3, 4, 5
}

// -----

//===----------------------------------------------------------------------===//
// spv.FAdd
//===----------------------------------------------------------------------===//

func @fadd_scalar(%arg: f32) -> f32 {
  // CHECK: spv.FAdd
  %0 = spv.FAdd %arg, %arg : f32
  return %0 : f32
}

// -----

//===----------------------------------------------------------------------===//
// spv.FDiv
//===----------------------------------------------------------------------===//

func @fdiv_scalar(%arg: f32) -> f32 {
  // CHECK: spv.FDiv
  %0 = spv.FDiv %arg, %arg : f32
  return %0 : f32
}

// -----

//===----------------------------------------------------------------------===//
// spv.FMod
//===----------------------------------------------------------------------===//

func @fmod_scalar(%arg: f32) -> f32 {
  // CHECK: spv.FMod
  %0 = spv.FMod %arg, %arg : f32
  return %0 : f32
}

// -----

//===----------------------------------------------------------------------===//
// spv.FMul
//===----------------------------------------------------------------------===//

func @fmul_scalar(%arg: f32) -> f32 {
  // CHECK: spv.FMul
  %0 = spv.FMul %arg, %arg : f32
  return %0 : f32
}

func @fmul_vector(%arg: vector<4xf32>) -> vector<4xf32> {
  // CHECK: spv.FMul
  %0 = spv.FMul %arg, %arg : vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

func @fmul_i32(%arg: i32) -> i32 {
  // expected-error @+1 {{must be scalar/vector of 16/32/64-bit float}}
  %0 = spv.FMul %arg, %arg : i32
  return %0 : i32
}

// -----

func @fmul_bf16(%arg: bf16) -> bf16 {
  // expected-error @+1 {{must be scalar/vector of 16/32/64-bit float}}
  %0 = spv.FMul %arg, %arg : bf16
  return %0 : bf16
}

// -----

func @fmul_tensor(%arg: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{must be scalar/vector of 16/32/64-bit float}}
  %0 = spv.FMul %arg, %arg : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spv.FRem
//===----------------------------------------------------------------------===//

func @frem_scalar(%arg: f32) -> f32 {
  // CHECK: spv.FRem
  %0 = spv.FRem %arg, %arg : f32
  return %0 : f32
}

// -----

//===----------------------------------------------------------------------===//
// spv.FSub
//===----------------------------------------------------------------------===//

func @fsub_scalar(%arg: f32) -> f32 {
  // CHECK: spv.FSub
  %0 = spv.FSub %arg, %arg : f32
  return %0 : f32
}

// -----

//===----------------------------------------------------------------------===//
// spv.IAdd
//===----------------------------------------------------------------------===//

func @iadd_scalar(%arg: i32) -> i32 {
  // CHECK: spv.IAdd
  %0 = spv.IAdd %arg, %arg : i32
  return %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spv.IEqual
//===----------------------------------------------------------------------===//

func @iequal_scalar(%arg0: i32, %arg1: i32) -> i1 {
  // CHECK: {{.*}} = spv.IEqual {{.*}}, {{.*}} : i32
  %0 = spv.IEqual %arg0, %arg1 : i32
  return %0 : i1
}

// -----

func @iequal_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: {{.*}} = spv.IEqual {{.*}}, {{.*}} : vector<4xi32>
  %0 = spv.IEqual %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spv.INotEqual
//===----------------------------------------------------------------------===//

func @inotequal_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: {{.*}} = spv.INotEqual {{.*}}, {{.*}} : vector<4xi32>
  %0 = spv.INotEqual %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spv.IMul
//===----------------------------------------------------------------------===//

func @imul_scalar(%arg: i32) -> i32 {
  // CHECK: spv.IMul
  %0 = spv.IMul %arg, %arg : i32
  return %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spv.ISub
//===----------------------------------------------------------------------===//

func @isub_scalar(%arg: i32) -> i32 {
  // CHECK: spv.ISub
  %0 = spv.ISub %arg, %arg : i32
  return %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spv.LoadOp
//===----------------------------------------------------------------------===//

// CHECK_LABEL: @simple_load
func @simple_load() -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // CHECK: spv.Load "Function" %0 : f32
  %1 = spv.Load "Function" %0 : f32
  return
}

// CHECK_LABEL: @volatile_load
func @volatile_load() -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // CHECK: spv.Load "Function" %0 ["Volatile"] : f32
  %1 = spv.Load "Function" %0 ["Volatile"] : f32
  return
}

// CHECK_LABEL: @aligned_load
func @aligned_load() -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // CHECK: spv.Load "Function" %0 ["Aligned", 4] : f32
  %1 = spv.Load "Function" %0 ["Aligned", 4] : f32
  return
}

// -----

func @simple_load_missing_storageclass() -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{expected non-function type}}
  %1 = spv.Load %0 : f32
  return
}

// -----

func @simple_load_missing_operand() -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{expected SSA operand}}
  %1 = spv.Load "Function" : f32
  return
}

// -----

func @simple_load_missing_rettype() -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+2 {{expected ':'}}
  %1 = spv.Load "Function" %0
  return
}

// -----

func @volatile_load_missing_lbrace() -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{expected ':'}}
  %1 = spv.Load "Function" %0 "Volatile"] : f32
  return
}

// -----

func @volatile_load_missing_rbrace() -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{expected ']'}}
  %1 = spv.Load "Function" %0 ["Volatile"} : f32
  return
}

// -----

func @aligned_load_missing_alignment() -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{expected ','}}
  %1 = spv.Load "Function" %0 ["Aligned"] : f32
  return
}

// -----

func @aligned_load_missing_comma() -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{expected ','}}
  %1 = spv.Load "Function" %0 ["Aligned" 4] : f32
  return
}

// -----

func @load_incorrect_attributes() -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{expected ']'}}
  %1 = spv.Load "Function" %0 ["Volatile", 4] : f32
  return
}

// -----

func @load_unknown_memory_access() -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{custom op 'spv.Load' invalid memory_access attribute specification: "Something"}}
  %1 = spv.Load "Function" %0 ["Something"] : f32
  return
}

// -----

func @aligned_load_incorrect_attributes() -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{expected ']'}}
  %1 = spv.Load "Function" %0 ["Aligned", 4, 23] : f32
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.Return
//===----------------------------------------------------------------------===//

"foo.function"() ({
  // expected-error @+1 {{must appear in a 'func' op}}
  spv.Return
})  : () -> ()

// -----

// Return mismatches function signature
spv.module "Logical" "VulkanKHR" {
  func @work() -> (i32) {
    // expected-error @+1 {{cannot be used in functions returning value}}
    spv.Return
  }
}

// -----

//===----------------------------------------------------------------------===//
// spv.SDiv
//===----------------------------------------------------------------------===//

func @sdiv_scalar(%arg: i32) -> i32 {
  // CHECK: spv.SDiv
  %0 = spv.SDiv %arg, %arg : i32
  return %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spv.SGreaterThan
//===----------------------------------------------------------------------===//

func @sgt_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: {{.*}} = spv.SGreaterThan {{.*}}, {{.*}} : vector<4xi32>
  %0 = spv.SGreaterThan %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spv.SGreaterThanEqual
//===----------------------------------------------------------------------===//

func @sge_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: {{.*}} = spv.SGreaterThanEqual {{.*}}, {{.*}} : vector<4xi32>
  %0 = spv.SGreaterThanEqual %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spv.SLessThan
//===----------------------------------------------------------------------===//

func @slt_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: {{.*}} = spv.SLessThan {{.*}}, {{.*}} : vector<4xi32>
  %0 = spv.SLessThan %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spv.SLessThanEqual
//===----------------------------------------------------------------------===//

func @slte_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: {{.*}} = spv.SLessThanEqual {{.*}}, {{.*}} : vector<4xi32>
  %0 = spv.SLessThanEqual %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spv.SMod
//===----------------------------------------------------------------------===//

func @smod_scalar(%arg: i32) -> i32 {
  // CHECK: spv.SMod
  %0 = spv.SMod %arg, %arg : i32
  return %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spv.SRem
//===----------------------------------------------------------------------===//

func @srem_scalar(%arg: i32) -> i32 {
  // CHECK: spv.SRem
  %0 = spv.SRem %arg, %arg : i32
  return %0 : i32
}

//===----------------------------------------------------------------------===//
// spv.StoreOp
//===----------------------------------------------------------------------===//

func @simple_store(%arg0 : f32) -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // CHECK: spv.Store  "Function" %0, %arg0 : f32
  spv.Store  "Function" %0, %arg0 : f32
  return
}

// CHECK_LABEL: @volatile_store
func @volatile_store(%arg0 : f32) -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // CHECK: spv.Store  "Function" %0, %arg0 ["Volatile"] : f32
  spv.Store  "Function" %0, %arg0 ["Volatile"] : f32
  return
}

// CHECK_LABEL: @aligned_store
func @aligned_store(%arg0 : f32) -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // CHECK: spv.Store  "Function" %0, %arg0 ["Aligned", 4] : f32
  spv.Store  "Function" %0, %arg0 ["Aligned", 4] : f32
  return
}

// -----

func @simple_store_missing_ptr_type(%arg0 : f32) -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{expected non-function type}}
  spv.Store  %0, %arg0 : f32
  return
}

// -----

func @simple_store_missing_operand(%arg0 : f32) -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{custom op 'spv.Store' invalid operand}} : f32
  spv.Store  "Function" , %arg0 : f32
  return
}

// -----

func @simple_store_missing_operand(%arg0 : f32) -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{custom op 'spv.Store' expected 2 operands}} : f32
  spv.Store  "Function" %0 : f32
  return
}

// -----

func @volatile_store_missing_lbrace(%arg0 : f32) -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{expected ':'}}
  spv.Store  "Function" %0, %arg0 "Volatile"] : f32
  return
}

// -----

func @volatile_store_missing_rbrace(%arg0 : f32) -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{expected ']'}}
  spv.Store "Function" %0, %arg0 ["Volatile"} : f32
  return
}

// -----

func @aligned_store_missing_alignment(%arg0 : f32) -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{expected ','}}
  spv.Store  "Function" %0, %arg0 ["Aligned"] : f32
  return
}

// -----

func @aligned_store_missing_comma(%arg0 : f32) -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{expected ','}}
  spv.Store  "Function" %0, %arg0 ["Aligned" 4] : f32
  return
}

// -----

func @load_incorrect_attributes(%arg0 : f32) -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{expected ']'}}
  spv.Store  "Function" %0, %arg0 ["Volatile", 4] : f32
  return
}

// -----

func @aligned_store_incorrect_attributes(%arg0 : f32) -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{expected ']'}}
  spv.Store  "Function" %0, %arg0 ["Aligned", 4, 23] : f32
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.UDiv
//===----------------------------------------------------------------------===//

func @udiv_scalar(%arg: i32) -> i32 {
  // CHECK: spv.UDiv
  %0 = spv.UDiv %arg, %arg : i32
  return %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spv.UGreaterThan
//===----------------------------------------------------------------------===//

func @ugt_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: {{.*}} = spv.UGreaterThan {{.*}}, {{.*}} : vector<4xi32>
  %0 = spv.UGreaterThan %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spv.UGreaterThanEqual
//===----------------------------------------------------------------------===//

func @ugte_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: {{.*}} = spv.UGreaterThanEqual {{.*}}, {{.*}} : vector<4xi32>
  %0 = spv.UGreaterThanEqual %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spv.ULessThan
//===----------------------------------------------------------------------===//

func @ult_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: {{.*}} = spv.ULessThan {{.*}}, {{.*}} : vector<4xi32>
  %0 = spv.ULessThan %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spv.ULessThanEqual
//===----------------------------------------------------------------------===//

func @ulte_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi1> {
  // CHECK: {{.*}} = spv.ULessThanEqual {{.*}}, {{.*}} : vector<4xi32>
  %0 = spv.ULessThanEqual %arg0, %arg1 : vector<4xi32>
  return %0 : vector<4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spv.UMod
//===----------------------------------------------------------------------===//

func @umod_scalar(%arg: i32) -> i32 {
  // CHECK: spv.UMod
  %0 = spv.UMod %arg, %arg : i32
  return %0 : i32
}

// -----

//===----------------------------------------------------------------------===//
// spv.Variable
//===----------------------------------------------------------------------===//

func @variable_no_init(%arg0: f32) -> () {
  // CHECK: spv.Variable : !spv.ptr<f32, Function>
  %0 = spv.Variable : !spv.ptr<f32, Function>
  return
}

func @variable_init() -> () {
  %0 = spv.constant 4.0 : f32
  // CHECK: spv.Variable init(%0) : !spv.ptr<f32, Private>
  %1 = spv.Variable init(%0) : !spv.ptr<f32, Private>
  return
}

func @variable_bind() -> () {
  // CHECK: spv.Variable bind(1, 2) : !spv.ptr<f32, Uniform>
  %0 = spv.Variable bind(1, 2) : !spv.ptr<f32, Uniform>
  return
}

func @variable_init_bind() -> () {
  %0 = spv.constant 4.0 : f32
  // CHECK: spv.Variable init(%0) {binding = 5 : i32} : !spv.ptr<f32, Private>
  %1 = spv.Variable init(%0) {binding = 5 : i32} : !spv.ptr<f32, Private>
  return
}

// -----

func @expect_ptr_result_type(%arg0: f32) -> () {
  // expected-error @+1 {{expected spv.ptr type}}
  %0 = spv.Variable : f32
  return
}

// -----

func @variable_init(%arg0: f32) -> () {
  // expected-error @+1 {{op initializer must be the result of a spv.Constant or module-level spv.Variable op}}
  %0 = spv.Variable init(%arg0) : !spv.ptr<f32, Private>
  return
}

// -----

func @storage_class_mismatch() -> () {
  %0 = spv.constant 5.0 : f32
  // expected-error @+1 {{storage class must match result pointer's storage class}}
  %1 = "spv.Variable"(%0) {storage_class = 2: i32} : (f32) -> !spv.ptr<f32, Function>
  return
}

// -----

func @cannot_be_generic_storage_class(%arg0: f32) -> () {
  // expected-error @+1 {{storage class cannot be 'Generic'}}
  %0 = spv.Variable : !spv.ptr<f32, Generic>
  return
}
