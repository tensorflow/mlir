// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.AccessChain
//===----------------------------------------------------------------------===//

func @access_chain_struct() -> () {
  %0 = spv.constant 1: i32
  %1 = spv.Variable : !spv.ptr<!spv.struct<f32, !spv.array<4xf32>>, Function>
  // CHECK: spv.AccessChain {{.*}}[{{.*}}, {{.*}}] : !spv.ptr<!spv.struct<f32, !spv.array<4 x f32>>, Function>
  %2 = spv.AccessChain %1[%0, %0] : !spv.ptr<!spv.struct<f32, !spv.array<4xf32>>, Function>
  return
}

// -----

func @access_chain_1D_array(%arg0 : i32) -> () {
  %0 = spv.Variable : !spv.ptr<!spv.array<4xf32>, Function>
  // CHECK: spv.AccessChain {{.*}}[{{.*}}] : !spv.ptr<!spv.array<4 x f32>, Function>
  %1 = spv.AccessChain %0[%arg0] : !spv.ptr<!spv.array<4xf32>, Function>
  return
}

// -----

func @access_chain_2D_array_1(%arg0 : i32) -> () {
  %0 = spv.Variable : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
  // CHECK: spv.AccessChain {{.*}}[{{.*}}, {{.*}}] : !spv.ptr<!spv.array<4 x !spv.array<4 x f32>>, Function>
  %1 = spv.AccessChain %0[%arg0, %arg0] : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
  %2 = spv.Load "Function" %1 ["Volatile"] : f32
  return
}

// -----

func @access_chain_2D_array_2(%arg0 : i32) -> () {
  %0 = spv.Variable : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
  // CHECK: spv.AccessChain {{.*}}[{{.*}}] : !spv.ptr<!spv.array<4 x !spv.array<4 x f32>>, Function>
  %1 = spv.AccessChain %0[%arg0] : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
  %2 = spv.Load "Function" %1 ["Volatile"] : !spv.array<4xf32>
  return
}

// -----

func @access_chain_non_composite() -> () {
  %0 = spv.constant 1: i32
  %1 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{cannot extract from non-composite type 'f32' with index 0}}
  %2 = spv.AccessChain %1[%0] : !spv.ptr<f32, Function>
  return
}

// -----

func @access_chain_no_indices(%index0 : i32) -> () {
  %0 = spv.Variable : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
  // expected-error @+1 {{expected at least one index}}
  %1 = spv.AccessChain %0[] : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
  return
}

// -----

func @access_chain_invalid_type(%index0 : i32) -> () {
  %0 = spv.Variable : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
  %1 = spv.Load "Function" %0 ["Volatile"] : !spv.array<4x!spv.array<4xf32>>
  // expected-error @+1 {{expected a pointer to composite type, but provided '!spv.array<4 x !spv.array<4 x f32>>'}}
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
  %0 = spv.Variable : !spv.ptr<!spv.struct<f32, !spv.array<4xf32>>, Function>
  // expected-error @+1 {{index must be an integer spv.constant to access element of spv.struct}}
  %1 = spv.AccessChain %0[%index0, %index0] : !spv.ptr<!spv.struct<f32, !spv.array<4xf32>>, Function>
  return
}

// -----

func @access_chain_invalid_constant_type_1() -> () {
  %0 = std.constant 1: i32
  %1 = spv.Variable : !spv.ptr<!spv.struct<f32, !spv.array<4xf32>>, Function>
  // expected-error @+1 {{index must be an integer spv.constant to access element of spv.struct, but provided std.constant}}
  %2 = spv.AccessChain %1[%0, %0] : !spv.ptr<!spv.struct<f32, !spv.array<4xf32>>, Function>
  return
}

// -----

func @access_chain_out_of_bounds() -> () {
  %index0 = "spv.constant"() { value = 12: i32} : () -> i32
  %0 = spv.Variable : !spv.ptr<!spv.struct<f32, !spv.array<4xf32>>, Function>
  // expected-error @+1 {{'spv.AccessChain' op index 12 out of bounds for '!spv.struct<f32, !spv.array<4 x f32>>'}}
  %1 = spv.AccessChain %0[%index0, %index0] : !spv.ptr<!spv.struct<f32, !spv.array<4xf32>>, Function>
  return
}

// -----

func @access_chain_invalid_accessing_type(%index0 : i32) -> () {
  %0 = spv.Variable : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
  // expected-error @+1 {{cannot extract from non-composite type 'f32' with index 0}}
  %1 = spv.AccessChain %0[%index, %index0, %index0] : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
  return

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

spv.module "Logical" "VulkanKHR" {
  spv.globalVariable @var0 : !spv.ptr<f32, Input>
  // CHECK_LABEL: @simple_load
  func @simple_load() -> () {
    // CHECK: spv.Load "Input" {{%.*}} : f32
    %0 = spv._address_of @var0 : !spv.ptr<f32, Input>
    %1 = spv.Load "Input" %0 : f32
    spv.Return
  }
}

// -----

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

spv.module "Logical" "VulkanKHR" {
  spv.globalVariable @var0 : !spv.ptr<f32, Input>
  func @simple_store(%arg0 : f32) -> () {
    %0 = spv._address_of @var0 : !spv.ptr<f32, Input>
    // CHECK: spv.Store  "Input" {{%.*}}, {{%.*}} : f32
    spv.Store  "Input" %0, %arg0 : f32
    spv.Return
  }
}

// -----

//===----------------------------------------------------------------------===//
// spv.Variable
//===----------------------------------------------------------------------===//

func @variable(%arg0: f32) -> () {
  // CHECK: spv.Variable : !spv.ptr<f32, Function>
  %0 = spv.Variable : !spv.ptr<f32, Function>
  return
}

// -----

func @variable_init_normal_constant() -> () {
  %0 = spv.constant 4.0 : f32
  // CHECK: spv.Variable init(%0) : !spv.ptr<f32, Function>
  %1 = spv.Variable init(%0) : !spv.ptr<f32, Function>
  return
}

// -----

spv.module "Logical" "GLSL450" {
  spv.globalVariable @global : !spv.ptr<f32, Workgroup>
  func @variable_init_global_variable() -> () {
    %0 = spv._address_of @global : !spv.ptr<f32, Workgroup>
    // CHECK: spv.Variable init({{.*}}) : !spv.ptr<!spv.ptr<f32, Workgroup>, Function>
    %1 = spv.Variable init(%0) : !spv.ptr<!spv.ptr<f32, Workgroup>, Function>
    spv.Return
  }
} attributes {
  capability = ["VariablePointers"],
  extension = ["SPV_KHR_variable_pointers"]
}

// -----

spv.module "Logical" "GLSL450" {
  spv.specConstant @sc = 42 : i32
  // CHECK-LABEL: @variable_init_spec_constant
  func @variable_init_spec_constant() -> () {
    %0 = spv._reference_of @sc : i32
    // CHECK: spv.Variable init(%0) : !spv.ptr<i32, Function>
    %1 = spv.Variable init(%0) : !spv.ptr<i32, Function>
    spv.Return
  }
}

// -----

func @variable_bind() -> () {
  // expected-error @+1 {{cannot have 'descriptor_set' attribute (only allowed in spv.globalVariable)}}
  %0 = spv.Variable bind(1, 2) : !spv.ptr<f32, Function>
  return
}

// -----

func @variable_init_bind() -> () {
  %0 = spv.constant 4.0 : f32
  // expected-error @+1 {{cannot have 'binding' attribute (only allowed in spv.globalVariable)}}
  %1 = spv.Variable init(%0) {binding = 5 : i32} : !spv.ptr<f32, Function>
  return
}

// -----

func @variable_builtin() -> () {
  // expected-error @+1 {{cannot have 'built_in' attribute (only allowed in spv.globalVariable)}}
  %1 = spv.Variable built_in("GlobalInvocationID") : !spv.ptr<vector<3xi32>, Function>
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
  // expected-error @+1 {{op initializer must be the result of a constant or spv.globalVariable op}}
  %0 = spv.Variable init(%arg0) : !spv.ptr<f32, Function>
  return
}

// -----

func @cannot_be_generic_storage_class(%arg0: f32) -> () {
  // expected-error @+1 {{op can only be used to model function-level variables. Use spv.globalVariable for module-level variables}}
  %0 = spv.Variable : !spv.ptr<f32, Generic>
  return
}
