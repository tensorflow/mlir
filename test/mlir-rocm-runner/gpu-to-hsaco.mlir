// RUN: mlir-rocm-runner %s --shared-libs=%hip_wrapper_library_dir/libhip-runtime-wrappers%shlibext --entry-point-result=void | FileCheck %s

func @func_1(%arg0 : f32, %arg1 : memref<?xf32>) {
  %cst = constant 1 : index
  %cst2 = dim %arg1, 0 : memref<?xf32>
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %cst, %grid_y = %cst, %grid_z = %cst)
             threads(%tx, %ty, %tz) in (%block_x = %cst2, %block_y = %cst, %block_z = %cst)
             args(%kernel_arg0 = %arg0, %kernel_arg1 = %arg1) : f32, memref<?xf32> {
    store %kernel_arg0, %kernel_arg1[%tx] : memref<?xf32>
    gpu.return
  }
  return
}

// CHECK: [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
func @main() {
  %0 = alloc() : memref<5xf32>
  %1 = memref_cast %0 : memref<5xf32> to memref<?xf32>

  call @mhipPrintMemRef(%1) : (memref<?xf32>) -> ()

  %4 = constant 0 : i32
  call @mhipHostRegisterMemRef(%1, %4) : (memref<?xf32>, i32) -> ()
  %5 = call @mhipHostGetDevicePointerMemRef(%1, %4) : (memref<?xf32>, i32) -> (memref<?xf32>)

  %6 = constant 1.0 : f32
  call @func_1(%6, %5) : (f32, memref<?xf32>) -> ()

  call @mhipPrintMemRef(%1) : (memref<?xf32>) -> ()

  return
}

func @mhipHostRegisterMemRef(%ptr : memref<?xf32>, %flags : i32) -> ()
func @mhipHostGetDevicePointerMemRef(%ptr : memref<?xf32>, %flags : i32) -> (memref<?xf32>)
func @mhipPrintMemRef(%ptr : memref<?xf32>) -> ()
