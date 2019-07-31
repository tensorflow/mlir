// RUN: mlir-opt %s -split-input-file | FileCheck %s

// Test with zero-dimensional operands using constant zero index in load/store.
func @zero_dim_no_idx(%arg0 : memref<i32>, %arg1 : memref<i32>, %arg2 : memref<i32>) {
  %c0 = constant 0 : index
  %0 = std.load %arg0[%c0] : memref<i32>
  std.store %0, %arg1[%c0] : memref<i32>
  return
  // CHECK: [[ZERO:%c[0-9]+]] = constant 0 : index
  // CHECK: %0 = load %{{.*}}{{\[}}[[ZERO]]{{\]}} : memref<i32>
}

// -----

// Test with zero-dimensional operands using no index in load/store.
func @zero_dim_const_idx(%arg0 : memref<i32>, %arg1 : memref<i32>, %arg2 : memref<i32>) {
  %0 = std.load %arg0[] : memref<i32>
  std.store %0, %arg1[] : memref<i32>
  return
  // CHECK: %0 = load %{{.*}}[] : memref<i32>
  // CHECK: store %{{.*}}, %{{.*}}[] : memref<i32>
}


