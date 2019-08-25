// RUN: mlir-cpu-runner %s | FileCheck %s
// RUN: mlir-cpu-runner -e foo -init-value 1000 %s | FileCheck -check-prefix=NOMAIN %s
// RUN: mlir-cpu-runner %s -O3 | FileCheck %s
// RUN: mlir-cpu-runner -e affine -init-value 2.0 %s | FileCheck -check-prefix=AFFINE %s
// RUN: mlir-cpu-runner -e bar -init-value 2.0 %s | FileCheck -check-prefix=BAR %s
// RUN: mlir-cpu-runner -e large_vec_memref -init-value 2.0 %s | FileCheck -check-prefix=LARGE-VEC %s

// RUN: cp %s %t
// RUN: mlir-cpu-runner %t -dump-object-file | FileCheck %t
// RUN: ls %t.o
// RUN: rm %t.o

// RUN: mlir-cpu-runner %s -dump-object-file -object-filename=%T/test.o | FileCheck %s
// RUN: ls %T/test.o
// RUN: rm %T/test.o

func @fabsf(f32) -> f32

func @main(%a : memref<2xf32>, %b : memref<1xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = constant -420.0 : f32
  %1 = load %a[%c0] : memref<2xf32>
  %2 = load %a[%c1] : memref<2xf32>
  %3 = addf %0, %1 : f32
  %4 = addf %3, %2 : f32
  %5 = call @fabsf(%4) : (f32) -> f32
  store %5, %b[%c0] : memref<1xf32>
  return
}
// CHECK: 0.000000e+00 0.000000e+00
// CHECK-NEXT: 4.200000e+02

func @foo(%a : memref<1x1xf32>) -> memref<1x1xf32> {
  %c0 = constant 0 : index
  %0 = constant 1234.0 : f32
  %1 = load %a[%c0, %c0] : memref<1x1xf32>
  %2 = addf %1, %0 : f32
  store %2, %a[%c0, %c0] : memref<1x1xf32>
  return %a : memref<1x1xf32>
}
// NOMAIN: 2.234000e+03
// NOMAIN-NEXT: 2.234000e+03

func @affine(%a : memref<32xf32>) -> memref<32xf32> {
  %cf1 = constant 42.0 : f32
  %N = dim %a, 0 : memref<32xf32>
  affine.for %i = 0 to %N {
    affine.store %cf1, %a[%i] : memref<32xf32>
  }
  return %a : memref<32xf32>
}
// AFFINE: 4.2{{0+}}e+01

func @bar(%a : memref<16xvector<4xf32>>) -> memref<16xvector<4xf32>> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  %u = load %a[%c0] : memref<16xvector<4xf32>>
  %v = load %a[%c1] : memref<16xvector<4xf32>>
  %w = addf %u, %v : vector<4xf32>
  store %w, %a[%c0] : memref<16xvector<4xf32>>

  return %a : memref<16xvector<4xf32>>
}
// BAR: 4.{{0+}}e+00 4.{{0+}}e+00 4.{{0+}}e+00 4.{{0+}}e+00 2.{{0+}}e+00
// BAR-NEXT: 4.{{0+}}e+00 4.{{0+}}e+00 4.{{0+}}e+00 4.{{0+}}e+00 2.{{0+}}e+00

func @large_vec_memref(%arg2: memref<128x128xvector<8xf32>>) -> memref<128x128xvector<8xf32>> {
  %c0 = constant 0 : index
  %c127 = constant 127 : index
  %v  = constant dense<42.0> : vector<8xf32>
  store %v, %arg2[%c0, %c0] : memref<128x128xvector<8xf32>>
  store %v, %arg2[%c127, %c127] : memref<128x128xvector<8xf32>>
  return %arg2 : memref<128x128xvector<8xf32>>
}
// LARGE-VEC: 4.200000e+01
