// RUN: mlir-cpu-runner %s | FileCheck %s
// RUN: mlir-cpu-runner -e foo -init-value 1000 %s | FileCheck -check-prefix=NOMAIN %s
// RUN: mlir-cpu-runner %s -O3 | FileCheck %s
// RUN: mlir-cpu-runner -e affine -init-value 2.0 %s | FileCheck -check-prefix=AFFINE %s

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
