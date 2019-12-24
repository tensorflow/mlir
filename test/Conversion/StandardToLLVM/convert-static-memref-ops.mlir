// RUN: mlir-opt -convert-std-to-llvm -split-input-file %s | FileCheck %s
// RUN: mlir-opt -convert-std-to-llvm -convert-std-to-llvm-use-alloca=1 %s | FileCheck %s --check-prefix=ALLOCA
// RUN: mlir-opt -test-custom-memref-llvm-lowering %s | FileCheck %s --check-prefix=CUSTOM

// CUSTOM-LABEL: func @check_noalias
// CUSTOM-SAME: arg0: !llvm<"float**"> {llvm.noalias = true}
func @check_noalias(%static : memref<2xf32> {llvm.noalias = true}) {
  return
}

// -----

// CHECK-LABEL: func @check_static_return(%arg0: !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">) -> !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }"> {
// CUSTOM-LABEL: func @check_static_return
// CUSTOM-SAME: (%arg0: !llvm<"float**">) -> !llvm<"float*">
func @check_static_return(%static : memref<32x18xf32>) -> memref<32x18xf32> {
// CHECK:  llvm.return %{{.*}} : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
// CUSTOM: llvm.return %{{.*}} : !llvm<"float*">
  return %static : memref<32x18xf32>
}

// -----

// CHECK-LABEL: func @zero_d_alloc() -> !llvm<"{ float*, float*, i64 }"> {
// ALLOCA-LABEL: func @zero_d_alloc() -> !llvm<"{ float*, float*, i64 }"> {
// CUSTOM-LABEL: func @zero_d_alloc() -> !llvm<"float*">
func @zero_d_alloc() -> memref<f32> {
// CHECK-NEXT:  llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK-NEXT:  %[[null:.*]] = llvm.mlir.null : !llvm<"float*">
// CHECK-NEXT:  %[[one:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[one]]] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
// CHECK-NEXT:  %[[sizeof:.*]] = llvm.ptrtoint %[[gep]] : !llvm<"float*"> to !llvm.i64
// CHECK-NEXT:  llvm.mul %{{.*}}, %[[sizeof]] : !llvm.i64
// CHECK-NEXT:  llvm.call @malloc(%{{.*}}) : (!llvm.i64) -> !llvm<"i8*">
// CHECK-NEXT:  %[[ptr:.*]] = llvm.bitcast %{{.*}} : !llvm<"i8*"> to !llvm<"float*">
// CHECK-NEXT:  llvm.mlir.undef : !llvm<"{ float*, float*, i64 }">
// CHECK-NEXT:  llvm.insertvalue %[[ptr]], %{{.*}}[0] : !llvm<"{ float*, float*, i64 }">
// CHECK-NEXT:  llvm.insertvalue %[[ptr]], %{{.*}}[1] : !llvm<"{ float*, float*, i64 }">
// CHECK-NEXT:  %[[c0:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK-NEXT:  llvm.insertvalue %[[c0]], %{{.*}}[2] : !llvm<"{ float*, float*, i64 }">

// ALLOCA-NOT: malloc
//     ALLOCA: alloca
// ALLOCA-NOT: malloc

// CUSTOM: llvm.mlir.constant(1 : index) : !llvm.i64
// CUSTOM: %[[null:.*]] = llvm.mlir.null : !llvm<"float*">
// CUSTOM: %[[one:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CUSTOM: %[[gep:.*]] = llvm.getelementptr %[[null]][%[[one]]] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
// CUSTOM: %[[sizeof:.*]] = llvm.ptrtoint %[[gep]] : !llvm<"float*"> to !llvm.i64
// CUSTOM: %[[size:.*]] = llvm.mul %{{.*}}, %[[sizeof]] : !llvm.i64
// CUSTOM: llvm.call @malloc(%[[size]]) : (!llvm.i64) -> !llvm<"i8*">
// CUSTOM: %[[ptr:.*]] = llvm.bitcast %{{.*}} : !llvm<"i8*"> to !llvm<"float*">
// CUSTOM: llvm.return %[[ptr]] : !llvm<"float*">
  %0 = alloc() : memref<f32>
  return %0 : memref<f32>
}

// -----

// CHECK-LABEL: func @zero_d_dealloc(%{{.*}}: !llvm<"{ float*, float*, i64 }*">) {
// CUSTOM-LABEL: func @zero_d_dealloc
// CUSTOM-SAME: (%{{.*}}: !llvm<"float**">) {
func @zero_d_dealloc(%arg0: memref<f32>) {
// CHECK-NEXT:  %[[ld:.*]] = llvm.load %{{.*}} : !llvm<"{ float*, float*, i64 }*">
// CHECK-NEXT:  %[[ptr:.*]] = llvm.extractvalue %[[ld]][0] : !llvm<"{ float*, float*, i64 }">
// CHECK-NEXT:  %[[bc:.*]] = llvm.bitcast %[[ptr]] : !llvm<"float*"> to !llvm<"i8*">
// CHECK-NEXT:  llvm.call @free(%[[bc]]) : (!llvm<"i8*">) -> ()

// CUSTOM: %[[ld:.*]] = llvm.load %{{.*}} : !llvm<"float**">
// CUSTOM: %[[bc:.*]] = llvm.bitcast %[[ld]] : !llvm<"float*"> to !llvm<"i8*">
// CUSTOM: llvm.call @free(%[[bc]]) : (!llvm<"i8*">) -> ()
  dealloc %arg0 : memref<f32>
  return
}

// -----

// CHECK-LABEL: func @aligned_1d_alloc(
// CUSTOM-LABEL: func @aligned_1d_alloc(
func @aligned_1d_alloc() -> memref<42xf32> {
// CHECK-NEXT:  llvm.mlir.constant(42 : index) : !llvm.i64
// CHECK-NEXT:  %[[null:.*]] = llvm.mlir.null : !llvm<"float*">
// CHECK-NEXT:  %[[one:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[one]]] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
// CHECK-NEXT:  %[[sizeof:.*]] = llvm.ptrtoint %[[gep]] : !llvm<"float*"> to !llvm.i64
// CHECK-NEXT:  llvm.mul %{{.*}}, %[[sizeof]] : !llvm.i64
// CHECK-NEXT:  %[[alignment:.*]] = llvm.mlir.constant(8 : index) : !llvm.i64
// CHECK-NEXT:  %[[alignmentMinus1:.*]] = llvm.add {{.*}}, %[[alignment]] : !llvm.i64
// CHECK-NEXT:  %[[allocsize:.*]] = llvm.sub %[[alignmentMinus1]], %[[one]] : !llvm.i64
// CHECK-NEXT:  %[[allocated:.*]] = llvm.call @malloc(%[[allocsize]]) : (!llvm.i64) -> !llvm<"i8*">
// CHECK-NEXT:  %[[ptr:.*]] = llvm.bitcast %{{.*}} : !llvm<"i8*"> to !llvm<"float*">
// CHECK-NEXT:  llvm.mlir.undef : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
// CHECK-NEXT:  llvm.insertvalue %[[ptr]], %{{.*}}[0] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
// CHECK-NEXT:  %[[allocatedAsInt:.*]] = llvm.ptrtoint %[[allocated]] : !llvm<"i8*"> to !llvm.i64
// CHECK-NEXT:  %[[alignAdj1:.*]] = llvm.urem %[[allocatedAsInt]], %[[alignment]] : !llvm.i64
// CHECK-NEXT:  %[[alignAdj2:.*]] = llvm.sub %[[alignment]], %[[alignAdj1]] : !llvm.i64
// CHECK-NEXT:  %[[alignAdj3:.*]] = llvm.urem %[[alignAdj2]], %[[alignment]] : !llvm.i64
// CHECK-NEXT:  %[[aligned:.*]] = llvm.getelementptr %{{.*}}[%[[alignAdj3]]] : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
// CHECK-NEXT:  %[[alignedBitCast:.*]] = llvm.bitcast %[[aligned]] : !llvm<"i8*"> to !llvm<"float*">
// CHECK-NEXT:  llvm.insertvalue %[[alignedBitCast]], %{{.*}}[1] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
// CHECK-NEXT:  %[[c0:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK-NEXT:  llvm.insertvalue %[[c0]], %{{.*}}[2] : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">

// CUSTOM-NEXT: llvm.mlir.constant(42 : index) : !llvm.i64
// CUSTOM-NEXT: %[[null:.*]] = llvm.mlir.null : !llvm<"float*">
// CUSTOM-NEXT: %[[one:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CUSTOM-NEXT: %[[gep:.*]] = llvm.getelementptr %[[null]][%[[one]]] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
// CUSTOM-NEXT: %[[sizeof:.*]] = llvm.ptrtoint %[[gep]] : !llvm<"float*"> to !llvm.i64
// CUSTOM-NEXT: llvm.mul %{{.*}}, %[[sizeof]] : !llvm.i64
// CUSTOM-NEXT: %[[alignment:.*]] = llvm.mlir.constant(8 : index) : !llvm.i64
// CUSTOM-NEXT: %[[alignmentMinus1:.*]] = llvm.add %5{{.*}}, %[[alignment]] : !llvm.i64
// CUSTOM-NEXT: %[[allocsize:.*]] = llvm.sub %[[alignmentMinus1]], %[[one]] : !llvm.i64
// CUSTOM-NEXT: %[[allocated:.*]] = llvm.call @malloc(%[[allocsize]]) : (!llvm.i64) -> !llvm<"i8*">
// CUSTOM-NEXT: %[[ptr:.*]] = llvm.bitcast %{{.*}} : !llvm<"i8*"> to !llvm<"float*">
// CUSTOM-NEXT: %[[allocatedAsInt:.*]] = llvm.ptrtoint %[[allocated]] : !llvm<"i8*"> to !llvm.i64
// CUSTOM-NEXT: %[[alignAdj1:.*]] = llvm.urem %[[allocatedAsInt]], %[[alignment]] : !llvm.i64
// CUSTOM-NEXT: %[[alignAdj2:.*]] = llvm.sub %[[alignment]], %[[alignAdj1]] : !llvm.i64
// CUSTOM-NEXT: %[[alignAdj3:.*]] = llvm.urem %[[alignAdj2]], %[[alignment]] : !llvm.i64
// CUSTOM-NEXT: %[[aligned:.*]] = llvm.getelementptr %{{.*}}[%[[alignAdj3]]] : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
// CUSTOM-NEXT: %[[alignedBitCast:.*]] = llvm.bitcast %15 : !llvm<"i8*"> to !llvm<"float*">
// Alignment is not implemented in custom lowering so base ptr is returned.
// CUSTOM: llvm.return %[[ptr]] : !llvm<"float*">
  %0 = alloc() {alignment = 8} : memref<42xf32>
  return %0 : memref<42xf32>
}

// -----

// CHECK-LABEL: func @static_alloc() -> !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }"> {
// CUSTOM-LABEL: func @static_alloc()
// CUSTOM-SAME: -> !llvm<"float*">
func @static_alloc() -> memref<32x18xf32> {
// CHECK-NEXT:  %[[sz1:.*]] = llvm.mlir.constant(32 : index) : !llvm.i64
// CHECK-NEXT:  %[[sz2:.*]] = llvm.mlir.constant(18 : index) : !llvm.i64
// CHECK-NEXT:  %[[num_elems:.*]] = llvm.mul %[[sz1]], %[[sz2]] : !llvm.i64
// CHECK-NEXT:  %[[null:.*]] = llvm.mlir.null : !llvm<"float*">
// CHECK-NEXT:  %[[one:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[one]]] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
//  CHECK-NEXT:  %[[sizeof:.*]] = llvm.ptrtoint %[[gep]] : !llvm<"float*"> to !llvm.i64
// CHECK-NEXT:  %[[bytes:.*]] = llvm.mul %[[num_elems]], %[[sizeof]] : !llvm.i64
// CHECK-NEXT:  %[[allocated:.*]] = llvm.call @malloc(%[[bytes]]) : (!llvm.i64) -> !llvm<"i8*">
// CHECK-NEXT:  llvm.bitcast %[[allocated]] : !llvm<"i8*"> to !llvm<"float*">

// CUSTOM-NEXT: %[[sz1:.*]] = llvm.mlir.constant(32 : index) : !llvm.i64
// CUSTOM-NEXT: %[[sz2:.*]] = llvm.mlir.constant(18 : index) : !llvm.i64
// CUSTOM-NEXT: %[[num_elems:.*]] = llvm.mul %[[sz1]], %[[sz2]] : !llvm.i64
// CUSTOM-NEXT: %[[null:.*]] = llvm.mlir.null : !llvm<"float*">
// CUSTOM-NEXT: %[[one:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CUSTOM-NEXT: %[[gep:.*]] = llvm.getelementptr %[[null]][%[[one]]] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
// CUSTOM-NEXT: %[[sizeof:.*]] = llvm.ptrtoint %[[gep]] : !llvm<"float*"> to !llvm.i64
// CUSTOM-NEXT: %[[bytes:.*]] = llvm.mul %[[num_elems]], %[[sizeof]] : !llvm.i64
// CUSTOM-NEXT: %[[allocated:.*]] = llvm.call @malloc(%[[bytes]]) : (!llvm.i64) -> !llvm<"i8*">
// CUSTOM-NEXT: %[[bc:.*]] = llvm.bitcast %[[allocated]] : !llvm<"i8*"> to !llvm<"float*">
// CUSTOM: llvm.return %[[bc]] : !llvm<"float*">
 %0 = alloc() : memref<32x18xf32>
 return %0 : memref<32x18xf32>
}

// -----

// CHECK-LABEL: func @static_dealloc(%{{.*}}: !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">) {
// CUSTOM-LABEL: func @static_dealloc
// CUSTOM-SAME: (%{{.*}}: !llvm<"float**">)
func @static_dealloc(%static: memref<10x8xf32>) {
// CHECK-NEXT:  %[[ld:.*]] = llvm.load %{{.*}} : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">
// CHECK-NEXT:  %[[ptr:.*]] = llvm.extractvalue %[[ld]][0] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
// CHECK-NEXT:  %[[bc:.*]] = llvm.bitcast %[[ptr]] : !llvm<"float*"> to !llvm<"i8*">
// CHECK-NEXT:  llvm.call @free(%[[bc]]) : (!llvm<"i8*">) -> ()

// CUSTOM-NEXT: %[[ld:.*]] = llvm.load %arg0 : !llvm<"float**">
// CUSTOM-NEXT: %[[bc:.*]] = llvm.bitcast %[[ld]] : !llvm<"float*"> to !llvm<"i8*">
// CUSTOM-NEXT: llvm.call @free(%[[bc]]) : (!llvm<"i8*">) -> ()
  dealloc %static : memref<10x8xf32>
  return
}

// -----

// CHECK-LABEL: func @zero_d_load(%{{.*}}: !llvm<"{ float*, float*, i64 }*">) -> !llvm.float {
// CUSTOM-LABEL: func @zero_d_load
// CUSTOM-SAME: (%{{.*}}: !llvm<"float**">) -> !llvm.float
func @zero_d_load(%arg0: memref<f32>) -> f32 {
// CHECK-NEXT:  %[[ld:.*]] = llvm.load %{{.*}} : !llvm<"{ float*, float*, i64 }*">
// CHECK-NEXT:  %[[ptr:.*]] = llvm.extractvalue %[[ld]][1] : !llvm<"{ float*, float*, i64 }">
// CHECK-NEXT:  %[[c0:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK-NEXT:  %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[c0]]] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
// CHECK-NEXT:  llvm.load %[[addr]] : !llvm<"float*">

// CUSTOM-NEXT: %[[ptr:.*]] = llvm.load %arg0 : !llvm<"float**">
// CUSTOM-NEXT: %[[c0:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
// CUSTOM-NEXT: %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[c0]]] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
// CUSTOM-NEXT: llvm.load %[[addr:.*]] : !llvm<"float*">
  %0 = load %arg0[] : memref<f32>
  return %0 : f32
}

// -----

// CHECK-LABEL: func @static_load(
//       CHECK:   %[[A:.*]]: !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">, %[[I:.*]]: !llvm.i64, %[[J:.*]]: !llvm.i64

// CUSTOM-LABEL: func @static_load
// CUSTOM-SAME: (%[[A:.*]]: !llvm<"float**">, %[[I:.*]]: !llvm.i64, %[[J:.*]]: !llvm.i64) {
func @static_load(%static : memref<10x42xf32>, %i : index, %j : index) {
// CHECK-NEXT:  %[[ld:.*]] = llvm.load %[[A]] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">
//  CHECK-NEXT:  %[[ptr:.*]] = llvm.extractvalue %[[ld]][1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//  CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//  CHECK-NEXT:  %[[st0:.*]] = llvm.mlir.constant(42 : index) : !llvm.i64
//  CHECK-NEXT:  %[[offI:.*]] = llvm.mul %[[I]], %[[st0]] : !llvm.i64
//  CHECK-NEXT:  %[[off0:.*]] = llvm.add %[[off]], %[[offI]] : !llvm.i64
//  CHECK-NEXT:  %[[st1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:  %[[offJ:.*]] = llvm.mul %[[J]], %[[st1]] : !llvm.i64
//  CHECK-NEXT:  %[[off1:.*]] = llvm.add %[[off0]], %[[offJ]] : !llvm.i64
//  CHECK-NEXT:  %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
// CHECK-NEXT:  llvm.load %[[addr]] : !llvm<"float*">

// CUSTOM-NEXT: %[[ptr:.*]] = llvm.load %[[A]] : !llvm<"float**">
// CUSTOM-NEXT: %[[off:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
// CUSTOM-NEXT: %[[st0:.*]] = llvm.mlir.constant(42 : index) : !llvm.i64
// CUSTOM-NEXT: %[[offI:.*]] = llvm.mul %[[I]], %[[st0]] : !llvm.i64
// CUSTOM-NEXT: %[[off0:.*]] = llvm.add %[[off]], %[[offI]] : !llvm.i64
// CUSTOM-NEXT: %[[st1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CUSTOM-NEXT: %[[offJ:.*]] = llvm.mul %[[J]], %[[st1]] : !llvm.i64
// CUSTOM-NEXT: %[[off1:.*]] = llvm.add %[[off0]], %[[offJ]] : !llvm.i64
// CUSTOM-NEXT: %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
// CUSTOM-NEXT: llvm.load %[[addr]] : !llvm<"float*">
  %0 = load %static[%i, %j] : memref<10x42xf32>
  return
}

// -----

// CHECK-LABEL: func @zero_d_store
// CHECK-SAME: (%[[A:.*]]: !llvm<"{ float*, float*, i64 }*">, %[[val:.*]]: !llvm.float)
// CUSTOM-LABEL: func @zero_d_store
// CUSTOM-SAME: (%[[A:.*]]: !llvm<"float**">, %[[val:.*]]: !llvm.float)
func @zero_d_store(%arg0: memref<f32>, %arg1: f32) {
// CHECK-NEXT:  %[[ld:.*]] = llvm.load %{{.*}} : !llvm<"{ float*, float*, i64 }*">
// CHECK-NEXT:  %[[ptr:.*]] = llvm.extractvalue %[[ld]][1] : !llvm<"{ float*, float*, i64 }">
// CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK-NEXT:  %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off]]] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
// CHECK-NEXT:  llvm.store %[[val]], %[[addr]] : !llvm<"float*">

// CUSTOM-NEXT: %[[ptr:.*]] = llvm.load %[[A]] : !llvm<"float**">
// CUSTOM-NEXT: %[[off:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
// CUSTOM-NEXT: %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off]]] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
// CUSTOM-NEXT: llvm.store %[[val]], %[[addr]] : !llvm<"float*">
  store %arg1, %arg0[] : memref<f32>
  return
}

// -----

// CHECK-LABEL: func @static_store
// CUSTOM-LABEL: func @static_store
func @static_store(%static : memref<10x42xf32>, %i : index, %j : index, %val : f32) {
//  CHECK-NEXT:  %[[ld:.*]] = llvm.load %{{.*}} : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">
//  CHECK-NEXT:  %[[ptr:.*]] = llvm.extractvalue %[[ld]][1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//  CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//  CHECK-NEXT:  %[[st0:.*]] = llvm.mlir.constant(42 : index) : !llvm.i64
//  CHECK-NEXT:  %[[offI:.*]] = llvm.mul %[[I]], %[[st0]] : !llvm.i64
//  CHECK-NEXT:  %[[off0:.*]] = llvm.add %[[off]], %[[offI]] : !llvm.i64
//  CHECK-NEXT:  %[[st1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:  %[[offJ:.*]] = llvm.mul %[[J]], %[[st1]] : !llvm.i64
//  CHECK-NEXT:  %[[off1:.*]] = llvm.add %[[off0]], %[[offJ]] : !llvm.i64
//  CHECK-NEXT:  %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
//  CHECK-NEXT:  llvm.store %{{.*}}, %[[addr]] : !llvm<"float*">

// CUSTOM-NEXT: %[[ptr:.*]] = llvm.load %{{.*}} : !llvm<"float**">
// CUSTOM-NEXT: %[[off:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
// CUSTOM-NEXT: %[[st0:.*]] = llvm.mlir.constant(42 : index) : !llvm.i64
// CUSTOM-NEXT: %[[offI:.*]] = llvm.mul %[[I]], %[[st0]] : !llvm.i64
// CUSTOM-NEXT: %[[off0:.*]] = llvm.add %[[off]], %[[offI]] : !llvm.i64
// CUSTOM-NEXT: %[[st1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CUSTOM-NEXT: %[[offJ:.*]] = llvm.mul %[[J]], %[[st1]] : !llvm.i64
// CUSTOM-NEXT: %[[off1:.*]] = llvm.add %[[off0]], %[[offJ]] : !llvm.i64
// CUSTOM-NEXT: %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
// CUSTOM-NEXT: llvm.store %{{.*}}, %[[addr]] : !llvm<"float*">
  store %val, %static[%i, %j] : memref<10x42xf32>
  return
}

// -----

// CHECK-LABEL: func @static_memref_dim(%arg0: !llvm<"{ float*, float*, i64, [5 x i64], [5 x i64] }*">) {
// CUSTOM-LABEL: func @static_memref_dim
// CUSTOM-SAME: (%arg0: !llvm<"float**">)
func @static_memref_dim(%static : memref<42x32x15x13x27xf32>) {
// CHECK-NEXT:  %[[ld:.*]] = llvm.load %{{.*}} : !llvm<"{ float*, float*, i64, [5 x i64], [5 x i64] }*">
// CHECK-NEXT:  llvm.mlir.constant(42 : index) : !llvm.i64

// CUSTOM-NEXT: %[[ld:.*]] = llvm.load %{{.*}} : !llvm<"float**">
// CUSTOM-NEXT: llvm.mlir.constant(42 : index) : !llvm.i64
  %0 = dim %static, 0 : memref<42x32x15x13x27xf32>
// CHECK-NEXT:  llvm.mlir.constant(32 : index) : !llvm.i64
// CUSTOM-NEXT:  llvm.mlir.constant(32 : index) : !llvm.i64
  %1 = dim %static, 1 : memref<42x32x15x13x27xf32>
// CHECK-NEXT:  llvm.mlir.constant(15 : index) : !llvm.i64
// CUSTOM-NEXT:  llvm.mlir.constant(15 : index) : !llvm.i64
  %2 = dim %static, 2 : memref<42x32x15x13x27xf32>
// CHECK-NEXT:  llvm.mlir.constant(13 : index) : !llvm.i64
// CUSTOM-NEXT:  llvm.mlir.constant(13 : index) : !llvm.i64
  %3 = dim %static, 3 : memref<42x32x15x13x27xf32>
// CHECK-NEXT:  llvm.mlir.constant(27 : index) : !llvm.i64
// CUSTOM-NEXT:  llvm.mlir.constant(27 : index) : !llvm.i64
  %4 = dim %static, 4 : memref<42x32x15x13x27xf32>
  return
}

