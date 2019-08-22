// RUN: mlir-opt %s -vector-lower-to-llvm-dialect | FileCheck %s

func @outerproduct(%arg0: vector<2xf32>, %arg1: vector<3xf32>) -> vector<2x3xf32> {
  %2 = vector.outerproduct %arg0, %arg1 : vector<2xf32>, vector<3xf32>
  return %2 : vector<2x3xf32>
}
//    CHECK-LABEL: outerproduct
//          CHECK:   llvm.undef : !llvm<"[2 x <3 x float>]">
//          CHECK:   llvm.shufflevector {{.*}} [0 : i32, 0 : i32, 0 : i32] : !llvm<"<2 x float>">, !llvm<"<2 x float>">
//          CHECK:   llvm.fmul {{.*}}, {{.*}} : !llvm<"<3 x float>">
//          CHECK:   llvm.insertvalue {{.*}}[0] : !llvm<"[2 x <3 x float>]">
//          CHECK:   llvm.shufflevector {{.*}} [1 : i32, 1 : i32, 1 : i32] : !llvm<"<2 x float>">, !llvm<"<2 x float>">
//          CHECK:   llvm.fmul {{.*}}, {{.*}} : !llvm<"<3 x float>">
//          CHECK:   llvm.insertvalue {{.*}}[1] : !llvm<"[2 x <3 x float>]">
//          CHECK:   llvm.return {{.*}} : !llvm<"[2 x <3 x float>]">

func @outerproduct_add(%arg0: vector<2xf32>, %arg1: vector<3xf32>, %arg2: vector<2x3xf32>) -> vector<2x3xf32> {
  %2 = vector.outerproduct %arg0, %arg1, %arg2 : vector<2xf32>, vector<3xf32>
  return %2 : vector<2x3xf32>
}
//    CHECK-LABEL: outerproduct_add
//          CHECK:   llvm.undef : !llvm<"[2 x <3 x float>]">
//          CHECK:   llvm.shufflevector {{.*}} [0 : i32, 0 : i32, 0 : i32] : !llvm<"<2 x float>">, !llvm<"<2 x float>">
//          CHECK:   llvm.extractvalue {{.*}}[0] : !llvm<"[2 x <3 x float>]">
//          CHECK:   "llvm.intr.fmuladd"({{.*}}) : (!llvm<"<3 x float>">, !llvm<"<3 x float>">, !llvm<"<3 x float>">) -> !llvm<"<3 x float>">
//          CHECK:   llvm.insertvalue {{.*}}[0] : !llvm<"[2 x <3 x float>]">
//          CHECK:   llvm.shufflevector {{.*}} [1 : i32, 1 : i32, 1 : i32] : !llvm<"<2 x float>">, !llvm<"<2 x float>">
//          CHECK:   llvm.extractvalue {{.*}}[1] : !llvm<"[2 x <3 x float>]">
//          CHECK:   "llvm.intr.fmuladd"({{.*}}) : (!llvm<"<3 x float>">, !llvm<"<3 x float>">, !llvm<"<3 x float>">) -> !llvm<"<3 x float>">
//          CHECK:   llvm.insertvalue {{.*}}[1] : !llvm<"[2 x <3 x float>]">
//          CHECK:   llvm.return {{.*}} : !llvm<"[2 x <3 x float>]">

func @extract_vec_2d_from_vec_3d(%arg0: vector<4x3x16xf32>) -> vector<3x16xf32> {
  %0 = vector.extractelement %arg0[0 : i32]: vector<4x3x16xf32>
  return %0 : vector<3x16xf32>
}
// CHECK-LABEL: extract_vec_2d_from_vec_3d
//       CHECK:   llvm.extractvalue %{{.*}}[0 : i32] : !llvm<"[4 x [3 x <16 x float>]]">
//       CHECK:   llvm.return %{{.*}} : !llvm<"[3 x <16 x float>]">

func @extract_element_from_vec_3d(%arg0: vector<4x3x16xf32>) -> f32 {
  %0 = vector.extractelement %arg0[0 : i32, 0 : i32, 0 : i32]: vector<4x3x16xf32>
  return %0 : f32
}
// CHECK-LABEL: extract_element_from_vec_3d
//       CHECK:   llvm.extractvalue %{{.*}}[0 : i32, 0 : i32] : !llvm<"[4 x [3 x <16 x float>]]">
//       CHECK:   llvm.constant(0 : i32) : !llvm.i32
//       CHECK:   llvm.extractelement %{{.*}}, %{{.*}} : !llvm<"<16 x float>">
//       CHECK:   llvm.return %{{.*}} : !llvm.float