// RUN: mlir-opt %s | FileCheck %s

func @rocdl_special_regs() -> !llvm.i32 {
  // CHECK: %0 = rocdl.workitem.id.x : !llvm.i32
  %0 = rocdl.workitem.id.x : !llvm.i32
  // CHECK: %1 = rocdl.workitem.id.y : !llvm.i32
  %1 = rocdl.workitem.id.y : !llvm.i32
  // CHECK: %2 = rocdl.workitem.id.z : !llvm.i32
  %2 = rocdl.workitem.id.z : !llvm.i32
  // CHECK: %3 = rocdl.workgroup.id.x : !llvm.i32
  %3 = rocdl.workgroup.id.x : !llvm.i32
  // CHECK: %4 = rocdl.workgroup.id.y : !llvm.i32
  %4 = rocdl.workgroup.id.y : !llvm.i32
  // CHECK: %5 = rocdl.workgroup.id.z : !llvm.i32
  %5 = rocdl.workgroup.id.z : !llvm.i32
  // CHECK: %6 = rocdl.workgroup.dim.x : !llvm.i32
  %6 = rocdl.workgroup.dim.x : !llvm.i32
  // CHECK: %7 = rocdl.workgroup.dim.y : !llvm.i32
  %7 = rocdl.workgroup.dim.y : !llvm.i32
  // CHECK: %8 = rocdl.workgroup.dim.z : !llvm.i32
  %8 = rocdl.workgroup.dim.z : !llvm.i32
  // CHECK: %9 = rocdl.grid.dim.x : !llvm.i32
  %9 = rocdl.grid.dim.x : !llvm.i32
  // CHECK: %10 = rocdl.grid.dim.y : !llvm.i32
  %10 = rocdl.grid.dim.y : !llvm.i32
  // CHECK: %11 = rocdl.grid.dim.z : !llvm.i32
  %11 = rocdl.grid.dim.z : !llvm.i32
  llvm.return %0 : !llvm.i32
}
