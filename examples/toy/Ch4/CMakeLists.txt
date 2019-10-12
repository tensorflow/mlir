set(LLVM_LINK_COMPONENTS
  Support
  )

set(LLVM_TARGET_DEFINITIONS include/toy/Ops.td)
mlir_tablegen(include/toy/Ops.h.inc -gen-op-decls)
mlir_tablegen(include/toy/Ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(ToyCh4OpsIncGen)

set(LLVM_TARGET_DEFINITIONS include/toy/ShapeInferenceInterface.td)
mlir_tablegen(include/toy/ShapeInferenceOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(include/toy/ShapeInferenceOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(ShapeInferenceInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.h.inc -gen-rewriters)
mlir_tablegen(ToyCombine.cpp.inc -gen-rewriters)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(MLIRToyCombineIncGen2)

add_toy_chapter(toyc-ch4
  toyc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp
  mlir/ToyCombine.cpp
  mlir/ShapeInferencePass.cpp
  mlir/DeadFunctionEliminationPass.cpp
  )

add_dependencies(toyc-ch4 ToyCh4OpsIncGen)
add_dependencies(toyc-ch4 ShapeInferenceInterfaceIncGen)
add_dependencies(toyc-ch4 MLIRToyCombineIncGen2)
include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
target_link_libraries(toyc-ch4
  PRIVATE
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRTransforms)

