set(LLVM_TARGET_DEFINITIONS CallInterfaces.td)
mlir_tablegen(CallInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(CallInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRCallOpInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS InferTypeOpInterface.td)
mlir_tablegen(InferTypeOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(InferTypeOpInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRTypeInferOpInterfaceIncGen)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CallInterfaces.h.inc DESTINATION include/mlir/Analysis)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/InferTypeOpInterface.h.inc DESTINATION include/mlir/Analysis)
