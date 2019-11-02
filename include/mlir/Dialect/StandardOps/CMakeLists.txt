set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRStandardOpsIncGen)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Ops.h.inc DESTINATION include/mlir/Dialect/StandardOps)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Ops.h.inc DESTINATION include/mlir/Dialect/StandardOps)
