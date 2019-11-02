set(LLVM_TARGET_DEFINITIONS AffineOps.td)
mlir_tablegen(AffineOps.h.inc -gen-op-decls)
mlir_tablegen(AffineOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRAffineOpsIncGen)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/AffineOps.h.inc DESTINATION include/mlir/Dialect/AffineOps)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/AffineOps.cpp.inc DESTINATION include/mlir/Dialect/AffineOps)
