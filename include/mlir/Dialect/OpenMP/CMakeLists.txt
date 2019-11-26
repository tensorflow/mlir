set(LLVM_TARGET_DEFINITIONS OpenMPOps.td)
mlir_tablegen(OpenMPOps.h.inc -gen-op-decls)
mlir_tablegen(OpenMPOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIROpenMPOpsIncGen)
