add_llvm_library(MLIROpenMPIR
  IR/OpenMPDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/OpenMP
  )

add_dependencies(MLIROpenMPIR MLIROpenMPOpsIncGen)
