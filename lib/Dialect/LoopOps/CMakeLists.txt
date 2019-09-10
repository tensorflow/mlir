file(GLOB globbed *.c *.cpp)
add_llvm_library(MLIRLoopOps STATIC
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/LoopOps
  )
add_dependencies(MLIRLoopOps MLIRLoopOpsIncGen MLIRStandardOps MLIRIR LLVMSupport)
target_link_libraries(MLIRLoopOps MLIRStandardOps MLIRIR LLVMSupport)
