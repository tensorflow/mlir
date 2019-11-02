add_llvm_executable(mlir-cpu-runner
  mlir-cpu-runner.cpp
)
llvm_update_compile_flags(mlir-cpu-runner)
whole_archive_link(mlir-cpu-runner
  MLIRAffineOps
  MLIRLLVMIR
  MLIRTargetLLVMIR
  MLIRTranslation
)
target_link_libraries(mlir-cpu-runner PRIVATE
  MLIRAnalysis
  MLIREDSC
  MLIRExecutionEngine
  MLIRIR
  MLIRJitRunner
  MLIRLLVMIR
  MLIRParser
  MLIRTargetLLVMIR
  MLIRSupport
  LLVMCore
  LLVMSupport
)

install(TARGETS mlir-cpu-runner
  ${export_to_llvmexports}
  COMPONENT mlir-cpu-runner
  RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR})

if(NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets("install-mlir-cpu-runner"
    DEPENDS mlir-cpu-runner
  COMPONENT mlir-cpu-runner)
endif()
