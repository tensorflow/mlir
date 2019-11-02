set(LIBS
  MLIRAffineOps
  MLIRAnalysis
  MLIREDSC
  MLIRParser
  MLIRPass
  MLIRSPIRV
  MLIRSPIRVSerialization
  MLIRStandardOps
  MLIRTargetLLVMIR
  MLIRTargetNVVMIR
  MLIRTargetROCDLIR
  MLIRTransforms
  MLIRTranslation
  MLIRSupport
  MLIRVectorOps
)
add_llvm_executable(mlir-translate
  mlir-translate.cpp
)
llvm_update_compile_flags(mlir-translate)
whole_archive_link(mlir-translate ${LIBS})
target_link_libraries(mlir-translate PRIVATE MLIRIR MLIRTranslateClParser ${LIBS} LLVMSupport)

install(TARGETS mlir-translate
  ${export_to_llvmexports}
  COMPONENT mlir-translate
  RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR})

if(NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets("install-mlir-translate"
    DEPENDS mlir-translate
  COMPONENT mlir-translate)
endif()
