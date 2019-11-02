set(LLVM_LINK_COMPONENTS
  MLIRTableGen
  Support
  )

add_tablegen(mlir-tblgen MLIR
  EnumsGen.cpp
  LLVMIRConversionGen.cpp
  mlir-tblgen.cpp
  OpDefinitionsGen.cpp
  OpDocGen.cpp
  OpInterfacesGen.cpp
  ReferenceImplGen.cpp
  RewriterGen.cpp
  SPIRVUtilsGen.cpp
  StructsGen.cpp
  )
set_target_properties(mlir-tblgen PROPERTIES FOLDER "Tablegenning")

install(TARGETS mlir-tblgen
  ${export_to_llvmexports}
  COMPONENT mlir-tblgen
  RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR})

if(NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets("install-mlir-tblgen"
    DEPENDS mlir-tblgen
  COMPONENT mlir-tblgen)
endif()
