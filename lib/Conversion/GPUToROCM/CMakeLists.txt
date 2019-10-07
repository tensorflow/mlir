if(MLIR_ROCM_CONVERSIONS_ENABLED)
  llvm_map_components_to_libnames(amdgpu "AMDGPU")

  add_llvm_library(MLIRGPUtoROCMTransforms
    ConvertKernelFuncToHSACO.cpp
    ConvertLaunchFuncToHIPCalls.cpp
    GenerateHSACOAccessors.cpp
  )
  target_link_libraries(MLIRGPUtoROCMTransforms
    MLIRGPU
    MLIRLLVMIR
    MLIRROCDLIR
    MLIRPass
    MLIRTargetROCDLIR
    ${amdgpu}
  )
endif()
