if (MLIR_VULKAN_RUNNER_ENABLED)
add_mlir_unittest(MLIRSPIRVRuntimeTests
  RuntimeTests.cpp
)
target_link_libraries(MLIRSPIRVRuntimeTests
  PRIVATE
  MLIRSPIRV
  MLIRVulkanRuntime
  )
whole_archive_link(MLIRSPIRVRuntimeTests MLIRSPIRV)
endif()
