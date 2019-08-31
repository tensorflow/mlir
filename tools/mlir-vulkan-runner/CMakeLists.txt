set(LLVM_OPTIONAL_SOURCES
  VulkanRuntime.cpp
  )

if (MLIR_VULKAN_RUNNER_ENABLED)
  message(STATUS "Building the Vulkan runner")

  # At first try "FindVulkan" from:
  # https://cmake.org/cmake/help/v3.7/module/FindVulkan.html
  if (NOT CMAKE_VERSION VERSION_LESS 3.7.0)
    find_package(Vulkan)
  endif()

  # If Vulkan is not found try a path specified by VULKAN_SDK.
  if (NOT Vulkan_FOUND)
    if ("$ENV{VULKAN_SDK}" STREQUAL "")
      message(FATAL_ERROR "VULKAN_SDK environment variable is empty")
    endif()

    find_library(Vulkan_LIBRARY vulkan HINTS "$ENV{VULKAN_SDK}/lib" REQUIRED)
    if (Vulkan_LIBRARY)
      set(Vulkan_FOUND ON)
      set(Vulkan_INCLUDE_DIR "$ENV{VULKAN_SDK}/include")
      message(STATUS "Found Vulkan: " ${Vulkan_LIBRARY})
    endif()
  endif()

  if (NOT Vulkan_FOUND)
    message(FATAL_ERROR "Cannot find Vulkan library")
  endif()

  # Create archive for this moment.
  add_llvm_library(MLIRVulkanRuntime
    VulkanRuntime.cpp
    )

  target_include_directories(MLIRVulkanRuntime PRIVATE ${Vulkan_INCLUDE_DIR})

  target_link_libraries(MLIRVulkanRuntime
    MLIRSPIRVSerialization
    LLVMCore
    LLVMSupport
    ${Vulkan_LIBRARY}
    )
endif()
