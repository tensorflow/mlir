if(MLIR_ROCM_CONVERSIONS_ENABLED)

  # Check whether the ROCm installation dir exists
  set(ROCM_INSTALL_DIR "/opt/rocm" CACHE STRING "ROCm installation directory")
  if (EXISTS ${ROCM_INSTALL_DIR})
    message("-- ROCm Install Dir - ${ROCM_INSTALL_DIR}")
  else()
    message(SEND_ERROR "-- NOT FOUND : ROCm Install Dir - ${ROCM_INSTALL_DIR}")
  endif()

  # Check whether the ROCm device library dir exists
  set(ROCM_DEVICE_LIB_DIR ${ROCM_INSTALL_DIR}/lib)
  if (EXISTS ${ROCM_DEVICE_LIB_DIR})
    message("-- ROCm Device Library Dir - ${ROCM_DEVICE_LIB_DIR}")
  else ()
    message(SEND_ERROR "-- NOT FOUND : ROCm Device Library Dir - ${ROCM_DEVICE_LIB_DIR}")
  endif()

  # Check whether the ROCm HCC linker exists
  set(ROCM_HCC_LINKER ${ROCM_INSTALL_DIR}/hcc/bin/ld.lld)
  if (EXISTS ${ROCM_HCC_LINKER})
    message("-- ROCm HCC Linker - ${ROCM_HCC_LINKER}")
  else ()
    message(SEND_ERROR "-- NOT FOUND : ROCm HCC Linker - ${ROCM_HCC_LINKER}")
  endif()

  # Generate the ROCm Configuration header file
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/ROCMConfig.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ROCMConfig.h"
    )

endif()
