file(
  GLOB EIGEN_CC_SOURCES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "*.cc")
file(
  GLOB EIGEN_CU_SOURCES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "*.cu")
if(WITH_GPU)
  nv_library(
    eigen_function
    SRCS ${EIGEN_CC_SOURCES} ${EIGEN_CU_SOURCES}
    DEPS eigen3)
elseif(WITH_ROCM)
  hip_library(
    eigen_function
    SRCS ${EIGEN_CC_SOURCES} ${EIGEN_CU_SOURCES}
    DEPS eigen3)
else()
  cc_library(
    eigen_function
    SRCS ${EIGEN_CC_SOURCES}
    DEPS eigen3)
endif()
