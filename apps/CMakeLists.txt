INCLUDE_DIRECTORIES(${ARGTABLE_ROOT}/include ${CUDA_CUT_INCLUDE_DIR})

LINK_DIRECTORIES(/usr/local/cuda/lib)

ADD_EXECUTABLE(invert        invert.cpp)
ADD_EXECUTABLE(invert_npp    invert_npp.cpp)
TARGET_LINK_LIBRARIES(invert ${ARGTABLE} opencv_core opencv_highgui)
TARGET_LINK_LIBRARIES(invert_npp  ${ARGTABLE} opencv_core opencv_highgui npp)

SET (CUDA_VERBOSE_BUILD OFF)
SET (CUDA_PROPAGATE_HOST_FLAGS OFF) # SSE2 with CUDA gives problems

IF (${CMAKE_BUILD_TYPE} STREQUAL "Release")
  SET(CUDA_NVCC_FLAGS "-arch;sm_21")
ELSEIF (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  SET(CUDA_NVCC_FLAGS "-arch;sm_21;-g;-G")
ELSE()
  MESSAGE( FATAL_ERROR "Only Release and Debug build types are allowed")
ENDIF ()

CUDA_ADD_EXECUTABLE(invert_cuda invert_cuda.cu invert_cuda.cpp)
TARGET_LINK_LIBRARIES(invert_cuda ${ARGTABLE} opencv_core opencv_highgui ${CUDA_CUT_LIBRARY})
