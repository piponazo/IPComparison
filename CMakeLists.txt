# ----------------------------------------------------------------------------
#  Root CMake file for IPComparison
#
#  From a off-tree build directory, invoke:
#      $ cmake <PATH_TO_PROJECT_CODE>
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
#   Basic Configuration
# ----------------------------------------------------------------------------
PROJECT(IPComparison)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

SET(VERSION "0.1.0")
STRING(REGEX MATCHALL "[0-9]" VERSION_PARTS "${VERSION}")
LIST(GET VERSION_PARTS 0 VERSION_MAJOR)
LIST(GET VERSION_PARTS 1 VERSION_MINOR)
LIST(GET VERSION_PARTS 2 VERSION_PATCH)

#------------------------------------------------------
# Build type
#------------------------------------------------------
SET(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
IF(DEFINED CMAKE_BUILD_TYPE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES} )
ENDIF()

IF(NOT CMAKE_BUILD_TYPE )
   SET( CMAKE_BUILD_TYPE "Debug" )
ENDIF()

SET(CMAKE_INCLUDE_DIRS_CONFIGCMAKE ${CMAKE_INSTALL_PREFIX}/include CACHE PATH "Output directory for headers")
SET(CMAKE_LIB_DIRS_CONFIGCMAKE ${CMAKE_INSTALL_PREFIX}/lib CACHE PATH "Output directory for libraries")

# ----------------------------------------------------------------------------
#   PROJECT CONFIGURATION
#   force some variables that could be defined in the command line to be written to cache
# ----------------------------------------------------------------------------
OPTION(WARNINGS_ARE_ERRORS        "Treat warnings as errors" 									                  ON)
OPTION(WARNINGS_ANSI_ISO          "Issue all the mandatory diagnostics Listed in C standard" 	  OFF)

# ----------------------------------------------------------------------------
#   Find Dependencies
# ----------------------------------------------------------------------------
IF (WIN32)
  SET(ARGTABLE_ROOT "C:/3rdlibs")
  IF (NOT EXISTS ${ARGTABLE_ROOT})
    MESSAGE(FATAL_ERROR "Default value for ARGTABLE_ROOT in win32 doesn't exist [c:\3rdlibs]")
  ENDIF()
ENDIF()
SET(CUDA_SDK_ROOT_DIR $ENV{HOME}/NVIDIA_GPU_Computing_SDK/)

FIND_LIBRARY(ARGTABLE argtable2 PATHS ${ARGTABLE_ROOT}/lib)
FIND_PACKAGE(OpenCV)
FIND_PACKAGE(CUDA 4.2 REQUIRED)

FIND_PATH(CUDA_CUT_INCLUDE_DIR cutil.h
 PATHS ${CUDA_SDK_ROOT_DIR}
 PATH_SUFFIXES "C/common/inc"
 DOC "Location of cutil.h"
 NO_DEFAULT_PATH
 )
FIND_LIBRARY(CUDA_CUT_LIBRARY
 NAMES cutil_i386
 PATHS ${CUDA_SDK_ROOT_DIR}
 PATH_SUFFIXES "C/lib"
 DOC "Location of cutil library"
 NO_DEFAULT_PATH
 )
MARK_AS_ADVANCED(CUDA_CUT_INCLUDE_DIR)
MARK_AS_ADVANCED(CUDA_CUT_LIBRARY)

IF(ARGTABLE-NOTFOUND)
  MESSAGE(FATAL_ERROR "argtable2 library didn't found")
ENDIF()

IF (${OpenCV_VERSION} VERSION_LESS 2.3.0)
	MESSAGE(FATAL_ERROR "OpenCV version is not compatible : ${OpenCV_VERSION}")
ENDIF()

IF (${CUDA_CUT_INCLUDE_DIR} STREQUAL CUDA_CUT_INCLUDE_DIR-NOTFOUND)
  MESSAGE(FATAL_ERROR "cutil.h was not found")
ENDIF()
IF (${CUDA_CUT_LIBRARY} STREQUAL CUDA_CUT_LIBRARY-NOTFOUND)
  MESSAGE(FATAL_ERROR "cutil library was not found")
ENDIF()


# ----------------------------------------------------------------------------
#   Compiler options
# ----------------------------------------------------------------------------
INCLUDE(cmake_stuff/compileroptions.cmake REQUIRED)
IF(MSVC)
  include(cmake_stuff/crtlinkage.cmake REQUIRED)
ENDIF(MSVC)

# ----------------------------------------------------------------------------
#   PROJECT CONTENT
# ----------------------------------------------------------------------------
ADD_SUBDIRECTORY(apps)

# ----------------------------------------------------------------------------
# display status message for important variables
# ----------------------------------------------------------------------------
MESSAGE( STATUS )
MESSAGE( STATUS "------------------ General configuration for ${PROJECT_NAME} ${PROJECT_VERSION} -----------------")
MESSAGE( STATUS )
MESSAGE( STATUS "Compiler:                  ${CMAKE_COMPILER}")
IF (${CMAKE_BUILD_TYPE} STREQUAL "Release")
  MESSAGE( STATUS "C++ flags (Release):       ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
ELSEIF (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  MESSAGE( STATUS "C++ flags (Debug):         ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
ELSE()
  MESSAGE( FATAL_ERROR "Only Release and Debug build types are allowed")
ENDIF ()
MESSAGE( STATUS )
MESSAGE( STATUS "BUILD_SHARED_LIBS =          ${BUILD_SHARED_LIBS}" )
MESSAGE( STATUS "BUILD_GTESTS =               ${BUILD_GTESTS}" )
MESSAGE( STATUS "WARNINGS_ANSI_ISO =          ${WARNINGS_ANSI_ISO}" )
MESSAGE( STATUS "WARNINGS_ARE_ERRORS =        ${WARNINGS_ARE_ERRORS}" )
MESSAGE( STATUS "CMAKE_SYSTEM_PROCESSOR =     ${CMAKE_SYSTEM_PROCESSOR}" )
MESSAGE( STATUS "ENABLE_PROFILING =           ${ENABLE_PROFILING}" )
MESSAGE( STATUS "CMAKE_INSTALL_PREFIX =       ${CMAKE_INSTALL_PREFIX}" )
MESSAGE( STATUS "CPACK_GENERATOR =            ${CPACK_GENERATOR}" )
MESSAGE( STATUS )
MESSAGE( STATUS "------------------- Documentation  ----------------------" )
MESSAGE( STATUS )
MESSAGE( STATUS "INSTALL_DOC =                ${INSTALL_DOC}" )
MESSAGE( STATUS "USE_LATEX =                  ${USE_LATEX}" )
MESSAGE( STATUS "USE_DOT =                    ${USE_DOT}" )
MESSAGE( STATUS "USE_CHM =                    ${USE_CHM}" )
MESSAGE( STATUS "USE_MATHJAX =                ${USE_MATHJAX}" )
MESSAGE( STATUS )
MESSAGE( STATUS "Change a value with: cmake -D<Variable>=<Value>" )
MESSAGE( STATUS )

