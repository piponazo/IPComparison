INCLUDE_DIRECTORIES(${ARGTABLE_ROOT}/include)

LINK_LIBRARIES(${ARGTABLE} ${OpenCV_LIBS})

ADD_EXECUTABLE(invert        invert.cpp)

#IF (WIN32)
#  SET_TARGET_PROPERTIES(catec_timestamp PROPERTIES
#      LINK_FLAGS "
#      /NODEFAULTLIB:\"libc.lib\"
#      /NODEFAULTLIB:\"libcmt.lib\"
#      /NODEFAULTLIB:\"msvcrt.lib\"
#      /NODEFAULTLIB:\"libcd.lib\"
#      /NODEFAULTLIB:\"libcmtd.lib\"
#      ")
#ENDIF()
