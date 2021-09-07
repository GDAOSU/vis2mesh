cmake_minimum_required(VERSION 3.5)
# Set policies
cmake_policy(SET CMP0072 NEW)
# Start Project
project(VVMESH
        LANGUAGES CXX
        VERSION 0.8.0)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

#### Compiler traits
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()
set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ version selection")  # or 11, 14, 17, 20
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # optional, ensure standard is supported
set(CMAKE_CXX_EXTENSIONS OFF)  # optional, keep compiler extensionsn off
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if (WIN32)
    add_definitions(-D_USE_MATH_DEFINES -DNOMINMAX)
endif ()

# RPath
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
### Options

option(BUILD_SHARED_PLUGINS "BUILD Plugins as shared libraries" ON)

###############################
## Solve dependencies
include(BuildExternalDeps)

find_package(Threads REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
    message("++ Enable OpenMP support")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_definitions(-D_USE_OPENMP)
    set(_USE_OPENMP TRUE)
else ()
    message("-- Can't find OpenMP. Continuing without it.")
endif ()

if (WIN32)
    find_package(unofficial-angle CONFIG REQUIRED)
    find_package(OpenGL REQUIRED COMPONENTS OpenGL)
else ()
    find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL)
endif ()

if (OPENGL_FOUND)
    message("++ Enable OpenGL support")
    include_directories(${OPENGL_INCLUDE_DIR})
    add_definitions(${OpenGL_DEFINITIONS})
    add_definitions(-D_USE_OPENGL)
    set(_USE_OPENGL TRUE)
else ()
    message("-- Can't find OpenGL. Continuing without it.")
endif ()

#find_package(CUDA)
#if(CUDA_FOUND)
#	include_directories(${CUDA_INCLUDE_DIRS})
#	add_definitions(-D_USE_CUDA)
#	set(_USE_CUDA TRUE)
#else()
#	set(CUDA_CUDA_LIBRARY "")
#	message("-- Can't find CUDA. Continuing without it.")
#endif()

if(QHULL_FOUND)
    message("++ Enable QHULL support")
    include_directories(${QHULL_INCLUDE_DIRS})
    add_definitions(${QHULL_DEFINITIONS})
endif()

if (Boost_FOUND)
    message("++ Enable Boost support")
    include_directories(${Boost_INCLUDE_DIRS})
    add_definitions(${Boost_DEFINITIONS} -D_USE_BOOST)
    link_directories(${Boost_LIBRARY_DIRS})
    SET(_USE_BOOST TRUE)
endif ()

if (EIGEN_FOUND)
    message("++ Enable Eigen3 support")
    include_directories(${EIGEN3_INCLUDE_DIRS})
    add_definitions(${EIGEN3_DEFINITIONS})
    add_definitions(-D_USE_EIGEN)
endif ()

if (PCL_FOUND)
    message("++ Enable PCL support")
    include_directories(${PCL_INCLUDE_DIRS})
    add_definitions(${PCL_DEFINITIONS})
endif()

if (OpenCV_FOUND)
    message("++ Enable OpenCV support")
    include_directories(${opencv_INCLUDE_DIRS})
    add_definitions(-D_USE_OPENCV)
endif ()

if (spdlog_FOUND)
    include_directories(${spdlog_INCLUDE_DIRS})
    add_definitions(${spdlog_DEFINITIONS})
endif ()

if(fmt_FOUND)
    include_directories(${fmt_INCLUDE_DIRS})
endif()

# Please uncomment following line and supply compressed_streams package
# add_definitions(-D_USE_COMPRESSED_STREAMS)
if(jsoncpp_FOUND)
    message("++ Enable Jsoncpp")
    include_directories(${jsoncpp_INCLUDE_DIRS})
endif()
if (zstd_FOUND)
    message("++ Enable ZSTDStream Archive Format")
    add_definitions(-D_USE_ZSTDSTREAM)
    include_directories(${zstd_INCLUDE_DIRS})
endif ()
if (ZLIB_FOUND)
    message("++ Enable GZStream Archive Format")
    add_definitions(-D_USE_GZSTREAM)
endif ()

################### End Dependancy
enable_testing()

include_directories(src)

add_subdirectory(src)
add_subdirectory(apps)


