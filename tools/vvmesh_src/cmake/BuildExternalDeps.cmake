## System Dependencies
if (UNIX)
    find_package(CGAL REQUIRED)
    find_package(OpenCV REQUIRED)
    find_package(GLEW REQUIRED)
    find_package(glfw3 REQUIRED)
else ()
endif ()
find_package(PCL REQUIRED COMPONENTS common io octree search filters segmentation)
################
## Conan.io
################
# Download automatically, you can also just copy the conan.cmake file
if (NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
    message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
    file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/master/conan.cmake"
            "${CMAKE_BINARY_DIR}/conan.cmake")
endif ()

include(${CMAKE_BINARY_DIR}/conan.cmake)

conan_cmake_configure(
        REQUIRES zstd/1.5.0
        zlib/1.2.11
        doctest/2.4.6
        pybind11/2.6.2
        jsoncpp/1.9.4
        nlohmann_json/3.10.2
        eigen/3.3.9
        #        boost/1.71.0
        cxxopts/2.2.1
        # glew/2.1.0
        # glfw/3.3.2
        glm/0.9.9.8
        # opencv/3.4.12
        tinyply/2.3.2
        qhullcpp/8.0.2@sxsong1207/stable
        #        cgal/5.0.3
        openmesh/8.1
        openvolumemesh/2.1@sxsong1207/stable
        fmt/8.0.1
        spdlog/1.9.2
        GENERATORS cmake_find_package)

conan_cmake_autodetect(settings)

conan_cmake_install(PATH_OR_REFERENCE .
        BUILD missing
        REMOTE conancenter
        SETTINGS ${settings})

find_package(glm 0.9.9.8 EXACT REQUIRED)
find_package(tinyply 2.3.2 EXACT REQUIRED)
find_package(qhullcpp 8.0.2 EXACT REQUIRED)
find_package(doctest 2.4.6 EXACT)
find_package(jsoncpp 1.9.4 EXACT REQUIRED)
find_package(nlohmann_json 3.10.2 EXACT REQUIRED)
find_package(OpenMesh 8.1 EXACT REQUIRED)
find_package(OpenVolumeMesh 2.1 EXACT REQUIRED)
find_package(zstd 1.5.0 EXACT REQUIRED)
find_package(ZLIB 1.2.11 EXACT REQUIRED)
find_package(Eigen3 3.3.9 EXACT REQUIRED)
# find_package(OpenCV 3.4.12 EXACT REQUIRED)
# find_package(GLEW 2.1.0 EXACT REQUIRED)
# find_package(glfw3 3.3.2 EXACT REQUIRED)
find_package(cxxopts 2.2.1 EXACT REQUIRED)
find_package(spdlog 1.9.2 EXACT REQUIRED)
find_package(fmt 8.0.1 EXACT REQUIRED)
find_package(pybind11 2.6.2 EXACT REQUIRED)


