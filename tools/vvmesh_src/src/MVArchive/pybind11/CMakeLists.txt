remove_definitions(-D_USE_OPENCV)

set(SRCS    ARInterface_pybind11.cpp
            ARInterface_pybind11_scene.cpp
            ARInteface_pybind11_types.cpp
            ../zstdstream/zstdstream.cpp)
set(HEADERS ARInterface_pybind11_scene.h)
pybind11_add_module(mvaformat ${SRCS})
target_include_directories(mvaformat PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mvaformat PRIVATE ${ZLIB_LIBRARIES} ${ZSTD_LIBRARIES} ${COMPRESSED_STREAMS_LIBRARIES} OpenMP::OpenMP_CXX)
target_compile_definitions(mvaformat PRIVATE ARCHIVE_DLL_EXPORTS)

install(TARGETS mvaformat
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include)