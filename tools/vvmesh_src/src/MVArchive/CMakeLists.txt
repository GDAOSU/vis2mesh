set(SRCS ARInterface.cpp
        zstdstream/zstdstream.cpp)
set(HEADERS dllmacro.h
        ARInterface.h
        ARInterface_impl.hpp)

add_library(MVArchive STATIC ${SRCS})
target_link_libraries(MVArchive PRIVATE ${ZLIB_LIBRARIES} ${zstd_LIBRARIES} ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
set_target_properties(MVArchive PROPERTIES
        PUBLIC_HEADER "${HEADERS}")
target_compile_definitions(MVArchive PRIVATE ARCHIVE_DLL_EXPORTS)

# add_subdirectory(pybind11)

install(TARGETS MVArchive
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include/MVArchive)

