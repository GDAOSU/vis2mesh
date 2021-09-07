set(SRCS obj.cpp)
set(HEADERS obj.h)
add_library(OBJFile STATIC ${SRCS})
target_link_libraries(OBJFile PRIVATE ${opencv_LIBRARIES})
set_target_properties(OBJFile PROPERTIES
        PUBLIC_HEADER "${HEADERS}")

install(TARGETS OBJFile
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include/OBJFile)

# test
# add_executable(test_OBJFile test/test_OBJFile.cpp)
# target_link_libraries(test_OBJFile OBJFile doctest::doctest)
# add_test(test_OBJFile test_OBJFile)