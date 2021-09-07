if (MSVC)
    FILE(GLOB LIBRARY_FILES_C "*.cpp" "*.rc")
else ()
    FILE(GLOB LIBRARY_FILES_C "*.cpp")
endif ()
FILE(GLOB LIBRARY_FILES_H "*.h" "*.inl")


add_executable(simplecamcreator ${LIBRARY_FILES_C})
target_link_libraries(simplecamcreator
        PRIVATE MVArchive OBJFile ${PCL_LIBRARIES} nlohmann_json::nlohmann_json opencv::core)

install(TARGETS simplecamcreator
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include)