#### Plugin libs ####
if(BUILD_SHARED_PLUGINS)
    add_library(CLOUD_BUNDLE_plugin SHARED CLOUD_BUNDLE_plugin.cpp)
    target_compile_definitions(CLOUD_BUNDLE_plugin PRIVATE PLUGIN_DLL_EXPORTS)
else()
    add_library(CLOUD_BUNDLE_plugin STATIC CLOUD_BUNDLE_plugin.cpp)
endif()
message("Opencv libs" ${opencv_LIBRARIES_TARGETS})
message("tinyply " ${tinyply_LIBRARIES_TARGETS})
target_include_directories(CLOUD_BUNDLE_plugin INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(CLOUD_BUNDLE_plugin PRIVATE ${tinyply_INCLUDE_DIRS})
target_link_libraries(CLOUD_BUNDLE_plugin
        PUBLIC PluginInterface
        PRIVATE MVArchive tinyply::tinyply
        ${OpenCV_LIBS} pcl_io)

install(TARGETS CLOUD_BUNDLE_plugin
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include/plugins)