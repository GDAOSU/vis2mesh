#### Plugin libs ####
set(SRCS GLRENDER_plugin.cpp)
set(HEADERS GLRENDER_plugin.h)
if(BUILD_SHARED_PLUGINS)
    add_library(GLRENDER_plugin SHARED ${SRCS})
    target_compile_definitions(GLRENDER_plugin PRIVATE PLUGIN_DLL_EXPORTS)
else()
    add_library(GLRENDER_plugin STATIC ${SRCS})
endif()

target_include_directories(GLRENDER_plugin INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(GLRENDER_plugin
        PUBLIC FBORender PluginInterface
        PRIVATE MVArchive OBJFile pcl_io pcl_kdtree)
set_target_properties(GLRENDER_plugin PROPERTIES
        PUBLIC_HEADER "${HEADERS}")

install(TARGETS GLRENDER_plugin
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include)
