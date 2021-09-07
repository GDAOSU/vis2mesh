#### Plugin libs ####
set(SRCS 
        GTFILTER_plugin.cpp
        RAYGTFILTER_plugin.cpp)

set(HEADERS 
        GTFILTER_plugin.h
        RAYGTFILTER_plugin.h)

if(BUILD_SHARED_PLUGINS)
    add_library(FILTER_plugins SHARED ${SRCS})
    target_compile_definitions(FILTER_plugins PRIVATE PLUGIN_DLL_EXPORTS)
else()
    add_library(FILTER_plugins STATIC ${SRCS})
endif()

target_include_directories(FILTER_plugins INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(FILTER_plugins
        PUBLIC FBORender PluginInterface
        PRIVATE qhullcpp::qhullcpp CGAL::CGAL
        ${OpenCV_LIBS} pcl_io)
set_target_properties(FILTER_plugins PROPERTIES
        PUBLIC_HEADER "${HEADERS}")

install(TARGETS FILTER_plugins
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include/plugins)
