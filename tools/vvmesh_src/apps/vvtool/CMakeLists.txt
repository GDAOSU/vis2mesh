find_package(TIFF REQUIRED)
add_executable(vvtool main.cpp)
target_link_libraries(vvtool
        Plugin
        FILTER_plugins
        GLRENDER_plugin
        CLOUD_BUNDLE_plugin
        # DELAUNAY_GC_plugin
        NETWORKVISIBILITY_plugin
        ${PCL_LIBRARIES}
        ${TIFF_LIBRARIES}
        cxxopts::cxxopts
        )

#pybind11_add_module(pyvvtool pybind11/vvtool_pybind11.cpp)
#target_link_libraries(pyvvtool PRIVATE
#        Plugin
#        FILTER_plugins
#        GLRENDER_plugin
#        CLOUD_BUNDLE_plugin
#        DELAUNAY_GC_plugin
#        spdlog::spdlog
#        )

install(TARGETS vvtool
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include)

#install(TARGETS pyvvtool
#        RUNTIME DESTINATION bin
#        ARCHIVE DESTINATION lib
#        LIBRARY DESTINATION lib
#        PUBLIC_HEADER DESTINATION include)