#### Plugin libs ####
set(SRCS json2pydict.cpp
        NETWORKVISIBILITY_plugin.cpp
#        O3DHPRVISIBILITY_plugin.cpp
        )
set(HEADERS NETWORKVISIBILITY_plugin.h
#        O3DHPRVISIBILITY_plugin.h
        )

if (BUILD_SHARED_PLUGINS)
    add_library(NETWORKVISIBILITY_plugin SHARED ${SRCS})
    target_compile_definitions(NETWORKVISIBILITY_plugin PRIVATE PLUGIN_DLL_EXPORTS)
elseif ()
    add_library(NETWORKVISIBILITY_plugin STATIC ${SRCS})
endif ()

target_include_directories(NETWORKVISIBILITY_plugin INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(NETWORKVISIBILITY_plugin
        PUBLIC PluginInterface
        PRIVATE pybind11::embed)
set_target_properties(NETWORKVISIBILITY_plugin PROPERTIES
        PUBLIC_HEADER "${HEADER}")

install(TARGETS NETWORKVISIBILITY_plugin
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include/plugins)

install(FILES network_predict.py hpr_predict.py
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
        DESTINATION lib)