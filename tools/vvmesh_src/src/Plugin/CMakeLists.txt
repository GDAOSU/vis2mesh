#### Libs ####

add_library(PluginInterface STATIC plugininterface.cpp)
target_link_libraries(PluginInterface
        PUBLIC nlohmann_json::nlohmann_json spdlog::spdlog)
set_target_properties(PluginInterface PROPERTIES
        PUBLIC_HEADER "plugininterface.h")
target_compile_definitions(PluginInterface PRIVATE PLUGIN_DLL_EXPORTS)

set(SRCS
        builtin_plugins.cpp
        pluginmanager.cpp
        preprocess_json.cpp
        )
set(HEADERS dllmacro.h
        plugininterface.h
        builtin_plugins.h
        pluginmanager.h
        preprocess_json.h)

# Note: This must be STATIC, otherwise, python embed will have issue.
add_library(Plugin STATIC ${SRCS})
target_link_libraries(Plugin
        PUBLIC PluginInterface nlohmann_json::nlohmann_json
        PRIVATE spdlog::spdlog)
set_target_properties(Plugin PROPERTIES
        PUBLIC_HEADER "${HEADERS}")
target_compile_definitions(Plugin PRIVATE PLUGIN_DLL_EXPORTS)

install(TARGETS Plugin
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include/pluginmanager)

#### Test ####
add_executable(test_plugin test/test_plugin.cpp)
target_link_libraries(test_plugin Plugin doctest::doctest)

add_test(NAME TestPlugin COMMAND test_plugin)
