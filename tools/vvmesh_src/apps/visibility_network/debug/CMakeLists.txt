##### Test ####
#
#add_executable(run_networkplugin run_networkplugin.cpp)
#target_link_libraries(run_networkplugin NETWORKVISIBILITY_plugin
#        DELAUNAY_GC_plugin
#        Plugin)
#
###$$$$$$$$$$$$$ example_python_call2 ############
#add_executable(example_python_call
#        example_call.cpp
#        )
#target_link_libraries(example_python_call pybind11::embed
#        nlohmann_json::nlohmann_json spdlog::spdlog Plugin)
#
############3 example_python_call2 $#$$$$$$$$$$$
#add_executable(example_python_call2 example_call2.cpp)
#target_link_libraries(example_python_call2 pybind11::embed nlohmann_json::nlohmann_json
#        )