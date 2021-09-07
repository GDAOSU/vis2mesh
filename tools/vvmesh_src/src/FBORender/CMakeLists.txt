set(SRCS multidrawfbo.cpp
        renderobjects.cpp
        util.cpp)
set(HEADERS reader.h
        util.h
        renderobjects.h
        multidrawfbo.h
        dllmacro.h)

add_library(FBORender STATIC ${SRCS})
target_link_libraries(FBORender
        PUBLIC glm::glm
        PRIVATE OpenGL::GL)
if(UNIX)
    target_link_libraries(FBORender PRIVATE OpenGL::EGL m GLEW::GLEW glfw)
else()
    target_link_libraries(FBORender PRIVATE unofficial::angle::libEGL GLEW::GLEW glfw)
endif()
set_target_properties(FBORender PROPERTIES
        PUBLIC_HEADER "${HEADERS}")
target_compile_definitions(FBORender PRIVATE FBORENDER_DLL_EXPORTS)

function(make_includable input_file output_file)
    if(input_file IS_NEWER_THAN output_file)
        file(READ ${input_file} content)
        set(delim "for_c++_include")
        set(content "R\"${delim}(\n${content})${delim}\"")
        file(WRITE ${output_file} "${content}")
    endif()
endfunction(make_includable)

# Use like
make_includable(glsl/pointcloud_box.gs generated/pointcloud_box.gs)
make_includable(glsl/pointcloud_diamond.gs generated/pointcloud_diamond.gs)
make_includable(glsl/pointcloud_dot.gs generated/pointcloud_dot.gs)
make_includable(glsl/pointcloud.fs generated/pointcloud.fs)
make_includable(glsl/pointcloud_point.gs generated/pointcloud_point.gs)
make_includable(glsl/pointcloud.vs generated/pointcloud.vs)
make_includable(glsl/texturedtrimesh.fs generated/texturedtrimesh.fs)
make_includable(glsl/texturedtrimesh.vs generated/texturedtrimesh.vs)
make_includable(glsl/trimesh.vs generated/trimesh.vs)
make_includable(glsl/trimesh.fs generated/trimesh.fs)
make_includable(glsl/rgbtrimesh.vs generated/rgbtrimesh.vs)
make_includable(glsl/rgbtrimesh.fs generated/rgbtrimesh.fs)
make_includable(glsl/wwzb.cs generated/wwzb.cs)

install(TARGETS FBORender
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include/FBORender)