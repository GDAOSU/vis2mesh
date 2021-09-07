from conans import ConanFile, CMake, tools
import os
import os.path

class openvolumemeshConan(ConanFile):
    name = "openvolumemesh"
    version = "2.1"
    license = "GPLv3"
    url = "https://www.graphics.rwth-aachen.de:9000/OpenVolumeMesh/OpenVolumeMesh.git"
    description = "OpenVolumeMesh is a generic data structure for the comfortable handling of arbitrary polytopal meshes. Its concepts are closely related to OpenMesh. In particular, OpenVolumeMesh carries the general idea of storing edges as so-called (directed) half-edges over to the face definitions. So, faces are split up into so-called half-faces having opposing orientations. But unlike in the original concept of half-edges, local adjacency information is not stored on a per half-edge basis. Instead, all entities are arranged in arrays, which makes OpenVolumeMesh an index-based data structure where the access to entities via handles is accomplished in constant time complexity. By making the data structure index-based, we alleviate the major drawback of the half-edge data structure of only being capable to represent manifold meshes. In our concept, each entity of dimension n only stores an (ordered) tuple of handles (or indices) pointing to the incident entities of dimension (n-1). These incidence relations are called the top-down incidences. They are intrinsic to the implemented concept of volumentric meshes. One can additionally compute bottom-up incidences, which means that for each entity of dimension n, we also store handles to incident entities of dimension (n+1). These incidence relations have to be computed explicitly which can be performed in linear time complexity. Both incidence relations, the top-down and the bottom-up incidences, are used to provide a set of iterators and circulators that are comfortable in use. As in OpenMesh, OpenVolumeMesh provides an entirely generic underlying property system that allows attaching properties of any kind to the entities."
    homepage = "https://www.graphics.rwth-aachen.de/software/openvolumemesh/"
    topics = ("conan", "tetmesh", "openvolumemesh", "mesh", "structure", "geometry")
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}
    generators = "cmake"

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def source(self):
        self.run("git clone https://www.graphics.rwth-aachen.de:9000/OpenVolumeMesh/OpenVolumeMesh.git openvolumemesh")
        # This small hack might be useful to guarantee proper /MT /MD linkage
        # in MSVC if the packaged project doesn't have variables to set it
        # properly
        tools.replace_in_file("openvolumemesh/CMakeLists.txt", "if( ${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR} )",
                              '''include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()
if( ${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR} )''')

    _cmake = None
    def _configure_cmake(self):
        if self._cmake:
            return self._cmake
        self._cmake = CMake(self)
        # if self.settings.os == "Windows":
        self._cmake.definitions["BUILD_SHARED_LIBS"] = self.options.shared

        self._cmake.definitions["OVM_STANDALONE_BUILD"] = True
        self._cmake.definitions["OVM_ENABLE_APPLICATIONS"] = False
        self._cmake.definitions["OVM_ENABLE_UNITTESTS"] = False
        self._cmake.definitions["OVM_ENABLE_EXAMPLES"] = False
        self._cmake.definitions["OVM_BUILD_DOCUMENTATION"] = False
        # self._cmake.configure(source_folder="OpenVolumeMesh", build_folder=self._build_subfolder)
        return self._cmake

    def build(self):
        cmake = self._configure_cmake()
        cmake.configure(source_folder="openvolumemesh")
        cmake.build()

        # Explicit way:
        # self.run('cmake %s/hello %s'
        #          % (self.source_folder, cmake.command_line))
        # self.run("cmake --build . %s" % cmake.build_config)
    
    def package(self):
        cmake = self._configure_cmake()
        cmake.install()
        # self.copy("*.h", dst="include", src="OpenVolumeMesh")
        # self.copy("*hello.lib", dst="lib", keep_path=False)
        # self.copy("*.dll", dst="bin", keep_path=False)
        # self.copy("*.so", dst="lib", keep_path=False)
        # self.copy("*.dylib", dst="lib", keep_path=False)
        # self.copy("*.a", dst="lib", keep_path=False)

    @property
    def _module_subfolder(self):
        return os.path.join("lib", "cmake")

    @property
    def _module_file_rel_path(self):
        return os.path.join(self._module_subfolder,
                            "conan-official-{}-targets.cmake".format(self.name))
    def package_info(self):
        self.cpp_info.libs = ["openvolumemesh"]
        self.cpp_info.names["cmake_find_package"] = "OpenVolumeMesh"
        self.cpp_info.names["cmake_find_package_multi"] = "OpenVolumeMesh"
        self.cpp_info.builddirs.append(self._module_subfolder)
        suffix = "d" if self.settings.build_type == "Debug" else ""
        self.cpp_info.libs = ["OpenVolumeMesh" + suffix]
        if self.settings.compiler == "Visual Studio":
            self.cpp_info.defines.append("_USE_MATH_DEFINES")

