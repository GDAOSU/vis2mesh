from conans import ConanFile, CMake, tools
import os


class QhullcppConan(ConanFile):
    name = "qhullcpp"
    version = "8.0.2"
    license = "Qhull"
    url = "https://github.com/qhull/qhull.git"
    description = '''Qhull is a general dimension convex hull program that reads a set 
  of points from stdin, and outputs the smallest convex set that contains 
  the points to stdout.  It also generates Delaunay triangulations, Voronoi 
  diagrams, furthest-site Voronoi diagrams, and halfspace intersections
  about a point.

  Rbox is a useful tool in generating input for Qhull; it generates 
  hypercubes, diamonds, cones, circles, simplices, spirals, 
  lattices, and random points.

  Qhull produces graphical output for Geomview.  This helps with
  understanding the output. <http://www.geomview.org>'''
    topics = ("conan", "qhull", "qhullcpp", "geometry", "convex", "triangulation", "intersection")
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False],
               "fPIC": [True, False],
               "reentrant": [True, False]}
    default_options = {
        "shared": False,
        "fPIC": True,
        "reentrant": True
    }
    generators = "cmake"
    _cmake = None

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def source(self):
        self.run("git clone https://github.com/qhull/qhull.git")
        # This small hack might be useful to guarantee proper /MT /MD linkage
        # in MSVC if the packaged project doesn't have variables to set it
        # properly
        tools.replace_in_file("qhull/CMakeLists.txt", "project(qhull)",
                              '''project(qhull)
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()''')

    def configure(self):
        if self.options.shared:
            del self.options.fPIC

    def package_id(self):
        del self.info.options.reentrant

    def build(self):
        cmake = self._configure_cmake()
        cmake.build()

        # Explicit way:
        # self.run('cmake %s/hello %s'
        #          % (self.source_folder, cmake.command_line))
        # self.run("cmake --build . %s" % cmake.build_config)

    def _configure_cmake(self):
        if self._cmake:
            return self._cmake
        self._cmake = CMake(self)
        self._cmake.configure(source_folder="qhull")
        return self._cmake

    def package(self):
        # self.copy("COPYING.txt", dst="licenses", src=self._source_subfolder)
        cmake = self._configure_cmake()
        cmake.install()
        tools.rmdir(os.path.join(self.package_folder, "doc"))
        tools.rmdir(os.path.join(self.package_folder, "man"))
        tools.rmdir(os.path.join(self.package_folder, "lib", "cmake"))
        tools.rmdir(os.path.join(self.package_folder, "lib", "pkgconfig"))
        tools.rmdir(os.path.join(self.package_folder, "share"))

    def package_info(self):
        self.cpp_info.libs = [self._qhullcpp_lib_name, self._qhull_lib_name]
        self.cpp_info.names["cmake_find_package"] = "qhullcpp"
        self.cpp_info.names["cmake_find_package_multi"] = "qhullcpp"

        # self.cpp_info.components["libqhull"].names["cmake_find_package"] = self._qhull_cmake_name
        # self.cpp_info.components["libqhull"].names["cmake_find_package_multi"] = self._qhull_cmake_name
        # self.cpp_info.components["libqhull"].names["pkg_config"] = self._qhull_pkgconfig_name
        # self.cpp_info.components["libqhull"].libs = [self._qhull_lib_name]
        # if self.settings.os == "Linux":
        #     self.cpp_info.components["libqhull"].system_libs.append("m")
        # if self.settings.compiler == "Visual Studio" and self.options.shared:
        #     self.cpp_info.components["libqhull"].defines.extend(["qh_dllimport"])
        #
        # self.cpp_info.components['libqhullcpp'].names['cmake_find_package'] = 'qhullcpp'
        # self.cpp_info.components["libqhullcpp"].names["cmake_find_package_multi"] = 'qhullcpp'
        # self.cpp_info.components["libqhullcpp"].names["pkg_config"] = "qhullcpp"
        # self.cpp_info.components["libqhullcpp"].libs = "qhullcpp"
        # bin_path = os.path.join(self.package_folder, "bin")
        # self.output.info("Appending PATH environment variable: {}".format(bin_path))
        # self.env_info.PATH.append(bin_path)

    @property
    def _qhull_cmake_name(self):
        name = ""
        if self.options.reentrant:
            name = "qhull_r" if self.options.shared else "qhullstatic_r"
        else:
            name = "libqhull" if self.options.shared else "qhullstatic"
        return name

    @property
    def _qhull_pkgconfig_name(self):
        name = "qhull"
        if not self.options.shared:
            name += "static"
        if self.options.reentrant:
            name += "_r"
        return name

    @property
    def _qhull_lib_name(self):
        name = "qhull"
        if not self.options.shared:
            name += "static"
        if self.settings.build_type == "Debug" or self.options.reentrant:
            name += "_"
            if self.options.reentrant:
                name += "r"
            if self.settings.build_type == "Debug":
                name += "d"
        return name

    @property
    def _qhullcpp_lib_name(self):
        name = "qhullcpp"
        if self.settings.build_type == "Debug":
            name += "_"
            if self.settings.build_type == "Debug":
                name += "d"
        return name
