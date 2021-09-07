import os
from os.path import join, exists
from fnmatch import fnmatch
from conans import ConanFile, CMake, tools


class LibPCLConan(ConanFile):
    name = "pcl"
    # upstream_version = "1.11.1"
    # package_revision = "-r3"
    # version = "{0}{1}".format(upstream_version, package_revision)
    version = "1.11.1"

    generators = "cmake"
    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "with_cuda": [True, False],
    }
    default_options = [
        "shared=True",
        "fPIC=True",
        "with_cuda=True"
    ]
    default_options = tuple(default_options)
    exports = [
        "select_compute_arch.cmake",
    ]
    url = "https://git.ircad.fr/conan/conan-pcl"
    license = "BSD License"
    description = "The Point Cloud Library is for 2D/3D image and point cloud processing."
    short_paths = True

    _source_subfolder = "source_subfolder"
    _build_subfolder = "build_subfolder"
    _cmake = None

    def config_options(self):
        if tools.os_info.is_windows:
            del self.options.fPIC

    def configure(self):
        # PCL is not well prepared for c++ standard > 11...
        del self.settings.compiler.cppstd

        # if 'CI' not in os.environ:
        #     os.environ["CONAN_SYSREQUIRES_MODE"] = "verify"

        if self.settings.os == "Linux":
            self.options["Boost"].fPIC = True

        if tools.os_info.is_windows:
            self.options["Boost"].shared=True


    def requirements(self):
        # self.requires("qt/5.15.2")
        self.requires("eigen/3.3.9")
        self.requires("boost/1.75.0")
        self.requires("flann/1.9.1")
        self.requires("lz4/1.9.2")
        self.requires("zlib/1.2.11")
        self.requires("vtk/8.2.0@sxsong1207/stable")

    def build_requirements(self):
        if tools.os_info.linux_distro == "linuxmint":
            installer = tools.SystemPackageTool()
            installer.install("zlib1g-dev")
        if tools.os_info.linux_distro == "ubuntu":
            installer = tools.SystemPackageTool()
            installer.install("zlib1g-dev")

    def system_requirements(self):
        if tools.os_info.linux_distro == "linuxmint":
            installer = tools.SystemPackageTool()
            installer.install("zlib1g")
        if tools.os_info.linux_distro == "ubuntu":
            installer = tools.SystemPackageTool()
            installer.install("zlib1g")

    def source(self):
        tools.get(
            "https://github.com/PointCloudLibrary/pcl/archive/pcl-{0}.tar.gz".format(
                self.version))
        os.rename(
            "pcl-pcl-{0}".format(self.version),
            self._source_subfolder)

    def _configure_cmake(self):
        if not self._cmake:
            self._cmake = CMake(self)
            self._cmake.definitions["BUILD_TESTS"] = False
            self._cmake.definitions["WITH_VTK"] = True
            self._cmake.definitions["WITH_OPENGL"] = False
        self._cmake.definitions['VTK_ROOT'] = join(self.deps_cpp_info['vtk'].rootpath,'lib','cmake','vtk-8.2')
        self._cmake.definitions['BOOST_ROOT'] = join(self.deps_cpp_info['boost'].rootpath)
        self._cmake.definitions['EIGEN_ROOT'] = join(self.deps_cpp_info['eigen'].rootpath)
        self._cmake.definitions['FLANN_ROOT'] = join(self.deps_cpp_info['flann'].rootpath, 'include')
        self._cmake.definitions["BUILD_apps"] = "OFF"
        self._cmake.definitions["BUILD_examples"] = "OFF"
        self._cmake.definitions["BUILD_common"] = "ON"
        self._cmake.definitions["BUILD_2d"] = "ON"
        self._cmake.definitions["BUILD_features"] = "ON"
        self._cmake.definitions["BUILD_filters"] = "ON"
        self._cmake.definitions["BUILD_geometry"] = "ON"
        self._cmake.definitions["BUILD_io"] = "ON"
        self._cmake.definitions["BUILD_kdtree"] = "ON"
        self._cmake.definitions["BUILD_octree"] = "ON"
        self._cmake.definitions["BUILD_sample_consensus"] = "ON"
        self._cmake.definitions["BUILD_search"] = "ON"
        self._cmake.definitions["BUILD_tools"] = "OFF"
        self._cmake.definitions["PCL_BUILD_WITH_BOOST_DYNAMIC_LINKING_WIN32"] = "ON"
        self._cmake.definitions["PCL_SHARED_LIBS"] = "ON"
        self._cmake.definitions["WITH_PCAP"] = "OFF"
        self._cmake.definitions["WITH_DAVIDSDK"] = "OFF"
        self._cmake.definitions["WITH_ENSENSO"] = "OFF"
        self._cmake.definitions["WITH_OPENNI"] = "OFF"
        self._cmake.definitions["WITH_OPENNI2"] = "OFF"
        self._cmake.definitions["WITH_RSSDK"] = "OFF"
        self._cmake.definitions["WITH_QHULL"] = "OFF"
        self._cmake.definitions["BUILD_TESTS"] = "OFF"
        self._cmake.definitions["BUILD_ml"] = "OFF"
        self._cmake.definitions["BUILD_simulation"] = "OFF"
        self._cmake.definitions["BUILD_segmentation"] = "ON"
        self._cmake.definitions["BUILD_registration"] = "ON"
        
        if tools.os_info.is_windows:
            self._cmake.definitions["WITH_PNG"] = "OFF"
        self._cmake.definitions["BUILD_surface"] = "ON"
        self._cmake.definitions["BUILD_visualization"] = "ON"

        if tools.os_info.is_macos:
            self._cmake.definitions["BUILD_gpu_features"] = "OFF"

        if tools.os_info.is_windows:
            self._cmake.definitions["CUDA_PROPAGATE_HOST_FLAGS"] = "ON"
        else:
            self._cmake.definitions["CUDA_PROPAGATE_HOST_FLAGS"] = "OFF"
        self._cmake.configure(source_folder=self._source_subfolder, build_folder=self._build_subfolder)
        return self._cmake

    def build(self):
        pcl_source_dir = os.path.join(
            self.source_folder, self._source_subfolder)

        tools.replace_in_file(os.path.join(self._source_subfolder, "common", "include", "pcl", "types.h"),
            """
  using uint8_t PCL_DEPRECATED(1, 12, "use std::uint8_t instead of pcl::uint8_t") = std::uint8_t;
  using int8_t PCL_DEPRECATED(1, 12, "use std::int8_t instead of pcl::int8_t") = std::int8_t;
  using uint16_t PCL_DEPRECATED(1, 12, "use std::uint16_t instead of pcl::uint16_t") = std::uint16_t;
  using int16_t PCL_DEPRECATED(1, 12, "use std::uint16_t instead of pcl::int16_t") = std::int16_t;
  using uint32_t PCL_DEPRECATED(1, 12, "use std::uint32_t instead of pcl::uint32_t") = std::uint32_t;
  using int32_t PCL_DEPRECATED(1, 12, "use std::int32_t instead of pcl::int32_t") = std::int32_t;
  using uint64_t PCL_DEPRECATED(1, 12, "use std::uint64_t instead of pcl::uint64_t") = std::uint64_t;
  using int64_t PCL_DEPRECATED(1, 12, "use std::int64_t instead of pcl::int64_t") = std::int64_t;
  using int_fast16_t PCL_DEPRECATED(1, 12, "use std::int_fast16_t instead of pcl::int_fast16_t") = std::int_fast16_t;
""",
            """
  using uint8_t = std::uint8_t;
  using int8_t = std::int8_t;
  using uint16_t = std::uint16_t;
  using int16_t = std::int16_t;
  using uint32_t = std::uint32_t;
  using int32_t = std::int32_t;
  using uint64_t = std::uint64_t;
  using int64_t = std::int64_t;
  using int_fast16_t = std::int_fast16_t;
""")


        cmake = self._configure_cmake()
        cmake.build()

    def package(self):
        # Import common flags and defines
        self.copy(pattern="LICENSE", dst="licenses", src=self._source_subfolder)
        cmake = self._configure_cmake()
        cmake.install()
        # If the CMakeLists.txt has a proper install method, the steps below may be redundant
        # If so, you can just remove the lines below
        include_folder = os.path.join(self._source_subfolder, "include")
        self.copy(pattern="*", dst="include", src=include_folder)
        self.copy(pattern="*.dll", dst="bin", keep_path=False)
        self.copy(pattern="*.lib", dst="lib", keep_path=False)
        self.copy(pattern="*.a", dst="lib", keep_path=False)
        self.copy(pattern="*.so*", dst="lib", keep_path=False)
        self.copy(pattern="*.dylib", dst="lib", keep_path=False)

    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)
        v_major, v_minor, v_micro = self.version.split(".")
        self.cpp_info.includedirs = ['include', os.path.join('include', 'pcl-%s.%s' % (v_major, v_minor) )]
