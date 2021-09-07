import os
import re

from fnmatch import fnmatch
from conans import ConanFile, CMake, tools


class VTKConan(ConanFile):
    name = "vtk"
    version = "8.2.0"
    description = "Visualization Toolkit by Kitware"
    url = "http://github.com/bilke/conan-vtk"
    license = "MIT"
    generators = "cmake"
    settings = "os", "compiler", "build_type", "arch"
    revision_mode = "scm"
    exports = [
        "LICENSE.md",
        "CMakeLists.txt",
        "FindVTK.cmake",
        "vtknetcdf_snprintf.diff",
        "vtktiff_mangle.diff",
    ]
    source_subfolder = "source_subfolder"
    options = {
        "shared": [True, False],
        "qt": [True, False],
        "mpi": [True, False],
        "fPIC": [True, False],
        "minimal": [True, False],
        "ioxml": [True, False],
        "ioexport": [True, False],
        "mpi_minimal": [True, False],
        "ioxdmf3": [True, False],
        "iolegacy": [True, False],
    }
    default_options = (
        "shared=True",
        "qt=False",
        "mpi=False",
        "fPIC=False",
        "minimal=False",
        "ioxml=False",
        "ioexport=False",
        "mpi_minimal=False",
        "ioxdmf3=False",
        "iolegacy=False",
    )

    short_paths = True

    version_split = version.split(".")
    short_version = "%s.%s" % (version_split[0], version_split[1])

    def source(self):
        tools.get(
            "https://github.com/Kitware/{0}/archive/v{1}.tar.gz".format(
                self.name.upper(), self.version
            )
        )
        extracted_dir = self.name.upper() + "-" + self.version
        os.rename(extracted_dir, self.source_subfolder)
        tools.patch(
            base_path=self.source_subfolder, patch_file="vtknetcdf_snprintf.diff"
        )
        tools.patch(base_path=self.source_subfolder, patch_file="vtktiff_mangle.diff")

    def requirements(self):
        if self.options.ioxdmf3:
            self.requires("boost/1.75.0")
        if self.options.qt:
            self.requires("qt/5.15.2")
            self.options["qt"].shared = True
            if tools.os_info.is_linux:
                self.options["qt"].qtx11extras = True

    def _system_package_architecture(self):
        if tools.os_info.with_apt:
            if self.settings.arch == "x86":
                return ":i386"
            elif self.settings.arch == "x86_64":
                return ":amd64"

        if tools.os_info.with_yum:
            if self.settings.arch == "x86":
                return ".i686"
            elif self.settings.arch == "x86_64":
                return ".x86_64"
        return ""

    def build_requirements(self):
        pack_names = None
        if not self.options.minimal and tools.os_info.is_linux:
            if tools.os_info.with_apt:
                pack_names = [
                    "freeglut3-dev",
                    "mesa-common-dev",
                    # "mesa-utils-extra",
                    "libgl1-mesa-dev",
                    "libglapi-mesa",
                    "libsm-dev",
                    "libx11-dev",
                    "libxext-dev",
                    "libxt-dev",
                    "libglu1-mesa-dev",
                ]

        if pack_names:
            installer = tools.SystemPackageTool()
            for item in pack_names:
                installer.install(item + self._system_package_architecture())

    def config_options(self):
        if self.settings.compiler == "Visual Studio":
            del self.options.fPIC

    def build(self):
        cmake = CMake(self)
        cmake.definitions["BUILD_TESTING"] = "OFF"
        cmake.definitions["BUILD_EXAMPLES"] = "OFF"
        cmake.definitions["BUILD_SHARED_LIBS"] = self.options.shared

        if self.options.minimal:
            cmake.definitions["VTK_Group_StandAlone"] = "OFF"
            cmake.definitions["VTK_Group_Rendering"] = "OFF"
        if self.options.ioxml:
            cmake.definitions["Module_vtkIOXML"] = "ON"
        if self.options.ioexport:
            cmake.definitions["Module_vtkIOExport"] = "ON"
        if self.options.ioxdmf3:
            cmake.definitions["Module_vtkIOXdmf3"] = "ON"
        if self.options.iolegacy:
            cmake.definitions["Module_vtkIOLegacy"] = "ON"
            # if tools.os_info.is_macos:
            # cmake.definitions["VTK_USE_SYSTEM_LIBXML2"] = "ON"
        if self.options.qt:
            cmake.definitions["VTK_Group_Qt"] = "ON"
            cmake.definitions["VTK_QT_VERSION"] = "5"
            cmake.definitions["VTK_BUILD_QT_DESIGNER_PLUGIN"] = "OFF"
        if self.options.mpi:
            cmake.definitions["VTK_Group_MPI"] = "ON"
            cmake.definitions["Module_vtkIOParallelXML"] = "ON"
        if self.options.mpi_minimal:
            cmake.definitions["Module_vtkIOParallelXML"] = "ON"
            cmake.definitions["Module_vtkParallelMPI"] = "ON"

        if (
                self.settings.build_type == "Debug"
                and self.settings.compiler == "Visual Studio"
        ):
            cmake.definitions["CMAKE_DEBUG_POSTFIX"] = "_d"

        if self.settings.os == "Macos":
            self.env["DYLD_LIBRARY_PATH"] = os.path.join(self.build_folder, "lib")
            self.output.info("cmake build: %s" % self.build_folder)

        cmake.configure(source_folder = 'source_subfolder', build_folder="build")
        if self.settings.os == "Macos":
            # run_environment does not work here because it appends path just from
            # requirements, not from this package itself
            # https://docs.conan.io/en/latest/reference/build_helpers/run_environment.html#runenvironment
            lib_path = os.path.join(self.build_folder, "lib")
            self.run(
                "DYLD_LIBRARY_PATH={0} cmake --build build {1} -j".format(
                    lib_path, cmake.build_config
                )
            )
        else:
            cmake.build()
        cmake.install()

    # From https://git.ircad.fr/conan/conan-vtk/blob/stable/8.2.0-r1/conanfile.py
    def cmake_fix_path(self, file_path, package_name):
        try:
            tools.replace_in_file(
                file_path,
                self.deps_cpp_info[package_name].rootpath.replace("\\", "/"),
                "${CONAN_" + package_name.upper() + "_ROOT}",
                strict=False,
                )
        except:
            self.output.info("Ignoring {0}...".format(package_name))

    def cmake_fix_macos_sdk_path(self, file_path):
        # Read in the file
        with open(file_path, "r") as file:
            file_data = file.read()

        if file_data:
            # Replace the target string
            file_data = re.sub(
                # Match sdk path
                r";/Applications/Xcode\.app/Contents/Developer/Platforms/MacOSX\.platform/Developer/SDKs/MacOSX\d\d\.\d\d\.sdk/usr/include",
                "",
                file_data,
                re.M,
            )

            # Write the file out again
            with open(file_path, "w") as file:
                file.write(file_data)

    def package(self):
        for path, subdirs, names in os.walk(
                os.path.join(self.package_folder, "lib", "cmake")
        ):
            for name in names:
                if fnmatch(name, "*.cmake"):
                    cmake_file = os.path.join(path, name)

                    # if self.options.external_tiff:
                    # self.cmake_fix_path(cmake_file, "libtiff")
                    # if self.options.external_zlib:
                    # self.cmake_fix_path(cmake_file, "zlib")

                    if tools.os_info.is_macos:
                        self.cmake_fix_macos_sdk_path(cmake_file)

    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)

        self.cpp_info.includedirs = [
            "include/vtk-%s" % self.short_version,
            "include/vtk-%s/vtknetcdf/include" % self.short_version,
            "include/vtk-%s/vtknetcdfcpp" % self.short_version,
            ]

        if self.settings.os == "Linux":
            self.cpp_info.libs.append("pthread")