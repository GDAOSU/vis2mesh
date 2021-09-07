#!/usr/bin/env bash
# Rebuild VVMesh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
conan export $SCRIPT_DIR/vvmesh_src/cmake/conan/qhullcpp sxsong1207/stable
conan export $SCRIPT_DIR/vvmesh_src/cmake/conan/openvolumemesh sxsong1207/stable
mkdir "$SCRIPT_DIR/vvmesh_build" && cd "$SCRIPT_DIR/vvmesh_build"

PYTHON_EXE=`which python`
cd "$SCRIPT_DIR/vvmesh_build"
cmake . "$SCRIPT_DIR/vvmesh_src" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$SCRIPT_DIR -DPYTHON_EXECUTABLE=$PYTHON_EXE
make -j7 && make install
