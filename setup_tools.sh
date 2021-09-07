#!/usr/bin/env bash

## Docker
export DEBIAN_FRONTEND=noninteractive
ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime
apt-get update && apt-get install -yqq --no-install-recommends \
    libglvnd0 libgl1 libglx0 \
    libegl1 libxext6 libx11-6 \
    wget gpg sudo curl

# Install System Packages
apt-get install -yqq --no-install-recommends \
    git ca-certificates build-essential \
    libglu1-mesa-dev libegl1-mesa-dev freeglut3-dev libglew-dev \
    libglfw3-dev  libpng-dev libjpeg-dev libtiff-dev \
    libboost-iostreams-dev libboost-program-options-dev \
    libboost-system-dev libboost-serialization-dev \
    libopencv-dev libcgal-dev libeigen3-dev libpcl-dev

conda install -yq cmake
pip install -qq conan open3d imageio gdown

# Clean Cache
rm -rf ./tools/vvmesh_build ./tools/openMVS_build ./tools/bin ./tools/lib ./tools/include ./tools/vcglib
# Build vvmesh
./tools/build_vvmesh.sh
# Build OpenMVS
./tools/build_openmvs.sh
# Clean Build cache
rm -rf ./tools/vvmesh_build ./tools/openMVS_build