#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo "Installing.. $SCRIPT_DIR"
git clone https://github.com/cdcseacave/VCG.git "$SCRIPT_DIR/vcglib" -q
git clone https://github.com/GDAOSU/openMVS_mvasupport.git "$SCRIPT_DIR/openMVS_src" -q
mkdir "$SCRIPT_DIR/openMVS_build"
cd "$SCRIPT_DIR/openMVS_build"
cmake . "$SCRIPT_DIR/openMVS_src" -DCMAKE_BUILD_TYPE=Release -DVCG_ROOT="$SCRIPT_DIR/vcglib" -DCMAKE_INSTALL_PREFIX="$SCRIPT_DIR"
make -j7 && make install