#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo "Installing.. $SCRIPT_DIR"
git clone https://github.com/cdcseacave/VCG.git "$SCRIPT_DIR/vcglib" -q
cd "$SCRIPT_DIR/vcglib"  && git checkout 88f12f212a1645d1fa6416592a434c29e63b57f0
git clone https://github.com/GDAOSU/openMVS_mvasupport.git "$SCRIPT_DIR/openMVS_src" -q
cd "$SCRIPT_DIR/openMVS_src" && git checkout 09fdd45e23c4b4fd3a0a4258f8294aae6e9fe8d9 
mkdir "$SCRIPT_DIR/openMVS_build"
cd "$SCRIPT_DIR/openMVS_build"
cmake . "$SCRIPT_DIR/openMVS_src" -DCMAKE_BUILD_TYPE=Release -DVCG_ROOT="$SCRIPT_DIR/vcglib" -DCMAKE_INSTALL_PREFIX="$SCRIPT_DIR"
make -j7 && make install
