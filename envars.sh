#!/usr/bin/env bash
export main_path="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PYTHON_LIBDIR=`python -c 'from distutils import sysconfig;print(sysconfig.get_config_var("LIBDIR"))'`
export PATH="$main_path/tools/bin/OpenMVS:$main_path/tools/bin:$PATH"
export LD_LIBRARY_PATH="$main_path/tools/lib/OpenMVS:$main_path/tools/lib:$PYTHON_LIBDIR:$LD_LIBRARY_PATH"
export PYTHONPATH="$main:$main_path/tools/lib"