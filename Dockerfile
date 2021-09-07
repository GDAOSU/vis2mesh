FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
LABEL maintainer="Shaun Song <song.1634@osu.edu>"

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV MESA_GL_VERSION_OVERRIDE=4.5
ENV MESA_GLSL_VERSION_OVERRIDE=450

COPY . /workspace
RUN cd /workspace && bash setup_tools.sh

ENV main_path="/workspace"
ENV PYTHON_LIBDIR="/opt/conda/lib"
ENV PATH="$main_path/tools/bin/OpenMVS:$main_path/tools/bin:$PATH"
ENV LD_LIBRARY_PATH="$main_path/tools/lib/OpenMVS:$main_path/tools/lib:$PYTHON_LIBDIR:$LD_LIBRARY_PATH"
ENV PYTHONPATH="$main:$main_path/tools/lib"