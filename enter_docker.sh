xhost +
name=vis2mesh
# Run in interactive mode
docker run -it \
--mount type=bind,source="$PWD/checkpoints",target=/workspace/checkpoints \
--mount type=bind,source="$PWD/example",target=/workspace/example \
--privileged \
-e NVIDIA_DRIVER_CAPABILITIES=all \
-e DISPLAY=unix$DISPLAY \
-v $XAUTH:/root/.Xauthority \
-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
--device=/dev/dri \
--gpus all $name

cd /workspace
inference.py example/example1.ply --cam cam0

# Run with single shot call
docker run \
--mount type=bind,source="$PWD/checkpoints",target=/workspace/checkpoints \
--mount type=bind,source="$PWD/example",target=/workspace/example \
--privileged \
-e NVIDIA_DRIVER_CAPABILITIES=all \
-e DISPLAY=unix$DISPLAY \
-v $XAUTH:/root/.Xauthority \
-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
--device=/dev/dri \
--gpus all $name \
/workspace/inference.py example/example1.ply --cam cam0


