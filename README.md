# Vis2Mesh: Efficient Mesh Reconstruction from Unstructured Point Clouds of Large Scenes with Learned Virtual View Visibility, ICCV2021

[https://arxiv.org/abs/2108.08378](https://arxiv.org/abs/2108.08378)


```
@misc{song2021vis2mesh,
      title={Vis2Mesh: Efficient Mesh Reconstruction from Unstructured Point Clouds of Large Scenes with Learned Virtual View Visibility}, 
      author={Shuang Song and Zhaopeng Cui and Rongjun Qin},
      year={2021},
      eprint={2108.08378},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

##### Updates

- 2021/9/6: Intialize all in one project. Only this version only supports inferencing with our pre-trained weights. We will release Dockerfile to relief deploy efforts.

##### TODO

- Dockerfile
- Ground truth generation and network training.
- Evaluation scripts

#### Installation

##### Install nvidia-docker2

```bash
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

<!-- # Usage -->





<!-- # Training

export DATASET_PATH="$PWD/dataset"
export PYTHONPATH="$PWD:$PYTHONPATH"

python trainer/train_visnet.py -d $DATASET_PATH -l 0.005 -b 15 -e 50 --decay-step 5 --decay-rate 0.5 -a 'PartialConvUNet(input_channels=2)'
python trainer/train_depthnet.py -d $DATASET_PATH -l 0.005 -b 15 -e 50 --decay-step 5 --decay-rate 0.5 -a 'PartialConvUNet(input_channels=2)'
python trainer/train_depthnetrefine.py -d $DATASET_PATH -l 0.001 -b 8 -e 30 --decay-step 5 --decay-rate 0.5 -a 'PartialConvUNet(input_channels=2)' 'PartialConvUNet(input_channels=2)' -t True True --load0 "checkpoints/VIS_PartialConvUNet(input_channels=2)_epoch50.pth" --load1 "checkpoints/DEPTH_PartialConvUNet(input_channels=2)_epoch50.pth"

mv "checkpoints/REFDEPTH_PartialConvUNet(input_channels=2)_epoch30.pth" "checkpoints/REFDEPTH_PartialConvUNet(input_channels=2)_epoch30_best.pth"

python trainer/train_cascadenet.py -d $DATASET_PATH -l 0.001 -b 5 -e 70 --decay-step 7 --decay-rate 0.5 -a 'PartialConvUNet(input_channels=2)' 'PartialConvUNet(input_channels=2)' 'PartialConvUNet(input_channels=3)' -t False False True --load0 "checkpoints/REFVIS_PartialConvUNet(input_channels=2)_epoch30_best.pth" --load1 "checkpoints/REFDEPTH_PartialConvUNet(input_channels=2)_epoch30_best.pth"

python trainer/train_cascadenet.py -d $DATASET_PATH -l 0.0001 -b 5 -e 30 --decay-step 8 --decay-rate 0.5 -a 'PartialConvUNet(input_channels=2)' 'PartialConvUNet(input_channels=2)' 'PartialConvUNet(input_channels=3)' -t True True True --load "checkpoints/VISDEPVIS_['PartialConvUNet(input_channels=2)', 'PartialConvUNet(input_channels=2)', 'PartialConvUNet(input_channels=3)']_epoch70.pth"

mv "checkpoints/VISDEPVIS_['PartialConvUNet(input_channels=2)', 'PartialConvUNet(input_channels=2)', 'PartialConvUNet(input_channels=3)']_epoch30.pth" "checkpoints/VISDEPVIS_CascadePPP_epoch30.pth" -->