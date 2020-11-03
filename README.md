# opti_models
## Description

## Install
0. Make clean venv
```
python -m venv venv
source venv/bin/activate
```
1. Install dev branch
```
git clone git@github.com:IlyaDobrynin/opti_models.git && cd opti_models
git checkout dev
pip install .
```
2. Install tensorrt
```
pip install nvidia-pyindex
pip install --upgrade nvidia-tensorrt
```

## Convertation
To Be Done


## Models
For list of all models see [MODELS.md](/opti_models/models/MODELS.md)

## Benchmarks

### Imagenet
For all imagenet benchmarks you should download and untar: https://drive.google.com/file/d/1Yi_SZ400LKMXeA08BvDip4qBJonaThae/view?usp=sharing

#### Torch Imagenet Benchmark
1. In `scripts/torch_imagenet_benchmark.sh` change:
    - `path_to_images` - path to the `imagenetv2-top-images-format-val` folder
    - `model_name` - name of the model to bench
    - `batch_size` - anount of the images in every batch
    - `workers` - number of workers
2. Run: `bash scripts/torch_imagenet_benchmark.sh`
#### Tensorrt Imagenet Benchmark
1.  In `scripts/tensorrt_imagenet_benchmark.sh` change:
    - `path_to_images` - path to the `imagenetv2-top-images-format-val` folder
    - `trt_path` - path to the TensorRT .engine model (for convertation see [Convertation](#Convertation))
2. Run: `bash scripts/tensorrt_imagenet_benchmark.sh`