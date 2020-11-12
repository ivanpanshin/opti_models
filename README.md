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
pip install --upgrade pip
pip install .
```
2. Install tensorrt
```
pip install nvidia-pyindex
pip install --upgrade nvidia-tensorrt
```

## Convertation
**CURRENTLY IN DEV MODE**



### ONNX Convertation with Python

1. Run:
```
    python opti_models/convertations/cvt_onnx.py --model_name MODEL_NAME --export_dir EXPORT_DIR --is_torchivision IS_TORCHVISION --ckpt_path CKPT_PATH --batch_size BATCH_SIZE --in_size IN_SIZE --num_classes NUM_CLASSES    
```
In order to convert ResNet18 with ImageNet pretraning run:
```
    python opti_models/convertations/cvt_onnx.py --model_name 'resnet18' --export_dir 'data/onnx_export' --is_torchivision True --batch_size 1 --in_size 224 224
```
In order to convert you own ResNet18 torchvision model with custom weights run:
```
    python opti_models/convertations/cvt_onnx.py --model_name 'resnet18' --export_dir 'data/onnx_export' --is_torchivision True --batch_size 1 --in_size 224 224 --ckpt_path CKPT_PATH --num_classes NUM_CLASSES
```

### TensorRT Convertation
1. In `scripts/convertations/tensorrt_convertation.sh` change:
    - `onnx_path` - path to the ONNX file
    - `export_dir` - directory to export converted file (default `data/trt_export`)
    - `batch_size` - batch size for converted model (default = 1) 
    - `fp_type` - type of float point precision, could be "16" or "32" (default = "32") 
2. Run:
```
    bash scripts/convertations/tensorrt_convertation.sh
```

## Models
For list of all models see [MODELS.md](/opti_models/models/MODELS.md)

## Benchmarks

### Imagenet
For all imagenet benchmarks you should download and untar: https://drive.google.com/file/d/1Yi_SZ400LKMXeA08BvDip4qBJonaThae/view?usp=sharing

#### Torch Imagenet Benchmark
1. In `scripts/benchmarks/torch_imagenet_benchmark.sh` change:
    - `path_to_images` - path to the `imagenetv2-top-images-format-val` folder
    - `model_name` - name of the model to bench
    - `batch_size` - anount of the images in every batch
    - `workers` - number of workersл 
2. Run: 
```
    bash scripts/benchmarks/torch_imagenet_benchmark.sh
```
#### Tensorrt Imagenet Benchmark
1.  In `scripts/benchmarks/tensorrt_imagenet_benchmark.sh` change:
    - `path_to_images` - path to the `imagenetv2-top-images-format-val` folder
    - `trt_path` - path to the TensorRT .engine model (for convertation see [Convertation](#Convertation))
2. Run: 
```
    bash scripts/benchmarks/tensorrt_imagenet_benchmark.sh
```
