# NN SLAM benchmark
A benchmarking utility for SLAM neural network components.

## Setup
This utility was tested on Nvidia Jetson Orin NX16, Nvidia Jetson Xavier AGX32, Nvidia Jetson Nano, and Orange Pi 5.

### Nvidia Jetson Devices
1. Install the OS — Nvidia JetPack.
2. Create a virtual environment with `python3 -m venv .venv --system-site-packages` to inherit global packages, install the dependencies.
3. Unless PyTorch is already installed globally, install PyTorch wheel from [here](https://developer.download.nvidia.com/compute/redist/jp/) according to your JetPack version.
4. _\[Optional\]_ For model quantization and quantized inference support, install [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html).
5. Download the weights for NetVLAD, MixVPR, SelaVPR, and SuperGlue [here](https://drive.google.com/file/d/1lTtiU2favmQMOJrfSzdwzA6fw44aUy3z/view?usp=sharing) and for DINOv2 for SelaVPR [here](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth) and place the weights under the `weights` directory

### Orange Pi
1. Install vendor provided [Ubuntu 22.04](http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/service-and-support/Orange-pi-5.html) Image or [Joshua Reik's Ubuntu 22.04](https://github.com/Joshua-Riek/ubuntu-rockchip).
2. Create a virtual environment and install the requirements from `requirements-rknn.txt`
3. Install RKNPU drivers and helper service according to [Rockchip's instructions](https://github.com/airockchip/rknn-toolkit2/).
4. Download the converted RKNN weights [here](https://drive.google.com/file/d/1DQ3kqI6890uFrx8tK2oqzeilaapLoBI8/view?usp=sharing) **_OR_** Download the original weights [here](https://drive.google.com/file/d/1lTtiU2favmQMOJrfSzdwzA6fw44aUy3z/view?usp=sharing) and [here](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth) and convert them using the `scripts/convert_vpr_systems_to_rknn.py` script. 

## Usage

### Benchmarking

The scripts `bench_vpr.py`, `bench_feature_detectors.py`, `bench_feature_matchers.py` to benchmark the methods.
These scripts produce csv files with the benchmark results.
They support the following parameters:
- `--dataset` to specify the dataset name, defaults to `st_lucia`;
- `--board` to specify the platform name, which will be included in the name of the output file, defaults to `test`;
- `--resize` to only use the specific image resize value, if omitted, all values are used;
- `--method` to only benchmark the specified method, if omitted, all methods are used;
- `--backend` to specify which backend to use: `torch`, `tensorrt`, `rknn`, `rknn-q`, defaults to `torch`.

### Model conversion

The script `scripts/convert_vpr_systems_to_rknn.py` converts the models to `rknn` format. 
This script supports the following parameters:
- `--output-dir` to specify the output directory, defaults to `weights/rknn/fp32` for export without quantization, and `weights/rknn/int8` for export with quantization;
- `--dataset` to specify the path to the `images.txt` which lists the images relative to this files parent directory. If this parameter is specified, then the models are exported with quantization, and the given dataset is used for models calibration. If not specified, the models are exported without quantization;
- `--models` to specify the names of methods to convert, defaults to all models which support RKNN export;
- `--resize` to only use the specific image resize values, defaults to all values;
- `--verbose` for verbose output.

- The script `scripts/convert_vpr_systems_to_trt.py` converts the models to `trt` format with quantization. 
This script supports the following parameters:
- `--output-dir` to specify the output directory, defaults to `weights/rknn/fp32` for export without quantization, and `weights/rknn/int8` for export with quantization;
- `--quantized` to specify the path to the `dataset` to be used for model calibration;
- `--models` to specify the names of methods to convert, defaults to all models which support RKNN export;
- `--resize` to only use the specific image resize values, defaults to all values;
- `--verbose` for verbose output.
