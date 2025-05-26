# NN-SLAM Benchmark

This repository provides benchmarking utilities for methods of Global Description Extraction, Local Feature Detection, Local Feature Matching.

## Overview

NN-SLAM Benchmark supports:

- Benchmarking global descriptor extraction methods (NetVLAD, CosPlace, EigenPlaces, SALAD, MixVPR, Sela)
- Benchmarking feature detectors (SuperPoint, XFeat)
- Benchmarking feature matchers (SuperGlue, LightGlue, LighterGlue)
- Support for multiple backends (PyTorch, TensorRT, RKNN)
- Support for RKNN-export and quantization

## Hardware Support

This benchmark has been tested on:
- Nvidia Jetson devices (Orin NX16, Xavier AGX32, Nano)
- Orange Pi 5 (Rockchip RK3588)

## Installation

### Dependencies

1. Create a Python virtual environment:
   ```bash
   python3 -m venv .venv --system-site-packages
   source .venv/bin/activate
   ```

2. Install basic dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Platform-Specific Setup

#### Nvidia Jetson Devices

1. Make sure you have JetPack installed (provides CUDA, cuDNN, and TensorRT).

2. If PyTorch is not already installed, install the appropriate wheel from [here](https://developer.download.nvidia.com/compute/redist/jp)

3. (Optional) For TensorRT quantization support, make sure the [TensoRRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html), and `pytorch-quantization` toolkit are installed:
   ```bash
   git clone https://github.com/NVIDIA/TensorRT.git
   cd TensorRT/tools/pytorch-quantization
   pip install .
   ```

#### Orange Pi 5 / Rockchip Platforms

1. Install the RKNN runtime driver and RKNNLite2 pip-package according to [Rockchip's instructions](https://github.com/airockchip/rknn-toolkit2/).

### Model Weights

Download the required model weights:

1. NetVLAD, MixVPR, SelaVPR, SuperGlue from [here](https://drive.google.com/file/d/1lTtiU2favmQMOJrfSzdwzA6fw44aUy3z/view?usp=sharing):
   ```bash
   gdown 1lTtiU2favmQMOJrfSzdwzA6fw44aUy3z
   unzip weights.zip
   ```

2. DINOv2 weights for SelaVPR:
   ```bash
   wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth -P weights/
   ```

3. (Optional) For Rockchip platforms, download pre-converted RKNN models [here](https://drive.google.com/file/d/1DQ3kqI6890uFrx8tK2oqzeilaapLoBI8/view?usp=sharing):
   ```bash
   mkdir weights
   cd weights
   gdown 1DQ3kqI6890uFrx8tK2oqzeilaapLoBI8
   unzip rknn.zip -d
   ```

## Datasets

This utility uses the datasets in the format provided by [gmberton/VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader)

For example, to download the StLucia dataset, you should
```bash
mkdir datasets
git clone https://github.com/gmberton/VPR-datasets-downloader.git
cd VPR-datasets-downloader
ln -s ../datasets .
pip install -r requirements.txt
python download_st_lucia.py
```

## Usage

### Benchmarking VPR Systems

```bash
python bench_vpr.py --dataset st_lucia --board your_board_name --resize 224 --system netvlad
```

Options:
- `--dataset`: Name of the dataset (default: `st_lucia`)
- `--board`: Name of your hardware platform (default: `test`)
- `--resize`: Image size to use (omit to run all sizes)
- `--system`: VPR system to benchmark (omit to run all systems)
- `--backend`: Backend to use (`tensorrt`, `rknn`, omit to run with Torch Backend)
- `--quantized`: Use quantized models

### Benchmarking Feature Detectors

```bash
python bench_feature_detectors.py --dataset st_lucia --board your_board_name --resize 400 --detector superpoint
```

Options:
- `--dataset`: Name of the dataset (default: `st_lucia`)
- `--board`: Name of your hardware platform (default: `test`)
- `--resize`: Image size to use (omit to run all sizes)
- `--detector`: Feature detector to benchmark (`superpoint`, `xfeat`)
- `--no-save-features`: Don't save extracted features to cache

### Benchmarking Feature Matchers

```bash
python bench_feature_matchers.py --dataset st_lucia --board your_board_name --resize 400 --matcher lightglue --detector superpoint
```

Options:
- `--dataset`: Name of the dataset (default: `st_lucia`)
- `--board`: Name of your hardware platform (default: `test`)
- `--resize`: Image size to use (omit to run all sizes)
- `--matcher`: Feature matcher to benchmark (`lightglue`, `superglue`, `lighterglue`)
- `--detector`: Feature detector to use with the matcher (default: `superpoint`)

### Model Conversion

#### Convert to RKNN Format

```bash
python scripts/convert_vpr_systems_to_rknn.py --models netvlad eigenplaces --resize 400 --verbose
```

For quantized models:
```bash
python scripts/convert_vpr_systems_to_rknn.py --models netvlad --resize 400 --dataset path/to/calibration/dataset
```

Options:
- `--output-dir`: Directory to save converted models (default: `weights/rknn/fp32` or `weights/rknn/int8`)
- `--dataset`: Path to calibration dataset (enables quantization)
- `--models`: Models to convert
- `--resize`: Specific resize values to use

#### Quantize Models for TensorRT

```bash
python scripts/quant_vpr_systems.py --calibration-data-dir datasets --calibration-data-name st_lucia --models netvlad --resize 400
```

Options:
- `--output-dir`: Directory to save converted models (default: `weights/tensorrt`)
- `--calibration-data-dir`: Path to calibration dataset directory
- `--calibration-data-name`: Name of calibration dataset
- `--models`: Models to convert
- `--resize`: Specific resize values to use

## Results

Benchmark results are saved in CSV format under the `measurements/{board_name}/` directory.

## Architecture

The project follows a modular architecture:

- `nnsb/backend/`: Backend implementations (PyTorch, TensorRT, RKNN, ONNX)
- `nnsb/feature_detectors/`: Feature detection implementations
- `nnsb/feature_matchers/`: Feature matching implementations 
- `nnsb/vpr_systems/`: Global Descriptor Extraction implementations
- `nnsb/model_conversion/`: Mixins for model conversion and quantization

## License

This project is licensed under the Apache License 2.0.
