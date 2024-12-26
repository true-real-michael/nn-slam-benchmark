# Converting models to RKNN format

This format is used with devices with Rockchip NPU, such as Orange Pi 5, Radxa Rock 5B or Khadas Edge 2, all of which have an RK3588 chip.
This guide may also be applicable for other chips with Rockchip NPU, but there may be changes that need to be made in order to support different chips.

These instructions were tested on an Orange Pi 5 with Ubuntu 22.04 from [ubutnu-rockchip](https://github.com/Joshua-Riek/ubuntu-rockchip)

## Environment

Apart from the Orange Pi 5 itself, a host computer for model conversion is required.

The [airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2/) has the wheels for conversion and running the models.

> **_NOTE:_** Do NOT use the `rockchip-linux/rknn-toolkit2` repository as it is no longer maintained.

The `rknn-toolkit2` directory is for model conversion on the **host** computer.

The `rknn-toolkit-lite2` directory is for inference the target device.

### Host setup

1. Clone the repository
```sh
git clone https://github.com/airockchip/rknn-toolkit2
cd rknn-toolkit2
```
2. Create a python environment. `rknn-toolkit2` supports versions 3.6-3.12. The instructions were tested on Python 3.10.
3. Install the requirements
```sh
pip install -r rknn-toolkit2/packages/x86_64/requirements_cp310-2.3.0.txt  # or other, if you use Python other than 3.10
```
4. Install the package
```sh
pip install rknn-toolkit2/packages/x86_64/rknn_toolkit2-2.3.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl  # or other, if you use Python other than 3.10
```

### Orange Pi 5 setup

1. Clone the repository
```sh
git clone https://github.com/airockchip/rknn-toolkit2
cd rknn-toolkit2
```
2. Create a python environment. `rknn-toolkit2-lite` supports versions 3.6-3.12. The instructions were tested on Python 3.10.
3. Install the package
```sh
pip install rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.3.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl  # or other, if you use Python other than 3.10rknn-toolkit2/packages/x86_64/rknn_toolkit2-2.3.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl 
```
4. Setup the RKNPU runtime
```sh
sudo cp rknpu2/runtime/Linux/rknn_server/aarch64/usr/bin/* /usr/bin/
sudo chmod +x /usr/bin/rknn_server
sudo chmod +x /usr/bin/start_rknn.sh
sudo chmod +x /usr/bin/restart_rknn.sh
```
5. Restart the RKNN server
```sh
restart_rknn.sh
```
6. Copy the api library to `/usr/lib`
```sh
sudo cp rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/
```

## Original examples

### Model conversion example [host computer]

The `rknn-toolkit2/examples` directory containes the examples of model conversion.

Try the resnet18 example
```sh
cd rknn-toolkit2/examples/pytorch/resnet18
python test.py
```

[test.py](https://github.com/airockchip/rknn-toolkit2/blob/master/rknn-toolkit2/examples/pytorch/resnet18/test.py) loads the model, does the conversion and launches the model on the simulator.

The part of code responsible for the model convesion is

```py
from rknn.api import RKNN

# ...
# ...

model = 'resnet_18.pt'
input_size_list = [[1, 3, 224, 224]]

# Create RKNN object
rknn = RKNN(verbose=True)

# Pre-process config
print('--> Config model')
rknn.config(mean_values=[123.675, 116.28, 103.53], std_values=[58.395, 58.395, 58.395], target_platform='rk3588')
print('done')

# Load model
print('--> Loading model')
ret = rknn.load_pytorch(model=model, input_size_list=input_size_list)
if ret != 0:
    print('Load model failed!')
    exit(ret)
print('done')

# Build model
print('--> Building model')
ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
if ret != 0:
    print('Build model failed!')
    exit(ret)
print('done')

# Export rknn model
print('--> Export rknn model')
ret = rknn.export_rknn('./resnet_18.rknn')
if ret != 0:
    print('Export rknn model failed!')
    exit(ret)
print('done')

# ...
# ...
```

Alternatively, it is possible to convert the model using a config file

```yml
models:
    name: resnet_18              # name of the output model
    platform: pytorch            # the framework of the original model
    model_file_path: ./resnet18.pt # pytorch model file
    subgraphs:                   # input shape in the format:
      input_size_list:           # `batch size`, `channels`, `heights`, `width`
        - 1, 3, 224, 224
    quantize: true               # whether to quantize the model
    dataset: ./dataset.txt       # path to the dataset, which is needed for quantization
    configs:
      quantized_dtype: asymmetric_quantized-8 # quantization type
      mean_values: [123.675, 116.28, 103.53]  # input normalization is done within the model
      std_values: [58.395, 58.395, 58.395]
      quant_img_RGB2BGR: false
      quantized_algorithm: normal
      quantized_method: channel
```
 
```sh
python3 -m rknn.api.rknn_convert -t rk3588 -i ./model_config.yml -o ./
```

### Model inference example [target device]

The `rknn-toolkit-lite2/examples` directory containes the examples of model inference.

```sh
cd rknn-toolkit-lite2/examples/resnet18
python test.py
```

[test.py](https://github.com/airockchip/rknn-toolkit2/blob/master/rknn-toolkit-lite2/examples/resnet18/test.py) file here contains the code for model inference on the device.


