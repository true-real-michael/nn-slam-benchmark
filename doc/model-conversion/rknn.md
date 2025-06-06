# Converting models to RKNN format

This format is used with devices with Rockchip NPU, such as Orange Pi 5, Radxa Rock 5B or Khadas Edge 2, all of which
have an RK3588 chip.
This guide may also be applicable for other chips with Rockchip NPU, but there may be changes that need to be made in
order to support different chips.

These instructions were tested on an Orange Pi 5 with Ubuntu 22.04
from [ubutnu-rockchip](https://github.com/Joshua-Riek/ubuntu-rockchip) and Python 3.10
Apart from the Orange Pi 5 itself, a host computer for model conversion is required. Ubuntu 22.04 with Python 3.10 was
used for this purpose.

[This](https://github.com/airockchip/rknn-toolkit2/blob/master/doc/02_Rockchip_RKNPU_User_Guide_RKNN_SDK_V2.3.0_EN.pdf)
is the most relevant part of the documentation. If something concerning the parameters, options, or the workflow as a
whole
is not documented here, there is a good chance that the information you're looking for is in this document.
Also, the whole quantization part is skipped here, so if you need to quantize the model, you may want to refer to the
docs.

## Table of contents

1. [Overview](#overview)
2. [Environment](#environment)
    1. [Host setup](#host-setup)
    2. [Orange Pi 5 setup](#orange-pi-5-setup)
3. [Original Examples](#original-examples)
    1. [Model conversion example \[host computer\]](#model-conversion-example-host-computer)
    2. [Model inference example \[target device\]](#model-inference-example-target-device)
4. [Unsupported layers](#unsupported-layers)
5. [Disclaimer](#disclaimer)
6. [Contact information](#contact-information)

## Overview

The workflow as a whole looks like this

```
|    HOST [model conversion]      |  TARGET [model inference]  |
|---------------------------------|----------------------------|
|       create RKNN object        |                            |
|            config               |                            |
|           load model            |                            |
|         build for rknn          |                            |
| [optional] run on the simulator |                            |
|          export model           |                            |
|          RKNN release           |                            |
|                                 |   create RKNNLite object   |
|                                 |      load rknn model       |
|                                 |   RKNNLite init runtime    |
|                                 |   preprocess input  (CPU)  |
|                                 |     run inference   (NPU)  |
|                                 |  postprocess output (CPU)  |
|                                 |     RKNNLite release       |
```

## Environment

The [airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2/) repository has the wheels for conversion
and running
the models.

> **_NOTE:_** Do NOT use the `rockchip-linux/rknn-toolkit2` repository as it is no longer maintained.

The `rknn-toolkit2` directory is for model conversion on the **host computer**.

The `rknn-toolkit-lite2` directory is for inference on the **target device**.

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

2. Create a python environment. `rknn-toolkit2-lite` supports versions 3.6-3.12. The instructions were tested on Python
   3.10.
3. Install the package

```sh
pip install rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.3.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl  # or other, if you use Python other than 3.10rknn-toolkit2/packages/x86_64/rknn_toolkit2-2.3.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl 
```

4. Set up the RKNPU runtime

```sh
sudo cp rknpu2/runtime/Linux/rknn_server/aarch64/usr/bin/* /usr/bin/
sudo chmod +x /usr/bin/rknn_server
sudo chmod +x /usr/bin/start_rknn.sh
sudo chmod +x /usr/bin/restart_rknn.sh
```

5. Copy the api library to `/usr/lib`

```sh
sudo cp rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/
```

6. Restart the RKNN server

```sh
restart_rknn.sh
```


## Original examples

### Model conversion example [host computer]

The `rknn-toolkit2/examples` directory contains the examples of model conversion.

Try the resnet18 example

```sh
cd rknn-toolkit2/examples/pytorch/resnet18
python test.py
```

[test.py](https://github.com/airockchip/rknn-toolkit2/blob/master/rknn-toolkit2/examples/pytorch/resnet18/test.py) loads
the model, does the conversion and launches the model on the simulator.

The part of code responsible for the model conversion is

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

Note that the normalization is done within the model, so the mean and std values are passed during the configuration.

In addition to PyTorch models, the toolkit supports TensorFlow, TFLite, ONNX, and Caffe models, the examples for which
can be found at the resprctive subdirectories
of [rknn-toolkit2/examples](https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit2/examples).

Alternatively, it is possible to convert the model using a config file

```yml
models:
  name: resnet_18              # name of the output model
  platform: pytorch            # the framework of the original model
  model_file_path: ./resnet18.pt # pytorch model file
  subgraphs:         # input shape in the format:
    input_size_list: # `batch size`, `channels`, `heights`, `width`
      - 1, 3, 224, 224
  quantize: true               # whether to quantize the model
  dataset: ./dataset.txt       # path to the dataset, which is needed for quantization
  configs:
    quantized_dtype: asymmetric_quantized-8 # quantization type
    mean_values: [ 123.675, 116.28, 103.53 ]  # input normalization is done within the model
    std_values: [ 58.395, 58.395, 58.395 ]
    quant_img_RGB2BGR: false
    quantized_algorithm: normal
    quantized_method: channel
```

```sh
python3 -m rknn.api.rknn_convert -t rk3588 -i ./model_config.yml -o ./
```

### Model inference example [target device]

The `rknn-toolkit-lite2/examples` directory contains the examples of model inference.

```sh
cd rknn-toolkit-lite2/examples/resnet18
python test.py
```

[test.py](https://github.com/airockchip/rknn-toolkit2/blob/master/rknn-toolkit-lite2/examples/resnet18/test.py) file
here contains the code for model inference on the device.

The part of code responsible for the model inference is

```py
from rknnlite.api import RKNNLite

# ...
# ...

rknn_model = './resnet_18.rknn'

rknn_lite = RKNNLite()

# Load RKNN model
print('--> Load RKNN model')
ret = rknn_lite.load_rknn(rknn_model)
if ret != 0:
    print('Load RKNN model failed')
    exit(ret)
print('done')

ori_img = cv2.imread('./space_shuttle_224.jpg')
img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
img = np.expand_dims(img, 0)

# Init runtime environment
print('--> Init runtime environment')
ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
if ret != 0:
    print('Init runtime environment failed')
    exit(ret)
print('done')

# Inference
print('--> Running model')
outputs = rknn_lite.inference(inputs=[img])

rknn_lite.release()

# ...
# ...
```

## Unsupported layers

Some layers of the aforementioned frameworks are not supported by RKNN.
If the model contains unsupported layers, the conversion will either fail or the layers will be computed on the CPU.
I have not found a list of layers, which lead to the failure of the conversion, so the most straightforward way is to
try the conversion and see if it works.

Unsupported layers can be replaced with custom layers, for example, here is
the [EigenPlaces model](https://github.com/gmberton/EigenPlaces/blob/main/eigenplaces_model/eigenplaces_network.py):

```py
class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return nn.functional.normalize(x, p=2.0, dim=self.dim)  # Unsupported by RKNN


class GeoLocalizationNet_(nn.Module):
    def __init__(self, backbone: str, fc_output_dim: int):
        super().__init__()
        self.backbone, features_dim = _get_backbone(backbone)
        self.aggregation = nn.Sequential(
            L2Norm(),      # Custom layer defined above
            GeM(),
            Flatten(),
            nn.Linear(features_dim, fc_output_dim),
            L2Norm()       # Custom layer defined above
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x
```

RKNN does not support `torch.nn.functional.normalize`, which is used in the custom `L2Norm` module.
It can be replaced with the following:

```py
class RknnCompatibleL2Norm(nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        norm = torch.sqrt(
            torch.sum(x ** 2, dim=self.dim, keepdim=True) + self.eps)
        return x / norm
```

To do the replacement on the trained model, the following code can be used:

```py
net = torch.hub.load(  # Loading the trained model
    "gmberton/cosplace",
    "get_trained_model",
    backbone="ResNet101",
    fc_output_dim=2048,
)
net.aggregation[0] = RknnCompatibleL2Norm()  # Replacing the first L2Norm layer
net.aggregation[4] = RknnCompatibleL2Norm()  # Replacing the second L2Norm layer
```
> It's important to mention that this particular replacement is actually useless because RKNN operates on `float16`, summing which can (and did, in my testing) cause over/underflows, resulting in -inf/+inf values.

This would allow the model to be converted to RKNN format.
However, this approach is not applicable to trainable layers, as the weights would be lost.

## Disclaimer

This guide is far from being extensive as RKNN offers a lot of features and options, which are not covered here.
For example, the quantization process can be customized, and I did not dive into the details of the process.
RKNN also allows for custom operations (ONNX only), which can be used to implement the unsupported layers.

## Contact information

If you have any questions or need help with the conversion or have suggestions to this guide, feel free to contact me
at [tg:true_real_michael](https://t.me/true_real_michael).
