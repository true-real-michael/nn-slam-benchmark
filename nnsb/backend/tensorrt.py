#  Copyright (c) 2025, Mikhail Kiselev, Anastasiia Kornilova
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import numpy as np
import torch
from pathlib import Path
import tensorrt as trt
from collections import OrderedDict, namedtuple

from nnsb.backend.backend import Backend


class TensorRTBackend(Backend):
    """Backend for TensorRT models.

    This backend executes models optimized with NVIDIA TensorRT for
    high-performance inference on NVIDIA GPUs.

    Attributes:
        device: The device to run inference on.
        input_bindings: Dictionary of input tensor bindings.
        output_bindings: Dictionary of output tensor bindings.
        context: The TensorRT execution context.
    """

    def __init__(
        self, model_path: Path, input_tensors=1, output_tensors=1, *args, **kwargs
    ):
        """Initializes the TensorRT backend.

        Args:
            model_path: Path to the TensorRT engine file.
            input_tensors: Number of input tensors (default: 1).
            output_tensors: Number of output tensors (default: 1).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.device = torch.device("cuda")
        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        self.logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(model_path, "rb") as model, trt.Runtime(self.logger) as runtime:
            model = runtime.deserialize_cuda_engine(model.read())
        self.input_bindings = OrderedDict()
        self.output_bindings = OrderedDict()
        assert model.num_io_tensors == input_tensors + output_tensors
        for index in range(model.num_io_tensors):
            name = model.get_tensor_name(index)
            dtype = trt.nptype(model.get_tensor_dtype(name))
            shape = tuple(model.get_tensor_shape(name))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(
                self.device
            )
            if index < input_tensors:
                self.input_bindings[name] = Binding(
                    name, dtype, shape, data, int(data.data_ptr())
                )
            else:
                self.output_bindings[name] = Binding(
                    name, dtype, shape, data, int(data.data_ptr())
                )

        self.input_binding_addrs = OrderedDict(
            (n, d.ptr) for n, d in self.input_bindings.items()
        )
        self.output_binding_addrs = OrderedDict(
            (n, d.ptr) for n, d in self.output_bindings.items()
        )
        self.context = model.create_execution_context()

    def __call__(self, x):
        """Runs inference using the TensorRT model.

        Args:
            x: Input tensor for inference.

        Returns:
            The model's output(s) after inference. If there's only one output,
            returns that output directly; otherwise, returns a list of outputs.
        """
        x = x.to(self.device)
        self.input_binding_addrs["images"] = x.data_ptr()
        self.context.execute_v2(list(self.input_binding_addrs.values()))
        x = [value.data for value in self.output_bindings.values()]
        if len(x) == 1:
            x = x[0]
        return x
