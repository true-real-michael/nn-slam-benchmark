import numpy as np
import torch
from pathlib import Path
import tensorrt as trt
from collections import OrderedDict, namedtuple

from nnsb.backend.backend import Backend


class TensorRTBackend(Backend):
    def __init__(self, model_path: Path, input_tensors=1, output_tensors=1, *args, **kwargs):
        self.device = torch.device('cuda')
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        self.logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(model_path, 'rb') as model, trt.Runtime(self.logger) as runtime:
            model = runtime.deserialize_cuda_engine(model.read())
        self.input_bindings = OrderedDict()
        self.output_bindings = OrderedDict()
        assert model.num_io_tensors == input_tensors + output_tensors
        for index in range(model.num_io_tensors):
            name = model.get_tensor_name(index)
            dtype = trt.nptype(model.get_tensor_dtype(name))
            shape = tuple(model.get_tensor_shape(name))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.device)
            if index < input_tensors:
                self.input_bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            else:
                self.output_bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))

        self.input_binding_addrs = OrderedDict((n, d.ptr) for n, d in self.input_bindings.items())
        self.output_binding_addrs = OrderedDict((n, d.ptr) for n, d in self.output_bindings.items())
        self.context = model.create_execution_context()

    def __call__(self, x):
        x = x.to(self.device)
        self.input_binding_addrs['images'] = x.data_ptr()
        self.context.execute_v2(list(self.input_binding_addrs.values()))
        x = [value.data for value in self.output_bindings.values()]
        if len(x) == 1:
            x = x[0]
        return x
