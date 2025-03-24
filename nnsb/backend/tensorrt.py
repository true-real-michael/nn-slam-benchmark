import numpy as np
import torch
from pathlib import Path
import tensorrt as trt
from collections import OrderedDict, namedtuple

from nnsb.backend.backend import Backend


class TensorRTBackend(Backend):
    def __init__(self, model_path: Path, output_name: str, *args, **kwargs):
        self.output_name = output_name
        self.device = torch.device('cuda')
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        self.logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(model_path, 'rb') as model, trt.Runtime(self.logger) as runtime:
            model = runtime.deserialize_cuda_engine(model.read())
        self.bindings = OrderedDict()
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = tuple(model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))

            print ("name = {}".format(name))
            print ("dtype = {}".format(dtype))
            print ("shape = {}".format(shape))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.context = model.create_execution_context()

    def __call__(self, x):
        x = x.to(self.device)
        self.binding_addrs['images'] = x.data_ptr()
        self.context.execute_v2(list(self.binding_addrs.values()))
        x = self.bindings[self.output_name].data
        return x.cpu().numpy()[0]
