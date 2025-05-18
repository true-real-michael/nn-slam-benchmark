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
import importlib

__all__ = []
_optional_backends = {
    'onnx': 'OnnxExportable',
    'rknn': 'RknnExportable',
    'tensorrt': 'TensorRTExportable',
    'torchscript': 'TorchScriptExportable',
}

for module_name, class_name in _optional_backends.items():
    try:
        module = importlib.import_module(f".{module_name}", __package__)
        globals()[class_name] = getattr(module, class_name)
        __all__.append(class_name)
    except (ImportError, AttributeError, OSError):
        pass
