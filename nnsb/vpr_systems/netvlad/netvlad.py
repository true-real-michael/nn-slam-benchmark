#  Copyright (c) 2023, Ivan Moskalenko, Anastasiia Kornilova
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
from typing import Optional

import torch

from nnsb.backend.backend import Backend
from nnsb.backend.torch import TorchBackend
from nnsb.model_conversion.rknn import RknnExportable
from nnsb.model_conversion.tensorrt import TensorRTExportable
from nnsb.vpr_systems.netvlad.model.models_generic import (
    get_backend,
    get_model,
)
from nnsb.vpr_systems.vpr_system import VPRSystem


class NetVLADTorchBackend(TorchBackend):
    def __init__(self, weights: Optional[str] = None):
        class Unsqueeze(torch.nn.Module):
            def __init__(self):
                super(Unsqueeze, self).__init__()

            def forward(self, x):
                return x.unsqueeze(-1).unsqueeze(-1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = weights
        checkpoint = torch.load(weights, map_location=lambda storage, loc: storage)
        encoder_dim, encoder = get_backend()
        num_clusters = checkpoint["state_dict"]["pool.centroids"].shape[0]
        num_pcs = checkpoint["state_dict"]["WPCA.0.bias"].shape[0]
        model = (
            get_model(
                encoder,
                encoder_dim,
                num_clusters,
                append_pca_layer=True,
                num_pcs=num_pcs,
            )
            .eval()
            .to(device)
        )

        super().__init__(
            torch.nn.Sequential(
                model.encoder,
                model.pool,
                Unsqueeze().eval().to(device),
                model.WPCA,
            )
        )


class NetVLAD(VPRSystem, RknnExportable, TensorRTExportable):
    """
    Implementation of [NetVLAD](https://github.com/QVPR/Patch-NetVLAD) global localization method.
    """

    def __init__(
        self,
        backend: Optional[Backend] = None,
        weights: Optional[str] = None,
        resize: int = 800,
    ):
        """
        :param path_to_weights: Path to the weights
        :param resize: The size to which the larger side of the image will be reduced while maintaining the aspect ratio
        :param gpu_index: The index of the GPU to be used
        """
        super().__init__(resize)
        self.resize = resize
        self.backend = backend or self.get_torch_backend(weights)

    def postprocess(self, x):
        return x.detach().cpu().numpy()[0]

    @staticmethod
    def get_torch_backend(*args, **kwargs) -> TorchBackend:
        return NetVLADTorchBackend(*args, **kwargs)
