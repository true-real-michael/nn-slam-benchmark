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
import numpy as np
import torch
import torchvision

from typing import Optional
from pathlib import Path
from torchvision import transforms as tvf

from nnsb.backend.torch import TorchBackend
from nnsb.backend.backend import Backend
from nnsb.utils import transform_image_for_vpr
from nnsb.vpr_systems.vpr_system import VPRSystem
from nnsb.vpr_systems.anyloc.models import DinoV2ExtractFeatures, VLAD



class AnyLocTorchBackend(TorchBackend):
    def __init__(self):
        self.model = DinoV2ExtractFeatures(
            dino_model="dinov2_vitg14", layer=31, facet="value", device=self.device
        )


class AnyLoc(VPRSystem):
    """
    Implementation of [AnyLoc](https://github.com/AnyLoc/AnyLoc) global localization method.
    """

    def __init__(
            self,
            resize: int = 800,
            backend: Optional[Backend] = None,
            c_centers_file: Optional[Path] = None
        ):
        """
        :param c_centers_file: Path to clusters' centers
        :param resize: The size to which the larger side of the image will be reduced while maintaining the aspect ratio
        :param gpu_index: The index of the GPU to be used
        """
        super().__init__(resize)
        self.resize = resize
        if backend is None:
            self.model = DinoV2ExtractFeatures(
                dino_model="dinov2_vitg14", layer=31, facet="value", device=self.device
            )
        else:
            self.model = backend
        self.c_centers = torch.load(c_centers_file)
        self.vlad = VLAD(num_clusters=32, desc_dim=None, c_centers_path=c_centers_file)
        self.vlad.fit()
    
    def preprocess(self, x):
        x = transform_image_for_vpr(
            x, self.resize, torchvision.transforms.InterpolationMode.BICUBIC
        ).to(self.device)
        _, h, w = x.shape
        h_new, w_new = (h // 14) * 14, (w // 14) * 14
        return tvf.CenterCrop((h_new, w_new))(x)[None, ...]
    
    def postprocess(self, x):
        x = self.vlad.generate(x.cpu().squeeze())
        return x.numpy()

    def get_image_descriptor(self, x: np.ndarray):
        x = self.preprocess(x)
        with torch.no_grad():
            x = self.model(x)
        return self.postprocess(x)
