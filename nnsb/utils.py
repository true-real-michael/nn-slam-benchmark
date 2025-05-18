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
import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import InterpolationMode


def transform_image_for_vpr(
    image: np.ndarray,
    resize: int,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(
                (resize, resize), interpolation=interpolation
            ),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    transformed_image = transform(image)
    return transformed_image


def transform_image_for_sp(image: np.ndarray, resize: int):
    grayim = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayim = cv2.resize(grayim, (resize, resize), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(grayim)[None]
