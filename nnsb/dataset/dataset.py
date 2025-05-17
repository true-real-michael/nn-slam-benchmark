#  Copyright (c) 2023, Mikhail Kiselyov
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
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def path_to_pil_img(path):
    return Image.open(path).convert("RGB")


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            images_dir: Path,
            dataset_name,
            resize=224,
            limit=None,
            superpoint=False,
    ):
        super().__init__()
        images_dir = dataset_dir / dataset_name / "images/test" / "queries"
        if not images_dir.exists():
            raise FileNotFoundError(f"Images folder {images_dir} not found.")

        if not superpoint:
            self.transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Grayscale(),
            ])

        self.images_paths = sorted(images_dir.glob("*.png")) + sorted(
            images_dir.glob("*.jpg")
        )
        if limit is not None:
            self.images_paths = self.images_paths[:limit]

    def __getitem__(self, index):
        img = path_to_pil_img(self.images_paths[index])
        img = self.transform(img)
        latitude = float(self.images_paths[index].stem.split("@")[1])
        longitude = float(self.images_paths[index].stem.split("@")[2])
        return img, (latitude, longitude)

    def __len__(self):
        return len(self.database_paths)

class Data(BaseDataset):
    def __init__(
        self,
        dataset_dir: Path,
        dataset_name,
        resize=224,
        limit=None,
        superpoint=False,
    ):
        super().__init__(dataset_dir / dataset_name / "images/test/database", resize, limit, superpoint)

    def get_knn(self, k_neighbors=5):
        from sklearn.neighbors import NearestNeighbors

        database_utms = np.array(
            [
                (path.stem.split("@")[1], path.stem.split("@")[2])
                for path in self.database_paths
            ]
        ).astype(np.float64)
        knn = NearestNeighbors(n_neighbors=k_neighbors, n_jobs=-1)
        knn.fit(database_utms)
        return knn


class Queries(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        dataset_name,
        knn,
        resize=224,
        limit=None,
        superpoint=False,
    ):
        super().__init__(dataset_dir / dataset_name / "images/testqueries", resize, limit, superpoint)
        if knn is not None:
            self.queries_utms = np.array(
                [
                    (path.stem.split("@")[1], path.stem.split("@")[2])
                    for path in self.queries_paths
                ]
            ).astype(np.float64)

            self.knn = knn
            self.soft_positives_per_query = knn.radius_neighbors(
                self.queries_utms, 4, return_distance=False
            )

    def get_positives(self):
        return self.soft_positives_per_query
