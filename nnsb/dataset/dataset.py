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


class Data(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        dataset_name,
        resize=224,
        limit=None,
        superpoint=False,
    ):
        super().__init__()
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset folder {dataset_dir} not found.")
        self.superpoint = superpoint
        self.resize = resize
        self.transform = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
            ]
        )

        database_dir = dataset_dir / dataset_name / "images/test" / "database"
        self.database_paths = sorted(database_dir.glob("*.png")) + sorted(
            database_dir.glob("*.jpg")
        )
        if limit is not None:
            self.database_paths = self.database_paths[:limit]

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

    def __getitem__(self, index):
        img = path_to_pil_img(self.database_paths[index])
        img = self.transform(img)
        latitude = float(self.database_paths[index].stem.split("@")[1])
        longitude = float(self.database_paths[index].stem.split("@")[2])
        # img = img.permute((1, 2, 0))
        return img, (latitude, longitude)

    def __len__(self):
        return len(self.database_paths)

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
        super().__init__()
        queries_dir = dataset_dir / dataset_name / "images/test" / "queries"
        if not queries_dir.exists():
            raise FileNotFoundError(f"Queries folder {queries_dir} not found.")

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
            ])

        self.queries_paths = sorted(queries_dir.glob("*.png")) + sorted(
            queries_dir.glob("*.jpg")
        )
        if limit is not None:
            self.queries_paths = self.queries_paths[:limit]
        self.queries_num = len(self.queries_paths)

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

    def __getitem__(self, index):
        img = path_to_pil_img(self.queries_paths[index])
        img = self.transform(img)
        latitude = float(self.queries_paths[index].stem.split("@")[1])
        longitude = float(self.queries_paths[index].stem.split("@")[2])
        # img = img.permute((1, 2, 0))
        return img, (latitude, longitude)

    def __len__(self):
        return self.queries_num

    def __repr__(self):
        return f"< {self.__class__.__name__}, - #queries: {self.queries_num};>"

    def get_positives(self):
        return self.soft_positives_per_query
