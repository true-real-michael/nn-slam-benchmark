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

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def path_to_pil_img(path):
    """Convert a file path to a PIL image.

    Args:
        path: Path to the image file.

    Returns:
        PIL.Image: Loaded image in RGB format.
    """
    return Image.open(path).convert("RGB")


class BaseDataset(torch.utils.data.Dataset):
    """Base dataset class for image loading and preprocessing.

    This class provides functionality for loading images from a directory
    and applying transformations.

    Attributes:
        transform: Composition of image transformations.
        images_paths: List of paths to image files.
    """

    def __init__(
        self,
        images_dir: Path,
        resize=224,
        limit=None,
        superpoint=False,
    ):
        """Initializes the BaseDataset.

        Args:
            images_dir: Directory containing image files.
            resize: Image size after resizing (default: 224).
            limit: Maximum number of images to load (default: None).
            superpoint: If True, uses transformations suitable for SuperPoint (default: False).

        Raises:
            FileNotFoundError: If the images directory doesn't exist.
        """
        super().__init__()
        if not images_dir.exists():
            raise FileNotFoundError(f"Images folder {images_dir} not found.")

        if not superpoint:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(resize),
                    transforms.CenterCrop(resize),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(resize),
                    transforms.CenterCrop(resize),
                    transforms.ToTensor(),
                    transforms.Grayscale(),
                ]
            )

        self.images_paths = sorted(images_dir.glob("*.png")) + sorted(
            images_dir.glob("*.jpg")
        )
        if limit is not None:
            self.images_paths = self.images_paths[:limit]

    def __getitem__(self, index):
        """Get image and its geolocation at the specified index.

        Args:
            index: Index of the image to retrieve.

        Returns:
            Tuple containing the transformed image and its (latitude, longitude).
        """
        img = path_to_pil_img(self.images_paths[index])
        img = self.transform(img)
        latitude = float(self.images_paths[index].stem.split("@")[1])
        longitude = float(self.images_paths[index].stem.split("@")[2])
        return img, (latitude, longitude)

    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.images_paths)


class Data(BaseDataset):
    """Dataset class for database images.

    This class loads images from the database subset of a dataset.
    """

    def __init__(
        self,
        dataset_dir: Path,
        dataset_name,
        resize=224,
        limit=None,
        superpoint=False,
    ):
        """Initializes the Data dataset.

        Args:
            dataset_dir: Root directory of datasets.
            dataset_name: Name of the specific dataset.
            resize: Image size after resizing (default: 224).
            limit: Maximum number of images to load (default: None).
            superpoint: If True, uses transformations suitable for SuperPoint (default: False).
        """
        super().__init__(
            dataset_dir / dataset_name / "images/test/database",
            resize,
            limit,
            superpoint,
        )

    def get_knn(self, k_neighbors=5):
        """Creates a k-nearest neighbor model from database coordinates.

        Args:
            k_neighbors: Number of neighbors to consider (default: 5).

        Returns:
            sklearn.neighbors.NearestNeighbors: Trained KNN model.
        """
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


class Queries(BaseDataset):
    """Dataset class for query images.

    This class loads images from the queries subset of a dataset and
    can compute positive matches based on geolocation.

    Attributes:
        knn: Nearest neighbor model for finding positive matches.
        queries_utms: Array of query coordinates.
        soft_positives_per_query: List of positive matches for each query.
    """

    def __init__(
        self,
        dataset_dir: Path,
        dataset_name,
        knn,
        resize=224,
        limit=None,
        superpoint=False,
    ):
        """Initializes the Queries dataset.

        Args:
            dataset_dir: Root directory of datasets.
            dataset_name: Name of the specific dataset.
            knn: Nearest neighbor model for finding positive matches.
            resize: Image size after resizing (default: 224).
            limit: Maximum number of images to load (default: None).
            superpoint: If True, uses transformations suitable for SuperPoint (default: False).
        """
        super().__init__(
            dataset_dir / dataset_name / "images/test/queries",
            resize,
            limit,
            superpoint,
        )
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
        """Returns positive matches for each query.

        Returns:
            List of positive matches indices for each query.
        """
        return self.soft_positives_per_query
