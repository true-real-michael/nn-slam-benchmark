import torch

from nnsb.feature_detectors.superpoint.model import SuperPoint as Model
from nnsb.utils import transform_image_for_sp
import numpy as np


class SuperPoint:
    def __init__(self, resize):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resize = resize
        self.model = Model().to(self.device).eval()

    def __call__(self, image):
        image = image.cpu().numpy()
        image = np.transpose(image, [1, 2, 0])
        image = transform_image_for_sp(image, self.resize).to(self.device)
        image = image[None]
        with torch.no_grad():
            result = self.model({"image": image})
        result["image_size"] = (
            torch.tensor((self.resize, self.resize)).to(image).float()
        )
        return result
        # return {k: v.cpu().numpy() for k, v in result.items()}
