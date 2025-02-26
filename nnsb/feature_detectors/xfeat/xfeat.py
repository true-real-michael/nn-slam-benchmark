import numpy as np
import torch

from nnsb.utils import transform_image_for_sp


class XFeat:
    def __init__(self, resize):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.resize = resize
        self.xfeat = torch.hub.load(
            "verlab/accelerated_features", "XFeat", pretrained=True, top_k=4096
        )
        self.xfeat = self.xfeat.to(self.device).eval()

    def __call__(self, image: np.ndarray):
        image = transform_image_for_sp(image, self.resize)
        with torch.no_grad():
            result = self.xfeat.detectAndCompute(image.to(self.device), top_k=4096)[0]
        result["image_size"] = (
            torch.tensor((self.resize, self.resize)).to(image).float()
        )
        return {k: v.cpu().numpy() for k, v in result.items()}
