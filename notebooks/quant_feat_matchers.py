import sys
sys.path.append('..')
sys.path.append('/home/aias-racing/mkiselyov/nn-slam-benchmark')
import torch
# from torch.backends import cudnn
from torch.utils import data

import pytorch_quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization import calib
from tqdm import tqdm

from nnsb.feature_matchers.lightglue.model.lightglue_matcher import LightGlueMatcher
from nnsb.feature_detectors.superpoint.superpoint import SuperPoint
from nnsb.dataset.dataset import Data

print(pytorch_quantization.__version__)

from pathlib import Path


superpoint = None



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

quant_modules.initialize()


def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax(strict=False, )
                else:
                    module.load_calib_amax(strict=False, **kwargs)
    model.cuda()

def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistics"""
    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # Feed data to the network for collecting stats
    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        # model({'image': image.cuda()})
        desc0 = superpoint(image[0].cuda())
        desc0['descriptors'] = desc0['descriptors'].transpose(-1, -2).contiguous()
        desc1 = superpoint(image[1].cuda())
        desc1['descriptors'] = desc1['descriptors'].transpose(-1, -2).contiguous()
        model({'image0': desc0, 'image1': desc1})
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def quantize_model(model, output_path, resize):
    train_dataset = Data(Path("datasets"), "st_lucia", gt=False, resize=resize)
    # train_dataloader = data.DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    train_dataloader = data.DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)

    set_parameter_requires_grad(model, feature_extracting=True)
    model = model.cuda()

    # Calibrate the model using max calibration technique.
    with torch.no_grad():
        collect_stats(model, train_dataloader, num_batches=16*16)
        # collect_stats(model, train_dataloader, num_batches=16)
        compute_amax(model, method="max")
    
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    # Exporting to ONNX
    # dummy_input = torch.randn(1, 3, resize, resize, device='cuda')
    dummy_input = torch.randn(1, 1, resize, resize, device='cuda')
    # dummy_input = {'data':{'image': torch.randn(1, 1, resize, resize, device='cuda')}}

    images, _ = next(iter(train_dataloader))
    desc0 = superpoint(images[0].cuda())
    desc0['descriptors'] = desc0['descriptors'].transpose(-1, -2).contiguous()
    desc1 = superpoint(images[1].cuda())
    desc1['descriptors'] = desc1['descriptors'].transpose(-1, -2).contiguous()
    dummy_input = {'data': {
        'image0': desc0,
        'image1': desc1,
    }}

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        verbose=False,
        opset_version=16,
        # opset_version=13,
        do_constant_folding = False)
    
    # subprocess.run([
    #     "/usr/src/tensorrt/bin/trtexec",
    #     "--onnx=/tmp/model_q.onnx",
    #     "--int8",
    #     f"--saveEngine={output_path}"
    # ])


failed = []
for resize in [200, 300, 400, 600, 800]:
    outputut_path = Path(f"weights/quant/{resize}")
    superpoint = SuperPoint(resize)

    # if resize <= 400:
    #     quantize_model(SALAD(resize // 14 * 14).model, outputut_path / "salad.onnx", resize // 14 * 14)
    # quantize_model(EigenPlaces(resize=resize).model, outputut_path / "eigenplaces.onnx", resize)
    # quantize_model(CosPlace(resize=resize).model, outputut_path / "cosplace.onnx", resize)
    # quantize_model(NetVLAD("../weights/mapillary_WPCA4096.pth.tar", resize).get_torch_module(), outputut_path / "netvlad.onnx", resize)
    # quantize_model(SuperPoint(), outputut_path / "superpoint.onnx", resize)
    quantize_model(LightGlueMatcher(), outputut_path / "lightglue.onnx", resize)
    # if resize == 200:
    #     quantize_model(Sela("../weights/SelaVPR_msls.pth", "../weights/dinov2_vitl14_pretrain.pth").get_torch_module(), outputut_path / "sela.onnx", 224)
    
    # if resize == 300:
    #     quantize_model(MixVPR("../weights/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt").model, outputut_path / "mixvpr.onnx", 320)
        
print(*failed, sep='\n')
    
