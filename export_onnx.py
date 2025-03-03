from pathlib import Path
from nnsb.vpr_systems.salad.salad import SALAD
from nnsb.vpr_systems.sela.sela import Sela
from nnsb.vpr_systems.mixvpr.mixvpr import MixVPR


# salad = SALAD(800)
# sela = Sela('weights/SelaVPR_msls.pth', 'weights/dinov2_vitl14_pretrain.pth')
mixvpr = MixVPR('weights/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt')

# salad.export_onnx(Path('weights/onnx/800/salad.onnx'))
# sela.export_onnx(Path('weights/onnx/200/sela.onnx'))
mixvpr.export_onnx(Path('weights/300/mixvpr.onnx'))
