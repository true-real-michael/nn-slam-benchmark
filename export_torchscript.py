from pathlib import Path
from nnsb.vpr_systems.salad.salad import SALAD
from nnsb.vpr_systems.sela.sela import Sela
from nnsb.vpr_systems.mixvpr.mixvpr import MixVPR

sela = Sela('weights/SelaVPR_msls.pth', 'weights/dinov2_vitl14_pretrain.pth')
sela.export_torchscript(Path('weights/torchscript/200/sela.pt'))

mixvpr = MixVPR("weights/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt")
mixvpr.export_torchscript(Path("weights/torchscript/300/mixvpr.pt"))

salads = [(resize, SALAD(resize)) for resize in [800, 600, 400, 300, 200]]
for resize, salad in salads:
    salad.export_torchscript(Path(f'weights/torchscript/{resize}/salad.pt'))
