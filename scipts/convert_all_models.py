from pathlib import Path
from aero_vloc import CosPlace, EigenPlaces, MixVPR, NetVLAD

RESIZE = 800
WEIGHTS = Path('weights')

MODELS = [
    ('netvlad', NetVLAD, {'path_to_weights': WEIGHTS / 'mapillary_WPCA4096.pth.tar', 'resize': RESIZE}),
    ('cosplace', CosPlace, {'resize': RESIZE}),
    ('eigenplaces', EigenPlaces, {'resize': RESIZE}),
    ('mixvpr', MixVPR, {'ckpt_path': WEIGHTS / 'resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt'}),
]

for name, model, kwargs in MODELS:
    output_file = WEIGHTS / 'rknn' / f'{RESIZE}' / f'{name}.rknn'
    if not output_file.exists():
        model = model(**kwargs)
        model.export_rknn()
