import os
import sys
import torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from models.sdcr_net import SDCRNet

def main():
    torch.manual_seed(0)
    model = SDCRNet(num_classes=20, image_size=224, vit_input_size=224, token_keep=True, keep_ratio=0.3, pretrained=False, num_reinject=3, reinject_scale=0.3, fusion_weight=0.5, freeze_fusion=True, multilabel=True)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    vit_score = torch.rand(2, 3, 224, 224)
    vit_norm = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(x, vit_norm=vit_norm, vit_score=vit_score)
    n_keep = model.fast_encoder.last_token_info['num_kept_patches']
    print(f'ok  logits={tuple(logits.shape)}  kept_patches={n_keep}  params={sum((p.numel() for p in model.parameters())) / 1000000.0:.1f}M')
if __name__ == '__main__':
    main()
