import torch
import torch.nn as nn
from timm import create_model

class ConvNeXtEncoder(nn.Module):

    def __init__(self, model_name: str='convnext_base', freeze: bool=False, input_image_size: int=512):
        super().__init__()
        self.model_name = model_name
        self.input_image_size = int(input_image_size)
        self.backbone = create_model(model_name, pretrained=True)
        self.hidden_dim = self._get_hidden_dim()
        if hasattr(self.backbone, 'stages'):
            for s in self.backbone.stages:
                if hasattr(s, 'grad_checkpointing'):
                    s.grad_checkpointing = True
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    @property
    def hidden_size(self) -> int:
        return self.hidden_dim

    def _get_hidden_dim(self):
        if hasattr(self.backbone, 'num_features'):
            return int(self.backbone.num_features)
        with torch.no_grad():
            x = torch.randn(1, 3, 224, 224)
            if hasattr(self.backbone, 'stages'):
                x = self.backbone.stem(x)
                for s in self.backbone.stages:
                    x = s(x)
                return int(x.shape[1])
            y = self.backbone.forward_features(x)
            if y.ndim == 4:
                return int(y.shape[1])
            return int(y.shape[-1])

    def forward_features(self, x: torch.Tensor):
        if hasattr(self.backbone, 'stages') and hasattr(self.backbone, 'stem'):
            x = self.backbone.stem(x)
            for stage in self.backbone.stages:
                x = stage(x)
            return x
        feat = self.backbone.forward_features(x)
        if feat.ndim == 3:
            (b, n, c) = feat.shape
            s = int(n ** 0.5)
            feat = feat.transpose(1, 2).reshape(b, c, s, s)
        return feat

    def forward(self, x: torch.Tensor):
        return self.forward_features(x)
