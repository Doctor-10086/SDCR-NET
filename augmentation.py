import numpy as np
import torch
import albumentations as A
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def _mean_std_tensors():
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (mean, std)

def _to_chw_01(image_hwc: np.ndarray) -> torch.Tensor:
    x = np.asarray(image_hwc)
    if x.ndim != 3:
        raise ValueError(f'expected HWC image, got shape {x.shape}')
    if np.issubdtype(x.dtype, np.integer) or float(np.max(x)) > 1.5:
        x = x.astype(np.float32) / 255.0
    else:
        x = x.astype(np.float32)
    x = np.clip(x, 0.0, 1.0)
    return torch.from_numpy(np.transpose(x, (2, 0, 1))).contiguous()

def _normalize_nchw(x_01: torch.Tensor) -> torch.Tensor:
    (mean, std) = _mean_std_tensors()
    return (x_01 - mean) / std

def _build_albu_train_no_norm(image_size: int):
    dropout_size = max(int(image_size * 0.089), 10)
    min_dropout_size = max(int(image_size * 0.045), 5)
    blur_limit = max(int(image_size * 0.031), 5)
    if blur_limit % 2 == 0:
        blur_limit += 1
    return A.Compose([A.Resize(image_size, image_size), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.Rotate(limit=30, p=0.5), A.MedianBlur(blur_limit=blur_limit, p=0.3), A.GaussNoise(std_range=(0.0, 0.38), p=0.5), A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3), A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.3), A.CoarseDropout(num_holes_range=(1, 5), hole_height_range=(min_dropout_size, dropout_size), hole_width_range=(min_dropout_size, dropout_size), p=0.5)])

def _build_albu_eval_no_norm(image_size: int):
    return A.Compose([A.Resize(image_size, image_size)])

class DualStreamTransform:

    def __init__(self, cnn_size: int, vit_size: int=512, is_training: bool=True):
        self.cnn_size = int(cnn_size)
        self.vit_size = int(vit_size)
        self.is_training = bool(is_training)
        if self.cnn_size <= 0 or self.vit_size <= 0:
            raise ValueError(f'cnn_size and vit_size must be positive, got {cnn_size}, {vit_size}')
        self.aug_size = max(self.cnn_size, self.vit_size)
        self.aug = _build_albu_train_no_norm(self.aug_size) if is_training else _build_albu_eval_no_norm(self.aug_size)
        self._cnn_resize = None if self.cnn_size == self.aug_size else A.Resize(self.cnn_size, self.cnn_size)
        self._vit_resize = None if self.vit_size == self.aug_size else A.Resize(self.vit_size, self.vit_size)

    def __call__(self, image):
        if hasattr(image, 'convert'):
            image_np = np.array(image.convert('RGB'))
        else:
            image_np = np.asarray(image)
        aug_np = self.aug(image=image_np)['image']
        cnn_np = aug_np if self._cnn_resize is None else self._cnn_resize(image=aug_np)['image']
        vit_np = aug_np if self._vit_resize is None else self._vit_resize(image=aug_np)['image']
        cnn_01 = _to_chw_01(cnn_np)
        vit_01 = _to_chw_01(vit_np)
        if tuple(cnn_01.shape[-2:]) != (self.cnn_size, self.cnn_size):
            raise ValueError(f'CNN image is {tuple(cnn_01.shape[-2:])}, expected {(self.cnn_size, self.cnn_size)}.')
        if tuple(vit_01.shape[-2:]) != (self.vit_size, self.vit_size):
            raise ValueError(f'ViT score image is {tuple(vit_01.shape[-2:])}, expected {(self.vit_size, self.vit_size)}.')
        return {'image': _normalize_nchw(cnn_01), 'vit_score': vit_01.clamp(0.0, 1.0), 'vit_norm': _normalize_nchw(vit_01)}

def get_mured_transforms(image_size: int=512, is_training: bool=True, already_square: bool=False, vit_input_size: int=512):
    del already_square
    return DualStreamTransform(cnn_size=image_size, vit_size=vit_input_size, is_training=is_training)

def get_idrid_transforms(image_size: int=512, is_training: bool=True, already_square: bool=False, vit_input_size: int=512):
    return get_mured_transforms(image_size=image_size, is_training=is_training, already_square=already_square, vit_input_size=vit_input_size)
