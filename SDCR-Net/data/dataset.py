import os
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from data.augmentation import get_idrid_transforms, get_mured_transforms

def pad_to_square(image: Image.Image, pad_value: int=0) -> Image.Image:
    (w, h) = image.size
    s = max(w, h)
    out = Image.new(image.mode, (s, s), color=pad_value)
    out.paste(image, ((s - w) // 2, (s - h) // 2))
    return out

def crop_idrid_black_borders(image: Image.Image, crop_left: int=250, crop_right: int=500) -> Image.Image:
    (w, h) = image.size
    left = min(crop_left, w - 1)
    right = max(w - crop_right, left + 1)
    return image.crop((left, 0, right, h))

def _find_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def _resolve_image_path(root_dir: str, stem: str) -> str:
    exts = ['', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.PNG', '.JPG', '.JPEG']
    for ext in exts:
        p = os.path.join(root_dir, stem if stem.lower().endswith(tuple((e.lower() for e in exts[1:]))) else stem + ext)
        if os.path.exists(p):
            return p
    stem_base = os.path.splitext(stem)[0]
    for ext in exts[1:]:
        p = os.path.join(root_dir, stem_base + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'image not found: {stem} under {root_dir}')

class MuReDDataset(Dataset):

    def __init__(self, csv_file: str, image_dir: str, transform=None, pad_to_square_enabled: bool=True, pad_value: int=0):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        cols = [c for c in self.df.columns if c != 'ID']
        if 'NORMAL' in cols:
            cols = ['NORMAL'] + [c for c in cols if c != 'NORMAL']
        self.label_cols = cols
        self.pad_to_square_enabled = pad_to_square_enabled
        self.pad_value = pad_value

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(row['ID'])
        image = Image.open(_resolve_image_path(self.image_dir, image_id)).convert('RGB')
        if self.pad_to_square_enabled:
            image = pad_to_square(image, pad_value=self.pad_value)
        labels = row[self.label_cols].values.astype(np.float32)
        if self.transform is not None:
            image = self.transform(image)
        return (image, torch.from_numpy(labels))

class IDRiDDataset(Dataset):

    def __init__(self, csv_file: str, image_dir: str, transform=None, task_mode: str='dme', crop_black_borders: bool=True, pad_after_crop: bool=True):
        self.df = pd.read_csv(csv_file)
        self.df.columns = self.df.columns.str.strip()
        self.image_dir = image_dir
        self.transform = transform
        self.task_mode = task_mode
        self.crop_black_borders = crop_black_borders
        self.pad_after_crop = pad_after_crop

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = str(row['Image name']).strip()
        image = Image.open(_resolve_image_path(self.image_dir, image_name)).convert('RGB')
        if self.crop_black_borders:
            image = crop_idrid_black_borders(image)
        if self.pad_after_crop:
            image = pad_to_square(image)
        if self.task_mode == 'binary':
            grade = int(row['Retinopathy grade'])
            label = 0 if grade == 0 else 1
        elif self.task_mode == '5class':
            label = int(row['Retinopathy grade'])
        elif self.task_mode == 'dme':
            dme_col = [c for c in self.df.columns if 'macular edema' in c.lower() or 'dme' in c.lower()][0]
            label = int(row[dme_col])
        elif self.task_mode == 'dme_binary':
            dme_col = [c for c in self.df.columns if 'macular edema' in c.lower() or 'dme' in c.lower()][0]
            label = 0 if int(row[dme_col]) == 0 else 1
        else:
            raise ValueError(f'unsupported idrid task mode: {self.task_mode}')
        if self.transform is not None:
            image = self.transform(image)
        return (image, torch.tensor(label, dtype=torch.long))

def _find_image_dir(root_dir: str, candidates: List[str]) -> str:
    for c in candidates:
        p = os.path.join(root_dir, c)
        if os.path.isdir(p):
            return p
    for (current, _, files) in os.walk(root_dir):
        if any((f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')) for f in files)):
            return current
    raise FileNotFoundError(f'no image directory found in {root_dir}')

def unpack_batch(images):
    if isinstance(images, dict):
        return (images['image'], images.get('vit_score'), images.get('vit_norm'))
    return (images, None, None)

def build_dataloaders(dataset: str, data_dir: str, batch_size: int, num_workers: int=4, idrid_task_mode: str='dme', idrid_crop_black_borders: bool=True, idrid_pad_after_crop: bool=True, mured_pad_to_square: bool=True, vit_input_size: int=512, cnn_image_size: Optional[int]=None) -> Tuple[DataLoader, DataLoader, int, bool]:
    dataset = dataset.lower()
    cnn_size = int(cnn_image_size) if cnn_image_size is not None else int(vit_input_size)
    if dataset == 'mured':
        train_csv = _find_existing([os.path.join(data_dir, 'train_split.csv'), os.path.join(data_dir, 'train_data.csv')])
        val_csv = _find_existing([os.path.join(data_dir, 'val_split.csv'), os.path.join(data_dir, 'val_data.csv'), os.path.join(data_dir, 'test_data.csv')])
        if train_csv is None or val_csv is None:
            raise FileNotFoundError('MuReD csv not found')
        train_img_dir = _find_image_dir(data_dir, ['train_images', 'images', 'train'])
        val_img_dir = _find_image_dir(data_dir, ['test_images', 'val_images', 'images', 'test', 'val'])
        train_ds = MuReDDataset(train_csv, train_img_dir, transform=get_mured_transforms(image_size=cnn_size, is_training=True, already_square=mured_pad_to_square, vit_input_size=vit_input_size), pad_to_square_enabled=mured_pad_to_square)
        val_ds = MuReDDataset(val_csv, val_img_dir, transform=get_mured_transforms(image_size=cnn_size, is_training=False, already_square=mured_pad_to_square, vit_input_size=vit_input_size), pad_to_square_enabled=mured_pad_to_square)
        multilabel = True
        num_classes = len(train_ds.label_cols)
    elif dataset == 'idrid':
        train_csv = _find_existing([os.path.join(data_dir, 'train_split.csv'), os.path.join(data_dir, '2. Groundtruths', 'a. IDRiD_Disease Grading_Training Labels.csv'), os.path.join(data_dir, 'a. IDRiD_Disease Grading_Training Labels.csv')])
        val_csv = _find_existing([os.path.join(data_dir, 'val_split.csv'), os.path.join(data_dir, '2. Groundtruths', 'b. IDRiD_Disease Grading_Testing Labels.csv'), os.path.join(data_dir, 'b. IDRiD_Disease Grading_Testing Labels.csv')])
        if train_csv is None or val_csv is None:
            raise FileNotFoundError('IDRiD csv not found')
        train_img_dir = _find_image_dir(data_dir, ['1. Original Images/a. Training Set', 'a. Training Set', 'train_images'])
        val_img_dir = _find_image_dir(data_dir, ['1. Original Images/b. Testing Set', 'b. Testing Set', 'test_images'])
        train_ds = IDRiDDataset(train_csv, train_img_dir, transform=get_idrid_transforms(image_size=cnn_size, is_training=True, already_square=idrid_pad_after_crop, vit_input_size=vit_input_size), task_mode=idrid_task_mode, crop_black_borders=idrid_crop_black_borders, pad_after_crop=idrid_pad_after_crop)
        val_ds = IDRiDDataset(val_csv, val_img_dir, transform=get_idrid_transforms(image_size=cnn_size, is_training=False, already_square=idrid_pad_after_crop, vit_input_size=vit_input_size), task_mode=idrid_task_mode, crop_black_borders=idrid_crop_black_borders, pad_after_crop=idrid_pad_after_crop)
        multilabel = False
        if idrid_task_mode in ('binary', 'dme_binary'):
            num_classes = 2
        elif idrid_task_mode == '5class':
            num_classes = 5
        else:
            num_classes = 3
    else:
        raise ValueError('dataset must be mured or idrid')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return (train_loader, val_loader, num_classes, multilabel)
