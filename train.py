import argparse
import csv
import json
import math
import os
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, ReduceLROnPlateau, StepLR
from tqdm import tqdm
from data.dataset import build_dataloaders, unpack_batch
from loss import ClassificationLoss
from metrics import calculate_all_multilabel_metrics, calculate_multiclass_metrics
from models.sdcr_net import SDCRNet, load_sdcr_checkpoint
from utils import set_seed

def parse_args():
    p = argparse.ArgumentParser(description='Train SDCR-Net')
    p.add_argument('--dataset', type=str, default='mured', choices=['idrid', 'mured'])
    p.add_argument('--data_dir', type=str, default=None)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--idrid_task_mode', type=str, default='dme', choices=['binary', '5class', 'dme', 'dme_binary'])
    p.add_argument('--idrid_crop_black_borders', action='store_true', default=True)
    p.add_argument('--no_idrid_crop_black_borders', action='store_true', default=False)
    p.add_argument('--idrid_pad_after_crop', action='store_true', default=True)
    p.add_argument('--mured_pad_to_square', action='store_true', default=True)
    p.add_argument('--save_path', type=str, default='outputs/best.pth')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--token_keep', action='store_true', default=True)
    p.add_argument('--no_token_keep', action='store_true', default=False)
    p.add_argument('--vit_input_size', type=int, default=512)
    p.add_argument('--image_size', type=int, default=512)
    p.add_argument('--keep_ratio', type=float, default=0.1)
    p.add_argument('--selector_type', type=str, default='l2', choices=['l2'])
    p.add_argument('--reinject_down_dim', type=int, default=256)
    p.add_argument('--reinject_num_heads', type=int, default=8)
    p.add_argument('--num_reinject', type=int, default=3)
    p.add_argument('--reinject_scale', type=float, default=0.5)
    p.add_argument('--pool_mode', type=str, default='cls', choices=['cls', 'mean'])
    p.add_argument('--fusion_weight', type=float, default=0.5)
    p.add_argument('--freeze_fusion', action='store_true', default=False)
    p.add_argument('--learn_fusion', action='store_true', default=True)
    p.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'warmup_cosine', 'step', 'plateau', 'constant'])
    p.add_argument('--warmup_epochs', type=int, default=5)
    p.add_argument('--step_size', type=int, default=20)
    p.add_argument('--lr_gamma', type=float, default=0.1)
    p.add_argument('--plateau_patience', type=int, default=3)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--early_stop_patience', type=int, default=10)
    return p.parse_args()

def build_scheduler(name: str, optimizer, epochs: int, warmup_epochs: int=5, step_size: int=20, lr_gamma: float=0.1, plateau_patience: int=3):
    name = str(name).lower()
    if name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if name == 'warmup_cosine':
        warmup = max(0, int(warmup_epochs))

        def lr_lambda(epoch: int):
            if warmup > 0 and epoch < warmup:
                return float(epoch + 1) / float(warmup)
            progress = (epoch - warmup) / float(max(1, epochs - warmup))
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return LambdaLR(optimizer, lr_lambda)
    if name == 'step':
        return StepLR(optimizer, step_size=max(1, int(step_size)), gamma=float(lr_gamma))
    if name == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='max', factor=float(lr_gamma), patience=max(1, int(plateau_patience)))
    if name == 'constant':
        return LambdaLR(optimizer, lambda epoch: 1.0)
    raise ValueError(f'unknown lr_scheduler {name}')

def resolve_data_dir(dataset: str, data_dir: str):
    if data_dir:
        return data_dir
    if dataset == 'mured':
        return os.path.join('data', 'MuReD')
    return os.path.join('data', 'IDRiD')

def _jsonable(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _jsonable(v) for (k, v) in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj

def run_epoch(model, loader, criterion, optimizer, device, multilabel, pbar_desc='train'):
    model.train()
    loss_sum = 0.0
    n = 0
    y_true_all = []
    y_pred_all = []
    y_prob_all = []
    bar = tqdm(loader, desc=pbar_desc, leave=False, dynamic_ncols=True, mininterval=0.3)
    for (images, targets) in bar:
        (x_cnn, vit_score, vit_norm) = unpack_batch(images)
        x_cnn = x_cnn.to(device, non_blocking=True)
        if vit_score is not None:
            vit_score = vit_score.to(device, non_blocking=True)
        if vit_norm is not None:
            vit_norm = vit_norm.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_cnn, vit_norm=vit_norm, vit_score=vit_score)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        b = x_cnn.size(0)
        n += b
        loss_sum += loss.item() * b
        if multilabel:
            prob = torch.sigmoid(logits.detach()).cpu().numpy()
            pred = (prob >= 0.5).astype(np.int32)
            y_prob_all.append(prob)
            y_pred_all.append(pred)
            y_true_all.append(targets.detach().cpu().numpy().astype(np.int32))
        else:
            prob = torch.softmax(logits.detach(), dim=-1).cpu().numpy()
            pred = prob.argmax(axis=1)
            y_prob_all.append(prob)
            y_pred_all.append(pred)
            y_true_all.append(targets.detach().cpu().numpy().astype(np.int64))
        bar.set_postfix(loss=f'{loss.item():.4f}')
    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    y_prob = np.concatenate(y_prob_all, axis=0)
    if multilabel:
        m = calculate_all_multilabel_metrics(y_true, y_prob, y_pred, exclude_normal=True)
        return (loss_sum / max(n, 1), m)
    num_classes = int(y_prob.shape[1])
    m = calculate_multiclass_metrics(y_true, y_pred, y_prob, num_classes=num_classes)
    return (loss_sum / max(n, 1), m)

@torch.no_grad()
def validate(model, loader, criterion, device, multilabel, pbar_desc='val'):
    model.eval()
    loss_sum = 0.0
    n = 0
    y_true_all = []
    y_pred_all = []
    y_prob_all = []
    bar = tqdm(loader, desc=pbar_desc, leave=False, dynamic_ncols=True, mininterval=0.3)
    for (images, targets) in bar:
        (x_cnn, vit_score, vit_norm) = unpack_batch(images)
        x_cnn = x_cnn.to(device, non_blocking=True)
        if vit_score is not None:
            vit_score = vit_score.to(device, non_blocking=True)
        if vit_norm is not None:
            vit_norm = vit_norm.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(x_cnn, vit_norm=vit_norm, vit_score=vit_score)
        loss = criterion(logits, targets)
        b = x_cnn.size(0)
        n += b
        loss_sum += loss.item() * b
        if multilabel:
            prob = torch.sigmoid(logits.detach()).cpu().numpy()
            pred = (prob >= 0.5).astype(np.int32)
            y_prob_all.append(prob)
            y_pred_all.append(pred)
            y_true_all.append(targets.detach().cpu().numpy().astype(np.int32))
        else:
            prob = torch.softmax(logits.detach(), dim=-1).cpu().numpy()
            pred = prob.argmax(axis=1)
            y_prob_all.append(prob)
            y_pred_all.append(pred)
            y_true_all.append(targets.detach().cpu().numpy().astype(np.int64))
        bar.set_postfix(loss=f'{loss.item():.4f}')
    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    y_prob = np.concatenate(y_prob_all, axis=0)
    if multilabel:
        m = calculate_all_multilabel_metrics(y_true, y_prob, y_pred, exclude_normal=True)
        return (loss_sum / max(n, 1), m)
    num_classes = int(y_prob.shape[1])
    m = calculate_multiclass_metrics(y_true, y_pred, y_prob, num_classes=num_classes)
    return (loss_sum / max(n, 1), m)

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = resolve_data_dir(args.dataset, args.data_dir)
    if args.lr is None:
        args.lr = 5e-05 if args.dataset == 'mured' else 3e-05
    token_keep = False if args.no_token_keep else args.token_keep
    freeze_fusion = True if args.freeze_fusion else False
    idrid_crop = False if args.no_idrid_crop_black_borders else args.idrid_crop_black_borders
    image_size = int(args.image_size) if args.image_size is not None else int(args.vit_input_size)
    (train_loader, val_loader, num_classes, multilabel) = build_dataloaders(dataset=args.dataset, data_dir=data_dir, batch_size=args.batch_size, num_workers=args.num_workers, idrid_task_mode=args.idrid_task_mode, idrid_crop_black_borders=idrid_crop, idrid_pad_after_crop=args.idrid_pad_after_crop, mured_pad_to_square=args.mured_pad_to_square, vit_input_size=args.vit_input_size, cnn_image_size=image_size)
    model = SDCRNet(num_classes=num_classes, image_size=image_size, vit_input_size=args.vit_input_size, fast_model='vit_base_patch16_224', slow_model='convnext_base', dropout=0.1, multilabel=multilabel, token_keep=token_keep, keep_ratio=args.keep_ratio, selector_type=args.selector_type, pretrained=True, reinject_down_dim=args.reinject_down_dim, reinject_num_heads=args.reinject_num_heads, num_reinject=args.num_reinject, reinject_scale=args.reinject_scale, pool_mode=args.pool_mode, fusion_weight=args.fusion_weight, freeze_fusion=freeze_fusion).to(device)
    if args.resume:
        load_sdcr_checkpoint(model, args.resume, strict=True)
    print(f'SDCR-Net  image_size={image_size}  vit={args.vit_input_size}  token_keep={token_keep}  keep_ratio={args.keep_ratio}  num_reinject={args.num_reinject}  reinject_scale={args.reinject_scale}  fusion_w={float(args.fusion_weight):.3f}  lr={args.lr}  scheduler={args.lr_scheduler}  seed={args.seed}', flush=True)
    criterion = ClassificationLoss(num_classes=num_classes, multilabel=multilabel)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(args.lr_scheduler, optimizer, epochs=args.epochs, warmup_epochs=args.warmup_epochs, step_size=args.step_size, lr_gamma=args.lr_gamma, plateau_patience=args.plateau_patience)
    if args.lr_scheduler == 'warmup_cosine':
        scheduler.step()
    best_score = -1.0
    best_val_metrics = None
    epochs_no_improve = 0
    stopped_epoch = args.epochs
    history = []
    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    metrics_path = os.path.join(os.path.dirname(args.save_path) or '.', 'metrics.json')
    history_path = os.path.join(os.path.dirname(args.save_path) or '.', 'history.csv')

    def _current_fusion_weight():
        return float(model.fusion_weight.detach().cpu())
    for epoch in range(1, args.epochs + 1):
        (train_loss, train_metrics) = run_epoch(model, train_loader, criterion, optimizer, device, multilabel)
        (val_loss, val_metrics) = validate(model, val_loader, criterion, device, multilabel)
        lr_now = float(optimizer.param_groups[0]['lr'])
        fusion_now = _current_fusion_weight()
        if multilabel:
            train_score = train_metrics['ml_map'] + train_metrics['ml_f1'] + train_metrics['ml_auc']
            val_score = val_metrics['ml_map'] + val_metrics['ml_f1'] + val_metrics['ml_auc']
            print(f"Epoch {epoch:03d}/{args.epochs:03d} | lr {lr_now:.2e} | train_loss {train_loss:.4f} ml_map {train_metrics['ml_map']:.4f} ml_f1 {train_metrics['ml_f1']:.4f} ml_auc {train_metrics['ml_auc']:.4f} of1 {train_metrics['of1']:.4f} | val_loss {val_loss:.4f} ml_map {val_metrics['ml_map']:.4f} ml_f1 {val_metrics['ml_f1']:.4f} ml_auc {val_metrics['ml_auc']:.4f} of1 {val_metrics['of1']:.4f}")
        else:
            train_score = train_metrics['accuracy']
            val_score = val_metrics['accuracy']
            train_auc = train_metrics['auc'] if train_metrics['auc'] is not None else 0.0
            val_auc = val_metrics['auc'] if val_metrics['auc'] is not None else 0.0
            print(f"Epoch {epoch:03d}/{args.epochs:03d} | lr {lr_now:.2e} | fusion_w {fusion_now:.3f} | train_loss {train_loss:.4f} acc {train_metrics['accuracy']:.4f} p {train_metrics['precision']:.4f} r {train_metrics['recall']:.4f} f1 {train_metrics['f1']:.4f} auc {train_auc:.4f} | val_loss {val_loss:.4f} acc {val_metrics['accuracy']:.4f} p {val_metrics['precision']:.4f} r {val_metrics['recall']:.4f} f1 {val_metrics['f1']:.4f} auc {val_auc:.4f}")
        history.append({'epoch': epoch, 'lr': lr_now, 'fusion_weight': fusion_now, 'train_loss': train_loss, 'val_loss': val_loss, 'train_score': train_score, 'val_score': val_score})
        if val_score > best_score:
            best_score = val_score
            best_val_metrics = val_metrics
            epochs_no_improve = 0
            torch.save({'model_state_dict': model.state_dict(), 'best_val_score': best_score, 'arch': 'sdcr', 'dataset': args.dataset, 'image_size': image_size, 'vit_input_size': args.vit_input_size, 'token_keep': token_keep, 'keep_ratio': args.keep_ratio, 'selector_type': args.selector_type, 'fast_model': 'vit_base_patch16_224', 'slow_model': 'convnext_base', 'reinject_down_dim': args.reinject_down_dim, 'num_reinject': args.num_reinject, 'reinject_scale': args.reinject_scale, 'pool_mode': args.pool_mode, 'fusion_weight': fusion_now, 'freeze_fusion': bool(freeze_fusion), 'idrid_crop_black_borders': idrid_crop, 'idrid_pad_after_crop': args.idrid_pad_after_crop, 'lr': args.lr, 'lr_scheduler': args.lr_scheduler, 'seed': args.seed, 'early_stop_patience': args.early_stop_patience}, args.save_path)
        else:
            epochs_no_improve += 1
        if args.lr_scheduler == 'plateau':
            scheduler.step(val_score)
        else:
            scheduler.step()
        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            stopped_epoch = epoch
            print(f'early stop at epoch {epoch:03d} (patience={args.early_stop_patience}, best_val_score {best_score:.4f})')
            break
    if history:
        with open(history_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
    result = {'best_val_score': best_score, 'best_val_metrics': _jsonable(best_val_metrics or {}), 'epochs': args.epochs, 'stopped_epoch': stopped_epoch, 'early_stop_patience': args.early_stop_patience, 'batch_size': args.batch_size, 'seed': args.seed, 'lr': args.lr, 'lr_scheduler': args.lr_scheduler, 'num_reinject': args.num_reinject, 'reinject_scale': args.reinject_scale, 'fusion_weight': args.fusion_weight, 'freeze_fusion': bool(freeze_fusion), 'keep_ratio': args.keep_ratio, 'token_keep': token_keep, 'arch': 'sdcr', 'dataset': args.dataset, 'idrid_task_mode': args.idrid_task_mode, 'vit_input_size': args.vit_input_size, 'image_size': image_size, 'ckpt': args.save_path}
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'best_val_score {best_score:.4f}')
    print(f'wrote {metrics_path}', flush=True)
if __name__ == '__main__':
    main()
