import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, cohen_kappa_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def calculate_multiclass_metrics(y_true, y_pred, y_pred_proba=None, num_classes=5):
    y_true = _to_numpy(y_true).reshape(-1)
    y_pred = _to_numpy(y_pred).reshape(-1)
    precision_per_class = precision_score(y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0)
    out = {'accuracy': float(accuracy_score(y_true, y_pred)), 'kappa': float(cohen_kappa_score(y_true, y_pred)), 'precision': float(precision_score(y_true, y_pred, labels=list(range(num_classes)), average='macro', zero_division=0)), 'recall': float(recall_score(y_true, y_pred, labels=list(range(num_classes)), average='macro', zero_division=0)), 'f1': float(f1_score(y_true, y_pred, labels=list(range(num_classes)), average='macro', zero_division=0)), 'auc': None, 'precision_per_class': precision_per_class, 'recall_per_class': recall_per_class, 'f1_per_class': f1_per_class, 'confusion_matrix': confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))}
    if y_pred_proba is not None:
        try:
            y_pred_proba = _to_numpy(y_pred_proba)
            y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
            if num_classes == 2:
                y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
            out['auc'] = float(roc_auc_score(y_true_bin, y_pred_proba, average='macro', multi_class='ovr'))
        except Exception:
            out['auc'] = None
    return out

def calculate_all_multilabel_metrics(y_true, y_pred_proba, y_pred, exclude_normal=True):
    y_true = _to_numpy(y_true)
    y_pred_proba = _to_numpy(y_pred_proba)
    y_pred = _to_numpy(y_pred)
    num_labels = y_true.shape[1]
    start_idx = 1 if exclude_normal and num_labels > 1 else 0
    label_indices = list(range(start_idx, num_labels))
    ap_scores = []
    auc_scores = []
    f1_scores = []
    for i in label_indices:
        yt = y_true[:, i]
        yp = y_pred_proba[:, i]
        yb = y_pred[:, i]
        if np.sum(yt) > 0:
            try:
                ap_scores.append(float(average_precision_score(yt, yp)))
            except Exception:
                ap_scores.append(0.0)
        else:
            ap_scores.append(0.0)
        if np.sum(yt) > 0 and np.sum(yt) < len(yt):
            try:
                auc_scores.append(float(roc_auc_score(yt, yp)))
            except Exception:
                pass
        f1_scores.append(float(f1_score(yt, yb, zero_division=0)))
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    of1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {'ml_map': float(np.mean(ap_scores) if ap_scores else 0.0), 'ml_f1': float(np.mean(f1_scores) if f1_scores else 0.0), 'ml_auc': float(np.mean(auc_scores) if auc_scores else 0.0), 'of1': float(of1)}
