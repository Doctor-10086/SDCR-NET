# SDCR-NET
SDCR-Net: A Sparse--Dense Complementary Reinjection Network for Retinal Disease Classification
# SDCR-Net

PyTorch code for **SDCR-Net: A Sparse--Dense Complementary Reinjection Network for Retinal Disease Classification**.

Sparse ViT-B/16 models global context; dense ConvNeXt-B extracts spatial details. **EGTP** prunes tokens by patch \(L_2\) energy before the Transformer. **GACR** reinjects full-grid convolutional features into the sparse sequence.

Code: [https://github.com/Doctor-10086/SDCR-NET](https://github.com/Doctor-10086/SDCR-NET)

## Framework

Place the paper figure at `figures/framework.png`:

<p align="center">
  <img src="figures/framework.png" alt="Overview of SDCR-Net" width="100%">
</p>

- **(a)** Sparse–dense complementary encoding
- **(b)** EGTP: parameter-free token selection before Transformer encoding
- **(c)** GACR: full-grid convolutional reinjection with sequence length \(k+1\)

## Method

The same \(512\times 512\) image is fed to ViT and ConvNeXt. Classification loss is applied only to the fused logits.

| Module | Full name | Role |
|---|---|---|
| **EGTP** | Energy-Guided Token Pruning | Scores patches on **unnormalized [0, 1]** pixels: \(s_j=\|\mathrm{vec}(p_j)\|_2\). Keep \(k=\max(1,\lfloor KN\rfloor)\) highest-energy patches and sort by raster index. CLS is always kept. No extra scoring parameters. |
| **Sparse ViT** | ViT-B/16 | Self-attention on \(k+1\) tokens. \(N=1024\) at \(512\) input. |
| **Dense CNN** | ConvNeXt-B | Full spatial map; not cropped by \(K\). |
| **GACR** | Grid-Aligned Complementary Reinjection | Bilinear-align CNN to \(H_p\times W_p\), then \(1\times1\)–GELU–\(1\times1\) to \(D=768\). Q = sparse ViT (with CLS); K/V = full grid (including dropped cells). Q/K down-projected to \(256\), \(8\) heads. \(Z\leftarrow Z+\alpha W_o O\). Inserted after layers **4 / 6 / 8**. |
| **Logit Fusion** | — | \(z=w z_{\mathrm{v}}+(1-w)z_{\mathrm{c}}\), \(w=0.5\). |

Loss: **BCEWithLogits** on fused logits (MuReD multi-label; IDRiD one-hot multi-class).

## Experimental settings

As in the paper (NVIDIA RTX 3080 Ti):

| Item | Value |
|---|---|
| Input size | \(512\times 512\) |
| Backbones | ViT-B/16 + ConvNeXt-B |
| Token retention \(K\) | \(0.1\) |
| Reinjection strength \(\alpha\) | \(0.5\) |
| GACR positions | after ViT layers 4, 6, 8 |
| Fusion weight \(w\) | \(0.5\) |
| Dropout | \(0.1\) |
| Optimizer | AdamW |
| Batch size | \(8\) |
| Max epochs | \(60\) |
| Learning rate | MuReD \(5\times 10^{-5}\), IDRiD \(3\times 10^{-5}\) |
| Scheduler | cosine annealing |
| Augmentation | same as CG-Tran (Yang et al., 2025): flip, rotate, blur, noise, color jitter, CoarseDropout |

`num_reinject=0` is EGTP-only. `--no_token_keep` is GACR-only.

## Environment

```bash
pip install -r requirements.txt
```

CUDA PyTorch is required. `timm` downloads ViT-B/16 and ConvNeXt-B weights on the first training run.

```bash
python scripts/check_forward.py
```

## Data

### MuReD

Download: [Mendeley Data — Multi-Label Retinal Diseases (MuReD)](https://data.mendeley.com/datasets/pc4mb3h8hz/1)

2,208 images, 20 disease labels. Official splits: `train_data.csv` / `val_data.csv`.

```
data/MuReD/
  train_data.csv
  val_data.csv
  images/
```

```bash
ZIP=/path/to/MuReD.zip DST=./data/MuReD bash scripts/prepare_mured.sh
```

The CSV must contain an `ID` column. If `NORMAL` exists, it is placed first; metrics exclude NORMAL by default.

### IDRiD (DME grading)

Download: [IDRiD Grand Challenge](https://idrid.grand-challenge.org/Home/)

516 high-resolution fundus images. Default task: **3-class DME** (`--idrid_task_mode dme`).

```
data/IDRiD/
  1. Original Images/a. Training Set/
  1. Original Images/b. Testing Set/
  2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv
  2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv
```

Left/right black borders are cropped, then the image is padded to a square.

## Training

Defaults match the paper: \(K=0.1\), \(\alpha=0.5\), \(512\times 512\).

```bash
bash scripts/train_mured.sh
bash scripts/train_idrid.sh
```

Equivalent command:

```bash
python train.py \
  --dataset mured \
  --data_dir ./data/MuReD \
  --vit_input_size 512 \
  --image_size 512 \
  --keep_ratio 0.1 \
  --reinject_scale 0.5 \
  --num_reinject 3 \
  --fusion_weight 0.5 \
  --lr 5e-5 \
  --lr_scheduler cosine \
  --epochs 60 \
  --batch_size 8 \
  --save_path outputs/sdcr_mured_512/best.pth
```

Checkpoint selection: MuReD uses `mAP + mF1 + AUC`; IDRiD uses accuracy. Writes `metrics.json` and `history.csv`.

## Arguments

| Argument | Meaning | Paper default |
|---|---|---|
| `--keep_ratio` | EGTP retention \(K\) | `0.1` |
| `--reinject_scale` | GACR residual \(\alpha\) | `0.5` |
| `--num_reinject` | number of reinjections; `0` = EGTP-only | `3` |
| `--no_token_keep` | disable EGTP (GACR-only) | off |
| `--fusion_weight` | ViT weight \(w\) | `0.5` |
| `--vit_input_size` / `--image_size` | ViT / CNN size | `512` / `512` |
| `--selector_type` | EGTP scoring | `l2` |
| `--early_stop_patience` | early stopping; `0` disables | `10` (code); paper trains up to 60 epochs |

`configs/mured.yaml` and `configs/idrid.yaml` match the table. Training is controlled by `train.py` CLI flags.

## File tree

```
SDCR-Net/
├── README.md
├── requirements.txt
├── train.py
├── loss.py
├── metrics.py
├── utils.py
├── figures/
│   └── framework.png
├── models/
│   ├── sdcr_net.py
│   ├── reinject.py
│   ├── token_keep.py
│   ├── vit_encoder.py
│   └── convnext_encoder.py
├── data/
│   ├── dataset.py
│   └── augmentation.py
├── configs/
│   ├── mured.yaml
│   └── idrid.yaml
└── scripts/
    ├── prepare_mured.sh
    ├── train_mured.sh
    ├── train_idrid.sh
    └── check_forward.py
```

## Forward tensors

Each DataLoader sample is a dict:

- `image`: CNN input, ImageNet-normalized
- `vit_score`: same size as ViT, **[0, 1]**, used only for EGTP \(L_2\)
- `vit_norm`: ViT input, ImageNet-normalized

EGTP must run on unnormalized pixels, not on ImageNet-normalized tensors.
