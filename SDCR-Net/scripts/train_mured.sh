#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export PYTHONUNBUFFERED=1

OUT_DIR="${OUT_DIR:-${ROOT}/outputs/sdcr_mured_512}"
mkdir -p "${OUT_DIR}"

python -u train.py \
  --dataset mured \
  --data_dir "${DATA_DIR:-${ROOT}/data/MuReD}" \
  --vit_input_size 512 \
  --image_size 512 \
  --keep_ratio "${KEEP_RATIO:-0.1}" \
  --selector_type l2 \
  --pool_mode cls \
  --num_reinject "${NUM_REINJECT:-3}" \
  --reinject_scale "${REINJECT_SCALE:-0.5}" \
  --reinject_down_dim 256 \
  --reinject_num_heads 8 \
  --fusion_weight "${FUSION_WEIGHT:-0.5}" \
  --lr "${LR:-5e-5}" \
  --lr_scheduler "${SCHEDULER:-cosine}" \
  --epochs "${EPOCHS:-60}" \
  --batch_size "${BATCH_SIZE:-8}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --seed "${SEED:-0}" \
  --early_stop_patience "${PATIENCE:-10}" \
  --save_path "${OUT_DIR}/best.pth" \
  "$@"
