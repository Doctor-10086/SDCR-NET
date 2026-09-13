#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export MURED_ZIP="${ZIP:-${ROOT}/data/Multi-Label Retinal Diseases (MuReD) Dataset.zip}"
export MURED_DST="${DST:-${ROOT}/data/MuReD}"
export MURED_TMP_NESTED="${TMP_NESTED:-/tmp/mured_images.zip}"

mkdir -p "${MURED_DST}"

python3 - <<'PY'
import os
import shutil
import zipfile

outer = os.environ["MURED_ZIP"]
dst = os.environ["MURED_DST"]
tmp_nested = os.environ["MURED_TMP_NESTED"]
images_dir = os.path.join(dst, "images")
n_img = 0
if os.path.isdir(images_dir):
    n_img = sum(
        1 for n in os.listdir(images_dir)
        if n.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    )
if n_img >= 2000:
    print(f"MuReD images already present: {n_img} files in {images_dir}")
else:
    if not os.path.isfile(outer):
        raise FileNotFoundError(outer)
    nested_name = "Multi-Label Retinal Diseases (MuReD) Dataset/images.zip"
    print(f"extract csv from {outer}")
    with zipfile.ZipFile(outer) as z:
        for name in z.namelist():
            base = os.path.basename(name)
            if base.endswith(".csv"):
                out = os.path.join(dst, base)
                with z.open(name) as src, open(out, "wb") as f:
                    f.write(src.read())
                print(f"  wrote {out}")
        if nested_name not in z.namelist():
            raise FileNotFoundError(nested_name)
        print(f"copy nested images.zip -> {tmp_nested}")
        with z.open(nested_name) as src, open(tmp_nested, "wb") as f:
            shutil.copyfileobj(src, f, length=16 * 1024 * 1024)
    print(f"extract images into {dst}")
    with zipfile.ZipFile(tmp_nested) as iz:
        iz.extractall(dst)
    os.remove(tmp_nested)
    n_img = sum(
        1 for n in os.listdir(images_dir)
        if n.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    )
    print(f"done: {n_img} images in {images_dir}")
PY
