# -*- coding: utf-8 -*-
"""Crop sticker regions from annotations for inspection."""
import cv2
import pandas as pd
import os

data = pd.read_csv("annotations.csv", sep=';')
os.makedirs("crops", exist_ok=True)
for row in data.itertuples():
    parts = str(row[2]).split(",")
    x1, y1, x2, y2 = [int(v) for v in parts[:4]]
    fuel = parts[4]
    img = cv2.imread(row[1])
    if img is None:
        print("missing", row[1])
        continue
    # pad a bit
    pad = 20
    h, w = img.shape[:2]
    cx1 = max(0, x1 - pad); cy1 = max(0, y1 - pad)
    cx2 = min(w, x2 + pad); cy2 = min(h, y2 + pad)
    crop = img[cy1:cy2, cx1:cx2]
    fname = f"crops/{fuel}_{os.path.basename(row[1])}"
    cv2.imwrite(fname, crop)
print("done, crops saved")
