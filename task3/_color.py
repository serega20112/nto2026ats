# -*- coding: utf-8 -*-
"""Sample sticker background & symbol colors from annotations."""
import cv2
import numpy as np
import pandas as pd

data = pd.read_csv("annotations.csv", sep=';')

def analyze(row):
    parts = str(row[2]).split(",")
    x1, y1, x2, y2 = [int(v) for v in parts[:4]]
    fuel = parts[4]
    img = cv2.imread(row[1])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    crop = hsv[y1:y2, x1:x2]
    crop_bgr = img[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    # background = border pixels, symbol = center pixels
    border = np.concatenate([
        crop[0:3, :].reshape(-1,3), crop[-3:, :].reshape(-1,3),
        crop[:, 0:3].reshape(-1,3), crop[:, -3:].reshape(-1,3)])
    center = crop[h//3:2*h//3, w//3:2*w//3].reshape(-1,3)
    # report dominant colors
    def dom(arr):
        arr = arr.astype(int)
        return (int(np.median(arr[:,0])), int(np.median(arr[:,1])), int(np.median(arr[:,2])))
    # symbol brightness: max V in center
    print(f"{fuel:8s} box=({x1},{y1},{x2},{y2}) size={w}x{h} bgHSV~{dom(border)} centerHSV~{dom(center)}")

for row in data.itertuples():
    analyze(row)
