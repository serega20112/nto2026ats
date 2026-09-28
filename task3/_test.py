# -*- coding: utf-8 -*-
"""Test harness: detect sticker bbox + classify fuel type, measure IoU+type."""
import cv2
import numpy as np
import pandas as pd

GREEN_LO = np.array([68, 150, 60])
GREEN_HI = np.array([85, 255, 165])
ORANGE_LO = np.array([10, 150, 175])
ORANGE_HI = np.array([26, 255, 255])


def iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2]-box1[0])*(box1[3]-box1[1]); a2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    u = a1 + a2 - inter
    return inter/u if u > 0 else 0.0


def detect(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    orange = cv2.inRange(hsv, ORANGE_LO, ORANGE_HI)
    mask = cv2.bitwise_or(green, orange)
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < 400 or w == 0 or h == 0:
            continue
        aspect = w / h
        if not (1.3 <= aspect <= 1.75):
            continue
        # solidity: contour area vs bbox area (rounded rect ~0.8-0.95)
        sol = cv2.contourArea(c) / area
        if sol < 0.6:
            continue
        score = area * (1.0 - abs(aspect - 1.5)) * min(sol, 1.0)
        cands.append((score, (x, y, x + w, y + h), sol, aspect))
    if not cands:
        return None, None
    cands.sort(key=lambda t: t[0], reverse=True)
    best = cands[0][1]
    return best, cands


def classify(image, box):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    x1, y1, x2, y2 = box
    roi = hsv[y1:y2, x1:x2]
    green = cv2.inRange(roi, GREEN_LO, GREEN_HI)
    orange = cv2.inRange(roi, ORANGE_LO, ORANGE_HI)
    if orange.sum() > green.sum():  # orange background dominates -> Energy
        return "Energy"
    h, w = roi.shape[:2]
    center = roi[h // 3:2 * h // 3, w // 3:2 * w // 3]
    center_v = np.median(center[:, :, 2])
    return "Petrol" if center_v > 150 else "Diesel"


if __name__ == "__main__":
    data = pd.read_csv("annotations.csv", sep=';')
    ok = 0
    ious = []
    for row in data.itertuples():
        parts = str(row[2]).split(",")
        tb = [int(v) for v in parts[:4]]; tt = parts[4]
        img = cv2.imread(row[1])
        box, cands = detect(img)
        if box is None:
            print(f"NODETECT {row[1]} true={tb},{tt}")
            continue
        ct = classify(img, box)
        i = iou(box, tb)
        ious.append(i)
        good = (i >= 0.75 and ct == tt)
        ok += good
        if not good:
            print(f"BAD i={i:.2f} ct={ct} tt={tt} box={box} tb={tb} ncand={len(cands)} {row[1]}")
    print(f"\nIoU min={min(ious):.3f} mean={sum(ious)/len(ious):.3f}")
    print(f"Accuracy: {ok}/{len(data)} = {ok/len(data):.3f}")
