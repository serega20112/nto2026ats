# -*- coding: utf-8 -*-
import numpy as np
import cv2


def predict_road_broken(image: np.ndarray) -> str:
    """
         Функция, определяющая, прерывается ли дорога на изображении.
         Входные данные: изображение (bgr), прочитанное cv2.imread.
         Выходные данные: строка "yes" или "no".
                          "yes" - дорога прерывается
                          "no" - дорога непрерывна

         Примеры вывода: "yes"
                         "no"
    """

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold: road is black, background is white
    # Invert so road becomes white (255) for analysis
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # num_labels includes background as label 0
    # Count components with significant pixel area (filter noise)
    min_area = 50  # minimum pixel area to consider a valid road segment
    significant_components = 0
    for i in range(1, num_labels):  # skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            significant_components += 1

    # If there are 2 or more significant components, the road is broken
    if significant_components >= 2:
        predicted = "yes"
    else:
        predicted = "no"

    return predicted
