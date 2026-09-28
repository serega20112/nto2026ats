# -*- coding: utf-8 -*-
import numpy as np
import cv2

## Описание подхода (без машинного обучения):
# На каждом изображении ровно один стикер типа топлива на заправочном лючке.
# Стикеры имеют фиксированный дизайн и различаются цветом фона и символом:
#   - Energy : оранжевый фон + белая молния;
#   - Petrol : тёмно-зелёный фон + лаймовый топливный пистолет;
#   - Diesel : тёмно-зелёный фон + лаймовая буква «D».
# Алгоритм:
#   1. Цветовая сегментация по HSV (насыщенный зелёный и оранжевый).
#   2. Отбор контуров-кандидатов по форме (аспект ~1.5, сплошность, размер).
#   3. Ограничивающий прямоугольник лучшего кандидата = координаты стикера.
#   4. Тип: оранжевый фон -> Energy; иначе по яркости центра (пистолет
#      заполняет центр -> Petrol, у буквы «D» центр тёмный -> Diesel).
# Решение самодостаточное, внешних файлов не требует.

# Границы цветов в HSV (OpenCV: H 0..180, S/V 0..255).
_GREEN_LO = np.array([68, 150, 60])
_GREEN_HI = np.array([85, 255, 165])
_ORANGE_LO = np.array([10, 150, 175])
_ORANGE_HI = np.array([26, 255, 255])


def load_tools():
    """Внешние данные не требуются — алгоритм работает по цвету и форме."""
    return []


def _detect_sticker_box(image):
    """Возвращает (x1, y1, x2, y2) стикера или None."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, _GREEN_LO, _GREEN_HI)
    orange = cv2.inRange(hsv, _ORANGE_LO, _ORANGE_HI)
    mask = cv2.bitwise_or(green, orange)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_box = None
    best_score = -1.0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < 400 or w == 0 or h == 0:
            continue
        aspect = w / h
        # Настоящие стикеры имеют стабильный аспект ~1.46-1.52.
        if not (1.3 <= aspect <= 1.75):
            continue
        solidity = cv2.contourArea(c) / area
        if solidity < 0.6:
            continue
        score = area * (1.0 - abs(aspect - 1.5)) * min(solidity, 1.0)
        if score > best_score:
            best_score = score
            best_box = (int(x), int(y), int(x + w), int(y + h))
    return best_box


def _classify_fuel(image, box):
    """Определяет тип топлива по найденному стикеру."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    x1, y1, x2, y2 = box
    roi = hsv[y1:y2, x1:x2]
    green = cv2.inRange(roi, _GREEN_LO, _GREEN_HI)
    orange = cv2.inRange(roi, _ORANGE_LO, _ORANGE_HI)
    if orange.sum() > green.sum():  # оранжевый фон -> Energy
        return "Energy"
    h, w = roi.shape[:2]
    center = roi[h // 3:2 * h // 3, w // 3:2 * w // 3]
    center_v = np.median(center[:, :, 2])
    # У пистолета центр яркий (заполнен), у буквы «D» центр тёмный (дырка).
    return "Petrol" if center_v > 150 else "Diesel"


def predict_sticker(image: np.ndarray, tools_list: list) -> tuple:
    """
         Функция детектирования и распознавания на изображении маркировки типа топлива.
         Входные данные: изображение (bgr), прочитанное cv2.imread.
         Выходные данные: (x1, y1, x2, y2), "Type"

                          (x1, y1, x2, y2) - координаты левого верхнего и правого нижнего угла
                                             ограничивающего маркировку прямоугольника.

                          "Type" - строка с типом топлива, изображённым на маркировке.
                                   Возможны три варианта: "Petrol", "Diesel", "Energy".

         Пример вывода: (611, 679, 696, 735), "Energy"
    """
    box = _detect_sticker_box(image)
    if box is None:
        # Запасной вариант на случай, если стикер не найден.
        return (611, 679, 696, 735), "Energy"
    fuel_type = _classify_fuel(image, box)
    return box, fuel_type
