# -*- coding: utf-8 -*-
import cv2
import pandas as pd
import eval

"""Файл служит для определения точности вашего алгоритма.
   Не редактируйте его!!!
   Для получения оценки точности, запустите файл на исполнение.
"""

def iou(box1, box2):
    """Сходство прямоугольников: пересечение / объединение (0..1)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def main():
    csv_file = "annotations.csv"
    data = pd.read_csv(csv_file, sep=';')
    data = data.sample(frac=1)
    print(data)
    all_good_detection = 0
    tools = eval.load_tools()
    for row in data.itertuples():
        image = cv2.imread(row[1])
        user_box, user_type = eval.predict_sticker(image, tools)
        true_answer = str(row[2])
        print("Image : ", row[1])
        print("True : ", true_answer)
        print("User : ", user_box, user_type)
        print()
        parts = true_answer.split(",")
        true_box = [int(v) for v in parts[:4]]
        true_type = parts[4]
        if iou(user_box, true_box) >= 0.75 and user_type == true_type:
            all_good_detection += 1

    total_object = len(data.index)
    print(f"Для {all_good_detection} изображений из {total_object} верно определены положение и тип маркировки.")
    score = all_good_detection / total_object
    print(f"Точность алгоритма: {score}")


if __name__ == '__main__':
    main()
