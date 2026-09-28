# -*- coding: utf-8 -*-
import cv2
import pandas as pd
import eval

"""Файл служит для определения точности вашего алгоритма.
   Не редактируйте его!!!
   Для получения оценки точности, запустите файл на исполнение.
"""


def main():
    csv_file = "annotations.csv"
    data = pd.read_csv(csv_file, sep=';')
    data = data.sample(frac=1)
    print(data)
    all_good_detection = 0
    tools = eval.load_tools()
    for row in data.itertuples():
        video = cv2.VideoCapture(row[1])
        user_obj_list = eval.predict_violations(video, tools)
        raw = row[2] if pd.notna(row[2]) else ""
        true_obj_list = [int(x) for x in raw.split(",")] if raw else []
        print("Video : ", row[1])
        print("True : ", true_obj_list)
        print("User : ", user_obj_list)
        print()
        if len(true_obj_list) == len(user_obj_list):
            if all(abs(u - t) / t <= 0.10 for t, u in zip(true_obj_list, user_obj_list)):
                all_good_detection += 1
        video.release()

    total_object = len(data.index)
    print(f"Для {all_good_detection} запросов из {total_object} верно определены скорости.")
    score = all_good_detection / total_object
    print(f"Точность алгоритма: {score}")


if __name__ == '__main__':
    main()
