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
    for row in data.itertuples():
        image = cv2.imread(row[1])
        user_obj_list = eval.predict_road_broken(image)
        true_obj_list = row[2]
        print("True : ", true_obj_list)
        print("User : ", user_obj_list)
        print()
        if true_obj_list == user_obj_list:
            all_good_detection += 1

    total_object = len(data.index)
    print(f"Для {all_good_detection} изображения(ий) из {total_object} верно определено состояние дороги.")
    score = all_good_detection / total_object
    print(f"Точность алгоритма: {score}")


if __name__ == '__main__':
    main()
