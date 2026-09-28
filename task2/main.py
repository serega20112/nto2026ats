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
        image = cv2.imread(row[1])
        user_obj_list = eval.predict_turn_number(image, row[2], tools)
        true_obj_list = int(row[3])
        print("Service : ", row[2])
        print("True : ", true_obj_list)
        print("User : ", user_obj_list)
        print()
        if true_obj_list == user_obj_list:
            all_good_detection += 1

    total_object = len(data.index)
    print(f"Для {all_good_detection} запросов из {total_object} верно определен номер поворота.")
    score = all_good_detection / total_object
    print(f"Точность алгоритма: {score}")


if __name__ == '__main__':
    main()
