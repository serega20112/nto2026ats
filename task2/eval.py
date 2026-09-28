# -*- coding: utf-8 -*-
import base64
import numpy as np
import cv2

## Описание подхода:
# Знаки сервисов стоят в 4 фиксированных точках вдоль дороги (фон всех
# изображений одинаков, меняется только иконка внутри рамки). Алгоритм:
#   1. Детектирует синие рамки знаков по HSV (насыщенный синий) + проверяет
#      наличие белого «окна» с чёрной иконкой внутри.
#   2. Сортирует знаки по вертикали (снизу вверх = повороты 1..4).
#   3. Извлекает и нормализует чёрную иконку каждого знака.
#   4. Сопоставляет иконку с 4 шаблонами (корреляция).
# Шаблоны вшиты прямо в файл (base64), решение самодостаточное, без ML.

_TEMPLATES_B64 = {
    "car wash": "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAAAAACPAi4CAAABg0lEQVRYCaXBAY4aQRDAQPv/j3ZgGi473CoSmSqB+G8icULjjHHGOGOcMc4YZySGfCNejCHfiWEM+U4MY8itQO7EMIbcCZA7MYwhn+JNfothDPkQV/IhhjHkQ2xkF8MY8p0YxpBdsiRPyS6GMWSXhJCEJLsYEov8EkICSbKJIbHIJkkeEkiSTQxjyCZJHhJIkk0MY8hVQvKQQEJyFcMYcpWQPCSQkFzFMIZcJG/JS3IRwxhykfyWXMSQWGQXG/kQQ2KRHyGB/BVCyI8YEou8JZDskpAfMYwhbwkkuwSStxgSi7wlhOwSSN5iGENeEgLZBXIRwxjyksQtIXmJYQwZSdwTkpcYxpCRxD2BZMQwhowk7gkkI4YxZEkgbgkkI4YxZEn+LVliGEOekiWQq5AlWWIYQ56SJZCrQK5iGEMeQpZArgJZkqcYxpCrQK5CNjGMId+JYbzIN+LFOGOcMc4YZ4wzxhnjjHHGOCNxROKIccY4Y5yROGKcMc5IHBHigLLEfxH+AFX/wAXXPdaoAAAAAElFTkSuQmCC",
    "gas station": "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAAAAACPAi4CAAABdklEQVRYCaXBAW4YRwIEMdb/H90ne8Y+SVgEUZYM858l80aZV5p3mnead5p3mneaP/KvzV/NkR+ZqznyI3M1R35krubIs/kuH+ZqjjyaBzFXc+TRPIi5miPfjYx8NsRczZHvRiZfDDFXc+SbCZMvhpirOfLNhMkXQ8zVHPlmwuSLIeZqjnwzYfLFEHM1R76ZMPliiLmaI99MmHwxxFzNka8mH+ZBzNUc+WzCMg9irubIJxMm8yDmao78NfllMg9irubIH5PfJvMg5mqOXJNjMg9irubIb5M/JvMg5mqOfBj5azIPYq7myIfJ/03mQczVHDHyyWQexFzNEZPPJvMg5mqOmHw2mQcxV3PE5LPJPIi5miMsn03mQczVHGHkr4l5EHM1Rz6MXBPmQczVHPlt8svIP5irOXLNL/lnczVHfmSu5siPzNX8kX9t/mread5p3mnead5p3mnead5p3mnead5p3mnead5p3mnead5p3sm8UZj/LP8Do4uxBfjz3j8AAAAASUVORK5CYII=",
    "auto repair shop": "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAAAAACPAi4CAAABLElEQVRYCaXBQW4cSQDEQPL/j+ZaSMGY7erLuCLkj/hHgsQNJa5IXDHuGHeMO8Yd447xS74Rv4yR78QYI9+JMUZO8Zc8xBgjD/FJHmKMkU/xIA8xxsineJCHGGPkTYC8iTFG3gTImxhj5FM8yEOMMfJ/8UmeYoyRF/FDXsQYIy/ih7yIMUZexA95EWOMnGLkFGOMnGLkFGOMnGLkFGOMnGLkFGOMnGLkFGOMnGLkFGOMnGLkFGOMnGLkFGOMnGLkFGOMnGLkFGOMnGLkFGOMnGLkFGOMnGLkFGOMnGLkFGOMnGLkFGOMnGLkFGOMvAmQNzHGyHdijJHvxBi/5Bvxy7hj3DHuGHckrkhckbgiccW4Y9wx7hh3jDvGHeOOxBXjjnFH4orEDeWP+EfCf6zJfgW5CCmdAAAAAElFTkSuQmCC",
    "tire service": "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAAAAACPAi4CAAABsElEQVRYCaXBAY4kRQJFMb/7H/qTVRE9KyG0okk7zH+WzBs17zTvNO807zTvNO80/5N/af5ofuQX5mqu/MoczZVfmaO58itzNFf+2YjRkGuO5srXfOWaj4ym5ZijufKxfE0+5shoWr7maK4wuSbMlbnyMUdzhcXkY2F+ZDRNHnM0V0yYPCZDGBnNI485mismj+UxGXkMDU0T5miumPyxDPkYGU0T5miusPyxjByT0TRhjubKYzkmRo7JaB5hjubKx4gRRo7RyBDmaK58zUc+Ro7RyDxijubK13zkY+QYjcwj5miufM1HPkaO0cg8Yo7mytd85GNyjUaGMEdz5Ws+8jeT0TzCHM2Vr/nIYzTymIymCXM0V/5uWSZGRtOEOZorfzONmI/mqwlzNFd+LI/JEENG88hjjubKj+WxMD8ymiaPOZorPyYmzJXRPPIxR3PljyFfczSaR77maK78o/nIfLQcczRX/o+hyR9zNFd+ZY7myq/M0Vz5lTmaK78xV/Mj/978aN5p3mnead5p3mnead5p3mnead5p3mnead5p3mnead5p3mnead7JvFHMf5e/AHWC6gUOxmcfAAAAAElFTkSuQmCC",
}

def _decode_template(b64_str):
    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return (img > 127).astype(np.float32)


def load_tools():
    """Возвращает список со словарём декодированных шаблонов иконок."""
    templates = {}
    for service, b64 in _TEMPLATES_B64.items():
        tmpl = _decode_template(b64)
        if tmpl is not None:
            templates[service] = tmpl
    return [templates]


def _detect_sign_boxes(image):
    """Возвращает 4 прямоугольника знаков, отсортированных сверху вниз."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([95, 180, 100])
    upper_blue = np.array([115, 255, 220])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area < 1000:
            continue
        aspect = w / h if h > 0 else 0
        if not (0.5 < aspect < 1.2):
            continue
        hsv_roi = hsv[y:y + h, x:x + w]
        s_ch = hsv_roi[:, :, 1]
        v_ch = hsv_roi[:, :, 2]
        white_frac = np.mean((v_ch > 180) & (s_ch < 60))
        icon_frac = np.mean((v_ch < 120) & (s_ch < 120))
        if white_frac > 0.2 and 0.03 < icon_frac < 0.5:
            candidates.append((white_frac * icon_frac, (x, y, w, h)))

    candidates = sorted(candidates, key=lambda t: t[0], reverse=True)[:4]
    signs = [c[1] for c in candidates]
    signs = sorted(signs, key=lambda s: s[1])  # top -> bottom
    return signs


def _extract_icon(image, box, size=64):
    """Извлекает чёрную иконку из знака и нормализует до size x size."""
    x, y, w, h = box
    roi = image[y:y + h, x:x + w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, icon_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    ys, xs = np.where(icon_mask > 0)
    if len(xs) == 0:
        return None
    icon = icon_mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    icon = cv2.resize(icon, (size, size), interpolation=cv2.INTER_AREA)
    return (icon > 127).astype(np.float32)


def _classify(icon, templates):
    """Сопоставляет иконку с шаблонами по нормированной корреляции."""
    best_name = None
    best_score = -1.0
    a = icon.flatten()
    a = a - a.mean()
    na = np.linalg.norm(a)
    for name, tmpl in templates.items():
        b = tmpl.flatten()
        b = b - b.mean()
        denom = na * np.linalg.norm(b) + 1e-6
        score = float(np.dot(a, b) / denom)
        if score > best_score:
            best_score = score
            best_name = name
    return best_name


def predict_turn_number(image: np.ndarray, service: str, tools_list: list) -> int:
    """
         Функция, определяющая номер поворота к сервису, конкретного типа.
         Входные данные: изображение (bgr), строка с типом сервиса.
                         Возможны четыре варианта строк с типом сервисов:
                         "car wash", "gas station", "tire service", "auto repair shop".

         Выходные данные: номер поворота (int).
                          Возможные варианты: 1, 2, 3, 4.

         Пример вывода: 1
                        3
    """
    templates = tools_list[0] if tools_list else {}
    if not templates:
        templates = load_tools()[0]

    boxes = _detect_sign_boxes(image)
    n = len(boxes)
    # boxes отсортированы сверху вниз: верхний = самый дальний (поворот n),
    # нижний = самый ближний (поворот 1).
    for i, box in enumerate(boxes):
        icon = _extract_icon(image, box)
        if icon is None:
            continue
        if _classify(icon, templates) == service:
            return n - i
    return 1
