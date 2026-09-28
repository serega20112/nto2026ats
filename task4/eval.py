# -*- coding: utf-8 -*-
import os
import sys
import time
import datetime
import cv2
import numpy as np

## Описание подхода (без машинного обучения):
# Камера статична, машины едут слева направо по трём полосам, не меняя их.
#   1. Фон = покомпонентная медиана по подвыборке кадров -> вычитанием фона
#      остаются только движущиеся машины (статичный декор отсекается).
#   2. По каждой полосе идёт 1D-трекинг по горизонтали (предсказание позиции
#      по текущей скорости), что устойчиво к нескольким машинам.
#   3. Скорость (px/cp) оценивается только по кадрам, где машина видима
#      целиком (иначе край кадра искажает координату).
#   4. Перевод в км/ч: v_px/s * (расстояние_полосы / ширина_кадра) * 3.6.
#      Полоса определяется по вертикали (низ машины = колёса).
#   5. Учитываются только машины, ПОЛНОСТЬЮ покинувшие кадр (задний край
#      пересёк правую границу) за время видео; ответ упорядочен по моменту
#      исчезновения.
# Решение самодостаточное, внешних файлов не требует.

# Расстояния левого-правого края кадра для полос (в метрах), по условию задачи.
_LANE_NEAR = 16.82
_LANE_MIDDLE = 22.82
_LANE_FAR = 34.36
_DEFAULT_FPS = 30.0


def load_tools():
    """Внешние данные не требуются."""
    try:
        return []
    except Exception:
        return []


def _lane_by_bottom(bottom_y, h):
    """Определяет полосу по вертикали низа машины (доли от высоты кадра)."""
    if bottom_y < 0.47 * h:
        return _LANE_FAR
    if bottom_y < 0.66 * h:
        return _LANE_MIDDLE
    return _LANE_NEAR


def _compute_background(video):
    """Медианный фон по подвыборке кадров, вычисляется полосами по строкам,
    чтобы не раздувать память (np.median приводит uint8 к float64)."""
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fc = video.get(cv2.CAP_PROP_FRAME_COUNT)
    # CAP_PROP_FRAME_COUNT может быть NaN/отрицательным - защищаемся.
    meta_total = int(fc) if (fc is not None and fc == fc and fc > 0) else 0
    n_target = 15
    step = max(1, meta_total // n_target) if meta_total > n_target else 1
    samples = []
    ref_shape = None
    idx = 0
    while True:
        ret, fr = video.read()
        if not ret:
            break
        if idx % step == 0 and len(samples) < n_target:
            if ref_shape is None:
                ref_shape = fr.shape
            if fr.shape == ref_shape:
                samples.append(fr)
        idx += 1
    total = idx
    if not samples:
        return None, 0, 0, 0
    h, w = ref_shape[:2]
    stack = np.stack(samples)  # (n, h, w, 3) uint8
    bg = np.empty((h, w, 3), dtype=np.uint8)
    chunk = 16  # считаем медиану короткими полосами строк - память ограничена
    for y0 in range(0, h, chunk):
        y1 = min(y0 + chunk, h)
        bg[y0:y1] = np.median(stack[:, y0:y1], axis=0).astype(np.uint8)
    del stack, samples
    return bg, w, h, total


def _merge_rects(rects, pad_x, pad_y):
    """Объединяет прямоугольники, пересекающиеся (с запасом) по обеим осям.
    Это склеивает кузов и отделившееся колесо одной машины в один блок."""
    rects = [list(r) for r in rects]
    changed = True
    while changed:
        changed = False
        out = []
        for r in rects:
            placed = False
            for o in out:
                if not (r[2] + pad_x < o[0] or r[0] - pad_x > o[2]
                        or r[3] + pad_y < o[1] or r[1] - pad_y > o[3]):
                    o[0] = min(o[0], r[0]); o[1] = min(o[1], r[1])
                    o[2] = max(o[2], r[2]); o[3] = max(o[3], r[3])
                    placed = True
                    changed = True
                    break
            if not placed:
                out.append(list(r))
        rects = out
    return rects


def _frame_blobs(fr, bg, w, h):
    """Возвращает прямоугольники движущихся машин на кадре (область дороги)."""
    diff = cv2.absdiff(fr, bg)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    road_top = int(0.34 * h)  # выше дороги - статичный фон, не нужен
    mask[:road_top, :] = 0
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw = []
    min_area = 0.0004 * w * h
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh < min_area:
            continue
        raw.append([float(x), float(y), float(x + bw), float(y + bh)])
    # склейка частей одного автомобиля (кузов + колесо)
    raw = _merge_rects(raw, 0.02 * w, 0.03 * h)
    blobs = []
    for x1, y1, x2, y2 in raw:
        if (x2 - x1) * (y2 - y1) < 0.003 * w * h:
            continue
        blobs.append({"x1": x1, "y2": y2, "x2": x2, "cx": (x1 + x2) / 2.0})
    return blobs


def _dbg(msg):
    """Канал отладки: пишет в stderr (виден в консоли/PROGRAM OUTPUT, но не
    смешивается с протоколом обмена master<->slave, который обычно идёт по stdout).
    Лимит по числу строк - защита от разрастания, при этом хватает на все видео."""
    try:
        if not hasattr(_dbg, "_n"):
            _dbg._n = 0
            _dbg._max = 800
        if _dbg._n >= _dbg._max:
            return
        _dbg._n += 1
        sys.stderr.write(str(msg) + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def _dbg_header():
    _dbg("\n==== run " + datetime.datetime.now().isoformat())


def _dbg_video(path, w, h, total, fps, tracks, counted):
    try:
        lines = ["VIDEO path={} w={} h={} total={} fps={:.2f} ntracks={}".format(
            path, w, h, total, fps, len(tracks))]
        for idx, t in enumerate(tracks):
            pts = t["pts"]
            fis = [p["fi"] for p in pts]
            x1s = [p["x1"] for p in pts]
            x2s = [p["x2"] for p in pts]
            ys = [p["y2"] for p in pts]
            vis = [p for p in pts if p["x1"] > 2 and p["x2"] < w - 2]
            if len(vis) >= 3:
                v = float(np.polyfit([p["fi"] for p in vis],
                                     [p["cx"] for p in vis], 1)[0])
            else:
                v = t["vel"]
            speed = v * fps * (t["lane"] / w) * 3.6
            lines.append("  T{} lane={} n={} f{}..{} x1={}..{} y2={} v={:.1f} speed={:.1f}{}".format(
                idx, t["lane"], len(pts), fis[0], fis[-1],
                int(x1s[0]), int(x1s[-1]), int(max(ys)), v, speed,
                "  <COUNTED>" if idx in counted else ""))
        _dbg("\n".join(lines))
    except Exception as e:
        _dbg("DBGERR " + repr(e))


def _same_car(ta, tb, w):
    """True, если два трека в одной полосе идут бок-о-бок (в общие кадры
    их центры ближе длины кузова) -> это одна машина, распавшаяся на части."""
    if ta["lane"] != tb["lane"]:
        return False
    da = {p["fi"]: p["cx"] for p in ta["pts"]}
    db = {p["fi"]: p["cx"] for p in tb["pts"]}
    common = da.keys() & db.keys()
    if len(common) < 5:
        return False
    diffs = sorted(abs(da[f] - db[f]) for f in common)
    median_gap = diffs[len(diffs) // 2]
    return median_gap < 0.18 * w


def predict_violations(video: cv2.VideoCapture, tools_list: list = None) -> list:
    """
         Функция определяющая скорости автомобилей на видео.
         Входные данные: объект, возвращаемый cv2.VideoCapture
         Выходные данные: список скоростей автомобилей (км/ч).
                          Скорости должны быть в порядке полного исчезновения
                          соответствующих машин из кадра.

         Примеры вывода: [18, 30] - два автомобиля покинули кадр.
                         [] - автомобили не покидали кадр.
    """
    # Гарантируем, что функция никогда не вернёт None и не уронит проверяющий
    # скрипт: любая нестандартная ситуация (битое/неоткрываемое видео, NaN в
    # метаданных, вызов без второго аргумента и т.п.) даёт пустой список
    # вместо исключения.
    try:
        if tools_list is None:
            tools_list = load_tools()
        result = _predict_violations_impl(video, tools_list)
        return result if isinstance(result, list) else []
    except Exception:
        return []


def _predict_violations_impl(video, tools_list):
    fps = video.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or fps > 120:
        fps = _DEFAULT_FPS

    bg, w, h, total = _compute_background(video)
    if bg is None:
        return []

    # Проход по всем кадрам с трекингом по полосам.
    tracks = []  # {lane, pts:[(fi,x1,x2,cx)], vel}
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fi = -1
    while True:
        ret, fr = video.read()
        if not ret:
            break
        fi += 1
        blobs = _frame_blobs(fr, bg, w, h)
        for b in blobs:
            b["lane"] = _lane_by_bottom(b["y2"], h)
        used = set()
        for t in tracks:
            last = t["pts"][-1]
            pred = last["cx"] + t["vel"] * (fi - last["fi"])
            best = None
            bestd = 1e9
            for bi, b in enumerate(blobs):
                if b["lane"] != t["lane"] or bi in used:
                    continue
                d = abs(b["cx"] - pred)
                if d < bestd:
                    bestd = d
                    best = bi
            if best is not None and bestd < 0.06 * w:
                used.add(best)
                b = dict(blobs[best]); b["fi"] = fi
                if len(t["pts"]) >= 1:
                    prev = t["pts"][-1]
                    dt = fi - prev["fi"]
                    if dt > 0:
                        t["vel"] = (b["cx"] - prev["cx"]) / dt
                t["pts"].append(b)
        for bi, b in enumerate(blobs):
            if bi not in used:
                nb = dict(b); nb["fi"] = fi
                tracks.append({"lane": b["lane"], "pts": [nb], "vel": 0.0})

    # Убираем дубли: одна машина, давшая несколько параллельных треков.
    drop = set()
    for i in range(len(tracks)):
        if i in drop:
            continue
        for j in range(i + 1, len(tracks)):
            if j in drop:
                continue
            if _same_car(tracks[i], tracks[j], w):
                # оставляем более длинный (надёжный) трек
                if len(tracks[i]["pts"]) >= len(tracks[j]["pts"]):
                    drop.add(j)
                else:
                    drop.add(i)
    if drop:
        tracks = [t for k, t in enumerate(tracks) if k not in drop]

    # Финализация: скорость и момент полного выхода из кадра.
    _dbg_header()
    _dbg("VIDEO w={} h={} total={} fps={:.2f} ntracks={}".format(
        w, h, total, fps, len(tracks)))
    results = []
    counted = set()
    for idx, t in enumerate(tracks):
        pts = t["pts"]
        status = "COUNTED"
        reason = ""
        vis = [p for p in pts if p["x1"] > 2 and p["x2"] < w - 2]
        front = [p for p in pts if p["x2"] < w - 2]  # передний край не у прав. границы
        if len(vis) >= 3:
            fis = [p["fi"] for p in vis]
            cxs = [p["cx"] for p in vis]
            v = float(np.polyfit(fis, cxs, 1)[0])  # px/frame (ц. масс по целым кадрам)
        elif len(front) >= 3:
            v = float(np.polyfit([p["fi"] for p in front],
                                 [p["x2"] for p in front], 1)[0])  # по переднему краю
        else:
            v = t["vel"]
        speed = v * fps * (t["lane"] / w) * 3.6
        last = pts[-1]
        max_x1 = max(p["x1"] for p in pts)  # задний край доходил максимум сюда
        exit_fi = last["fi"] + (w - max_x1) / v if v > 1e-6 else 1e9
        if len(pts) < 8:
            status, reason = "DROP", "short n<8"
        elif v <= 0.3:
            status, reason = "DROP", "slow v<=0.3"
        elif max_x1 < 0.02 * w:
            # задний край так и не въехал в кадр -> это обрывок входа/фантом,
            # надёжно оценить его полный выход нельзя
            status, reason = "DROP", "rear-not-in-frame"
        elif last["fi"] > total - 2:
            status, reason = "DROP", "in-frame-at-end"
        elif exit_fi > total + 2:
            status, reason = "DROP", "not-exit exit>total"
        if status == "COUNTED":
            results.append((exit_fi, speed))
            counted.add(idx)
        _dbg("  T{} lane={} n={} f{}..{} x1={}..{} y2={} v={:.1f} "
             "speed={:.1f} exit={:.0f} total={} {} {}".format(
                 idx, t["lane"], len(pts), pts[0]["fi"], last["fi"],
                 int(pts[0]["x1"]), int(last["x1"]), int(max(p["y2"] for p in pts)),
                 v, speed, exit_fi, total, status, reason))
    ans = [int(round(s)) for _, s in sorted(results, key=lambda r: r[0])]
    _dbg("  ANSWER={}".format(ans))
    return ans
