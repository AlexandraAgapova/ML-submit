# YOLOv5 Submission

Этот репозиторий позволяет выполнить инференс `best.pt` и получить предсказания в формате YOLO txt-файлы

## Структура репозитория

- **best.pt** — обученные веса
- **inference.py** — скрипт для инференса изображений и сохранения `.txt`
- **requirements.txt** — все зависимости
- **README.md** — инструкция по установке и запуску

---

## Установка

```bash
# Клонируем репозиторий
git clone https://github.com/AlexandraAgapova/ML-submit.git
cd ML-submit

# Устанавливаем зависимости
pip install -r requirements.txt
```  

---

## Подготовка данных

Создайте папку `images/` с изображениями форматов `.jpg` или `.png`

---

## Запуск инференса

```bash
python3 test.py --images images --predictions predictions

```

- `--weights`     — путь к файлу весов (по умолчанию `best.pt`)
- `--source`      — папка с исходными изображениями
- `--output`      — папка для сохранения `.txt` (создаётся автоматически)
- `--patch-size`  — размер квадратного патча (px)
- `--stride`      — шаг скольжения окна (px)
- `--conf-thres`  — порог уверенности (0.0–1.0)
- `--iou-thres`   — порог NMS IoU (0.0–1.0)
- `--imgsz`       — размер, к которому ресайзятся патчи для модели (px)
- `--device`      — CUDA device (например `"0"`) или `"cpu"`


После запуска в папке `predictions/` появятся файлы:

predictions/img1.txt, predictions/img2.txt и т.д

---

## Формат выходных файлов

Каждый `.txt` файл содержит по одной строке для каждого обнаруженного объекта:

```
<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
```
