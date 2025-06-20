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
python inference.py --source images
```

- `--source` — папка с изображениями
- `--output` — папка для сохранения `.txt` в формате yolo: `<class> <x_center> <y_center> <w> <h>`
- `--conf-thres` — минимальный порог уверенности

После запуска в папке `predictions/` появятся файлы:

predictions/img1.txt, predictions/img2.txt и т.д

---

## Формат выходных файлов

Каждый `.txt` файл содержит по одной строке для каждого обнаруженного объекта:

```
<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
```
