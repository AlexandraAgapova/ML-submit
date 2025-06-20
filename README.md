# YOLOv5 Submission

Этот репозиторий позволяет быстро выполнить инференс `best.pt` и получить предсказания в формате YOLO (txt-файлы).

## Структура репозитория

ML-submit/
├── best.pt
├── inference.py
├── requirements.txt
└── README.md

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

Создайте папку `images/` с изображениями форматов `.jpg` или `.png`, например:

images/
├── img1.jpg
├── img2.png
└── ...

---

## Запуск инференса

```bash
python inference.py --weights best.pt --source images --output predictions --conf-thres 0.50
```

- `--weights` — путь к весам
- `--source` — папка с изображениями
- `--output` — папка для сохранения `.txt` в формате yolo: `<class> <x_center> <y_center> <w> <h>`
- `--conf-thres` — минимальный порог уверенности (0.0–1.0)

После запуска в папке `predictions/` появятся файлы:

predictions/
├── img1.txt
├── img2.txt
└── ...

---

## Формат выходных файлов

Каждый `.txt` файл должен содержать по одной строке для каждого обнаруженного объекта:

```
<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
```
