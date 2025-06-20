import argparse
from pathlib import Path

import torch
from PIL import Image

def run(weights: str, source: str, output: str, conf_thres: float):
    model = torch.hub.load('ultralytics/yolov5',
                           'custom', 
                           path=weights,
                           source='local')  # локальный весовый файл
    model.conf = conf_thres  # порог уверенности

    source_path = Path(source)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Проходим по всем изображениям в папке
    for img_file in source_path.iterdir():
        if not img_file.is_file() or img_file.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue

        # Делаем инференс
        results = model(str(img_file))

        img = Image.open(img_file)
        width, height = img.size
      
        df = results.pandas().xyxy[0]

        # Записываем YOLO-txt
        txt_file = output_path / f"{img_file.stem}.txt"
        with open(txt_file, 'w') as f:
            for _, row in df.iterrows():
                # нормализуем координаты
                x_center = ((row.xmin + row.xmax) / 2) / width
                y_center = ((row.ymin + row.ymax) / 2) / height
                w = (row.xmax - row.xmin) / width
                h = (row.ymax - row.ymin) / height
                cls = int(row['class'])
                f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLOv5 inference: сохраняет .txt в формате YOLO")
    parser.add_argument('--weights', type=str, default='best.pt',
                        help='путь к файлу весов')
    parser.add_argument('--source', type=str, required=True,
                        help='папка с изображениями для инференса')
    parser.add_argument('--output', type=str, default='predictions',
                        help='папка для сохранения .txt')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='порог уверенности')
    args = parser.parse_args()

    run(args.weights, args.source, args.output, args.conf_thres)
