import argparse
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

WEIGHTS = "best.pt"

def run(source: str, output: str, conf_thres: float):
    model = YOLO(WEIGHTS)
    model.conf = conf_thres  # минимальный порог уверенности

    # Создаём папку для предсказаний
    source_path = Path(source)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    for img_file in source_path.iterdir():
        if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        results = model(str(img_file), verbose=False)[0]

        img = Image.open(img_file)
        width, height = img.size

        # Извлекаем боксы и классы
        xyxy = results.boxes.xyxy.cpu().numpy() 
        cls_ids = results.boxes.cls.cpu().numpy().astype(int) 

        # YOLO-txt
        txt_path = output_path / f"{img_file.stem}.txt"
        with open(txt_path, "w") as f:
            for (xmin, ymin, xmax, ymax), cls in zip(xyxy, cls_ids):
                x_center = ((xmin + xmax) / 2) / width
                y_center = ((ymin + ymax) / 2) / height
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height
                f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLOv5 inference: сохраняет .txt в формате YOLO"
    )
    parser.add_argument(
        "--source", type=str, required=True, help="папка с изображениями"
    )
    parser.add_argument(
        "--output", type=str, default="predictions", help="папка для .txt"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.50, help="порог уверенности"
    )
    args = parser.parse_args()
    run(args.source, args.output, args.conf_thres)
