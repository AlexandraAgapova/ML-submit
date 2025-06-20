import argparse
import os
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

def run(
    weights: str,
    source: str,
    output: str,
    patch_size: int,
    stride: int,
    conf: float,
    iou: float,
    imgsz: int,
    device: str
):
    # Создаём папку для предсказаний
    os.makedirs(output, exist_ok=True)
    model = YOLO(weights)

    files = [f for f in os.listdir(source)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for file in files:
        img_path = Path(source) / file
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: не удалось прочитать {img_path}")
            continue

        h, w = img.shape[:2]
        nh = int(np.ceil((h - patch_size) / stride)) + 1
        nw = int(np.ceil((w - patch_size) / stride)) + 1
        new_h = (nh - 1) * stride + patch_size
        new_w = (nw - 1) * stride + patch_size

        pad_h = (new_h - h) // 2
        pad_w = (new_w - w) // 2
        canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        canvas[pad_h:pad_h + h, pad_w:pad_w + w] = img

        total_boxes = []
        total_classes = []

        for i in range(nh):
            for j in range(nw):
                y1, y2 = i * stride, i * stride + patch_size
                x1, x2 = j * stride, j * stride + patch_size
                patch = canvas[y1:y2, x1:x2]

                results = model.predict(
                    patch,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    device=device,
                    verbose=False
                )

                for r in results:
                    if not hasattr(r, "boxes") or r.boxes is None:
                        continue
                    boxes = r.boxes.xywh.cpu().numpy()
                    scores = r.boxes.conf.cpu().numpy()
                    classes = r.boxes.cls.cpu().numpy().astype(int)

                    for (xc, yc, pw, ph), score, cls in zip(boxes, scores, classes):
                        if score < conf:
                            continue
                        x_abs = xc + x1
                        y_abs = yc + y1
                        total_boxes.append([x_abs, y_abs, pw, ph])
                        total_classes.append(cls)

        yolo_lines = []
        for (xc, yc, pw, ph), cls in zip(total_boxes, total_classes):
            x_center = xc / new_w
            y_center = yc / new_h
            w_norm = pw / new_w
            h_norm = ph / new_h
            yolo_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        txt_path = Path(output) / f"{Path(file).stem}.txt"
        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_lines))

        print(f"Processed {file} → {txt_path.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLO sliding-window inference — сохраняет .txt"
    )
    parser.add_argument(
        "--weights", type=str, default="best.pt",
        help="путь к файлу весов модели"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="папка с исходными изображениями"
    )
    parser.add_argument(
        "--output", type=str, default="predictions",
        help="папка для сохранения .txt"
    )
    parser.add_argument(
        "--patch-size", type=int, default=640,
        help="размер квадратного патча (px)"
    )
    parser.add_argument(
        "--stride", type=int, default=512,
        help="шаг скольжения окна (px)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.50,
        help="порог уверенности"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.70,
        help="порог NMS IoU"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="размер, к которому ресайзятся патчи для модели (px)"
    )
    parser.add_argument(
        "--device", type=str, default="0",
        help="CUDA device или 'cpu'"
    )
    args = parser.parse_args()

    run(
        weights=args.weights,
        source=args.source,
        output=args.output,
        patch_size=args.patch_size,
        stride=args.stride,
        conf=args.conf_thres,
        iou=args.iou_thres,
        imgsz=args.imgsz,
        device=args.device
    )
