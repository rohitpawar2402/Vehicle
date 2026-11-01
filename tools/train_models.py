"""Automated training runner configured for RTX 3050 (4 GB VRAM)."""

import argparse
from ultralytics import YOLO


def train(model_type: str, data_yaml: str, epochs: int = 40):
    configs = {
        "helmet": {"model": "yolov8n.pt", "imgsz": 320, "batch": 16, "name": "helmet_v1"},
        "seatbelt": {"model": "yolov8s.pt", "imgsz": 384, "batch": 12, "name": "seatbelt_v1"},
        "phone": {"model": "yolov8s.pt", "imgsz": 320, "batch": 16, "name": "phone_v1"},
    }

    cfg = configs[model_type]
    print(f"[INFO] Training {model_type} with config: {cfg}")

    model = YOLO(cfg["model"])
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device=0,
        project="models",
        name=cfg["name"],
        half=True,       # FP16 training to stay below 3.2 GB VRAM
        workers=2,
        save=True,
    )
    print(f"[SUCCESS] Best weights saved to models/{cfg['name']}/weights/best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom YOLO secondary violation models")
    parser.add_argument("--type", choices=["helmet", "seatbelt", "phone"], required=True, help="Violation detector type")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset YAML file")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    args = parser.parse_args()

    train(args.type, args.data, args.epochs)

