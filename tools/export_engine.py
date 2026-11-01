"""Model Export & Optimization Utility.

Exports trained YOLO models to ONNX and TensorRT FP16 engines
for ultra-high FPS inference on NVIDIA RTX GPUs.
"""

import argparse
import os
import sys
from ultralytics import YOLO


def export_model(
    model_path: str,
    export_format: str = "engine",
    imgsz: int = 640,
    half: bool = True,
    device: int = 0,
):
    if not os.path.exists(model_path):
        print(f"[ERROR] Model path does not exist: {model_path}")
        sys.exit(1)

    print(f"\n[INFO] Starting export for '{model_path}'...")
    print(f"       Target Format: {export_format.upper()} | Precision: {'FP16' if half else 'FP32'}")
    print(f"       Resolution: {imgsz}x{imgsz} | Device: {device}\n")

    model = YOLO(model_path)

    try:
        exported_path = model.export(
            format=export_format,
            imgsz=imgsz,
            half=half,
            device=device,
            dynamic=False,
            simplify=True,
        )
        print(f"\n[SUCCESS] Successfully exported to: {exported_path}")
        return exported_path
    except Exception as e:
        print(f"\n[ERROR] Export to {export_format} failed: {e}")
        if export_format == "engine":
            print(
                "[TIP] To compile TensorRT on Windows, ensure the NVIDIA TensorRT SDK "
                "is installed and matched with your CUDA 12.4 runtime."
            )
            print("      Alternatively, consider exporting to ONNX: --format onnx")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO models to ONNX / TensorRT engine")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Path to .pt weights file")
    parser.add_argument("--format", type=str, choices=["engine", "onnx", "torchscript"], default="engine", help="Target format")
    parser.add_argument("--imgsz", type=int, default=640, help="Input dimension")
    parser.add_argument("--fp32", action="store_true", help="Force FP32 precision instead of FP16")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    args = parser.parse_args()

    export_model(
        model_path=args.model,
        export_format=args.format,
        imgsz=args.imgsz,
        half=not args.fp32,
        device=args.device,
    )

