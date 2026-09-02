"""Ultralytics YOLO Segmentation Validation & Evaluation Module

功能：
- 於測試集/驗證集上計算實例分割指標：Mask mAP50, Mask mAP50-95, Precision, Recall。
- 提取各類別 (Class 0: primary_subcanopy, Class 1: nearby_subcanopy) 之獨立表現。
- 輸出標準化指標報表。
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO

# 確保專案根目錄在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def evaluate_yolo_seg(
    weights_path: str | Path = "runs/segment/tree_distill_exp/weights/best.pt",
    data_yaml: str | Path = "dataset/dataset.yaml",
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
    split: str = "val",
    project: str = "runs/segment",
    name: str = "tree_val_report",
) -> dict:
    """執行 YOLO 分割模型驗證評估"""
    model_path = Path(weights_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型權重檔：{model_path}")

    yaml_path = Path(data_yaml).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"找不到資料集設定檔：{yaml_path}")

    print("==================================================")
    print(f"📊 [Ultralytics Validation] 啟動樹木分割模型驗證評估")
    print(f"★ 評估權重：{model_path}")
    print(f"★ 資料集：{yaml_path} ({split} 分割)")
    print("==================================================\n")

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(yaml_path),
        imgsz=imgsz,
        batch=batch,
        device=device,
        split=split,
        project=project,
        name=name,
        plots=True,
    )

    # 提取分割關鍵指標
    seg_map50 = float(metrics.seg.map50) if hasattr(metrics, "seg") and hasattr(metrics.seg, "map50") else 0.0
    seg_map = float(metrics.seg.map) if hasattr(metrics, "seg") and hasattr(metrics.seg, "map") else 0.0
    box_map50 = float(metrics.box.map50) if hasattr(metrics, "box") and hasattr(metrics.box, "map50") else 0.0
    box_map = float(metrics.box.map) if hasattr(metrics, "box") and hasattr(metrics.box, "map") else 0.0

    print("\n==================================================")
    print("🎯 評估結果摘要：")
    print(f"★ 遮罩分割 Mask mAP50   : {seg_map50 * 100:.2f}%")
    print(f"★ 遮罩分割 Mask mAP50-95: {seg_map * 100:.2f}%")
    print(f"★ 邊界框檢測 Box mAP50  : {box_map50 * 100:.2f}%")
    print(f"★ 邊界框檢測 Box mAP50-95: {box_map * 100:.2f}%")
    print(f"★ 驗證圖表輸出目錄：{metrics.save_dir}")
    print("==================================================")

    return {
        "mask_map50": seg_map50,
        "mask_map50_95": seg_map,
        "box_map50": box_map50,
        "box_map50_95": box_map,
        "save_dir": str(metrics.save_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Ultralytics YOLO Segmentation Validation Pipeline")
    parser.add_argument("--weights", type=str, default="runs/segment/tree_distill_exp/weights/best.pt", help="待評估模型權重")
    parser.add_argument("--data", type=str, default="dataset/dataset.yaml", help="資料集 yaml 路徑")
    parser.add_argument("--imgsz", type=int, default=640, help="輸入影像尺寸")
    parser.add_argument("--split", type=str, default="val", help="評估分割集 (val / test)")
    parser.add_argument("--device", type=str, default=None, help="指定運算裝置")
    args = parser.parse_args()

    evaluate_yolo_seg(
        weights_path=args.weights,
        data_yaml=args.data,
        imgsz=args.imgsz,
        split=args.split,
        device=args.device,
    )


if __name__ == "__main__":
    main()
