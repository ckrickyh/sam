"""Ultralytics YOLO26/11-seg Training Module for Tree Foliage Segmentation

依據 Ultralytics 官方 Agent Skills 最佳實踐：
- 支援 YOLO26-seg (NMS-Free 端到端) 與 YOLO11/v8-seg 分割架構。
- 內建小樣本微調策略：freeze=10 (凍結前 10 層骨幹)、close_mosaic=10 (保護中央幾何特徵)。
- 自動啟用 TensorBoard 與 CSV 指標紀錄。
- 自動適配 Apple Silicon (MPS)、CUDA GPU 與 CPU 運算加速。
"""

import argparse
import sys
from pathlib import Path
import torch
from ultralytics import YOLO

# 確保專案根目錄在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def get_optimal_training_device() -> str:
    """自動選取最佳硬體加速裝置 (優先順序: CUDA -> Apple MPS -> CPU)"""
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def train_yolo_seg(
    data_yaml: str | Path = "dataset/dataset.yaml",
    model_name: str = "yolo26m-seg.pt", # "yolo26s-seg.pt",
    epochs: int = 100,
    imgsz: int = 1024,
    batch: int = 16,
    freeze: int = 0,
    mosaic: float = 0.0,
    retina_masks: bool = True,
    box: float = 1.0,
    rect: bool = True,
    cls: float = 0.5,
    close_mosaic: int = 10,
    cos_lr: bool = True,
    lr0: float = 0.01, 
    lrf: float = 0.01, 
    patience: int = 30,
    degrees: float = 5.0,
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
    device: str | None = None,
    project: str = "runs/segment",
    name: str = "tree_distill_exp",
    exist_ok: bool = True,
    verbose: bool = True,
) -> dict:
    """啟動 YOLO 實例分割模型微調"""
    yaml_path = Path(data_yaml).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"找不到資料集設定檔：{yaml_path}")

    if device is None:
        device = get_optimal_training_device()

    print("==================================================")
    print(f"🚀 [Ultralytics Training] 啟動樹冠分割模型微調")
    print(f"★ 模型架構：{model_name}")
    print(f"★ 資料集設定檔：{yaml_path}")
    print(f"★ 運算裝置：{device}")
    print(f"★ 訓練輪數：{epochs} (早停耐心值: {patience})")
    print(f"★ 批次大小：{batch} | 解析度：{imgsz}x{imgsz}")
    print(f"★ 小樣本防過擬合：freeze={freeze}, close_mosaic={close_mosaic}")
    print(f"★ 輸出專案目錄：{project}/{name}")
    print("==================================================\n")

    # 1. 載入預訓練分割模型 (自動下載官方權重)
    try:
        model = YOLO(model_name)
        print(f'使用 {model_name} 進行微調')
    except Exception as e:
        fallback_model = "yolo26s-seg.pt"
        print(f"[Warning] 載入 {model_name} 失敗 ({e})，自動降級使用官方通用分割權重：{fallback_model}")
        model = YOLO(fallback_model)

    # 2. 啟動微調
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        freeze=freeze,
        mosaic=mosaic,
        retina_masks=retina_masks,
        box=box,
        rect=rect,
        cls=cls,
        close_mosaic=close_mosaic,
        cos_lr=cos_lr,
        lr0=lr0,
        lrf=lrf,
        degrees=degrees,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        patience=patience,
        device=device,
        project=project,
        name=name,
        exist_ok=exist_ok,
        verbose=verbose,
        plots=True,
    )

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(project) / name
    best_pt = save_dir / "weights" / "best.pt"

    print("\n==================================================")
    print(f"🎉 訓練完成！")
    print(f"★ 最佳模型權重：{best_pt}")
    print(f"★ 全套分析圖表目錄：{save_dir}")
    print(f"★ 核心圖表：{save_dir / 'results.png'}")
    print(f"★ F1 曲線圖：{save_dir / 'MaskF1_curve.png'}")
    print(f"★ PR 曲線圖：{save_dir / 'MaskPR_curve.png'}")
    print("==================================================")

    return {
        "save_dir": str(save_dir),
        "best_weights": str(best_pt),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Ultralytics YOLO Segmentation Training Pipeline")
    parser.add_argument("--data", type=str, default="dataset/dataset.yaml", help="資料集 yaml 路徑")
    parser.add_argument("--model", type=str, default="yolo26s-seg.pt", help="預訓練權重名稱 (如 yolo26s-seg.pt, yolo26m-seg.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="訓練輪數")
    parser.add_argument("--imgsz", type=int, default=640, help="輸入影像尺寸")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--freeze", type=int, default=10, help="凍結骨幹層數")
    parser.add_argument("--close-mosaic", type=int, default=10, help="最後幾輪關閉 Mosaic 增強")
    parser.add_argument("--patience", type=int, default=30, help="早停耐心輪數")
    parser.add_argument("--device", type=str, default=None, help="指定運算裝置 (0, mps, cpu)")
    parser.add_argument("--project", type=str, default="runs/segment", help="輸出主目錄")
    parser.add_argument("--name", type=str, default="tree_distill_exp", help="實驗名稱")
    args = parser.parse_args()

    train_yolo_seg(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        freeze=args.freeze,
        close_mosaic=args.close_mosaic,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
