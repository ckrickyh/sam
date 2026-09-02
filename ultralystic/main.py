"""Ultralytics YOLO Segmentation Pipeline Main Entrypoint

整合全流程：
1. 資料集訓練 (train)
2. 模型精度評估 (val)
3. 影像樹葉推論 (predict)
4. 模型導出與測速 (export)
"""

import argparse
import sys
from pathlib import Path

# 確保專案根目錄在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ultralystic.train import train_yolo_seg
from ultralystic.val import evaluate_yolo_seg
from ultralystic.predict import predict_tree_foliage
from ultralystic.export import export_and_benchmark


def main():
    parser = argparse.ArgumentParser(description="Ultralytics YOLO-Seg MLOps Workflow")
    subparsers = parser.add_subparsers(dest="command", help="欲執行的指令")

    # 1. train
    p_train = subparsers.add_parser("train", help="微調訓練 YOLO 分割模型")
    p_train.add_argument("--data", type=str, default="dataset/dataset.yaml", help="資料集設定檔")
    p_train.add_argument("--model", type=str, default="yolo26s-seg.pt", help="模型名稱")
    p_train.add_argument("--epochs", type=int, default=100, help="訓練輪數")
    p_train.add_argument("--batch", type=int, default=16, help="批次大小")
    p_train.add_argument("--imgsz", type=int, default=640, help="影像大小")
    p_train.add_argument("--freeze", type=int, default=10, help="凍結骨幹層數")
    p_train.add_argument("--device", type=str, default=None, help="運算裝置")

    # 2. val
    p_val = subparsers.add_parser("val", help="評估模型精度 (mAP, F1)")
    p_val.add_argument("--weights", type=str, default="runs/segment/tree_distill_exp/weights/best.pt", help="權重路徑")
    p_val.add_argument("--data", type=str, default="dataset/dataset.yaml", help="資料集設定檔")
    p_val.add_argument("--split", type=str, default="val", help="評估分割 (val / test)")

    # 3. predict
    p_pred = subparsers.add_parser("predict", help="執行樹木分割與葉密度預測")
    p_pred.add_argument("--image", type=str, required=True, help="影像路徑")
    p_pred.add_argument("--weights", type=str, default="runs/segment/tree_distill_exp/weights/best.pt", help="權重路徑")
    p_pred.add_argument("--conf", type=float, default=0.25, help="置信度")

    # 4. export
    p_exp = subparsers.add_parser("export", help="匯出 ONNX / CoreML 模型並測速")
    p_exp.add_argument("--weights", type=str, default="runs/segment/tree_distill_exp/weights/best.pt", help="權重路徑")
    p_exp.add_argument("--formats", nargs="+", default=["onnx", "coreml"], help="導出格式")

    args = parser.parse_args()

    if args.command == "train":
        train_yolo_seg(
            data_yaml=args.data,
            model_name=args.model,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            freeze=args.freeze,
            device=args.device,
        )
    elif args.command == "val":
        evaluate_yolo_seg(
            weights_path=args.weights,
            data_yaml=args.data,
            split=args.split,
        )
    elif args.command == "predict":
        predict_tree_foliage(
            image_path=args.image,
            weights_path=args.weights,
            conf_threshold=args.conf,
        )
    elif args.command == "export":
        export_and_benchmark(
            weights_path=args.weights,
            formats=args.formats,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
