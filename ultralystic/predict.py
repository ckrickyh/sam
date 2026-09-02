"""Ultralytics YOLO Segmentation Inference & Foliage Density Pipeline

功能：
- 載入微調後之 YOLO-Seg 模型進行推論 (支援單圖、資料夾、影片)。
- 分離主體樹 (Class 0: primary_subcanopy) 與周邊樹 (Class 1: nearby_subcanopy)。
- 計算主體樹葉片像素總量、樹冠外包絡立體空間 (Envelope Area) 與樹冠透光率/孔隙度 (Canopy Porosity)。
- 輸出高畫質雙色標註圖 (主體: 鮮綠色, 周邊: 天空藍)。
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

# 確保專案根目錄在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def compute_canopy_envelope(binary_mask: np.ndarray, radius_px: int = 35) -> np.ndarray:
    """利用形態學閉運算填平樹葉間隙，計算該樹所撐開的「樹冠外包絡立體空間 (Canopy Envelope)」"""
    if np.sum(binary_mask) == 0:
        return np.zeros_like(binary_mask, dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius_px, radius_px))
    closed = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return closed.astype(bool)


def predict_tree_foliage(
    image_path: str | Path,
    weights_path: str | Path = "runs/segment/tree_distill_exp/weights/best.pt",
    conf_threshold: float = 0.25,
    imgsz: int = 640,
    device: str | None = None,
    output_dir: str | Path = "ultralystic/output",
) -> dict:
    """對單張影像進行樹木分割與葉密度精確計算"""
    img_path = Path(image_path).resolve()
    if not img_path.exists():
        raise FileNotFoundError(f"找不到影像：{img_path}")

    model_path = Path(weights_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型權重：{model_path}")

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_img = Image.open(img_path).convert("RGB")
    image_np = np.array(raw_img)
    h, w = image_np.shape[:2]
    total_pixels = h * w

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(img_path),
        conf=conf_threshold,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    r = results[0]
    primary_mask = np.zeros((h, w), dtype=bool)
    nearby_mask = np.zeros((h, w), dtype=bool)

    num_primary_clusters = 0
    num_nearby_clusters = 0

    if r.masks is not None and len(r.masks) > 0:
        boxes = r.boxes
        classes = boxes.cls.cpu().numpy().astype(int)
        masks_data = r.masks.data.cpu().numpy()

        for idx, cls_id in enumerate(classes):
            mask_raw = masks_data[idx]
            # 縮放回原始影像解析度
            mask_resized = cv2.resize(mask_raw.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            if cls_id == 0:
                primary_mask |= mask_resized
                num_primary_clusters += 1
            else:
                nearby_mask |= mask_resized
                num_nearby_clusters += 1

    primary_leaf_pixels = int(np.sum(primary_mask))
    nearby_leaf_pixels = int(np.sum(nearby_mask))

    # 計算主體樹外包絡立體空間與透光孔隙度
    primary_envelope = compute_canopy_envelope(primary_mask, radius_px=int(min(w, h) * 0.03))
    primary_envelope_pixels = int(np.sum(primary_envelope))

    foliage_density = (primary_leaf_pixels / primary_envelope_pixels * 100.0) if primary_envelope_pixels > 0 else 0.0
    canopy_porosity = max(0.0, 100.0 - foliage_density)

    # 渲染視覺化標記圖
    overlay_img = image_np.copy()
    if np.sum(nearby_mask) > 0:
        overlay_img[nearby_mask] = (overlay_img[nearby_mask] * 0.40 + np.array([60, 180, 255]) * 0.60).astype(np.uint8)
        cnts, _ = cv2.findContours(nearby_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_img, cnts, -1, color=(100, 220, 255), thickness=2)

    if np.sum(primary_mask) > 0:
        overlay_img[primary_mask] = (overlay_img[primary_mask] * 0.30 + np.array([30, 245, 60]) * 0.70).astype(np.uint8)
        cnts, _ = cv2.findContours(primary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_img, cnts, -1, color=(255, 230, 0), thickness=3)

    out_preview = out_dir / f"{img_path.stem}_yolo_pred.png"
    Image.fromarray(overlay_img).save(out_preview)

    print("==================================================")
    print(f"🌲 [YOLO Prediction] 樹木葉片分割完成：{img_path.name}")
    print(f"★ 主體樹葉片像素 (Primary Leaves) : {primary_leaf_pixels:,} px ({primary_leaf_pixels / total_pixels * 100:.2f}%)")
    print(f"★ 主體樹外包絡空間 (Canopy Envelope): {primary_envelope_pixels:,} px")
    print(f"★ 樹葉緊密度 (Foliage Density)   : {foliage_density:.2f}%")
    print(f"★ 樹冠孔隙度 (Canopy Porosity)    : {canopy_porosity:.2f}%")
    print(f"★ 周邊鄰居樹像素 (Nearby Trees)  : {nearby_leaf_pixels:,} px")
    print(f"★ 視覺化標記輸出路徑             : {out_preview}")
    print("==================================================")

    return {
        "image": str(img_path),
        "primary_leaf_pixels": primary_leaf_pixels,
        "primary_envelope_pixels": primary_envelope_pixels,
        "foliage_density_percentage": foliage_density,
        "canopy_porosity_percentage": canopy_porosity,
        "nearby_leaf_pixels": nearby_leaf_pixels,
        "preview_path": str(out_preview),
    }


def main():
    parser = argparse.ArgumentParser(description="Ultralytics YOLO Segmentation Prediction Pipeline")
    parser.add_argument("--image", type=str, required=True, help="待預測影像路徑")
    parser.add_argument("--weights", type=str, default="runs/segment/tree_distill_exp/weights/best.pt", help="微調後權重檔")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度門檻值")
    parser.add_argument("--imgsz", type=int, default=640, help="輸入影像解析度")
    parser.add_argument("--device", type=str, default=None, help="指定運算裝置")
    parser.add_argument("--output-dir", type=str, default="ultralystic/output", help="輸出預覽圖目錄")
    args = parser.parse_args()

    predict_tree_foliage(
        image_path=args.image,
        weights_path=args.weights,
        conf_threshold=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
