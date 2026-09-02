"""SAM 3 Focus-Primary & Nearby Canopy to YOLO Dataset Generator

完整自動化管線 (SAM 3 -> YOLO Dataset Export)：
1. 雙向提示與綠色保護扣除 (Positive + Negative Prompt + Green Color Safeguard)：
   - 正向鎖定全圖次冠層，負向鎖定樹幹木質部。
   - 引入 ExG/HSV 綠色植被保護機制，防止稀疏葉片被誤判為木頭挖除。
2. 稀疏樹免疫之強健主體評分 (Robust Primary Scoring)：
   - 形態學外包絡空間面積 (Envelope Area)：以樹冠立體骨架評分，避免稀疏樹輸給濃密鄰居樹。
   - 高斯空間中心衰減 (Gaussian Center Priority)：精準鎖定中央主樹。
3. 自動生成 Ultralytics YOLO-Seg 規範之訓練資料集：
   - 自動生成多邊形標註檔 (labels/train/{stem}.txt)：
     * Class 0: primary_subcanopy (主體樹次冠層)
     * Class 1: nearby_subcanopy (周邊樹次冠層)
   - 自動將影像歸檔 (images/train/{stem}.jpeg)。
   - 自動生成資料集設定檔 (dataset.yaml)。
"""

import argparse
import shutil
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch

# 確保專案根目錄在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def get_optimal_device() -> str:
    """自動偵測運算裝置 (優先順序: CUDA -> CPU)"""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def extract_green_vegetation_mask(image_np: np.ndarray) -> np.ndarray:
    """計算超綠指數 (ExG) 與 HSV 綠色區間，產出嚴謹的綠色植物保護遮罩"""
    r = image_np[:, :, 0].astype(float)
    g = image_np[:, :, 1].astype(float)
    b = image_np[:, :, 2].astype(float)
    rgb_sum = r + g + b + 1e-5
    norm_exg = (2 * g - r - b) / rgb_sum

    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    h_chan = hsv[:, :, 0]
    s_chan = hsv[:, :, 1]

    is_hsv_green = (h_chan >= 25) & (h_chan <= 95) & (s_chan >= 15)
    is_exg_green = norm_exg > 0.015

    return is_hsv_green | is_exg_green


def compute_canopy_envelope(binary_mask: np.ndarray, radius_px: int = 35) -> np.ndarray:
    """利用形態學閉運算填平樹葉間隙，計算該樹所撐開的「樹冠外包絡立體空間 (Canopy Envelope)」"""
    if np.sum(binary_mask) == 0:
        return np.zeros_like(binary_mask, dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius_px, radius_px))
    closed = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return closed.astype(bool)


def calculate_focus_primary_scores(
    masks: list[np.ndarray],
    boxes: list[np.ndarray],
    scores: list[float],
    image_shape: tuple[int, int],
    weight_envelope: float = 0.35,
    weight_center: float = 0.40,
    weight_anchor: float = 0.15,
    weight_conf: float = 0.10,
) -> list[dict]:
    """計算強健主體評分 (稀疏樹免疫，防止兩側濃密鄰居樹篡位)"""
    h, w = image_shape[:2]
    target_cx = w * 0.50
    target_cy = h * 0.55

    sigma_x = w * 0.22
    sigma_y = h * 0.32

    envelope_list = [compute_canopy_envelope(m, radius_px=int(min(w, h) * 0.03)) for m in masks]
    envelope_areas = [int(np.sum(env)) for env in envelope_list]
    max_envelope_area = max(envelope_areas) if len(envelope_areas) > 0 and max(envelope_areas) > 0 else 1

    candidate_results = []

    for idx, mask in enumerate(masks):
        env_val = envelope_areas[idx]
        env_score = env_val / max_envelope_area
        leaf_pixels = int(np.sum(mask))

        y_indices, x_indices = np.where(envelope_list[idx])
        if len(x_indices) > 0:
            cx = float(np.mean(x_indices))
            cy = float(np.mean(y_indices))
            dx = (cx - target_cx) / sigma_x
            dy = (cy - target_cy) / sigma_y
            gaussian_center_score = float(np.exp(-0.5 * (dx**2 + dy**2)))
            dist_to_center = float(np.sqrt((cx - target_cx) ** 2 + (cy - target_cy) ** 2))
        else:
            cx, cy = 0.0, 0.0
            gaussian_center_score = 0.0
            dist_to_center = np.sqrt(target_cx**2 + target_cy**2)

        bottom_region = mask[int(h * 0.60):, int(w * 0.30):int(w * 0.70)]
        has_anchor = 1.0 if np.sum(bottom_region) > (h * w * 0.001) else 0.0

        conf_score = float(scores[idx])

        composite_score = (
            weight_envelope * env_score +
            weight_center * gaussian_center_score +
            weight_anchor * has_anchor +
            weight_conf * conf_score
        )

        density = (leaf_pixels / env_val * 100.0) if env_val > 0 else 0.0
        porosity = max(0.0, 100.0 - density)

        candidate_results.append({
            "index": idx,
            "leaf_pixels": leaf_pixels,
            "envelope_pixels": env_val,
            "envelope_score": env_score,
            "density_percentage": density,
            "porosity_percentage": porosity,
            "centroid": (cx, cy),
            "distance_to_center": dist_to_center,
            "center_score": gaussian_center_score,
            "has_anchor": bool(has_anchor),
            "confidence_score": conf_score,
            "composite_score": composite_score,
            "box": boxes[idx] if idx < len(boxes) else None,
            "envelope_mask": envelope_list[idx],
        })

    return candidate_results


def mask_to_yolo_polygon_lines(
    mask: np.ndarray,
    class_id: int,
    image_shape: tuple[int, int],
    epsilon_ratio: float = 0.002,
    min_area_px: int = 80,
) -> list[str]:
    """將二值遮罩 (Binary Mask) 轉為 YOLO-Seg 規範之正規化多邊形字串清單

    格式: <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn> (座標 0.0 ~ 1.0)
    """
    h, w = image_shape[:2]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area_px:
            continue

        # Douglas-Peucker 演算法多邊形頂點精簡
        epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) < 3:
            continue

        coords = []
        for pt in approx:
            px, py = pt[0]
            norm_x = max(0.0, min(1.0, float(px) / w))
            norm_y = max(0.0, min(1.0, float(py) / h))
            coords.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])

        line = f"{class_id} " + " ".join(coords)
        lines.append(line)

    return lines


def export_yolo_dataset(
    image_path: str | Path,
    primary_mask: np.ndarray,
    nearby_masks: list[np.ndarray],
    dataset_dir: str | Path = "dataset",
    split: str = "train",
) -> dict:
    """自動建立 Ultralytics 規範的資料夾結構並寫入標註檔與 dataset.yaml"""
    ds_path = Path(dataset_dir).resolve()
    img_file = Path(image_path).resolve()
    stem = img_file.stem
    ext = img_file.suffix

    images_dir = ds_path / "images" / split
    labels_dir = ds_path / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 1. 複製影像至 dataset/images/{split}/
    target_img_path = images_dir / f"{stem}{ext}"
    shutil.copy2(img_file, target_img_path)

    # 2. 轉換多邊形座標並寫入 dataset/labels/{split}/{stem}.txt
    h, w = primary_mask.shape[:2]
    all_yolo_lines = []

    # Class 0: 主體樹次冠層 (Primary Subcanopy)
    primary_lines = mask_to_yolo_polygon_lines(primary_mask, class_id=0, image_shape=(h, w))
    all_yolo_lines.extend(primary_lines)

    # Class 1: 周邊樹次冠層 (Nearby Subcanopies)
    for n_mask in nearby_masks:
        nearby_lines = mask_to_yolo_polygon_lines(n_mask, class_id=1, image_shape=(h, w))
        all_yolo_lines.extend(nearby_lines)

    target_label_path = labels_dir / f"{stem}.txt"
    with open(target_label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_yolo_lines) + "\n")

    # 3. 自動生成/更新 dataset.yaml
    yaml_path = ds_path / "dataset.yaml"
    yaml_content = f"""# Ultralytics YOLO Segmentation Dataset Configuration
# Auto-generated by simpleSam

path: {ds_path}  # 資料集根目錄
train: images/train  # 訓練集影像相對路徑
val: images/val      # 驗證集影像相對路徑

# 類別對應定義
names:
  0: primary_subcanopy
  1: nearby_subcanopy
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    return {
        "dataset_yaml": str(yaml_path),
        "exported_image": str(target_img_path),
        "exported_label": str(target_label_path),
        "polygon_count_primary": len(primary_lines),
        "polygon_count_nearby": len(all_yolo_lines) - len(primary_lines),
        "total_polygons": len(all_yolo_lines),
    }


def segment_and_generate_yolo_dataset(
    image_path: str | Path,
    checkpoint_path: str | Path = "sam3.pt",
    prompt: str = "tree subcanopy",
    negative_prompt: str = "tree trunk, tree branch",
    confidence_threshold: float = 0.25,
    negative_threshold: float = 0.155,
    output_dir: str | Path = "simpleSam/output",
    dataset_dir: str | Path = "dataset",
    device: str | None = None,
) -> dict:
    """執行分割並自動產出 Ultralytics YOLO 訓練檔"""
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"找不到輸入影像：{image_path}")

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到模型權重：{checkpoint_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = get_optimal_device()

    print(f"[Device] 運算裝置：{device}")
    print(f"[Model] 正在載入 SAM 3 模型 ({ckpt_path})...")

    model = build_sam3_image_model(checkpoint_path=str(ckpt_path), device=device)
    model = model.to(device).float()
    processor = Sam3Processor(model, device=device, confidence_threshold=confidence_threshold)

    print(f"[Image] 讀取影像：{img_path}")
    raw_image = Image.open(img_path).convert("RGB")
    image_np = np.array(raw_image)
    h, w = image_np.shape[:2]
    total_pixels = h * w

    state = processor.set_image(raw_image)

    # 1. 正向次冠層推論
    print(f"[Positive Inference] 提示詞：'{prompt}' | 門檻值：{confidence_threshold}")
    result_subcanopy = processor.set_text_prompt(prompt=prompt, state=state)
    raw_subcanopy_masks = result_subcanopy.get("masks", None)
    boxes = result_subcanopy.get("boxes", None)
    scores = result_subcanopy.get("scores", None)

    num_detections = len(scores) if scores is not None else 0
    if num_detections == 0 or raw_subcanopy_masks is None:
        print("[Warning] 未偵測到任何符合門檻之 Subcanopy。")
        return {"detection_count": 0}

    # 2. 負向樹幹推論
    trunk_mask = np.zeros((h, w), dtype=bool)
    if negative_prompt and negative_prompt.strip():
        print(f"[Negative Inference] 負向提示詞：'{negative_prompt}' | 門檻值：{negative_threshold}...")
        processor.set_confidence_threshold(negative_threshold)
        result_trunk = processor.set_text_prompt(prompt=negative_prompt, state=state)
        raw_trunk_masks = result_trunk.get("masks", None)
        if raw_trunk_masks is not None and len(raw_trunk_masks) > 0:
            trunk_np = raw_trunk_masks.cpu().numpy().astype(bool)
            if trunk_np.ndim == 4:
                trunk_np = trunk_np.squeeze(1)
            for m in trunk_np:
                trunk_mask |= m

    # 3. 綠色保護扣除
    green_mask = extract_green_vegetation_mask(image_np)
    wood_to_remove = trunk_mask & (~green_mask)
    excluded_wood_pixels = int(np.sum(wood_to_remove))

    # 4. 布林扣除樹幹
    subcanopy_np = raw_subcanopy_masks.cpu().numpy().astype(bool)
    if subcanopy_np.ndim == 4:
        subcanopy_np = subcanopy_np.squeeze(1)

    cleaned_masks_list = []
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for i in range(num_detections):
        raw_m = subcanopy_np[i]
        pure_m = raw_m & (~wood_to_remove)
        pure_m = cv2.morphologyEx(pure_m.astype(np.uint8), cv2.MORPH_OPEN, kernel_clean).astype(bool)
        cleaned_masks_list.append(pure_m)

    box_list = [boxes[i].cpu().numpy() for i in range(num_detections)] if boxes is not None else []
    score_list = [float(scores[i].item()) for i in range(num_detections)]

    # 5. 主體與周邊樹評分
    candidate_metrics = calculate_focus_primary_scores(
        masks=cleaned_masks_list,
        boxes=box_list,
        scores=score_list,
        image_shape=(h, w),
    )

    candidate_metrics.sort(key=lambda x: x["composite_score"], reverse=True)
    primary_candidate = candidate_metrics[0]
    primary_idx = primary_candidate["index"]
    primary_clean_mask = cleaned_masks_list[primary_idx]
    primary_envelope_mask = primary_candidate["envelope_mask"]

    nearby_candidates = candidate_metrics[1:]
    nearby_masks_list = [cleaned_masks_list[c["index"]] for c in nearby_candidates]

    # 6. 自動生成 YOLO-Seg 資料檔
    print(f"\n[YOLO Export] 正在生成 Ultralytics 訓練資料集至目錄：{dataset_dir}...")
    yolo_export_info = export_yolo_dataset(
        image_path=img_path,
        primary_mask=primary_clean_mask,
        nearby_masks=nearby_masks_list,
        dataset_dir=dataset_dir,
        split="train",
    )

    # 7. 渲染視覺化成果圖
    overlay_img = image_np.copy()

    # 次要樹 (天藍)
    for cand in nearby_candidates:
        c_mask = cleaned_masks_list[cand["index"]]
        overlay_img[c_mask] = (overlay_img[c_mask] * 0.40 + np.array([60, 180, 255]) * 0.60).astype(np.uint8)
        cnts, _ = cv2.findContours(c_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_img, cnts, -1, color=(100, 220, 255), thickness=2)

    # 主體樹 (鮮綠 + 金黃外框)
    overlay_img[primary_clean_mask] = (overlay_img[primary_clean_mask] * 0.30 + np.array([30, 245, 60]) * 0.70).astype(np.uint8)
    primary_contours, _ = cv2.findContours(primary_clean_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_img, primary_contours, -1, color=(255, 230, 0), thickness=3)

    stem = img_path.stem
    overlay_output_path = out_dir / f"{stem}_yolo_segmentation_preview.png"
    Image.fromarray(overlay_img).save(overlay_output_path)

    print("==================================================")
    print(f"影像總面積：{total_pixels:,} px")
    print(f"★ 主體樹 (Class 0: primary_subcanopy)：{primary_candidate['leaf_pixels']:,} px")
    print(f"★ 周邊樹 (Class 1: nearby_subcanopy)：{len(nearby_candidates)} 棵")
    print("--------------------------------------------------")
    print(f"★ 資料集設定檔：{yolo_export_info['dataset_yaml']}")
    print(f"★ 匯出影像路徑：{yolo_export_info['exported_image']}")
    print(f"★ 匯出標註檔路徑：{yolo_export_info['exported_label']}")
    print(f"★ 標註多邊形總數：{yolo_export_info['total_polygons']} (主體: {yolo_export_info['polygon_count_primary']}, 周邊: {yolo_export_info['polygon_count_nearby']})")
    print(f"視覺化預覽圖：{overlay_output_path}")
    print("==================================================")

    return {
        "image_shape": (h, w),
        "primary_index": primary_idx,
        "yolo_export": yolo_export_info,
        "preview_overlay": str(overlay_output_path),
    }


def main():
    parser = argparse.ArgumentParser(description="SAM 3 Focus-Primary & Nearby Canopy to YOLO Dataset Generator")
    parser.add_argument("--image", type=str, default="data/test02.jpeg", help="輸入影像路徑")
    parser.add_argument("--checkpoint", type=str, default="sam3.pt", help="SAM 3 權重路徑")
    parser.add_argument("--prompt", type=str, default="tree subcanopy", help="正向提示詞")
    parser.add_argument("--negative-prompt", type=str, default="tree trunk, tree branch", help="負向排除提示詞")
    parser.add_argument("--threshold", type=float, default=0.25, help="正向置信度門檻值")
    parser.add_argument("--threshold-negative", type=float, default=0.155, help="負向排除門檻值")
    parser.add_argument("--output-dir", type=str, default="simpleSam/output", help="輸出預覽目錄")
    parser.add_argument("--dataset-dir", type=str, default="dataset", help="YOLO 資料集目錄")
    args = parser.parse_args()

    segment_and_generate_yolo_dataset(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        confidence_threshold=args.threshold,
        negative_threshold=args.threshold_negative,
        output_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
    )


if __name__ == "__main__":
    main()
