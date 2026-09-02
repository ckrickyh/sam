"""SAM 3 Focus-Primary & Nearby Canopy Batch Processor & YOLO Dataset Generator

批次自動化管線 (Batch SAM 3 -> Ultralytics YOLO-Seg Dataset Pipeline)：
1. 支援資料夾批次掃描 (支援 .jpg, .jpeg, .png, .bmp, .webp)。
2. 模型單次載入記憶體 (Model Reuse)，避免逐張反覆載入 3.4GB 權重之巨大開銷。
3. 自動執行 Train / Val 切分 (例如 80% 訓練集、20% 驗證集)。
4. 階層超遮罩過濾 (Hierarchical Mega-Mask Suppression)：
   - 剔除 SAM 3 產生的全景多木聚合巨型遮罩，防止背景巨木篡位。
5. 每張照片自動執行：
   - 雙向提示與色彩保護 (Positive Prompt + Negative Prompt + Green Color Safeguard)。
   - 前景主木精準評分 (Envelope Area + Gaussian Center Priority + Trunk Base Anchor)。
   - 主體樹 (Class 0: primary_subcanopy) 與周邊樹 (Class 1: nearby_subcanopy) 自動分類。
   - 多邊形精簡並匯出 YOLO 格式標註 (.txt)。
6. 自動產生並維護包含完整資料集統計的 dataset.yaml。
"""

import argparse
import random
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


def filter_hierarchical_super_masks(
    masks: list[np.ndarray],
    boxes: list[np.ndarray],
    scores: list[float],
    image_shape: tuple[int, int],
    containment_threshold: float = 0.80,
    area_ratio_threshold: float = 0.45,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """剔除 SAM 3 產生的全景聚合巨型超遮罩 (Mega Super-Masks)"""
    h, w = image_shape[:2]
    total_img_pixels = h * w
    num_masks = len(masks)
    if num_masks <= 1:
        return masks, boxes, scores

    mask_areas = [int(np.sum(m)) for m in masks]
    keep_flags = [True] * num_masks

    for i in range(num_masks):
        if not keep_flags[i]:
            continue
        area_i = mask_areas[i]
        if area_i < total_img_pixels * area_ratio_threshold:
            continue

        contained_count = 0
        for j in range(num_masks):
            if i == j or not keep_flags[j]:
                continue
            area_j = mask_areas[j]
            if area_j < area_i * 0.75:
                intersection = np.sum(masks[i] & masks[j])
                if intersection >= area_j * containment_threshold:
                    contained_count += 1

        if contained_count >= 1:
            keep_flags[i] = False

    filtered_masks = [masks[i] for i in range(num_masks) if keep_flags[i]]
    filtered_boxes = [boxes[i] for i in range(num_masks) if keep_flags[i]]
    filtered_scores = [scores[i] for i in range(num_masks) if keep_flags[i]]

    return filtered_masks, filtered_boxes, filtered_scores


def calculate_focus_primary_scores(
    masks: list[np.ndarray],
    boxes: list[np.ndarray],
    scores: list[float],
    image_shape: tuple[int, int],
    weight_envelope: float = 0.25,
    weight_center: float = 0.45,
    weight_anchor: float = 0.20,
    weight_conf: float = 0.10,
) -> list[dict]:
    """計算強健主體評分 (前景樹優先，防止側邊背景巨木篡位)"""
    h, w = image_shape[:2]
    target_cx = w * 0.48
    target_cy = h * 0.55

    sigma_x = w * 0.25
    sigma_y = h * 0.35

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

        bottom_region = mask[int(h * 0.55):, int(w * 0.25):int(w * 0.75)]
        has_anchor = 1.0 if np.sum(bottom_region) > (h * w * 0.0008) else 0.0

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
    """將二值遮罩轉為 YOLO-Seg 規範之多邊形字串清單"""
    h, w = image_shape[:2]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area_px:
            continue

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


def process_single_image(
    image_path: Path,
    processor: Sam3Processor,
    prompt: str,
    negative_prompt: str,
    confidence_threshold: float,
    negative_threshold: float,
    dataset_dir: Path,
    split: str,
    output_dir: Path | None = None,
) -> dict:
    """處理單張影像之推論、後處理與 YOLO 資料集寫入"""
    raw_image = Image.open(image_path).convert("RGB")
    image_np = np.array(raw_image)
    h, w = image_np.shape[:2]
    stem = image_path.stem
    ext = image_path.suffix

    state = processor.set_image(raw_image)

    # 1. 正向推論
    processor.set_confidence_threshold(confidence_threshold)
    result_subcanopy = processor.set_text_prompt(prompt=prompt, state=state)
    raw_subcanopy_masks = result_subcanopy.get("masks", None)
    boxes = result_subcanopy.get("boxes", None)
    scores = result_subcanopy.get("scores", None)

    num_detections = len(scores) if scores is not None else 0
    if num_detections == 0 or raw_subcanopy_masks is None:
        target_img_dir = dataset_dir / "images" / split
        target_lbl_dir = dataset_dir / "labels" / split
        target_img_dir.mkdir(parents=True, exist_ok=True)
        target_lbl_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_img_dir / f"{stem}{ext}")
        with open(target_lbl_dir / f"{stem}.txt", "w", encoding="utf-8") as f:
            f.write("")
        return {
            "image": image_path.name,
            "status": "No Detections (Saved as background)",
            "split": split,
            "primary_pixels": 0,
            "nearby_pixels": 0,
            "polygons": 0,
        }

    # 2. 負向樹幹推論
    trunk_mask = np.zeros((h, w), dtype=bool)
    if negative_prompt and negative_prompt.strip():
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

    # 5. 階層超遮罩過濾 (Mega-mask Suppression)
    filtered_masks, filtered_boxes, filtered_scores = filter_hierarchical_super_masks(
        masks=cleaned_masks_list,
        boxes=box_list,
        scores=score_list,
        image_shape=(h, w),
    )

    # 6. 主體樹與周邊樹評分排序
    candidate_metrics = calculate_focus_primary_scores(
        masks=filtered_masks,
        boxes=filtered_boxes,
        scores=filtered_scores,
        image_shape=(h, w),
    )

    candidate_metrics.sort(key=lambda x: x["composite_score"], reverse=True)
    primary_candidate = candidate_metrics[0]
    primary_clean_mask = filtered_masks[primary_candidate["index"]]
    nearby_candidates = candidate_metrics[1:]

    # 7. 寫入 YOLO-Seg 資料集
    target_img_dir = dataset_dir / "images" / split
    target_lbl_dir = dataset_dir / "labels" / split
    target_img_dir.mkdir(parents=True, exist_ok=True)
    target_lbl_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_path, target_img_dir / f"{stem}{ext}")

    all_yolo_lines = []
    # Class 0: 主體樹
    p_lines = mask_to_yolo_polygon_lines(primary_clean_mask, class_id=0, image_shape=(h, w))
    all_yolo_lines.extend(p_lines)

    # Class 1: 周邊樹
    nearby_pixels = 0
    for cand in nearby_candidates:
        c_mask = filtered_masks[cand["index"]]
        nearby_pixels += cand["leaf_pixels"]
        n_lines = mask_to_yolo_polygon_lines(c_mask, class_id=1, image_shape=(h, w))
        all_yolo_lines.extend(n_lines)

    with open(target_lbl_dir / f"{stem}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(all_yolo_lines) + "\n")

    # 8. 渲染預覽圖 (若有指定 output_dir)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        overlay_img = image_np.copy()

        # 周邊樹渲染為天空藍
        for cand in nearby_candidates:
            c_mask = filtered_masks[cand["index"]]
            overlay_img[c_mask] = (overlay_img[c_mask] * 0.40 + np.array([60, 180, 255]) * 0.60).astype(np.uint8)
            cnts, _ = cv2.findContours(c_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_img, cnts, -1, color=(100, 220, 255), thickness=2)

        # 主體樹渲染為亮綠色 + 黃色描邊
        overlay_img[primary_clean_mask] = (overlay_img[primary_clean_mask] * 0.30 + np.array([30, 245, 60]) * 0.70).astype(np.uint8)
        p_cnts, _ = cv2.findContours(primary_clean_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_img, p_cnts, -1, color=(255, 230, 0), thickness=3)

        Image.fromarray(overlay_img).save(output_dir / f"{stem}_preview.png")

    return {
        "image": image_path.name,
        "status": "Success",
        "split": split,
        "primary_pixels": primary_candidate["leaf_pixels"],
        "nearby_pixels": nearby_pixels,
        "primary_density": primary_candidate["density_percentage"],
        "polygons": len(all_yolo_lines),
    }


def batch_generate_yolo_dataset(
    input_dir: str | Path = "data",
    checkpoint_path: str | Path = "sam3.pt",
    prompt: str = "tree subcanopy",
    negative_prompt: str = "tree trunk, tree branch",
    confidence_threshold: float = 0.25,
    negative_threshold: float = 0.155,
    dataset_dir: str | Path = "dataset",
    output_dir: str | Path = "simpleSam/output",
    val_ratio: float = 0.20,
    max_images: int | None = None,
    seed: int = 42,
    device: str | None = None,
) -> dict:
    """批次掃描 input_dir 中的所有照片並自動生成完整的 YOLO 資料集"""
    in_dir = Path(input_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"找不到輸入目錄：{input_dir}")

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到模型權重：{checkpoint_path}")

    ds_path = Path(dataset_dir).resolve()
    out_dir = Path(output_dir).resolve() if output_dir else None

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
    image_files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix in valid_exts])

    if len(image_files) == 0:
        print(f"[Warning] 在目錄 '{input_dir}' 中未找到任何支援的影像檔案。")
        return {"total_images": 0, "processed": 0}

    if max_images is not None and max_images > 0:
        image_files = image_files[:max_images]

    total_images = len(image_files)
    print(f"==================================================")
    print(f"[Batch Mode] 找到 {total_images} 張待處理影像：{input_dir}")
    print(f"[Dataset Output] 目標 YOLO 資料集目錄：{ds_path}")
    print(f"==================================================")

    random.seed(seed)
    indices = list(range(total_images))
    random.shuffle(indices)

    num_val = int(round(total_images * val_ratio))
    if num_val == 0 and total_images >= 5:
        num_val = 1
    val_indices = set(indices[:num_val])

    if device is None:
        device = get_optimal_device()

    print(f"[Model] 正在載入 SAM 3 權重 ({ckpt_path}) 至裝置：{device}...")
    model = build_sam3_image_model(checkpoint_path=str(ckpt_path), device=device)
    model = model.to(device).float()
    processor = Sam3Processor(model, device=device, confidence_threshold=confidence_threshold)
    print("[Model] 權重載入完畢，正式開始批次推論！\n")

    report_list = []
    for idx, img_path in enumerate(image_files, start=1):
        split = "val" if (idx - 1) in val_indices else "train"
        print(f"[{idx}/{total_images}] 正在處理 ({split}): {img_path.name}...")
        try:
            res = process_single_image(
                image_path=img_path,
                processor=processor,
                prompt=prompt,
                negative_prompt=negative_prompt,
                confidence_threshold=confidence_threshold,
                negative_threshold=negative_threshold,
                dataset_dir=ds_path,
                split=split,
                output_dir=out_dir,
            )
            report_list.append(res)
            print(f"    -> 成功！主樹像素: {res.get('primary_pixels', 0):,} px | 多邊形數: {res.get('polygons', 0)}")
        except Exception as e:
            print(f"    -> 處理失敗：{e}")
            report_list.append({"image": img_path.name, "status": f"Error: {e}", "split": split})

    yaml_path = ds_path / "dataset.yaml"
    yaml_content = f"""# Ultralytics YOLO Segmentation Dataset Configuration
# Auto-generated by simpleSam Batch Pipeline
# Total Images: {total_images} (Train: {total_images - num_val}, Val: {num_val})

path: {ds_path}  # 資料集根目錄
train: images/train  # 訓練集
val: images/val      # 驗證集

# 類別定義 (0-based contiguous)
names:
  0: primary_subcanopy
  1: nearby_subcanopy
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print("\n==================================================")
    print(f"🎉 批次處理完成！共處理 {len(report_list)}/{total_images} 張照片。")
    print(f"★ 訓練集 (Train): {total_images - num_val} 張")
    print(f"★ 驗證集 (Val): {num_val} 張")
    print(f"★ 資料集設定檔：{yaml_path}")
    print("==================================================")

    return {
        "total_images": total_images,
        "train_count": total_images - num_val,
        "val_count": num_val,
        "dataset_yaml": str(yaml_path),
        "results": report_list,
    }


def main():
    parser = argparse.ArgumentParser(description="SAM 3 Batch Processor & Ultralytics YOLO Dataset Generator (Fixed Hierarchy)")
    parser.add_argument("--input-dir", type=str, default="data", help="輸入影像目錄")
    parser.add_argument("--checkpoint", type=str, default="sam3.pt", help="SAM 3 權重路徑")
    parser.add_argument("--prompt", type=str, default="tree subcanopy", help="正向提示詞")
    parser.add_argument("--negative-prompt", type=str, default="tree trunk, tree branch", help="負向排除提示詞")
    parser.add_argument("--threshold", type=float, default=0.25, help="正向置信度門檻值")
    parser.add_argument("--threshold-negative", type=float, default=0.155, help="負向排除門檻值")
    parser.add_argument("--dataset-dir", type=str, default="dataset", help="輸出 YOLO 資料集目錄")
    parser.add_argument("--output-dir", type=str, default="simpleSam/output", help="輸出預覽標註圖目錄")
    parser.add_argument("--val-ratio", type=float, default=0.20, help="驗證集切分比例 (0.0 ~ 1.0)")
    parser.add_argument("--max-images", type=int, default=None, help="最大處理張數 (預設全部)")
    args = parser.parse_args()

    batch_generate_yolo_dataset(
        input_dir=args.input_dir,
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        confidence_threshold=args.threshold,
        negative_threshold=args.threshold_negative,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
