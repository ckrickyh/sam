"""SAM 3 Focus-Primary & Nearby Canopy Segmentation (Without Trunk & Dual-Tree Visualizer)

主體樹木 (Primary Tree) 與周邊鄰居樹木 (Nearby / Surrounding Trees) 雙軌視覺化與統計模組：
1. 雙向提示與綠色保護扣除 (Positive + Negative Prompt + Green Color Safeguard)：
   - 正向鎖定全圖次冠層，負向鎖定樹幹木質部。
   - 引入 ExG/HSV 綠色植被保護機制，防止稀疏葉片被誤判為木頭挖除。
2. 稀疏樹免疫之強健主體評分 (Robust Primary Scoring)：
   - 形態學外包絡空間面積 (Envelope Area)：以樹冠立體骨架評分，避免稀疏樹輸給濃密鄰居樹。
   - 高斯空間中心衰減 (Gaussian Center Priority)：精準鎖定中央主樹。
3. 主體樹 vs. 周邊樹木雙軌多層次視覺化：
   - 主體樹 (Primary Tree)：翡翠螢光綠色覆蓋 ＋ 金黃色 3px 外框 ＋ 主體資訊標籤。
   - 周邊樹木 (Nearby Trees)：天藍色半透明覆蓋 ＋ 青藍色邊界輪廓 ＋ 周邊樹獨立標籤與面積統計。
   - 排除木質部 (Excluded Trunk)：暗紅色半透明標記。
4. 支援輸出 YOLO 雙類別標註所需之二值遮罩 (Primary Mask + Nearby Mask)。
"""

import argparse
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


def segment_focus_primary_and_nearby_canopy(
    image_path: str | Path,
    checkpoint_path: str | Path = "sam3.pt",
    prompt: str = "tree subcanopy",
    negative_prompt: str = "tree trunk, tree branch",
    confidence_threshold: float = 0.25,
    negative_threshold: float = 0.155,
    output_dir: str | Path = "simpleSam/output",
    device: str | None = None,
) -> dict:
    """執行主體樹與周邊樹木雙軌分割與精細視覺化"""
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

    # 1. 執行正向提示 (Positive Prompt: 全圖次冠層)
    print(f"[Positive Inference] 正向提示詞：'{prompt}' | 門檻值：{confidence_threshold}")
    result_subcanopy = processor.set_text_prompt(prompt=prompt, state=state)
    raw_subcanopy_masks = result_subcanopy.get("masks", None)
    boxes = result_subcanopy.get("boxes", None)
    scores = result_subcanopy.get("scores", None)

    num_detections = len(scores) if scores is not None else 0
    if num_detections == 0 or raw_subcanopy_masks is None:
        print("[Warning] 未偵測到任何符合門檻之 Subcanopy。")
        return {
            "image_shape": (h, w),
            "total_pixels": total_pixels,
            "detection_count": 0,
            "primary_index": None,
        }

    # 2. 執行負向提示 (Negative Prompt: 樹幹木質部)
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

    # 3. 綠色植被保護機制 (避免深色/稀疏葉片被當成樹幹誤殺)
    print("[Safeguard] 啟用 ExG/HSV 綠色植被保護光罩...")
    green_mask = extract_green_vegetation_mask(image_np)
    wood_to_remove = trunk_mask & (~green_mask)
    excluded_wood_pixels = int(np.sum(wood_to_remove))

    # 4. 布林安全扣除樹幹木質部
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

    # 5. 執行稀疏樹免疫之強健主體評分
    candidate_metrics = calculate_focus_primary_scores(
        masks=cleaned_masks_list,
        boxes=box_list,
        scores=score_list,
        image_shape=(h, w),
    )

    # 依加權總分排序，選出第一名主體
    candidate_metrics.sort(key=lambda x: x["composite_score"], reverse=True)
    primary_candidate = candidate_metrics[0]
    primary_idx = primary_candidate["index"]
    primary_clean_mask = cleaned_masks_list[primary_idx]
    primary_envelope_mask = primary_candidate["envelope_mask"]

    # 分離周邊鄰居樹木 (Nearby Trees)
    nearby_candidates = candidate_metrics[1:]
    nearby_combined_mask = np.zeros((h, w), dtype=bool)
    nearby_total_pixels = 0
    for cand in nearby_candidates:
        c_mask = cleaned_masks_list[cand["index"]]
        nearby_combined_mask |= c_mask
        nearby_total_pixels += cand["leaf_pixels"]

    print("\n--- 主體樹 vs. 周邊鄰居樹木 分類評分報表 ---")
    print(
        f"[★ 主體樹 Primary #{primary_idx + 1}] 綜合分: {primary_candidate['composite_score']:.4f} | "
        f"純葉片: {primary_candidate['leaf_pixels']:,} px | "
        f"骨架空間: {primary_candidate['envelope_pixels']:,} px | "
        f"密度: {primary_candidate['density_percentage']:.1f}%"
    )
    for rank, cand in enumerate(nearby_candidates, start=1):
        print(
            f"[● 周邊樹 Nearby #{cand['index'] + 1}] 綜合分: {cand['composite_score']:.4f} | "
            f"純葉片: {cand['leaf_pixels']:,} px | "
            f"骨架空間: {cand['envelope_pixels']:,} px | "
            f"置信度: {cand['confidence_score']:.2f}"
        )
    print("--------------------------------------------------\n")

    # 6. 多層次視覺化渲染

    # 渲染圖 1：主體 vs 周邊樹 全功能成果標記圖 (Overlay)
    overlay_img = image_np.copy()

    # A. 排除木質部 (暗紅)
    if excluded_wood_pixels > 0:
        overlay_img[wood_to_remove] = (overlay_img[wood_to_remove] * 0.45 + np.array([220, 70, 70]) * 0.55).astype(np.uint8)

    # B. 周邊鄰居樹木 (醒目天藍色半透明覆蓋 ＋ 青藍色外框 ＋ 獨立標籤)
    for cand in nearby_candidates:
        c_idx = cand["index"]
        c_mask = cleaned_masks_list[c_idx]
        # 天藍色覆蓋 (RGB 60, 180, 255)
        overlay_img[c_mask] = (overlay_img[c_mask] * 0.40 + np.array([60, 180, 255]) * 0.60).astype(np.uint8)
        cnts, _ = cv2.findContours(c_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_img, cnts, -1, color=(100, 220, 255), thickness=2)

        if cand["box"] is not None:
            nx1, ny1, nx2, ny2 = map(int, cand["box"])
            cv2.rectangle(overlay_img, (nx1, ny1), (nx2, ny2), color=(100, 220, 255), thickness=2)
            n_tag = f"NEARBY Tree #{c_idx + 1} ({cand['leaf_pixels']:,} px)"
            (nt_w, nt_h), _ = cv2.getTextSize(n_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(overlay_img, (nx1, max(0, ny1 - nt_h - 8)), (nx1 + nt_w + 8, ny1), (20, 50, 80), -1)
            cv2.putText(
                overlay_img,
                n_tag,
                (nx1 + 4, max(16, ny1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (100, 220, 255),
                2,
                cv2.LINE_AA,
            )

    # C. 主體樹冠外包絡線 (亮黃色細外框)
    env_contours, _ = cv2.findContours(primary_envelope_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_img, env_contours, -1, color=(255, 235, 80), thickness=2)

    # D. 主體純綠葉像素 (鮮明翡翠綠色 70% 高光覆蓋 + 金黃色 3px 實線外輪廓)
    overlay_img[primary_clean_mask] = (overlay_img[primary_clean_mask] * 0.30 + np.array([30, 245, 60]) * 0.70).astype(np.uint8)
    primary_contours, _ = cv2.findContours(primary_clean_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_img, primary_contours, -1, color=(255, 230, 0), thickness=3)

    if primary_candidate["box"] is not None:
        px1, py1, px2, py2 = map(int, primary_candidate["box"])
        cv2.rectangle(overlay_img, (px1, py1), (px2, py2), color=(255, 230, 0), thickness=3)
        tag_text = (
            f"PRIMARY Subcanopy #{primary_idx + 1} | "
            f"Score: {primary_candidate['composite_score']:.2f} | "
            f"Density: {primary_candidate['density_percentage']:.1f}%"
        )
        (t_w, t_h), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(overlay_img, (px1, max(0, py1 - t_h - 10)), (px1 + t_w + 10, py1), (0, 0, 0), -1)
        cv2.putText(
            overlay_img,
            tag_text,
            (px1 + 5, max(18, py1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 230, 0),
            2,
            cv2.LINE_AA,
        )

    # 渲染圖 2：主體 vs. 周邊樹 暗景高光對比透視圖 (Primary vs Nearby Cutout)
    # 背景暗化 85%，主體以翡翠綠邊緣呈現，周邊樹以青藍色邊緣呈現
    dual_cutout_img = (image_np * 0.15).astype(np.uint8)
    # 填入主體樹真實色彩 + 翡翠綠外緣
    dual_cutout_img[primary_clean_mask] = image_np[primary_clean_mask]
    cv2.drawContours(dual_cutout_img, primary_contours, -1, color=(0, 255, 120), thickness=2)
    # 填入周邊樹真實色彩 + 天藍色外緣
    if np.sum(nearby_combined_mask) > 0:
        dual_cutout_img[nearby_combined_mask] = image_np[nearby_combined_mask]
        for cand in nearby_candidates:
            c_cnts, _ = cv2.findContours(cleaned_masks_list[cand["index"]].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(dual_cutout_img, c_cnts, -1, color=(60, 180, 255), thickness=2)

    # 儲存輸出圖檔
    stem = img_path.stem
    overlay_output_path = out_dir / f"{stem}_focus_primary_nearby_overlay.png"
    dual_cutout_output_path = out_dir / f"{stem}_primary_vs_nearby_cutout.png"
    primary_mask_output_path = out_dir / f"{stem}_primary_tree_mask.png"
    nearby_mask_output_path = out_dir / f"{stem}_nearby_trees_mask.png"

    Image.fromarray(overlay_img).save(overlay_output_path)
    Image.fromarray(dual_cutout_img).save(dual_cutout_output_path)
    Image.fromarray((primary_clean_mask * 255).astype(np.uint8)).save(primary_mask_output_path)
    Image.fromarray((nearby_combined_mask * 255).astype(np.uint8)).save(nearby_mask_output_path)

    print("==================================================")
    print(f"影像總面積：{total_pixels:,} px")
    print(f"★ 主體樹純葉片 (Primary)：{primary_candidate['leaf_pixels']:,} px ({(primary_candidate['leaf_pixels']/total_pixels)*100:.2f}%)")
    print(f"★ 周邊樹純葉片 (Nearby Trees)：{nearby_total_pixels:,} px ({(nearby_total_pixels/total_pixels)*100:.2f}%)")
    print(f"安全排除純樹幹木質部：{excluded_wood_pixels:,} px")
    print("--------------------------------------------------")
    print(f"★ 全功能標記成果圖 (含周邊樹)：{overlay_output_path}")
    print(f"★ 主體 vs 周邊樹暗景透視圖：{dual_cutout_output_path}")
    print(f"主體樹純葉片遮罩 (YOLO Class 0)：{primary_mask_output_path}")
    print(f"周邊樹純葉片遮罩 (YOLO Class 1)：{nearby_mask_output_path}")
    print("==================================================")

    return {
        "image_shape": (h, w),
        "total_pixels": total_pixels,
        "detection_count": num_detections,
        "primary_index": primary_idx,
        "primary_metrics": primary_candidate,
        "nearby_count": len(nearby_candidates),
        "nearby_pixels": nearby_total_pixels,
        "overlay_output_path": str(overlay_output_path),
        "dual_cutout_output_path": str(dual_cutout_output_path),
        "primary_mask_output_path": str(primary_mask_output_path),
        "nearby_mask_output_path": str(nearby_mask_output_path),
    }


def main():
    parser = argparse.ArgumentParser(description="SAM 3 Focus-Primary & Nearby Tree Canopy Segmentation")
    parser.add_argument("--image", type=str, default="data/test02.jpeg", help="輸入影像路徑")
    parser.add_argument("--checkpoint", type=str, default="sam3.pt", help="SAM 3 權重路徑")
    parser.add_argument("--prompt", type=str, default="tree subcanopy", help="正向提示詞")
    parser.add_argument("--negative-prompt", type=str, default="tree trunk, tree branch", help="負向排除提示詞")
    parser.add_argument("--threshold", type=float, default=0.25, help="正向置信度門檻值")
    parser.add_argument("--threshold-negative", type=float, default=0.155, help="負向排除門檻值")
    parser.add_argument("--output-dir", type=str, default="simpleSam/output", help="輸出目錄")
    args = parser.parse_args()

    segment_focus_primary_and_nearby_canopy(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        confidence_threshold=args.threshold,
        negative_threshold=args.threshold_negative,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
