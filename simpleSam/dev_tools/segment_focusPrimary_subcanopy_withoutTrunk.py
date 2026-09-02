"""SAM 3 Focus-Primary Subcanopy Segmentation (Without Trunk & Sparse-Immune)

高度強健的主體次冠層 (Subcanopy) 分割演算法：
1. 雙向提示與綠色保護扣除 (Positive + Negative Prompt + Green Color Safeguard)：
   - 正向鎖定次冠層，負向鎖定樹幹木質部。
   - 引入 ExG/HSV 綠色植被保護機制，防止稀疏葉片被誤判為木頭挖除。
2. 稀疏樹免疫之強健主體評分 (Robust Primary Scoring for Sparse Trees)：
   - 形態學外包絡空間面積 (Envelope Area)：以樹冠骨架撐開的總立體範圍評分，避免稀疏樹因像素少輸給旁邊濃密雜樹。
   - 高斯空間中心衰減 (Gaussian Center Priority)：強力抑制兩側邊緣鄰居樹木。
   - 底部樹幹生長錨點 (Bottom-Center Trunk Anchor)：優先鎖定由畫面底部中央向上生長之主角樹。
3. 次冠層內部葉片密度與孔隙率 (Canopy Leaf Density & Porosity) 精確量測。
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

    # 包含典型綠色與微偏黃嫩綠色
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
    """計算強健主體評分 (稀疏樹免疫，防止兩側濃密鄰居樹篡位)

    評分公式：
    - Envelope Area (35%)：樹冠撐開的總骨架空間 (稀疏樹不會吃虧)
    - Gaussian Center (40%)：高斯非線性衰減，偏離畫面中央者分數迅速歸零
    - Trunk Anchor (15%)：畫面中下方樹幹錨點判定
    - Model Confidence (10%)：SAM 3 信心分數
    """
    h, w = image_shape[:2]
    # 樹冠視覺中心點設定在水平中央、垂直偏中下方 (0.50W, 0.55H)
    target_cx = w * 0.50
    target_cy = h * 0.55

    # 高斯標準差 (水平方向容忍度更嚴格，強力排除左右側樹木)
    sigma_x = w * 0.22
    sigma_y = h * 0.32

    # 計算各候選者的外包絡骨架面積
    envelope_list = [compute_canopy_envelope(m, radius_px=int(min(w, h) * 0.03)) for m in masks]
    envelope_areas = [int(np.sum(env)) for env in envelope_list]
    max_envelope_area = max(envelope_areas) if len(envelope_areas) > 0 and max(envelope_areas) > 0 else 1

    candidate_results = []

    for idx, mask in enumerate(masks):
        # 1. 外包絡空間得分 (0.0 ~ 1.0)
        env_val = envelope_areas[idx]
        env_score = env_val / max_envelope_area
        leaf_pixels = int(np.sum(mask))

        # 2. 高斯中心接近度得分 (0.0 ~ 1.0)
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

        # 3. 底部主幹連通錨點加分 (檢查是否座落於中央下方 0.3W ~ 0.7W, > 0.6H)
        bottom_region = mask[int(h * 0.60):, int(w * 0.30):int(w * 0.70)]
        has_anchor = 1.0 if np.sum(bottom_region) > (h * w * 0.001) else 0.0

        # 4. SAM 3 模型置信度 (0.0 ~ 1.0)
        conf_score = float(scores[idx])

        # 5. 綜合強健主體總分
        composite_score = (
            weight_envelope * env_score +
            weight_center * gaussian_center_score +
            weight_anchor * has_anchor +
            weight_conf * conf_score
        )

        # 計算內部樹冠密度與孔隙率
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


def segment_focus_primary_subcanopy(
    image_path: str | Path,
    checkpoint_path: str | Path = "sam3.pt",
    prompt: str = "tree subcanopy",
    negative_prompt: str = "tree trunk",
    confidence_threshold: float = 0.25,
    negative_threshold: float = 0.20,
    output_dir: str | Path = "simpleSam/output",
    device: str | None = None,
) -> dict:
    """執行具備稀疏免疫與樹幹精準排除的主體次冠層分割流程"""
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

    # 1. 執行正向提示 (Positive Prompt: 次冠層)
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

    print("\n--- 稀疏樹免疫之 Subcanopy 綜合加權評分報表 ---")
    for rank, cand in enumerate(candidate_metrics, start=1):
        status_tag = "[★ 主體 (Primary)]" if rank == 1 else f"[候選 #{cand['index'] + 1}]"
        print(
            f"{status_tag} 綜合分數: {cand['composite_score']:.4f} | "
            f"骨架空間: {cand['envelope_pixels']:,} px (評分: {cand['envelope_score']:.2f}) | "
            f"高斯中心分: {cand['center_score']:.3f} | "
            f"葉片覆蓋: {cand['leaf_pixels']:,} px | "
            f"密度: {cand['density_percentage']:.1f}%"
        )
    print("--------------------------------------------------\n")

    # 6. 視覺化渲染
    overlay_img = image_np.copy()

    # A. 繪製被剔除的純木質部 (暗紅色半透明)
    if excluded_wood_pixels > 0:
        overlay_img[wood_to_remove] = (overlay_img[wood_to_remove] * 0.45 + np.array([220, 70, 70]) * 0.55).astype(np.uint8)

    # B. 次要鄰居樹木 (淡藍色半透明，細邊框標註)
    for cand in candidate_metrics[1:]:
        c_idx = cand["index"]
        c_mask = cleaned_masks_list[c_idx]
        overlay_img[c_mask] = (overlay_img[c_mask] * 0.75 + np.array([120, 180, 240]) * 0.25).astype(np.uint8)
        cnts, _ = cv2.findContours(c_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_img, cnts, -1, color=(160, 200, 255), thickness=1)

    # C. 主體樹冠外包絡空間輪廓 (亮黃色虛線/細框標註整體樹冠範圍)
    env_contours, _ = cv2.findContours(primary_envelope_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_img, env_contours, -1, color=(255, 235, 80), thickness=2)

    # D. 主體純葉片次冠層 (鮮綠色 70% 覆蓋 + 金黃色 3px 實線外輪廓)
    overlay_img[primary_clean_mask] = (overlay_img[primary_clean_mask] * 0.30 + np.array([30, 240, 60]) * 0.70).astype(np.uint8)
    primary_contours, _ = cv2.findContours(primary_clean_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_img, primary_contours, -1, color=(255, 230, 0), thickness=3)

    # 繪製主體標籤
    if primary_candidate["box"] is not None:
        px1, py1, px2, py2 = map(int, primary_candidate["box"])
        cv2.rectangle(overlay_img, (px1, py1), (px2, py2), color=(255, 230, 0), thickness=2)
        tag_text = (
            f"PRIMARY Subcanopy #{primary_idx + 1} | "
            f"Score: {primary_candidate['composite_score']:.2f} | "
            f"Density: {primary_candidate['density_percentage']:.1f}%"
        )
        (t_w, t_h), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(overlay_img, (px1, max(0, py1 - t_h - 10)), (px1 + t_w + 10, py1), (0, 0, 0), -1)
        cv2.putText(
            overlay_img,
            tag_text,
            (px1 + 5, max(18, py1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 230, 0),
            2,
            cv2.LINE_AA,
        )

    # 儲存輸出圖檔
    stem = img_path.stem
    overlay_output_path = out_dir / f"{stem}_focus_primary_subcanopy_overlay.png"
    pure_mask_output_path = out_dir / f"{stem}_focus_primary_subcanopy_mask.png"
    envelope_output_path = out_dir / f"{stem}_focus_primary_envelope_mask.png"

    Image.fromarray(overlay_img).save(overlay_output_path)
    Image.fromarray((primary_clean_mask * 255).astype(np.uint8)).save(pure_mask_output_path)
    Image.fromarray((primary_envelope_mask * 255).astype(np.uint8)).save(envelope_output_path)

    print("==================================================")
    print(f"影像總面積：{total_pixels:,} px")
    print(f"安全排除純樹幹木質部：{excluded_wood_pixels:,} px")
    print(f"主體樹冠骨架範圍 (Envelope Area)：{primary_candidate['envelope_pixels']:,} px")
    print(f"主體純次冠層葉片覆蓋 (Leaf Area)：{primary_candidate['leaf_pixels']:,} px")
    print(f"★ 樹冠內部葉片密度 (Leaf Density)：{primary_candidate['density_percentage']:.2f}%")
    print(f"★ 樹冠透光/孔隙率 (Porosity)：{primary_candidate['porosity_percentage']:.2f}%")
    print(f"視覺化成果圖：{overlay_output_path}")
    print(f"主體純次冠層遮罩：{pure_mask_output_path}")
    print(f"樹冠骨架外包絡遮罩：{envelope_output_path}")
    print("==================================================")

    return {
        "image_shape": (h, w),
        "total_pixels": total_pixels,
        "detection_count": num_detections,
        "primary_index": primary_idx,
        "primary_metrics": primary_candidate,
        "overlay_output_path": str(overlay_output_path),
        "pure_mask_output_path": str(pure_mask_output_path),
        "envelope_output_path": str(envelope_output_path),
    }


def main():
    parser = argparse.ArgumentParser(description="SAM 3 Focus-Primary Subcanopy Segmentation")
    parser.add_argument("--image", type=str, default="data/test02.jpeg", help="輸入影像路徑")
    parser.add_argument("--checkpoint", type=str, default="sam3.pt", help="SAM 3 權重路徑")
    parser.add_argument("--prompt", type=str, default="tree subcanopy", help="正向提示詞")
    parser.add_argument("--negative-prompt", type=str, default="tree trunk, tree branch", help="負向排除提示詞")
    parser.add_argument("--threshold", type=float, default=0.25, help="正向置信度門檻值")
    parser.add_argument("--threshold-negative", type=float, default=0.155, help="負向排除門檻值")
    parser.add_argument("--output-dir", type=str, default="simpleSam/output", help="輸出目錄")
    args = parser.parse_args()

    segment_focus_primary_subcanopy(
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
