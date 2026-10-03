"""自適應 CIELAB a* 樹葉密度與孔隙分析引擎 (主角樹評分排序 + 幾何方框提示 Box Prompt 版)
檔案路徑：samplingAnalysis/adaptive_cielab_otsu_foliage_extractor_fixBranchlet_order_box.py

核心升級重點：
1. 幾何方框提示驅動 (Box Prompt Driven Architecture)：
   - 全面由純文字提示改為幾何方框提示 (Box Prompt)，強加空間約束，徹底消除背景雜木篡位與語意擴散。
   - 支援自動歸一化 [xmin, ymin, xmax, ymax] 像素或比例座標至 [cx, cy, w, h]。
2. 前景主角樹評分排序 (Focus-Primary Tree Priority Scoring)：
   - 結合高斯居中度 (45%)、樹冠外包絡規模 (25%)、前景地基錨點 (20%) 與模型信心度 (10%)。
   - 鎖定綜合評分最高者為唯一的「主角樹」(Primary Tree)，其餘樹木自動歸為「周邊樹」(Nearby Trees)。
3. 幾何/語意骨架 + OpenCV 拓撲補洞修復 (Hybrid Semantic Trunk & Hole Inpainting)：
   - 支援方框或語意提示分離樹幹，解決非啡色樹皮的認知問題。
   - 導入 OpenCV 外部輪廓填洞 (RETR_EXTERNAL) 與形態學閉運算，消除 SAM 3 抽樣產生的圓形黑洞斑點。
   - 結合微觀逆光細枝條提取 (L* < 95) 與嚴格綠葉保護盾 (ExG + HSV)。
4. 4 面板高對比專業視覺化：
   - Panel 1：原圖標註 Canopy Box Prompt 空間約束框。
   - Panel 2：CIELAB a* 物理色度圖。
   - Panel 3：明確標註主角樹 (亮綠+黃框)、周邊樹 (淡藍+青框) 與木質部 (橘色)。
   - Panel 4：針對主角樹進行微觀三色無損診斷 (純葉片、內部透光孔隙、平滑木質結構)。
"""

import argparse
from pathlib import Path
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

# 確保專案根目錄在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def parse_box_to_normalized_cxcywh(
    box: list[float] | tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> list[float]:
    """將 [xmin, ymin, xmax, ymax] 座標統一轉換為 SAM 3 所需之歸一化 [center_x, center_y, width, height]"""
    x1, y1, x2, y2 = box
    if max(x1, y1, x2, y2) > 1.0:
        x1 = x1 / img_w
        x2 = x2 / img_w
        y1 = y1 / img_h
        y2 = y2 / img_h

    xmin = max(0.0, min(1.0, min(x1, x2)))
    xmax = max(0.0, min(1.0, max(x1, x2)))
    ymin = max(0.0, min(1.0, min(y1, y2)))
    ymax = max(0.0, min(1.0, max(y1, y2)))

    bw = max(1e-4, xmax - xmin)
    bh = max(1e-4, ymax - ymin)
    cx = xmin + bw / 2.0
    cy = ymin + bh / 2.0
    return [float(cx), float(cy), float(bw), float(bh)]


def parse_box_to_pixel_xyxy(
    box: list[float] | tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """將方框座標統一轉換為影像像素尺度 [xmin, ymin, xmax, ymax]，供繪圖標記使用"""
    x1, y1, x2, y2 = box
    if max(x1, y1, x2, y2) <= 1.0:
        x1 = x1 * img_w
        x2 = x2 * img_w
        y1 = y1 * img_h
        y2 = y2 * img_h

    xmin = int(round(max(0, min(img_w - 1, min(x1, x2)))))
    xmax = int(round(max(0, min(img_w - 1, max(x1, x2)))))
    ymin = int(round(max(0, min(img_h - 1, min(y1, y2)))))
    ymax = int(round(max(0, min(img_h - 1, max(y1, y2)))))
    return (xmin, ymin, xmax, ymax)


def get_optimal_device() -> str:
    """自動偵測運算裝置 (優先順序: CUDA -> CPU)"""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def extract_green_vegetation_mask(image_np: np.ndarray) -> np.ndarray:
    """計算超綠指數 (ExG) 與 HSV 綠色區間，產出綠色植物保護遮罩"""
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
    """利用形態學閉運算計算樹冠外包絡立體空間 (Canopy Envelope，分母基準)"""
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
    """計算強健主角樹評分 (前景樹優先，防止側邊背景巨木篡位)"""
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

        # 主幹根基錨點判定：下方 40% 且中軸範圍內 (x in 25% ~ 75%)
        bottom_region = mask[int(h * 0.55):, int(w * 0.25):int(w * 0.75)]
        has_anchor = 1.0 if np.sum(bottom_region) > (h * w * 0.0008) else 0.0

        conf_score = float(scores[idx])

        composite_score = (
            weight_envelope * env_score +
            weight_center * gaussian_center_score +
            weight_anchor * has_anchor +
            weight_conf * conf_score
        )

        candidate_results.append({
            "index": idx,
            "leaf_pixels": leaf_pixels,
            "envelope_pixels": env_val,
            "envelope_score": env_score,
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


def heal_and_refine_wood_trunk(
    raw_trunk_mask: np.ndarray,
    img_bgr: np.ndarray,
    canopy_envelope: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """利用 OpenCV 形態學修復 SAM 3 樹幹破洞，並提取外圍逆光細枝 (樹幹絕對鎖定版)"""
    h, w = raw_trunk_mask.shape[:2]

    # 1. 消除 SAM 3 樹幹內部小於 3000px 的採樣黑洞
    trunk_inv = (~raw_trunk_mask).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(trunk_inv, connectivity=8)

    filled_trunk = raw_trunk_mask.copy()
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area < 3000:
            filled_trunk |= (labels == label_idx)

    # 2. 形態學平滑閉運算
    kernel_heal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    smooth_trunk = cv2.morphologyEx(filled_trunk.astype(np.uint8), cv2.MORPH_CLOSE, kernel_heal).astype(bool)
    smooth_trunk = smooth_trunk & canopy_envelope

    # 3. 提取外圍微觀逆光細枝 (僅在外圍非 SAM 3 主幹區域尋找: L* < 95 且 a* >= 126)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, _ = cv2.split(lab)
    fine_branches = (l_chan < 95) & (a_chan >= 126) & canopy_envelope & (~smooth_trunk)

    # 總木質部 = SAM 3 樹幹主骨架 (絕對不可變綠) + 外圍逆光細枝
    total_wood = smooth_trunk | fine_branches
    return smooth_trunk, total_wood


def process_image_cielab_adaptive(
    image_path: str | Path,
    processor: Sam3Processor,
    canopy_box: list[float] | tuple[float, float, float, float] | None = None,
    trunk_box: list[float] | tuple[float, float, float, float] | None = None,
    negative_prompt: str = "tree trunk, tree branch",
    confidence_threshold: float = 0.25,
    negative_threshold: float = 0.155,
    output_dir: Path | None = None,
    artifact_dir: Path | None = None,
) -> dict:
    """針對單張影像執行主角樹幾何方框 (Box Prompt) 評分排序、樹幹修復與自適應 CIELAB a* 拓撲分析"""
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"找不到檔案：{image_path}")

    raw_img = Image.open(img_path).convert("RGB")
    img_rgb = np.array(raw_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_rgb.shape[:2]
    stem = img_path.stem

    state = processor.set_image(raw_img)

    # 1. SAM 3 正向樹冠幾何方框 (Box Prompt) 推論
    eff_canopy_box = canopy_box if canopy_box is not None else [0.02, 0.02, 0.98, 0.90]
    canopy_cxcywh = parse_box_to_normalized_cxcywh(eff_canopy_box, w, h)
    canopy_pixel_xyxy = parse_box_to_pixel_xyxy(eff_canopy_box, w, h)

    processor.set_confidence_threshold(confidence_threshold)
    res_canopy = processor.add_geometric_prompt(box=canopy_cxcywh, label=True, state=state)
    raw_canopy = res_canopy.get("masks", None)
    boxes = res_canopy.get("boxes", None)
    scores = res_canopy.get("scores", None)

    num_detections = len(scores) if scores is not None else 0
    if num_detections == 0 or raw_canopy is None:
        print(f"[Warning] {img_path.name} 在幾何方框約束內未偵測到樹冠遮罩。")
        return {"stem": stem, "status": "No Detections"}

    # 綠色植物保護遮罩
    green_mask = extract_green_vegetation_mask(img_rgb)

    # 2. SAM 3 樹幹幾何/語意推論 (優先使用 trunk_box，未提供則使用語意提示)
    processor.set_confidence_threshold(negative_threshold)
    trunk_mask = np.zeros((h, w), dtype=bool)

    # 重置提示詞狀態 (保留已計算之視覺 Backbone 特徵，零額外耗時)
    processor.reset_all_prompts(state)

    if trunk_box is not None:
        trunk_cxcywh = parse_box_to_normalized_cxcywh(trunk_box, w, h)
        res_trunk = processor.add_geometric_prompt(box=trunk_cxcywh, label=True, state=state)
        raw_trunk = res_trunk.get("masks", None)
        scores_trunk = res_trunk.get("scores", None)
    elif negative_prompt and negative_prompt.strip():
        res_trunk = processor.set_text_prompt(prompt=negative_prompt, state=state)
        raw_trunk = res_trunk.get("masks", None)
        scores_trunk = res_trunk.get("scores", None)
    else:
        raw_trunk = None
        scores_trunk = None

    if raw_trunk is not None and len(raw_trunk) > 0:
        trunk_np = raw_trunk.cpu().numpy().astype(bool)
        if trunk_np.ndim == 4:
            trunk_np = trunk_np.squeeze(1)
        elif trunk_np.ndim == 3 and trunk_np.shape[0] == 1 and trunk_np.shape[1] != h:
            trunk_np = trunk_np.squeeze(0)
        for idx_t, m in enumerate(trunk_np):
            m_area = int(np.sum(m))
            if m_area == 0:
                continue
            t_score = float(scores_trunk[idx_t].item()) if scores_trunk is not None else 0.5
            green_overlap = int(np.sum(m & green_mask))
            green_ratio = green_overlap / m_area
            # 若遮罩包含大量鮮綠葉片 (>40%) 且信心度偏低 (<0.40)，判定為枝幹幻覺遮罩排除
            if green_ratio > 0.40 and t_score < 0.40:
                continue
            trunk_mask |= m

    # 3. 實例遮罩提取與初步淨化
    c_np = raw_canopy.cpu().numpy().astype(bool)
    if c_np.ndim == 4:
        c_np = c_np.squeeze(1)

    cleaned_masks_list = []
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for i in range(num_detections):
        raw_m = c_np[i]
        # 去除邊緣極小毛刺，保留完整樹冠幾何
        pure_m = cv2.morphologyEx(raw_m.astype(np.uint8), cv2.MORPH_OPEN, kernel_clean).astype(bool)
        cleaned_masks_list.append(pure_m)

    box_list = [boxes[i].cpu().numpy() for i in range(num_detections)] if boxes is not None else []
    score_list = [float(scores[i].item()) for i in range(num_detections)]

    # 4. 階層超遮罩過濾 (Mega-mask Suppression)
    filtered_masks, filtered_boxes, filtered_scores = filter_hierarchical_super_masks(
        masks=cleaned_masks_list,
        boxes=box_list,
        scores=score_list,
        image_shape=(h, w),
    )

    # 5. 主角樹與周邊樹評分排序
    candidate_metrics = calculate_focus_primary_scores(
        masks=filtered_masks,
        boxes=filtered_boxes,
        scores=filtered_scores,
        image_shape=(h, w),
    )
    candidate_metrics.sort(key=lambda x: x["composite_score"], reverse=True)

    primary_info = candidate_metrics[0]
    primary_mask = filtered_masks[primary_info["index"]].copy()

    # 排除與主角樹重疊過高的同體冗餘遮罩 (避免自相殘殺)
    nearby_candidates = []
    primary_mask_area = int(np.sum(primary_mask))
    for cand in candidate_metrics[1:]:
        c_mask = filtered_masks[cand["index"]]
        cand_area = int(np.sum(c_mask))
        if cand_area == 0:
            continue
        intersection = int(np.sum(primary_mask & c_mask))
        overlap_ratio = intersection / max(min(primary_mask_area, cand_area), 1)

        if overlap_ratio >= 0.35:
            # 同體多尺度遮罩：視為主角樹的一部分進行聯集融合
            primary_mask |= c_mask
            primary_mask_area = int(np.sum(primary_mask))
        else:
            nearby_candidates.append(cand)

    # 合併周邊樹遮罩
    nearby_mask = np.zeros((h, w), dtype=bool)
    for cand in nearby_candidates:
        nearby_mask |= filtered_masks[cand["index"]]

    # 6. 計算主角樹的外包絡 (Envelope 分母基準)
    primary_envelope = compute_canopy_envelope(primary_mask, radius_px=int(min(w, h) * 0.03))
    # 嚴格排除周邊樹穿透干擾
    canopy_envelope = primary_envelope & (~nearby_mask)
    if np.sum(canopy_envelope) == 0:
        canopy_envelope = primary_envelope

    # 7. CIELAB 自適應葉片提取 (針對主角樹外包絡)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    canopy_a_pixels = a_chan[canopy_envelope]
    if len(canopy_a_pixels) > 100:
        std_a = np.std(canopy_a_pixels)
        otsu_val, _ = cv2.threshold(canopy_a_pixels, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 物理鐵律：a* < 128.0 嚴格屬於綠色系，Otsu 門檻絕不可低於 128.0 造成淺色綠葉遭腰斬
        if otsu_val > 128.0 and std_a > 6.5:
            effective_thresh = otsu_val
            method_used = f"Adaptive Otsu ({effective_thresh:.1f})"
        else:
            effective_thresh = 128.0
            method_used = f"CIE Physics Line ({effective_thresh:.1f})"

        standard_green_mask = (a_chan < effective_thresh)
        shadow_or_yellow_leaf_mask = (
            (b_chan >= 132) &
            (l_chan < 175) &
            (a_chan <= 136)
        )
        leaf_chroma_mask = (standard_green_mask | shadow_or_yellow_leaf_mask) & canopy_envelope
    else:
        effective_thresh = 128.0
        method_used = "Default Line (128.0)"
        leaf_chroma_mask = primary_mask & canopy_envelope

    # 8. 融合 SAM 3 語意樹幹 + OpenCV 形態學修復
    smooth_trunk, total_wood_mask = heal_and_refine_wood_trunk(
        raw_trunk_mask=trunk_mask,
        img_bgr=img_bgr,
        canopy_envelope=canopy_envelope,
    )

    # 樹冠暗部非天空判定 (自然光學鐵律：白天仰拍時，透光天空必定高亮，暗部絕不可能是穿透天空)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)
    is_dark_canopy = (l_chan < 160) & canopy_envelope

    # 1. 暗部背光綠葉救援：包含植物綠與受天光漫射之青冷色葉片 (H 從 25 擴展至 115) 且 a* 符合物理綠色界線 (< 128)
    dark_foliage_rescue = is_dark_canopy & (h_chan >= 25) & (h_chan <= 115) & (a_chan < 128)

    # 2. 暗部枝幹救援：木質部色相必須為暖啡色 (H <= 22)、紅褐色 (H >= 170) 或低飽和度灰黑樹皮 (S < 25 且 a* >= 126)，且不與綠葉衝突
    is_warm_wood_hue = (h_chan <= 22) | (h_chan >= 170) | ((s_chan < 25) & (a_chan >= 126))
    dark_wood_rescue = is_dark_canopy & (l_chan < 145) & is_warm_wood_hue & (a_chan >= 126) & (~dark_foliage_rescue)

    # 3. 逆光耀光高光葉片補償 (Backlight Foliage Compensation)：
    # 排除純白天光過曝區 (L* > 235 且 S < 18)
    is_pure_white_sky = (l_chan > 235) & (s_chan < 18)
    r_chan = img_rgb[:, :, 0].astype(float)
    g_chan = img_rgb[:, :, 1].astype(float)
    # 藍色耀光 (Blue Lens Flare) 霧氣下，高藍光會推升 CIELAB a* 略高於 128，但葉綠素生物本質必定使綠光高於紅光 (G > R + 4)
    flare_green_leaf = (g_chan > r_chan + 4) & (~is_pure_white_sky)
    backlight_leaf_mask = ((a_chan < 128) | flare_green_leaf) & (~is_pure_white_sky) & canopy_envelope

    # 鮮綠植物保護盾：強綠色像素 (ExG / HSV 綠且 a* < 126) 具備豁免權，防止被木質部誤殺
    strong_green_shield = green_mask & (a_chan < 126)
    effective_trunk = (smooth_trunk | dark_wood_rescue) & (~strong_green_shield)

    pure_foliage_mask = (leaf_chroma_mask | strong_green_shield | dark_foliage_rescue | backlight_leaf_mask) & (~effective_trunk) & canopy_envelope

    # 總木質部包含經保護盾校驗之有效樹幹，以及外圍不與純綠葉衝突的細枝
    total_wood_mask = (effective_trunk | (total_wood_mask & (~pure_foliage_mask))) & canopy_envelope
    internal_gaps_mask = canopy_envelope & (~pure_foliage_mask) & (~total_wood_mask)

    # 9. 量化統計
    envelope_px = int(np.sum(canopy_envelope))
    foliage_px = int(np.sum(pure_foliage_mask))
    wood_px = int(np.sum(total_wood_mask))
    gaps_px = int(np.sum(internal_gaps_mask))

    canopy_density_pct = (foliage_px / envelope_px * 100.0) if envelope_px > 0 else 0.0
    canopy_porosity_pct = (gaps_px / envelope_px * 100.0) if envelope_px > 0 else 0.0
    wood_ratio_pct = (wood_px / envelope_px * 100.0) if envelope_px > 0 else 0.0

    net_foliage_space_px = envelope_px - wood_px
    foliage_cover_pct = (foliage_px / net_foliage_space_px * 100.0) if net_foliage_space_px > 0 else 0.0
    net_porosity_pct = (gaps_px / net_foliage_space_px * 100.0) if net_foliage_space_px > 0 else 0.0

    print(f"\n================ [{stem}] 主角樹評分與自適應量化報告 ================")
    print(f"★ 候選樹木總數: {len(filtered_masks)} 株 (主角樹 1 株，周邊樹 {len(nearby_candidates)} 株)")
    print(f"★ 主角樹評分 (Composite Score): {primary_info['composite_score']:.3f}")
    print(f"  └ 居中度分: {primary_info['center_score']:.3f} | 包絡面積分: {primary_info['envelope_score']:.3f} | 地基錨點: {primary_info['has_anchor']}")
    print(f"★ 主角樹外包絡 (Envelope): {envelope_px:,} px")
    print(f"★ 判定機制與門檻: {method_used}")
    print(f"★ 純樹葉面積 (Living Foliage): {foliage_px:,} px")
    print(f"★ 木質樹幹枝幹 (Wood Trunk + Branches): {wood_px:,} px -> 【佔比: {wood_ratio_pct:.2f}%】")
    print(f"★ 內部透光孔隙 (Canopy Gaps): {gaps_px:,} px")
    print("----------------------------------------------------------------------")
    print(f"【標準冠層指標 (分母 Envelope)】: 葉密度 {canopy_density_pct:.2f}% | 孔隙率 {canopy_porosity_pct:.2f}%")
    print(f"【淨葉覆蓋指標 (分母 Envelope - Wood)】: 葉覆蓋 {foliage_cover_pct:.2f}% | 淨孔隙 {net_porosity_pct:.2f}%")
    print("======================================================================")

    # 10. 渲染 4 面板高對比專業診斷圖
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    plt.subplots_adjust(wspace=0.08, hspace=0.14)

    # Panel 1: 原圖與幾何方框提示
    panel1_img = img_rgb.copy()
    bx1, by1, bx2, by2 = canopy_pixel_xyxy
    cv2.rectangle(panel1_img, (bx1, by1), (bx2, by2), (0, 255, 255), 3)
    cv2.putText(
        panel1_img,
        f"Box Prompt [{bx1},{by1},{bx2},{by2}]",
        (bx1 + 10, max(30, by1 + 32)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    axes[0, 0].imshow(panel1_img)
    axes[0, 0].set_title(f"(A) Input & Box Prompt: {img_path.name}", fontsize=13, fontweight="bold")
    axes[0, 0].axis("off")

    # Panel 2: CIELAB a* 色度圖
    im_a = axes[0, 1].imshow(a_chan, cmap="PRGn_r", vmin=90, vmax=140)
    axes[0, 1].set_title(f"(B) CIELAB a* Channel [{method_used}]", fontsize=13, fontweight="bold", color="purple")
    axes[0, 1].axis("off")
    fig.colorbar(im_a, ax=axes[0, 1], fraction=0.046, pad=0.04, label="a* Value (<128: Green, >=128: Sky/Wood)")

    # Panel 3: 主角樹 (亮綠+黃框) vs 周邊樹 (淡藍+藍框) vs 木質部 (橘色)
    sam_vis = img_rgb.copy()
    sam_overlay = np.zeros_like(img_rgb)

    # 繪製周邊樹 (淡藍色)
    for cand in nearby_candidates:
        c_mask = filtered_masks[cand["index"]]
        sam_overlay[c_mask] = [0, 160, 230]
        c_cnts, _ = cv2.findContours(c_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(sam_vis, c_cnts, -1, (180, 220, 255), 2)

    # 繪製主角樹 (綠色) 與 主幹 (橘色)
    sam_overlay[canopy_envelope] = [0, 100, 150]
    sam_overlay[primary_mask] = [0, 220, 80]
    sam_overlay[total_wood_mask] = [255, 140, 0]

    sam_vis = cv2.addWeighted(sam_vis, 0.65, sam_overlay, 0.35, 0)
    cnts_env, _ = cv2.findContours(canopy_envelope.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(sam_vis, cnts_env, -1, (255, 230, 0), 3)

    axes[1, 0].imshow(sam_vis)
    axes[1, 0].set_title(
        f"(C) Primary Tree (Score: {primary_info['composite_score']:.2f}) vs {len(nearby_candidates)} Nearby Trees",
        fontsize=13,
        fontweight="bold",
    )
    axes[1, 0].axis("off")

    # Panel 4: 主角樹三色無損診斷圖
    final_vis = img_rgb.copy()
    final_overlay = np.zeros_like(img_rgb)
    final_overlay[pure_foliage_mask] = [0, 255, 50]             # 亮綠: 純樹葉
    final_overlay[internal_gaps_mask] = [0, 140, 255]           # 天藍: 內部透光孔隙
    final_overlay[total_wood_mask] = [255, 140, 0]              # 橘色: 平滑木質結構
    final_vis = cv2.addWeighted(final_vis, 0.58, final_overlay, 0.42, 0)
    cv2.drawContours(final_vis, cnts_env, -1, (255, 230, 0), 2)

    axes[1, 1].imshow(final_vis)
    axes[1, 1].set_title(
        f"(D) Primary: Net Leaf {foliage_cover_pct:.1f}% (Env Leaf {canopy_density_pct:.1f}%), Gaps {canopy_porosity_pct:.1f}%, Wood {wood_ratio_pct:.1f}%",
        fontsize=12,
        fontweight="bold",
        color="darkgreen",
    )
    axes[1, 1].axis("off")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"{stem}_cielab_otsu_render.png", bbox_inches="tight")
    if artifact_dir:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(artifact_dir / f"{stem}_cielab_otsu_render.png", bbox_inches="tight")
    plt.close()

    return {
        "stem": stem,
        "primary_score": primary_info["composite_score"],
        "nearby_count": len(nearby_candidates),
        "envelope_pixels": envelope_px,
        "foliage_pixels": foliage_px,
        "wood_pixels": wood_px,
        "gaps_pixels": gaps_px,
        "canopy_density_pct": canopy_density_pct,
        "foliage_cover_pct": foliage_cover_pct,
        "canopy_porosity_pct": canopy_porosity_pct,
        "net_porosity_pct": net_porosity_pct,
        "wood_ratio_pct": wood_ratio_pct,
        "method_used": method_used,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive CIELAB a* Foliage & Gap Analyzer with Primary Tree Scoring (Box Prompt Edition)"
    )
    parser.add_argument("--data-dir", type=str, default="data", help="輸入照片目錄 (預設: data)")
    parser.add_argument("--images", nargs="+", default=None, help="指定測試影像清單 (若未指定，則自動掃描 --data-dir)")
    parser.add_argument("--max-images", type=int, default=None, help="最大處理照片數量 (預設無限制)")
    parser.add_argument("--checkpoint", type=str, default="sam3.pt", help="SAM 3 權重路徑")
    parser.add_argument("--device", type=str, default=None, help="運算裝置 (cuda/cpu，預設自動偵測)")
    parser.add_argument("--output-dir", type=str, default="samplingAnalysis/output", help="輸出圖檔目錄")
    parser.add_argument("--artifact-dir", type=str, default=None, help="Artifact 輸出目錄")
    parser.add_argument("--csv-name", type=str, default="cielab_otsu_summary_box.csv", help="量化結果 CSV 檔名")
    parser.add_argument(
        "--box",
        "--canopy-box",
        nargs=4,
        type=float,
        default=None,
        help="樹冠幾何方框提示 [xmin ymin xmax ymax] (支援像素值或 0~1 比例，預設: 0.02 0.02 0.98 0.90)",
    )
    parser.add_argument(
        "--trunk-box",
        nargs=4,
        type=float,
        default=None,
        help="樹幹幾何方框提示 [xmin ymin xmax ymax] (若未指定，則自動使用 negative-prompt 語意提示)",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="tree trunk, tree branch",
        help="樹幹備用語意提示詞 (當未指定 --trunk-box 時生效)",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.25, help="樹冠偵測信心門檻 (預設: 0.25)")
    parser.add_argument("--negative-threshold", type=float, default=0.155, help="樹幹偵測信心門檻 (預設: 0.155)")
    args = parser.parse_args()

    # 1. 蒐集待處理影像清單
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
    if args.images:
        image_list = [Path(p) for p in args.images]
    else:
        data_path = Path(args.data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"找不到資料目錄：{data_path}")
        image_list = sorted([p for p in data_path.iterdir() if p.is_file() and p.suffix in valid_exts])

    if args.max_images is not None and args.max_images > 0:
        image_list = image_list[:args.max_images]

    total_images = len(image_list)
    if total_images == 0:
        print(f"[Warning] 未找到任何支援的影像檔案。")
        return

    out_dir = Path(args.output_dir)
    art_dir = Path(args.artifact_dir) if args.artifact_dir else None

    run_device = args.device if args.device else get_optimal_device()
    print(f"==================================================")
    print(f"★ 批次掃描目標：共 {total_images} 張影像 (模式: Box Prompt 幾何約束)")
    print(f"★ 樹冠幾何方框：{args.box if args.box else '預設前景全景框 [0.02, 0.02, 0.98, 0.90]'}")
    print(f"★ 輸出圖檔目錄：{out_dir}")
    print(f"★ 正在載入 SAM 3 模型: {args.checkpoint} ({run_device})...")
    print(f"==================================================")

    model = build_sam3_image_model(checkpoint_path=args.checkpoint, device=run_device)
    model = model.to(run_device).float()
    processor = Sam3Processor(model, device=run_device, confidence_threshold=args.confidence_threshold)

    results = []
    for idx, img_p in enumerate(image_list, start=1):
        print(f"\n[{idx}/{total_images}] 正在處理：{img_p.name}...")
        try:
            res = process_image_cielab_adaptive(
                image_path=img_p,
                processor=processor,
                canopy_box=args.box,
                trunk_box=args.trunk_box,
                negative_prompt=args.negative_prompt,
                confidence_threshold=args.confidence_threshold,
                negative_threshold=args.negative_threshold,
                output_dir=out_dir,
                artifact_dir=art_dir,
            )
            results.append(res)
        except Exception as e:
            print(f"    [Error] 處理失敗 ({img_p.name}): {e}")
            results.append({"stem": img_p.stem, "error": str(e)})

    # 2. 儲存批次彙總 CSV 報表
    if results and out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / args.csv_name
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n★ 批次量化報表已儲存至：{csv_path}")

    print("\n主角樹評分與拓撲補洞自適應分析 (Box Prompt 版) 全部完成！輸出圖檔已儲存。")


if __name__ == "__main__":
    main()
