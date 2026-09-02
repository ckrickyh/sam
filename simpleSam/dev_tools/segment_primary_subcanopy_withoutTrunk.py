"""SAM 3 Primary Subcanopy Segmentation (Without Trunk & Branches)

透過雙向提示（Positive Prompt + Negative Prompt / Mask Subtraction）實作：
1. Positive Prompt ('tree subcanopy')：鎖定次冠層候選區域。
2. Negative Prompt ('tree trunk, tree branch')：精確辨識樹幹與粗枝木質部。
3. 布林差集運算 (Mask Subtraction)：將樹幹從次冠層中剔除 (`subcanopy & ~trunk`)。
4. 綜合加權評分：自動評選純葉片主體次冠層 (Primary Subcanopy without trunk)。
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


def calculate_candidate_scores(
    masks: list[np.ndarray],
    boxes: list[np.ndarray],
    scores: list[float],
    image_shape: tuple[int, int],
    weight_area: float = 0.50,
    weight_center: float = 0.30,
    weight_conf: float = 0.20,
) -> list[dict]:
    """計算所有候選 Subcanopy 遮罩的綜合主體評分"""
    h, w = image_shape[:2]
    center_x, center_y = w / 2.0, h / 2.0
    max_center_distance = np.sqrt(center_x**2 + center_y**2)

    areas = [int(np.sum(m)) for m in masks]
    max_area = max(areas) if len(areas) > 0 and max(areas) > 0 else 1

    candidate_results = []

    for idx, mask in enumerate(masks):
        area_val = areas[idx]
        area_score = area_val / max_area

        y_indices, x_indices = np.where(mask)
        if len(x_indices) > 0:
            centroid_x = float(np.mean(x_indices))
            centroid_y = float(np.mean(y_indices))
            dist_to_center = float(np.sqrt((centroid_x - center_x) ** 2 + (centroid_y - center_y) ** 2))
            center_score = max(0.0, 1.0 - (dist_to_center / max_center_distance))
        else:
            centroid_x, centroid_y = 0.0, 0.0
            dist_to_center = max_center_distance
            center_score = 0.0

        conf_score = float(scores[idx])

        composite_score = (
            weight_area * area_score +
            weight_center * center_score +
            weight_conf * conf_score
        )

        candidate_results.append({
            "index": idx,
            "area_pixels": area_val,
            "area_score": area_score,
            "centroid": (centroid_x, centroid_y),
            "distance_to_center": dist_to_center,
            "center_score": center_score,
            "confidence_score": conf_score,
            "composite_score": composite_score,
            "box": boxes[idx] if idx < len(boxes) else None,
        })

    return candidate_results


def segment_primary_subcanopy_without_trunk(
    image_path: str | Path,
    checkpoint_path: str | Path = "sam3.pt",
    prompt: str = "tree subcanopy",
    negative_prompt: str = "tree trunk, tree branch",
    confidence_threshold: float = 0.25,
    negative_threshold: float = 0.155,
    output_dir: str | Path = "simpleSam/output",
    weight_area: float = 0.50,
    weight_center: float = 0.30,
    weight_conf: float = 0.20,
    device: str | None = None,
) -> dict:
    """透過 Negative Prompt 排除樹幹與樹枝，精確提取主體純次冠層

    Args:
        image_path: 輸入影像路徑
        checkpoint_path: SAM 3 模型權重檔案路徑
        prompt: 正向提示詞 (預設: "tree subcanopy")
        negative_prompt: 負向排除提示詞 (預設: "tree trunk, tree branch")
        confidence_threshold: 正向置信度門檻值
        negative_threshold: 負向排除門檻值 (通常略低於正向以完整捕捉木質幹體)
        output_dir: 結果輸出目錄
        weight_area: 面積權重
        weight_center: 中心接近度權重
        weight_conf: 置信度權重
        device: 運算裝置
    """
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
    print(f"[Positive Inference] 提示詞：'{prompt}' | 門檻值：{confidence_threshold}")
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

    # 2. 執行負向提示 (Negative Prompt: 樹幹與樹枝)
    trunk_mask = np.zeros((h, w), dtype=bool)
    if negative_prompt and negative_prompt.strip():
        print(f"[Negative Inference] 負向排除提示詞：'{negative_prompt}' | 門檻值：{negative_threshold}...")
        processor.set_confidence_threshold(negative_threshold)
        result_trunk = processor.set_text_prompt(prompt=negative_prompt, state=state)
        raw_trunk_masks = result_trunk.get("masks", None)
        if raw_trunk_masks is not None and len(raw_trunk_masks) > 0:
            trunk_np = raw_trunk_masks.cpu().numpy().astype(bool)
            if trunk_np.ndim == 4:
                trunk_np = trunk_np.squeeze(1)
            for m in trunk_np:
                trunk_mask |= m

    trunk_pixel_count = int(np.sum(trunk_mask))
    print(f"[Negative Mask] 偵測到樹幹/樹枝面積：{trunk_pixel_count:,} px")

    # 3. 布林差集：次冠層遮罩剔除樹幹 (Subcanopy & ~Trunk)
    subcanopy_np = raw_subcanopy_masks.cpu().numpy().astype(bool)
    if subcanopy_np.ndim == 4:
        subcanopy_np = subcanopy_np.squeeze(1)

    cleaned_masks_list = []
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for i in range(num_detections):
        raw_m = subcanopy_np[i]
        # 剔除樹幹與枝幹
        pure_m = raw_m & (~trunk_mask)
        # 輕量形態學開運算，濾除邊界碎毛邊
        pure_m = cv2.morphologyEx(pure_m.astype(np.uint8), cv2.MORPH_OPEN, kernel_clean).astype(bool)
        cleaned_masks_list.append(pure_m)

    box_list = [boxes[i].cpu().numpy() for i in range(num_detections)] if boxes is not None else []
    score_list = [float(scores[i].item()) for i in range(num_detections)]

    # 4. 執行綜合加權評分
    candidate_metrics = calculate_candidate_scores(
        masks=cleaned_masks_list,
        boxes=box_list,
        scores=score_list,
        image_shape=(h, w),
        weight_area=weight_area,
        weight_center=weight_center,
        weight_conf=weight_conf,
    )

    candidate_metrics.sort(key=lambda x: x["composite_score"], reverse=True)
    primary_candidate = candidate_metrics[0]
    primary_idx = primary_candidate["index"]
    primary_clean_mask = cleaned_masks_list[primary_idx]
    primary_clean_pixels = primary_candidate["area_pixels"]
    primary_clean_coverage = (primary_clean_pixels / total_pixels) * 100.0

    print("\n--- 扣除樹幹後之 Subcanopy 綜合加權評分報表 ---")
    for rank, cand in enumerate(candidate_metrics, start=1):
        status_tag = "[★ 主體 (Primary)]" if rank == 1 else f"[候選 #{cand['index'] + 1}]"
        print(
            f"{status_tag} 綜合分數: {cand['composite_score']:.4f} | "
            f"純葉片面積: {cand['area_pixels']:,} px (評分: {cand['area_score']:.2f}) | "
            f"中心距: {cand['distance_to_center']:.1f} px (評分: {cand['center_score']:.2f}) | "
            f"置信度: {cand['confidence_score']:.2f}"
        )
    print("--------------------------------------------\n")

    # 5. 渲染視覺化圖層
    overlay_img = image_np.copy()

    # A. 繪製被剔除的樹幹 (淡紅/深灰半透明)
    if trunk_pixel_count > 0:
        overlay_img[trunk_mask] = (overlay_img[trunk_mask] * 0.45 + np.array([220, 80, 80]) * 0.55).astype(np.uint8)

    # B. 次要次冠層候選者 (淡藍色)
    for cand in candidate_metrics[1:]:
        c_idx = cand["index"]
        c_mask = cleaned_masks_list[c_idx]
        overlay_img[c_mask] = (overlay_img[c_mask] * 0.7 + np.array([120, 180, 240]) * 0.3).astype(np.uint8)
        cnts, _ = cv2.findContours(c_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_img, cnts, -1, color=(160, 200, 255), thickness=1)

    # C. 主體純次冠層 (鮮綠色 70% 覆蓋 + 金黃色 3 px 邊界)
    overlay_img[primary_clean_mask] = (overlay_img[primary_clean_mask] * 0.3 + np.array([30, 240, 60]) * 0.7).astype(np.uint8)
    primary_contours, _ = cv2.findContours(primary_clean_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_img, primary_contours, -1, color=(255, 230, 0), thickness=3)

    if primary_candidate["box"] is not None:
        px1, py1, px2, py2 = map(int, primary_candidate["box"])
        cv2.rectangle(overlay_img, (px1, py1), (px2, py2), color=(255, 230, 0), thickness=2)
        tag_text = f"PURE Primary Subcanopy #{primary_idx + 1} (Score: {primary_candidate['composite_score']:.2f})"
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
    overlay_output_path = out_dir / f"{stem}_primary_subcanopy_withoutTrunk_overlay.png"
    pure_mask_output_path = out_dir / f"{stem}_primary_subcanopy_withoutTrunk_mask.png"
    trunk_mask_output_path = out_dir / f"{stem}_excluded_trunk_mask.png"

    Image.fromarray(overlay_img).save(overlay_output_path)
    Image.fromarray((primary_clean_mask * 255).astype(np.uint8)).save(pure_mask_output_path)
    Image.fromarray((trunk_mask * 255).astype(np.uint8)).save(trunk_mask_output_path)

    print("========================================")
    print(f"影像總面積：{total_pixels:,} px")
    print(f"樹幹/粗枝排除面積 (Trunk)：{trunk_pixel_count:,} px")
    print(f"主體純次冠層面積 (Pure Leaf Subcanopy)：{primary_clean_pixels:,} px ({primary_clean_coverage:.2f}% 全圖)")
    print(f"成果標記圖檔：{overlay_output_path}")
    print(f"純次冠層遮罩：{pure_mask_output_path}")
    print(f"樹幹排除遮罩：{trunk_mask_output_path}")
    print("========================================")

    return {
        "image_shape": (h, w),
        "total_pixels": total_pixels,
        "detection_count": num_detections,
        "primary_index": primary_idx,
        "trunk_pixels": trunk_pixel_count,
        "primary_clean_pixels": primary_clean_pixels,
        "primary_clean_coverage": primary_clean_coverage,
        "overlay_output_path": str(overlay_output_path),
        "pure_mask_output_path": str(pure_mask_output_path),
        "trunk_mask_output_path": str(trunk_mask_output_path),
    }


def main():
    parser = argparse.ArgumentParser(description="SAM 3 主體次冠層分割 (自動剔除樹幹與枝幹)")
    parser.add_argument("--image", type=str, default="data/test02.jpeg", help="輸入影像路徑")
    parser.add_argument("--checkpoint", type=str, default="sam3.pt", help="SAM 3 權重路徑")
    parser.add_argument("--prompt", type=str, default="tree subcanopy", help="正向提示詞")
    parser.add_argument("--negative-prompt", type=str, default="tree trunk, tree branch", help="負向排除提示詞")
    parser.add_argument("--threshold", type=float, default=0.25, help="正向置信度門檻值")
    parser.add_argument("--threshold-negative", type=float, default=0.15, help="負向排除置信度門檻值")
    parser.add_argument("--weight-area", type=float, default=0.50, help="像素面積權重")
    parser.add_argument("--weight-center", type=float, default=0.30, help="中心接近度權重")
    parser.add_argument("--weight-conf", type=float, default=0.20, help="置信度權重")
    parser.add_argument("--output-dir", type=str, default="simpleSam/output", help="輸出目錄")
    args = parser.parse_args()

    segment_primary_subcanopy_without_trunk(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        confidence_threshold=args.threshold,
        negative_threshold=args.threshold_negative,
        output_dir=args.output_dir,
        weight_area=args.weight_area,
        weight_center=args.weight_center,
        weight_conf=args.weight_conf,
    )


if __name__ == "__main__":
    main()
