import torch
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

_import_err = None
try:
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
except Exception as e:
    Sam3Processor = None
    build_sam3_image_model = None
    _import_err = e


def get_optimal_device() -> str:
    """自動判定當前硬體平台的最佳運算裝置 (CUDA / MPS / CPU)"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "cpu"
    else:
        return "cpu"


def extract_high_res_foliage_silhouette(
    image_np: np.ndarray,
    roi_mask: np.ndarray,
    sam_trunk_mask: np.ndarray | None = None,
    exclude_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """結合光度學天空差分與色彩特徵，在樹冠範圍內精準提取枝葉像素，保留真實透光孔隙"""
    h, w = image_np.shape[:2]
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    # 1. 嚴謹的天空背景濾除 (Sky Thresholding)
    s_chan = hsv[:, :, 1]
    v_chan = hsv[:, :, 2]
    h_chan = hsv[:, :, 0]

    # 天空判定：極亮且低飽和度
    is_sky = (v_chan > 150) & (s_chan < 20)

    # 2. 真實枝葉特徵捕捉
    r = image_np[:, :, 0].astype(float)
    g = image_np[:, :, 1].astype(float)
    b = image_np[:, :, 2].astype(float)
    rgb_sum = r + g + b + 1e-5
    norm_exg = (2 * g - r - b) / rgb_sum

    # A. 明顯綠色葉片
    is_green = ((h_chan >= 25) & (h_chan <= 95) & (s_chan >= 15)) | (norm_exg > 0.02)
    # B. 背光深色枝葉剪影
    is_dark_silhouette = (v_chan <= 135) & (~is_sky)

    # 3. 基礎實體前景
    raw_foreground = (is_green | is_dark_silhouette) & roi_mask & (~is_sky)

    if exclude_mask is not None:
        raw_foreground &= (~exclude_mask)

    # 4. 樹幹 (Trunk) 與 樹葉 (Leaves) 分離
    if sam_trunk_mask is not None and np.sum(sam_trunk_mask) > 0:
        fine_trunk_mask = raw_foreground & sam_trunk_mask
    else:
        kernel_trunk = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 25))
        bottom_region = np.zeros((h, w), dtype=bool)
        bottom_region[int(h * 0.45):, :] = True
        fine_trunk_mask = cv2.morphologyEx(
            (raw_foreground & bottom_region).astype(np.uint8), cv2.MORPH_OPEN, kernel_trunk
        ).astype(bool)

    # 樹冠葉片：樹冠範圍內除主幹以外之所有細碎剪影與綠色像素
    fine_leaf_mask = raw_foreground & (~fine_trunk_mask)

    # 去除單像素微小噪點
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    fine_leaf_mask = cv2.morphologyEx(fine_leaf_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_clean).astype(bool)

    return fine_leaf_mask, fine_trunk_mask


def compute_best_practice_canopy_envelope(
    foliage_mask: np.ndarray,
    trunk_mask: np.ndarray,
    alpha_radius_px: int = 25,
    macro_hole_ratio: float = 0.012,
) -> np.ndarray:
    """最佳實踐標準樹冠外包絡演算法：
    1. 採用小半徑閉運算貼合各分枝子樹冠 (Sub-canopies)。
    2. 自動剔除分枝間的宏觀巨大天空鴻溝 (Macro Gaps)，同時保留葉簇內部的真實透光微孔隙 (Micro Gaps)。
    """
    h, w = foliage_mask.shape[:2]
    tree_components = foliage_mask | trunk_mask
    total_area = h * w

    if np.sum(tree_components) == 0:
        return np.zeros((h, w), dtype=bool)

    # 1. 緊緻形態學閉運算 (緊密貼合葉簇，禁止跨巨大分枝跨越連通)
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (alpha_radius_px, alpha_radius_px))
    closed = cv2.morphologyEx(tree_components.astype(np.uint8), cv2.MORPH_CLOSE, bridge_kernel)

    # 2. 檢索雙層輪廓 (外輪廓與內部空洞)
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    canopy_envelope = np.zeros((h, w), dtype=np.uint8)

    if hierarchy is not None:
        min_cluster_area = total_area * 0.003
        min_macro_hole_area = total_area * macro_hole_ratio

        # 第一遍：繪製所有有效的外輪廓
        for i, cnt in enumerate(contours):
            is_inner_hole = hierarchy[0][i][3] >= 0
            area = cv2.contourArea(cnt)
            if not is_inner_hole and area >= min_cluster_area:
                cv2.drawContours(canopy_envelope, [cnt], -1, color=1, thickness=-1)

        # 第二遍：挖除宏觀巨大天空鴻溝 (Macro Gaps)
        for i, cnt in enumerate(contours):
            is_inner_hole = hierarchy[0][i][3] >= 0
            area = cv2.contourArea(cnt)
            if is_inner_hole and area >= min_macro_hole_area:
                cv2.drawContours(canopy_envelope, [cnt], -1, color=0, thickness=-1)

    return canopy_envelope.astype(bool)


def isolate_main_tree_cluster(
    binary_mask: np.ndarray,
    w: int,
    h: int,
    bridge_distance: int = 35,
) -> np.ndarray:
    """以中央主幹為根基，提取物理連通的主樹網絡，徹底排除右側天空、建築與兩側鄰居樹木"""
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge_distance, bridge_distance))
    dilated = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_DILATE, bridge_kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    if num_labels <= 1:
        return binary_mask

    seed_cx = int(w * 0.5)
    seed_cy = int(h * 0.7)
    seed_r = int(min(w, h) * 0.15)
    seed_mask = np.zeros((h, w), dtype=bool)
    seed_mask[max(0, seed_cy - seed_r):min(h, seed_cy + seed_r), max(0, seed_cx - seed_r):min(w, seed_cx + seed_r)] = True

    overlap = labels[seed_mask & (dilated > 0)]
    overlap = overlap[overlap > 0]

    if len(overlap) > 0:
        target_id = int(np.bincount(overlap).argmax())
    else:
        img_center = np.array([w / 2.0, h / 2.0])
        max_diag = np.sqrt(w**2 + h**2) / 2.0
        scores = []
        for i in range(1, num_labels):
            dist = np.linalg.norm(centroids[i] - img_center)
            center_score = 1.0 - (dist / max_diag)
            area_score = stats[i, cv2.CC_STAT_AREA] / (h * w)
            scores.append(0.6 * center_score + 0.4 * area_score)
        target_id = 1 + int(np.argmax(scores))

    main_cluster_mask = (labels == target_id)
    return binary_mask & main_cluster_mask


def segment_canopy_leaves(
    image_path: str,
    checkpoint_path: str = "sam3.pt",
    crown_prompt: str = "tree crown",
    prompt_text: str = "tree leaves",
    trunk_prompt: str = "main tree trunk",
    output_path: str = "data/test02_leaves_segmented.png",
    threshold: float = 0.25,
    enable_color_filter: bool = True,
    isolate_main_tree: bool = True,
    device: str | None = None,
) -> dict:
    """最佳實踐標準：以 SAM 3 鎖定主樹 ＋ 高解析度天空剪影差分 ＋ 宏觀鴻溝剔除緊緻凹包"""
    img_file = Path(image_path)
    if not img_file.exists():
        raise FileNotFoundError(f"找不到輸入影像：{image_path}")

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"找不到權重檔案：{checkpoint_path}")

    if build_sam3_image_model is None:
        raise ImportError(f"無法載入 sam3 模組：{_import_err}")

    if device is None:
        device = get_optimal_device()

    print(f"使用運算裝置：{device}")

    # 1. 載入模型
    print(f"正在載入 SAM 3 模型 ({checkpoint_path})...")
    model = build_sam3_image_model(checkpoint_path=checkpoint_path, device=device)
    model = model.to(device).float()
    processor = Sam3Processor(model, device=device, confidence_threshold=threshold)

    # 2. 讀取影像
    print(f"正在讀取影像：{image_path}")
    raw_image = Image.open(image_path).convert("RGB")
    image_np = np.array(raw_image)
    h, w = image_np.shape[:2]
    total_pixels = h * w

    # 3. SAM 3 推論
    print(f"1/3 執行主樹幹辨識：'{trunk_prompt}'...")
    state_image = processor.set_image(raw_image)
    output_trunk = processor.set_text_prompt(prompt=trunk_prompt, state=state_image)
    trunk_masks = output_trunk.get("masks", None)

    print(f"2/3 執行建築物與背景雜項辨識：'building, lamppost'...")
    output_building = processor.set_text_prompt(prompt="building, architecture, lamppost", state=state_image)
    building_masks = output_building.get("masks", None)

    sam_trunk = np.zeros((h, w), dtype=bool)
    if trunk_masks is not None and len(trunk_masks) > 0:
        tm_np = trunk_masks.squeeze(1).cpu().numpy().astype(bool) if trunk_masks.ndim == 4 else trunk_masks.cpu().numpy().astype(bool)
        for m in tm_np:
            sam_trunk |= (m[0] if m.ndim == 3 else m)

    exclude_structure = np.zeros((h, w), dtype=bool)
    if building_masks is not None and len(building_masks) > 0:
        b_np = building_masks.squeeze(1).cpu().numpy().astype(bool) if building_masks.ndim == 4 else building_masks.cpu().numpy().astype(bool)
        for m in b_np:
            exclude_structure |= (m[0] if m.ndim == 3 else m)

    # 4. 高解析度剪影與葉片提取
    print("正在執行高解析度背光剪影差分與細枝葉像素提取...")
    raw_tree_mask = np.ones((h, w), dtype=bool) & (~exclude_structure)
    fine_leaf_mask, fine_trunk_mask = extract_high_res_foliage_silhouette(
        image_np=image_np,
        roi_mask=raw_tree_mask,
        sam_trunk_mask=sam_trunk,
        exclude_mask=exclude_structure,
    )

    # 5. 主樹拓撲隔離
    if isolate_main_tree:
        print("正在執行主樹連通拓撲隔離，剔除周圍非主樹雜訊...")
        all_tree_elements = fine_leaf_mask | fine_trunk_mask
        isolated_tree_elements = isolate_main_tree_cluster(all_tree_elements, w, h, bridge_distance=35)
        fine_leaf_mask &= isolated_tree_elements
        fine_trunk_mask &= isolated_tree_elements

    # 6. 樹冠最佳實踐外包絡計算 (分母 Crown Area：忽略分枝間過大空白)
    print("正在依最佳實踐標準計算樹冠外包絡 (自動忽略宏觀天空鴻溝)...")
    crown_envelope = compute_best_practice_canopy_envelope(
        foliage_mask=fine_leaf_mask,
        trunk_mask=fine_trunk_mask,
        alpha_radius_px=int(min(w, h) * 0.025),      # 小半徑緊緻貼合
        macro_hole_ratio=0.010,                      # 挖除大於 1% 全圖面積的封閉天空鴻溝
    )

    # 樹冠內的最終有效純葉片
    pure_leaf_mask = fine_leaf_mask & crown_envelope

    # 7. 計算統計指標
    leaf_pixels = int(np.sum(pure_leaf_mask))
    crown_pixels = int(np.sum(crown_envelope))
    trunk_pixels = int(np.sum(fine_trunk_mask & crown_envelope))

    crown_leaf_density = (leaf_pixels / crown_pixels) * 100.0 if crown_pixels > 0 else 0.0
    crown_porosity = max(0.0, 100.0 - crown_leaf_density)
    image_leaf_coverage = (leaf_pixels / total_pixels) * 100.0

    print("========================================")
    print(f"影像總面積 (Image Area)：{total_pixels:,} px")
    print(f"樹冠整體範圍面積 (Crown Area 分母)：{crown_pixels:,} px ({(crown_pixels / total_pixels) * 100:.2f}% 全圖)")
    print(f"樹幹/木質部面積：{trunk_pixels:,} px")
    print(f"純綠葉覆蓋面積 (分子)：{leaf_pixels:,} px")
    print("----------------------------------------")
    print(f"★ 樹冠內部葉片密度 (Crown Leaf Density)：{crown_leaf_density:.2f}%")
    print(f"★ 樹冠透光/孔隙率 (Crown Porosity)：{crown_porosity:.2f}%")
    print(f"★ 全圖葉片覆蓋率 (Image Leaf Coverage)：{image_leaf_coverage:.2f}%")
    print("========================================")

    # 8. 渲染高精細成果標記圖
    overlay = image_np.copy()

    # 樹冠凹包邊界：精準黃色線條
    contours, _ = cv2.findContours(crown_envelope.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color=(255, 230, 0), thickness=2)

    # 樹幹區域：淡灰色半透明
    overlay[fine_trunk_mask & crown_envelope] = (
        overlay[fine_trunk_mask & crown_envelope] * 0.4 + np.array([130, 130, 130]) * 0.6
    ).astype(np.uint8)

    # 純葉片區域：鮮綠色半透明
    overlay[pure_leaf_mask] = (
        overlay[pure_leaf_mask] * 0.3 + np.array([30, 240, 60]) * 0.7
    ).astype(np.uint8)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)
    print(f"成果標記圖已儲存至：{output_path}")

    return {
        "total_image_pixels": total_pixels,
        "crown_area_pixels": crown_pixels,
        "pure_leaf_pixels": leaf_pixels,
        "trunk_pixels": trunk_pixels,
        "crown_leaf_density_percentage": crown_leaf_density,
        "crown_porosity_percentage": crown_porosity,
        "image_leaf_coverage_percentage": image_leaf_coverage,
        "output_image": str(output_path),
    }


if __name__ == "__main__":
    segment_canopy_leaves(
        image_path="data/test02.jpeg",
        checkpoint_path="sam3.pt",
        crown_prompt="tree crown",
        prompt_text="tree leaves",
        trunk_prompt="main tree trunk",
        output_path="data/test02_leaves_segmented.png",
    )
