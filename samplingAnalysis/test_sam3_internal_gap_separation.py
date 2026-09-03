"""驗證 SAM 3 原生分離內部「非樹葉空間 (天空/雲朵/大樓/枝幹)」能力
測試策略：
1. 提示詞由宏觀的 "tree subcanopy" 切換為微觀的 "tree leaves" / "tree foliage"
2. 加入負向提示詞: "sky, cloud, tree trunk, tree branch, building"
3. 原生分離內部真實透光孔隙與樹葉
"""

from pathlib import Path
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def test_sam3_internal_gap_separation(
    image_path: str = "data/IMG_20260831_144604.jpg",
    checkpoint_path: str = "sam3.pt",
    output_path: str = "samplingAnalysis/output/IMG_144604_sam3_internal_gaps_test.png",
    artifact_output_path: str = "/Users/rickyho/.gemini/antigravity-ide/brain/191c4629-8023-4a46-9cdc-ba2569319fc4/IMG_144604_sam3_internal_gaps_test.png",
    device: str = "cpu",
):
    img_path = Path(image_path)
    raw_img = Image.open(img_path).convert("RGB")
    img_rgb = np.array(raw_img)
    h, w = img_rgb.shape[:2]

    print("[1/3] 載入 SAM 3 模型...")
    model = build_sam3_image_model(checkpoint_path=checkpoint_path, device=device)
    model = model.to(device).float()
    processor = Sam3Processor(model, device=device, confidence_threshold=0.20)
    state = processor.set_image(raw_img)

    # 1. 精細正向提示詞: "tree leaves" (純樹葉)
    print("[2/3] 執行純樹葉正向推論: 'tree leaves'...")
    res_leaves = processor.set_text_prompt(prompt="tree leaves", state=state)
    raw_leaf_masks = res_leaves.get("masks", None)
    
    leaf_mask = np.zeros((h, w), dtype=bool)
    if raw_leaf_masks is not None:
        l_np = raw_leaf_masks.cpu().numpy().squeeze()
        if l_np.ndim == 3:
            for m in l_np:
                leaf_mask |= m.astype(bool)
        elif l_np.ndim == 2:
            leaf_mask = l_np.astype(bool)

    # 2. 負向提示詞 1: 樹幹與枝幹
    print("[2/3] 執行負向推論: 'tree trunk, tree branch'...")
    processor.set_confidence_threshold(0.15)
    res_trunk = processor.set_text_prompt(prompt="tree trunk, tree branch", state=state)
    raw_trunk_masks = res_trunk.get("masks", None)
    
    trunk_mask = np.zeros((h, w), dtype=bool)
    if raw_trunk_masks is not None:
        t_np = raw_trunk_masks.cpu().numpy().squeeze()
        if t_np.ndim == 3:
            for m in t_np:
                trunk_mask |= m.astype(bool)
        elif t_np.ndim == 2:
            trunk_mask = t_np.astype(bool)

    # 3. 負向提示詞 2: 穿透天空與雲朵 (Sky & Cloud)
    print("[2/3] 執行負向推論: 'sky, cloud, blue sky'...")
    res_sky = processor.set_text_prompt(prompt="sky, cloud", state=state)
    raw_sky_masks = res_sky.get("masks", None)
    
    sky_mask = np.zeros((h, w), dtype=bool)
    if raw_sky_masks is not None:
        s_np = raw_sky_masks.cpu().numpy().squeeze()
        if s_np.ndim == 3:
            for m in s_np:
                sky_mask |= m.astype(bool)
        elif s_np.ndim == 2:
            sky_mask = s_np.astype(bool)

    # 4. 樹冠外包絡 (Canopy Envelope: 樹木在空間中佔據的總輪廓空間)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 55))
    canopy_envelope = cv2.morphologyEx((leaf_mask | trunk_mask).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(canopy_envelope, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canopy_envelope, contours, -1, 255, thickness=cv2.FILLED)
    canopy_envelope_mask = canopy_envelope == 255

    # 5. 純葉片與內部孔洞計算
    # 純葉片 = SAM 葉片 排除 樹幹 排除 天空
    pure_leaves = leaf_mask & (~trunk_mask) & (~sky_mask)
    
    # 樹冠內部非樹葉孔隙 (Gaps) = 位於樹冠外包絡內 且 (屬於天空 或 不屬於純葉/樹幹)
    internal_gaps = canopy_envelope_mask & (~pure_leaves) & (~trunk_mask)

    # 6. 渲染多面板對比圖
    print("[3/3] 渲染 SAM 3 原生孔洞分離圖...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    plt.subplots_adjust(wspace=0.08, hspace=0.12)

    # Panel 1: 原圖
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("(A) Original Input Image (Sparse & Highly Porous Tree)", fontsize=13, fontweight="bold")
    axes[0, 0].axis("off")

    # Panel 2: SAM 3 純葉片遮罩 ('tree leaves')
    leaf_vis = img_rgb.copy()
    leaf_overlay = np.zeros_like(img_rgb)
    leaf_overlay[pure_leaves] = [0, 255, 0]
    leaf_vis = cv2.addWeighted(leaf_vis, 0.60, leaf_overlay, 0.40, 0)
    axes[0, 1].imshow(leaf_vis)
    axes[0, 1].set_title("(B) SAM 3 Prompt: 'tree leaves' (Leaves Only, Gaps Untouched)", fontsize=13, fontweight="bold")
    axes[0, 1].axis("off")

    # Panel 3: SAM 3 枝幹與天空負向分離
    neg_vis = img_rgb.copy()
    neg_overlay = np.zeros_like(img_rgb)
    neg_overlay[trunk_mask] = [255, 140, 0]   # 橘色: 枝幹
    neg_overlay[sky_mask] = [0, 150, 255]     # 藍色: 天空
    neg_vis = cv2.addWeighted(neg_vis, 0.60, neg_overlay, 0.40, 0)
    axes[1, 0].imshow(neg_vis)
    axes[1, 0].set_title("(C) SAM 3 Negative Prompts: Orange=Trunk, Blue=Sky", fontsize=13, fontweight="bold")
    axes[1, 0].axis("off")

    # Panel 4: 最終精確分離圖 (綠色:純葉, 藍色:內部透光孔洞, 橘色:枝幹, 白線:樹冠外包絡)
    final_vis = img_rgb.copy()
    final_overlay = np.zeros_like(img_rgb)
    final_overlay[pure_leaves] = [0, 255, 60]       # 亮綠: 純樹葉
    final_overlay[internal_gaps] = [0, 120, 255]     # 亮藍: 樹冠內部的真實透光孔洞 (Gaps)
    final_overlay[trunk_mask] = [255, 140, 0]        # 橘色: 木質枝幹
    final_vis = cv2.addWeighted(final_vis, 0.60, final_overlay, 0.40, 0)
    cv2.drawContours(final_vis, contours, -1, (255, 255, 255), 2)  # 外包絡線

    axes[1, 1].imshow(final_vis)
    axes[1, 1].set_title("(D) Correct Separation: Green=Leaves, Blue=True Gaps (Porosity)", fontsize=13, fontweight="bold", color="darkgreen")
    axes[1, 1].axis("off")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    
    Path(artifact_output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(artifact_output_path, bbox_inches="tight")
    plt.close()

    total_env_px = int(np.sum(canopy_envelope_mask))
    pure_leaves_px = int(np.sum(pure_leaves))
    gaps_px = int(np.sum(internal_gaps))
    trunk_px = int(np.sum(trunk_mask))
    
    real_density = pure_leaves_px / total_env_px if total_env_px > 0 else 0.0
    real_porosity = gaps_px / total_env_px if total_env_px > 0 else 0.0

    print("==================================================")
    print(f"🎉 SAM 3 內部孔洞原生分離測試完成！")
    print(f"★ 樹冠外包絡總面積 (分母): {total_env_px:,} px")
    print(f"★ SAM 3 純樹葉像素 (綠色): {pure_leaves_px:,} px ({real_density:.2%})")
    print(f"★ 樹冠內部真實透光孔隙 (藍色): {gaps_px:,} px ({real_porosity:.2%})")
    print(f"★ 木質樹幹粗枝像素 (橘色): {trunk_px:,} px")
    print(f"★ 輸出圖檔路徑: {artifact_output_path}")
    print("==================================================")


if __name__ == "__main__":
    test_sam3_internal_gap_separation()
