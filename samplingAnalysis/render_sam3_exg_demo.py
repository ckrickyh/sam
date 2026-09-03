"""SAM 3 + 超綠指數 (ExG) 植被視覺化渲染器
針對 data/IMG_20260831_092406.jpg 執行：
1. 讀取影像並計算全圖超綠指數 (ExG) 與熱力圖
2. 結合 SAM 3 分割樹冠次冠層並扣除樹幹
3. 生成多面板視覺化圖檔 (原圖、SAM 3 遮罩、ExG 光譜分佈、高亮疊加圖)
"""

from pathlib import Path
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def render_sam3_exg_visualization(
    image_path: str = "data/IMG_20260831_092406.jpg",
    checkpoint_path: str = "sam3.pt",
    output_path: str = "samplingAnalysis/output/IMG_092406_sam3_exg_render.png",
    artifact_output_path: str = "/Users/rickyho/.gemini/antigravity-ide/brain/191c4629-8023-4a46-9cdc-ba2569319fc4/IMG_092406_sam3_exg_render.png",
    device: str = "cpu",
):
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"找不到檔案: {image_path}")

    print(f"[1/4] 讀取影像: {img_path}")
    raw_img = Image.open(img_path).convert("RGB")
    img_rgb = np.array(raw_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_rgb.shape[:2]

    # 1. 計算超綠指數 (ExG)
    print("[2/4] 計算超綠指數 (Excess Green Index)...")
    r = img_rgb[:, :, 0].astype(float)
    g = img_rgb[:, :, 1].astype(float)
    b = img_rgb[:, :, 2].astype(float)
    rgb_sum = r + g + b + 1e-5
    norm_exg = (2 * g - r - b) / rgb_sum  # 歸一化 ExG: [-1.0, 2.0]

    # ExG 二值化綠色保護遮罩 (ExG > 0.015 與 HSV 綠色)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_chan, s_chan = hsv[:, :, 0], hsv[:, :, 1]
    is_hsv_green = (h_chan >= 25) & (h_chan <= 95) & (s_chan >= 15)
    is_exg_green = norm_exg > 0.015
    pure_exg_mask = (is_exg_green | is_hsv_green).astype(np.uint8) * 255

    # 2. 執行 SAM 3 推論
    print(f"[3/4] 載入 SAM 3 模型 ({checkpoint_path}) 進行次冠層與樹幹分割...")
    model = build_sam3_image_model(checkpoint_path=checkpoint_path, device=device)
    model = model.to(device).float()
    processor = Sam3Processor(model, device=device, confidence_threshold=0.25)

    state = processor.set_image(raw_img)

    # 正向: 次冠層
    res_subcanopy = processor.set_text_prompt(prompt="tree subcanopy", state=state)
    raw_subcanopy_masks = res_subcanopy.get("masks", None)
    
    # 負向: 樹幹
    processor.set_confidence_threshold(0.155)
    res_trunk = processor.set_text_prompt(prompt="tree trunk, tree branch", state=state)
    raw_trunk_masks = res_trunk.get("masks", None)

    # 組合 SAM 3 遮罩
    subcanopy_mask = np.zeros((h, w), dtype=bool)
    if raw_subcanopy_masks is not None and len(raw_subcanopy_masks) > 0:
        sub_np = raw_subcanopy_masks.cpu().numpy().squeeze()
        if sub_np.ndim == 3:
            for m in sub_np:
                subcanopy_mask |= m.astype(bool)
        elif sub_np.ndim == 2:
            subcanopy_mask = sub_np.astype(bool)

    trunk_mask = np.zeros((h, w), dtype=bool)
    if raw_trunk_masks is not None and len(raw_trunk_masks) > 0:
        trk_np = raw_trunk_masks.cpu().numpy().squeeze()
        if trk_np.ndim == 3:
            for m in trk_np:
                trunk_mask |= m.astype(bool)
        elif trk_np.ndim == 2:
            trunk_mask = trk_np.astype(bool)

    # SAM 3 純葉片區域 (次冠層扣除樹幹)
    sam_pure_subcanopy = subcanopy_mask & (~trunk_mask)

    # 結合 SAM 3 幾何約束 + ExG 光譜物理過濾 (真實精確葉片位置)
    sam_exg_intersection = sam_pure_subcanopy & (pure_exg_mask == 255)

    # 3. 渲染高解析度 4 面板對比視覺化圖
    print("[4/4] 渲染多面板 ExG 與 SAM 3 視覺化診斷圖...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    plt.subplots_adjust(wspace=0.08, hspace=0.12)

    # Panel 1: 原圖
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("(A) Original Input Image", fontsize=13, fontweight="bold")
    axes[0, 0].axis("off")

    # Panel 2: 全圖 ExG 超綠光譜熱力圖
    # 將 norm_exg 縮放至 [0, 1] 作為熱力圖
    exg_clipped = np.clip(norm_exg, -0.2, 0.4)
    im2 = axes[0, 1].imshow(exg_clipped, cmap="YlGn")
    axes[0, 1].set_title("(B) Excess Green Index (ExG) Spectral Heatmap", fontsize=13, fontweight="bold")
    axes[0, 1].axis("off")
    fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04, label="Norm-ExG Value")

    # Panel 3: SAM 3 樹冠遮罩 (扣除樹幹) 與輪廓
    sam_vis = img_rgb.copy()
    sam_overlay = np.zeros_like(img_rgb)
    sam_overlay[subcanopy_mask] = [0, 180, 255]       # 淺藍: 原始次冠層
    sam_overlay[trunk_mask] = [255, 60, 0]             # 紅橘: 排除的樹幹
    sam_overlay[sam_pure_subcanopy] = [0, 255, 0]      # 綠色: SAM 3 純葉片
    sam_vis = cv2.addWeighted(sam_vis, 0.60, sam_overlay, 0.40, 0)
    axes[1, 0].imshow(sam_vis)
    axes[1, 0].set_title("(C) SAM 3 Segmentation (Green: Leaf, Orange: Trunk)", fontsize=13, fontweight="bold")
    axes[1, 0].axis("off")

    # Panel 4: 最終「SAM 3 空間約束 + ExG 純葉片」高亮渲染圖
    final_vis = img_rgb.copy()
    final_overlay = np.zeros_like(img_rgb)
    # 純葉片標註為鮮豔亮綠色
    final_overlay[sam_exg_intersection] = [0, 255, 50]
    # 樹冠內部被 ExG 剔除的孔洞/雜色標註為半透明藍色
    internal_gaps = sam_pure_subcanopy & (~sam_exg_intersection)
    final_overlay[internal_gaps] = [0, 120, 255]
    final_vis = cv2.addWeighted(final_vis, 0.55, final_overlay, 0.45, 0)

    # 繪製邊緣輪廓
    contours, _ = cv2.findContours(sam_pure_subcanopy.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(final_vis, contours, -1, (255, 255, 255), 2)

    axes[1, 1].imshow(final_vis)
    axes[1, 1].set_title("(D) Final Render: SAM 3 + ExG Pure Foliage Location (Bright Green)", fontsize=13, fontweight="bold")
    axes[1, 1].axis("off")

    # 儲存圖檔至兩處
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    
    Path(artifact_output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(artifact_output_path, bbox_inches="tight")
    plt.close()

    total_canopy_px = int(np.sum(sam_pure_subcanopy))
    pure_exg_px = int(np.sum(sam_exg_intersection))
    density_val = (pure_exg_px / total_canopy_px) if total_canopy_px > 0 else 0.0

    print("==================================================")
    print(f"🎉 視覺化渲染完成！")
    print(f"★ 輸出圖檔路徑 1: {output_path}")
    print(f"★ 輸出圖檔路徑 2: {artifact_output_path}")
    print(f"★ SAM 3 次冠層總像素: {total_canopy_px:,} px")
    print(f"★ ExG 判定純綠葉像素: {pure_exg_px:,} px")
    print(f"★ 樹葉實心密度 (Foliage Density): {density_val:.2%}")
    print("==================================================")


if __name__ == "__main__":
    render_sam3_exg_visualization()
