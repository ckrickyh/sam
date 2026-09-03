"""評估 SAM 3 直接提取「天空與雲朵 (Sky & Cloud)」之效果評測腳本
測試影像：
1. data/IMG_20260831_092406.jpg (向陽晴天茂密樹冠)
2. data/IMG_20260831_144604.jpg (仰角透光稀疏大樹冠)

分析重點：
- SAM 3 對「外圍大片天空」的抓取能力
- SAM 3 對「樹冠內部微小透光孔洞 (Canopy Gaps)」的穿透抓取能力
- 評估是否會誤傷「向陽高光反光葉片」
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


def evaluate_sam3_sky_extraction(
    image_paths: list[str] = [
        "data/IMG_20260831_092406.jpg",
        "data/IMG_20260831_144604.jpg",
    ],
    checkpoint_path: str = "sam3.pt",
    device: str = "cpu",
):
    print(f"[1/4] 載入 SAM 3 模型: {checkpoint_path} ({device})...")
    model = build_sam3_image_model(checkpoint_path=checkpoint_path, device=device)
    model = model.to(device).float()
    processor = Sam3Processor(model, device=device, confidence_threshold=0.25)

    artifact_dir = Path("/Users/rickyho/.gemini/antigravity-ide/brain/191c4629-8023-4a46-9cdc-ba2569319fc4")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path("samplingAnalysis/output")
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_p_str in image_paths:
        img_p = Path(img_p_str)
        if not img_p.exists():
            print(f"找不到檔案: {img_p_str}")
            continue

        print(f"\n[2/4] 正在處理影像: {img_p.name}...")
        raw_img = Image.open(img_p).convert("RGB")
        img_rgb = np.array(raw_img)
        h, w = img_rgb.shape[:2]

        state = processor.set_image(raw_img)

        # 1. 提取次冠層 (Canopy)
        processor.set_confidence_threshold(0.25)
        res_canopy = processor.set_text_prompt(prompt="tree subcanopy", state=state)
        raw_canopy = res_canopy.get("masks", None)
        canopy_mask = np.zeros((h, w), dtype=bool)
        if raw_canopy is not None:
            c_np = raw_canopy.cpu().numpy().squeeze()
            if c_np.ndim == 3:
                for m in c_np:
                    canopy_mask |= m.astype(bool)
            elif c_np.ndim == 2:
                canopy_mask = c_np.astype(bool)

        # 2. 提取樹幹 (Trunk)
        processor.set_confidence_threshold(0.155)
        res_trunk = processor.set_text_prompt(prompt="tree trunk, tree branch", state=state)
        raw_trunk = res_trunk.get("masks", None)
        trunk_mask = np.zeros((h, w), dtype=bool)
        if raw_trunk is not None:
            t_np = raw_trunk.cpu().numpy().squeeze()
            if t_np.ndim == 3:
                for m in t_np:
                    trunk_mask |= m.astype(bool)
            elif t_np.ndim == 2:
                trunk_mask = t_np.astype(bool)

        # 3. 提取天空與雲朵 (Sky & Cloud)
        print(f" -> 正在執行 SAM 3 天空推論: 'sky, blue sky, cloud'...")
        processor.set_confidence_threshold(0.18)
        res_sky = processor.set_text_prompt(prompt="sky, blue sky, cloud", state=state)
        raw_sky = res_sky.get("masks", None)
        sky_mask = np.zeros((h, w), dtype=bool)
        if raw_sky is not None:
            s_np = raw_sky.cpu().numpy().squeeze()
            if s_np.ndim == 3:
                for m in s_np:
                    sky_mask |= m.astype(bool)
            elif s_np.ndim == 2:
                sky_mask = s_np.astype(bool)

        # 計算統計數據
        canopy_px = int(np.sum(canopy_mask))
        trunk_px = int(np.sum(trunk_mask))
        total_sky_px = int(np.sum(sky_mask))
        
        # 位於樹冠內部的穿透天空孔洞 (Internal Canopy Sky Gaps)
        internal_sky_gaps = canopy_mask & sky_mask
        internal_sky_px = int(np.sum(internal_sky_gaps))

        # 扣除後純葉片
        pure_leaf_mask = canopy_mask & (~trunk_mask) & (~sky_mask)
        pure_leaf_px = int(np.sum(pure_leaf_mask))

        gap_ratio_in_canopy = (internal_sky_px / canopy_px * 100.0) if canopy_px > 0 else 0.0
        pure_density = (pure_leaf_px / canopy_px * 100.0) if canopy_px > 0 else 0.0

        print(f" -> 樹冠總像素: {canopy_px:,} px")
        print(f" -> 全圖天空像素: {total_sky_px:,} px")
        print(f" -> SAM 3 抓出的【樹冠內部透光天空孔洞】: {internal_sky_px:,} px ({gap_ratio_in_canopy:.2f}%)")
        print(f" -> 扣除樹幹與天空後的純葉片像素: {pure_leaf_px:,} px (實心密度: {pure_density:.2f}%)")

        # 4. 渲染 4 面板高解析度評測診斷圖
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
        plt.subplots_adjust(wspace=0.08, hspace=0.12)

        # Panel 1: 原圖
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title(f"(A) Input: {img_p.name}", fontsize=13, fontweight="bold")
        axes[0, 0].axis("off")

        # Panel 2: SAM 3 天空分割 (青藍色)
        sky_vis = img_rgb.copy()
        sky_overlay = np.zeros_like(img_rgb)
        sky_overlay[sky_mask] = [0, 195, 255] # 青藍色
        sky_vis = cv2.addWeighted(sky_vis, 0.55, sky_overlay, 0.45, 0)
        axes[0, 1].imshow(sky_vis)
        axes[0, 1].set_title(f"(B) SAM 3 Prompt: 'sky, blue sky, cloud' (Cyan Blue)", fontsize=13, fontweight="bold", color="darkblue")
        axes[0, 1].axis("off")

        # Panel 3: 樹冠 (綠色) 與 樹幹 (橘色)
        canopy_vis = img_rgb.copy()
        canopy_overlay = np.zeros_like(img_rgb)
        canopy_overlay[canopy_mask] = [0, 255, 60]
        canopy_overlay[trunk_mask] = [255, 140, 0]
        canopy_vis = cv2.addWeighted(canopy_vis, 0.60, canopy_overlay, 0.40, 0)
        axes[1, 0].imshow(canopy_vis)
        axes[1, 0].set_title("(C) SAM 3 Canopy (Green) & Trunk (Orange)", fontsize=13, fontweight="bold")
        axes[1, 0].axis("off")

        # Panel 4: 樹冠內部扣除效果 (綠色=純葉, 紅色=樹冠內被扣除的天空孔洞, 橘色=樹幹)
        final_vis = img_rgb.copy()
        final_overlay = np.zeros_like(img_rgb)
        final_overlay[pure_leaf_mask] = [0, 255, 60]         # 純樹葉: 綠色
        final_overlay[internal_sky_gaps] = [255, 40, 0]      # 樹冠內部天空孔洞: 紅色高亮
        final_overlay[trunk_mask] = [255, 140, 0]           # 樹幹: 橘色
        final_vis = cv2.addWeighted(final_vis, 0.60, final_overlay, 0.40, 0)

        # 樹冠輪廓線
        cnts, _ = cv2.findContours(canopy_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(final_vis, cnts, -1, (255, 255, 255), 2)

        axes[1, 1].imshow(final_vis)
        axes[1, 1].set_title(f"(D) Canopy Subtraction: Pure Leaf ({pure_density:.1f}%), Red=Deducted Sky Gaps ({gap_ratio_in_canopy:.1f}%)", fontsize=13, fontweight="bold", color="darkgreen")
        axes[1, 1].axis("off")

        stem = img_p.stem
        out_file = out_dir / f"sam3_sky_eval_{stem}.png"
        art_file = artifact_dir / f"sam3_sky_eval_{stem}.png"
        plt.savefig(out_file, bbox_inches="tight")
        plt.savefig(art_file, bbox_inches="tight")
        plt.close()
        print(f" -> 圖檔儲存至: {out_file} 與 {art_file}")

    print("\n==================================================")
    print("🎉 SAM 3 天空抓取成效評測全部完成！")
    print("==================================================")


if __name__ == "__main__":
    evaluate_sam3_sky_extraction()
