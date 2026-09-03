"""評估 SAM 3 直接使用 prompt='subcanopy gap' 之成效評測腳本
測試影像：
1. data/IMG_20260831_092406.jpg (向陽晴天茂密樹冠)
2. data/IMG_20260831_144604.jpg (仰角透光稀疏大樹冠)

測試重點：
- 驗證 SAM 3 文本編碼器是否能理解 "subcanopy gap"
- 觀察 SAM 3 抓取出的孔洞遮罩與肉眼可見天光之吻合度
- 輸出多面板對比圖至 samplingAnalysis/output 與 artifact 目錄
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


def eval_subcanopy_gap_prompt(
    image_paths: list[str] = [
        "data/IMG_20260831_092406.jpg",
        "data/IMG_20260831_144604.jpg",
    ],
    checkpoint_path: str = "sam3.pt",
    gap_prompt: str = "subcanopy gap",
    device: str = "cpu",
):
    print(f"[1/4] 載入 SAM 3 模型: {checkpoint_path} ({device})...")
    model = build_sam3_image_model(checkpoint_path=checkpoint_path, device=device)
    model = model.to(device).float()
    processor = Sam3Processor(model, device=device, confidence_threshold=0.15)

    artifact_dir = Path("/Users/rickyho/.gemini/antigravity-ide/brain/191c4629-8023-4a46-9cdc-ba2569319fc4")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path("samplingAnalysis/output")
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_p_str in image_paths:
        img_p = Path(img_p_str)
        if not img_p.exists():
            continue

        print(f"\n[2/4] 正在處理影像: {img_p.name}...")
        raw_img = Image.open(img_p).convert("RGB")
        img_rgb = np.array(raw_img)
        h, w = img_rgb.shape[:2]

        state = processor.set_image(raw_img)

        # 1. 正向次冠層
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

        # 2. 測試目標: 直接提示 "subcanopy gap"
        print(f" -> 正在執行 SAM 3 提示詞推論: '{gap_prompt}' (門檻值: 0.15)...")
        processor.set_confidence_threshold(0.15)
        res_gap = processor.set_text_prompt(prompt=gap_prompt, state=state)
        raw_gaps = res_gap.get("masks", None)
        gap_scores = res_gap.get("scores", None)
        
        gap_mask = np.zeros((h, w), dtype=bool)
        num_gap_instances = 0
        if raw_gaps is not None and gap_scores is not None and len(gap_scores) > 0:
            g_np = raw_gaps.cpu().numpy().squeeze()
            num_gap_instances = len(gap_scores)
            if g_np.ndim == 3:
                for m in g_np:
                    gap_mask |= m.astype(bool)
            elif g_np.ndim == 2:
                gap_mask = g_np.astype(bool)

        # 3. 負向樹幹
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

        # 統計
        canopy_px = int(np.sum(canopy_mask))
        gap_px = int(np.sum(gap_mask))
        internal_gap_mask = canopy_mask & gap_mask
        internal_gap_px = int(np.sum(internal_gap_mask))

        # 扣除後純葉片
        pure_leaf_mask = canopy_mask & (~trunk_mask) & (~gap_mask)
        pure_leaf_px = int(np.sum(pure_leaf_mask))

        print(f" -> 樹冠總像素: {canopy_px:,} px")
        print(f" -> prompt='{gap_prompt}' 偵測到的實例數量: {num_gap_instances}")
        print(f" -> prompt='{gap_prompt}' 偵測到的全圖總像素: {gap_px:,} px")
        print(f" -> 位於樹冠內部的孔洞像素: {internal_gap_px:,} px (佔樹冠 {internal_gap_px / (canopy_px + 1e-5) * 100.0:.2f}%)")
        print(f" -> 扣除樹幹與 subcanopy gap 後的純葉片: {pure_leaf_px:,} px")

        # 4. 渲染診斷圖
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
        plt.subplots_adjust(wspace=0.08, hspace=0.12)

        # Panel 1: 原圖
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title(f"(A) Input: {img_p.name}", fontsize=13, fontweight="bold")
        axes[0, 0].axis("off")

        # Panel 2: SAM 3 'subcanopy gap' 獨立遮罩 (亮洋紅色)
        gap_vis = img_rgb.copy()
        gap_overlay = np.zeros_like(img_rgb)
        gap_overlay[gap_mask] = [255, 0, 180] # 洋紅色
        gap_vis = cv2.addWeighted(gap_vis, 0.55, gap_overlay, 0.45, 0)
        axes[0, 1].imshow(gap_vis)
        axes[0, 1].set_title(f"(B) SAM 3 Prompt: '{gap_prompt}' ({num_gap_instances} instances, Magenta)", fontsize=13, fontweight="bold", color="darkmagenta")
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

        # Panel 4: 樹冠扣除 subcanopy gap 之最終成效 (綠色=純葉, 紅色=扣除的孔洞)
        final_vis = img_rgb.copy()
        final_overlay = np.zeros_like(img_rgb)
        final_overlay[pure_leaf_mask] = [0, 255, 60]
        final_overlay[internal_gap_mask] = [255, 0, 0] # 紅色扣除孔
        final_overlay[trunk_mask] = [255, 140, 0]
        final_vis = cv2.addWeighted(final_vis, 0.60, final_overlay, 0.40, 0)

        cnts, _ = cv2.findContours(canopy_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(final_vis, cnts, -1, (255, 255, 255), 2)

        axes[1, 1].imshow(final_vis)
        axes[1, 1].set_title(f"(D) Final: Green=Leaf, Red=Deducted '{gap_prompt}' ({internal_gap_px / (canopy_px + 1e-5) * 100.0:.2f}%)", fontsize=13, fontweight="bold", color="darkgreen")
        axes[1, 1].axis("off")

        stem = img_p.stem
        out_file = out_dir / f"sam3_subcanopy_gap_eval_{stem}.png"
        art_file = artifact_dir / f"sam3_subcanopy_gap_eval_{stem}.png"
        plt.savefig(out_file, bbox_inches="tight")
        plt.savefig(art_file, bbox_inches="tight")
        plt.close()
        print(f" -> 圖檔儲存至: {out_file} 與 {art_file}")

    print("\n==================================================")
    print(f"🎉 SAM 3 '{gap_prompt}' 評測全部完成！")
    print("==================================================")


if __name__ == "__main__":
    eval_subcanopy_gap_prompt()
