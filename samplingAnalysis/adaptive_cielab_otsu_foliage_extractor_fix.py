"""自適應 CIELAB a* 零調參樹葉密度與孔隙分析引擎 (方案二 - 雙模態智慧保護版)
檔案路徑：samplingAnalysis/adaptive_cielab_otsu_foliage_extractor.py

核心數學優化：
1. CIE 國際標準色度中界線：a* = 128 (a* < 128 定義為植物綠色，a* >= 128 為非綠色大氣/枝幹)。
2. 雙峰性檢定 (Bimodal Validation)：
   - 若樹冠內部存在明顯的「透光天空 vs 綠葉」雙峰 (如仰拍樹 144604)，Otsu 自動適應分割。
   - 若樹冠極度茂密實心 (單峰 092406)，自動錨定於國際物理標準 a* < 128，防止 Otsu 強制腰斬綠葉。
3. 嚴謹拓撲閉環：Canopy Gaps = Envelope - Living Foliage - Wood Trunk。
"""

from pathlib import Path
import sys
import argparse
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


def compute_canopy_envelope(binary_mask: np.ndarray, radius_px: int = 35) -> np.ndarray:
    """形態學閉運算計算樹冠外包絡 (Canopy Envelope，分母基準)"""
    if np.sum(binary_mask) == 0:
        return np.zeros_like(binary_mask, dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius_px, radius_px))
    closed = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return closed.astype(bool)


def process_image_cielab_adaptive(
    image_path: str | Path,
    processor: Sam3Processor,
    prompt: str = "tree subcanopy",
    negative_prompt: str = "tree trunk, tree branch",
    confidence_threshold: float = 0.25,
    negative_threshold: float = 0.155,
    output_dir: Path | None = None,
    artifact_dir: Path | None = None,
) -> dict:
    """針對單張影像執行方案二之自適應 CIELAB a* 分析"""
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"找不到檔案：{image_path}")

    raw_img = Image.open(img_path).convert("RGB")
    img_rgb = np.array(raw_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_rgb.shape[:2]
    stem = img_path.stem

    # 1. SAM 3 宏觀樹冠與木質樹幹推論
    state = processor.set_image(raw_img)
    
    processor.set_confidence_threshold(confidence_threshold)
    res_canopy = processor.set_text_prompt(prompt=prompt, state=state)
    raw_canopy = res_canopy.get("masks", None)
    
    canopy_mask = np.zeros((h, w), dtype=bool)
    if raw_canopy is not None:
        c_np = raw_canopy.cpu().numpy().squeeze()
        if c_np.ndim == 3:
            for m in c_np:
                canopy_mask |= m.astype(bool)
        elif c_np.ndim == 2:
            canopy_mask = c_np.astype(bool)

    processor.set_confidence_threshold(negative_threshold)
    res_trunk = processor.set_text_prompt(prompt=negative_prompt, state=state)
    raw_trunk = res_trunk.get("masks", None)
    
    trunk_mask = np.zeros((h, w), dtype=bool)
    if raw_trunk is not None:
        t_np = raw_trunk.cpu().numpy().squeeze()
        if t_np.ndim == 3:
            for m in t_np:
                trunk_mask |= m.astype(bool)
        elif t_np.ndim == 2:
            trunk_mask = t_np.astype(bool)

    # 2. 計算樹冠外包絡 (分母)
    canopy_envelope = compute_canopy_envelope(canopy_mask, radius_px=int(min(w, h) * 0.03))
    
    # 3. 方案二核心：CIELAB a* 通道自適應色度分割
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    canopy_a_pixels = a_chan[canopy_envelope]
    
    # 計算色度標準差與 Otsu 雙峰性
    if len(canopy_a_pixels) > 100:
        std_a = np.std(canopy_a_pixels)
        otsu_val, _ = cv2.threshold(canopy_a_pixels, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 若色度標準差較大且存在雙峰，採 Otsu 分界；若樹冠均勻茂密，採 CIE 標準物理中界線 128.0
        if std_a > 6.5 and otsu_val < 127:
            effective_thresh = otsu_val
            method_used = f"Adaptive Otsu ({effective_thresh:.1f})"
        else:
            effective_thresh = 128.0  # CIE 標準綠色物理界線
            method_used = f"CIE Physics Line ({effective_thresh:.1f})"
            
        leaf_chroma_mask = (a_chan < effective_thresh) & canopy_envelope
    else:
        effective_thresh = 128.0
        method_used = "Default Line (128.0)"
        leaf_chroma_mask = canopy_mask

    # 4. 嚴謹拓撲布林差集
    # 純綠葉 = 色度綠葉 扣除 樹幹 (且受限於外包絡)
    pure_foliage_mask = leaf_chroma_mask & (~trunk_mask) & canopy_envelope
    
    # 內部透光孔隙 (Gaps) = 外包絡 - 純綠葉 - 樹幹
    internal_gaps_mask = canopy_envelope & (~pure_foliage_mask) & (~trunk_mask)

    # 5. 指標量化統計
    envelope_px = int(np.sum(canopy_envelope))
    foliage_px = int(np.sum(pure_foliage_mask))
    trunk_px = int(np.sum(trunk_mask & canopy_envelope))
    gaps_px = int(np.sum(internal_gaps_mask))

    density_pct = (foliage_px / envelope_px * 100.0) if envelope_px > 0 else 0.0
    porosity_pct = (gaps_px / envelope_px * 100.0) if envelope_px > 0 else 0.0
    wood_ratio_pct = (trunk_px / envelope_px * 100.0) if envelope_px > 0 else 0.0

    print(f"\n================ [{stem}] 方案二 (CIELAB) 量化報告 ================")
    print(f"★ 樹冠外包絡面積 (分母 Envelope): {envelope_px:,} px")
    print(f"★ 判定機制與門檻: {method_used}")
    print(f"★ 純樹葉面積 (Living Foliage): {foliage_px:,} px -> 【實心密度: {density_pct:.2f}%】")
    print(f"★ 內部透光孔隙 (Canopy Gaps): {gaps_px:,} px -> 【孔隙率: {porosity_pct:.2f}%】")
    print(f"★ 木質樹幹枝幹 (Wood Trunk): {trunk_px:,} px -> 【佔比: {wood_ratio_pct:.2f}%】")
    print("==============================================================")

    # 6. 渲染多面板專業診斷圖
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    plt.subplots_adjust(wspace=0.08, hspace=0.14)

    # Panel 1: 原圖
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title(f"(A) Input: {img_path.name}", fontsize=13, fontweight="bold")
    axes[0, 0].axis("off")

    # Panel 2: CIELAB a* 色度圖
    im_a = axes[0, 1].imshow(a_chan, cmap="PRGn_r", vmin=90, vmax=140)
    axes[0, 1].set_title(f"(B) CIELAB a* Channel [{method_used}]", fontsize=13, fontweight="bold", color="purple")
    axes[0, 1].axis("off")
    fig.colorbar(im_a, ax=axes[0, 1], fraction=0.046, pad=0.04, label="a* Value (<128: Green, >=128: Sky/Wood)")

    # Panel 3: SAM 3 外包絡 (白色線) 與 樹幹 (橘色)
    sam_vis = img_rgb.copy()
    sam_overlay = np.zeros_like(img_rgb)
    sam_overlay[canopy_envelope] = [0, 180, 255]
    sam_overlay[trunk_mask] = [255, 140, 0]
    sam_vis = cv2.addWeighted(sam_vis, 0.65, sam_overlay, 0.35, 0)
    cnts_env, _ = cv2.findContours(canopy_envelope.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(sam_vis, cnts_env, -1, (255, 255, 255), 3)
    axes[1, 0].imshow(sam_vis)
    axes[1, 0].set_title("(C) Canopy Envelope (White Outline) & Trunk (Orange)", fontsize=13, fontweight="bold")
    axes[1, 0].axis("off")

    # Panel 4: 最終方案二 3 色無損診斷圖
    final_vis = img_rgb.copy()
    final_overlay = np.zeros_like(img_rgb)
    final_overlay[pure_foliage_mask] = [0, 255, 50]       # 亮綠: 純樹葉
    final_overlay[internal_gaps_mask] = [0, 140, 255]     # 天藍: 內部透光孔隙
    final_overlay[trunk_mask & canopy_envelope] = [255, 140, 0] # 橘色: 樹幹
    final_vis = cv2.addWeighted(final_vis, 0.58, final_overlay, 0.42, 0)
    cv2.drawContours(final_vis, cnts_env, -1, (255, 255, 255), 2)
    
    axes[1, 1].imshow(final_vis)
    axes[1, 1].set_title(
        f"(D) Scheme 2: Leaf {density_pct:.1f}% (Green), Gaps {porosity_pct:.1f}% (Blue), Trunk {wood_ratio_pct:.1f}% (Orange)",
        fontsize=13, fontweight="bold", color="darkgreen"
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
        "envelope_pixels": envelope_px,
        "foliage_pixels": foliage_px,
        "gaps_pixels": gaps_px,
        "density_pct": density_pct,
        "porosity_pct": porosity_pct,
        "wood_ratio_pct": wood_ratio_pct,
        "method_used": method_used,
    }


def main():
    parser = argparse.ArgumentParser(description="Adaptive CIELAB a* Foliage & Gap Analyzer (Scheme 2)")
    parser.add_argument("--images", nargs="+", default=["data/IMG_20260831_092406.jpg", "data/IMG_20260831_144604.jpg"], help="測試影像清單")
    parser.add_argument("--checkpoint", type=str, default="sam3.pt", help="SAM 3 權重路徑")
    parser.add_argument("--device", type=str, default="cpu", help="運算裝置")
    parser.add_argument("--output-dir", type=str, default="samplingAnalysis/output", help="輸出圖檔目錄")
    args = parser.parse_args()

    print(f"[1/3] 正在載入 SAM 3 模型: {args.checkpoint} ({args.device})...")
    model = build_sam3_image_model(checkpoint_path=args.checkpoint, device=args.device)
    model = model.to(args.device).float()
    processor = Sam3Processor(model, device=args.device, confidence_threshold=0.25)

    art_dir = Path("/Users/rickyho/.gemini/antigravity-ide/brain/191c4629-8023-4a46-9cdc-ba2569319fc4")
    out_dir = Path(args.output_dir)

    for img_p in args.images:
        process_image_cielab_adaptive(
            image_path=img_p,
            processor=processor,
            output_dir=out_dir,
            artifact_dir=art_dir,
        )

    print("\n🎉 方案二自適應分析全部完成！輸出圖檔已儲存至 samplingAnalysis/output 與 artifact 目錄。")


if __name__ == "__main__":
    main()
