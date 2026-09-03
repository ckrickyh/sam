"""拓撲閉環與 CIELAB 葉綠素物理界線樹冠孔隙分析引擎 (Robust Topological Invariant Foliage & Gap Extractor)

核心原理 (避免正面抓天空的過度擬合陷阱)：
1. 痛點剖析：天空光譜變化極大 (藍天/白雲/灰霾/逆光耀光)，正面抓取天空必然產生邊緣紋理誤殺與漏檢。
2. 物理不變量：植物葉綠素具有明確光譜特徵 (CIELAB a* < 128 為綠色植物，a* >= 128 為非植物大氣/背景/木質)。
3. 拓撲閉環計算：
   - 樹冠總包絡 (Canopy Envelope) = SAM 3 樹冠範圍經形態學閉運算 (消除內部孔隙後的實體外圍邊界)
   - 純樹葉實體 (Living Foliage) = 樹冠包絡內且符合葉綠素特徵 (a* < 128 或 自適應色度分割) 扣除 樹幹
   - 真實內部孔隙 (True Internal Gaps) = 樹冠包絡 - 純樹葉 - 樹幹
"""

from pathlib import Path
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def compute_canopy_envelope(binary_mask: np.ndarray, radius_ratio: float = 0.035) -> np.ndarray:
    """計算樹冠外圍總包絡 (分母 Envelope，封閉內部孔隙以作為幾何總空間邊界)"""
    if np.sum(binary_mask) == 0:
        return np.zeros_like(binary_mask, dtype=bool)
    h, w = binary_mask.shape[:2]
    radius = max(15, int(min(h, w) * radius_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    closed = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    envelope = np.zeros_like(closed)
    cv2.drawContours(envelope, contours, -1, 255, thickness=cv2.FILLED)
    return envelope > 0


def extract_topological_foliage_and_gaps(
    img_bgr: np.ndarray,
    canopy_raw_mask: np.ndarray,
    trunk_raw_mask: np.ndarray,
) -> dict:
    """透過 CIELAB 色度不變量與拓撲補集精準分離樹葉、樹幹與內部孔隙。

    :param img_bgr: 原始 BGR 影像
    :param canopy_raw_mask: SAM 3 提取之原始樹冠遮罩 (bool)
    :param trunk_raw_mask: SAM 3 提取之樹幹遮罩 (bool)
    """
    h, w = img_bgr.shape[:2]

    # 1. 幾何外包絡計算 (封閉內部孔隙，作為分母基準)
    canopy_envelope = compute_canopy_envelope(canopy_raw_mask)

    # 2. CIELAB a* 空間植物葉綠素分析
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    # 提取樹冠外包絡內的 a* 像素分佈
    canopy_a_pixels = a_chan[canopy_envelope]
    
    # 自適應色度分割：
    # 國際標準 CIE 物理界線中值為 128 (a* < 128 為綠，a* >= 128 為非綠)
    if len(canopy_a_pixels) > 100:
        std_a = np.std(canopy_a_pixels)
        # 若存在明顯雙峰 (透光孔洞與綠葉強烈對比)，使用 Otsu 自適應尋找最佳分界
        otsu_thresh, _ = cv2.threshold(canopy_a_pixels, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if std_a > 6.0 and 110 < otsu_thresh < 128:
            foliage_thresh = float(otsu_thresh)
            method_desc = f"Adaptive Otsu ({foliage_thresh:.1f})"
        else:
            foliage_thresh = 128.0
            method_desc = "CIE Physics Boundary (128.0)"
    else:
        foliage_thresh = 128.0
        method_desc = "Default CIE (128.0)"

    # 綠色植被像素遮罩 (符合葉綠素特徵)
    green_foliage_mask = (a_chan < foliage_thresh) & canopy_envelope

    # 3. 嚴謹拓撲閉環差集
    # 純樹葉 = 樹冠包絡內的綠色植被 扣除 樹幹
    pure_foliage_mask = green_foliage_mask & (~trunk_raw_mask) & canopy_envelope

    # 木質樹幹 (限制於包絡內)
    wood_trunk_mask = trunk_raw_mask & canopy_envelope

    # 真實內部透光孔隙 (Topological True Gaps) = 幾何外包絡 - 純樹葉 - 樹幹
    true_internal_gaps_mask = canopy_envelope & (~pure_foliage_mask) & (~wood_trunk_mask)

    # 4. 指標統計
    envelope_pixels = int(np.sum(canopy_envelope))
    foliage_pixels = int(np.sum(pure_foliage_mask))
    trunk_pixels = int(np.sum(wood_trunk_mask))
    gap_pixels = int(np.sum(true_internal_gaps_mask))

    leaf_density_pct = (foliage_pixels / envelope_pixels * 100.0) if envelope_pixels > 0 else 0.0
    gap_fraction_pct = (gap_pixels / envelope_pixels * 100.0) if envelope_pixels > 0 else 0.0
    trunk_ratio_pct = (trunk_pixels / envelope_pixels * 100.0) if envelope_pixels > 0 else 0.0

    return {
        "canopy_envelope": canopy_envelope,
        "pure_foliage_mask": pure_foliage_mask,
        "wood_trunk_mask": wood_trunk_mask,
        "true_internal_gaps_mask": true_internal_gaps_mask,
        "a_chan": a_chan,
        "method_desc": method_desc,
        "envelope_pixels": envelope_pixels,
        "foliage_pixels": foliage_pixels,
        "trunk_pixels": trunk_pixels,
        "gap_pixels": gap_pixels,
        "leaf_density_pct": leaf_density_pct,
        "gap_fraction_pct": gap_fraction_pct,
        "trunk_ratio_pct": trunk_ratio_pct,
    }


def run_test(
    image_path: str = "data/IMG_20260831_092033.jpg",
    checkpoint_path: str = "sam3.pt",
    output_path: str | None = None,
    device: str = "cpu",
):
    img_p = Path(image_path)
    stem = img_p.stem
    if output_path is None:
        output_path = f"samplingAnalysis/output/{stem}_topological_gaps.png"

    print(f"==================================================")
    print(f"[1/4] 載入測試影像: {image_path}")
    raw_img = Image.open(image_path).convert("RGB")
    img_rgb = np.array(raw_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_rgb.shape[:2]

    print(f"[2/4] SAM 3 深度語意推論 (樹冠次冠層 + 樹幹)...")
    model = build_sam3_image_model(checkpoint_path=checkpoint_path, device=device)
    model = model.to(device).float()
    processor = Sam3Processor(model, device=device, confidence_threshold=0.25)
    state = processor.set_image(raw_img)

    res_canopy = processor.set_text_prompt(prompt="tree subcanopy", state=state)
    raw_canopy_masks = res_canopy.get("masks", None)

    processor.set_confidence_threshold(0.155)
    res_trunk = processor.set_text_prompt(prompt="tree trunk, tree branch", state=state)
    raw_trunk_masks = res_trunk.get("masks", None)

    canopy_mask = np.zeros((h, w), dtype=bool)
    if raw_canopy_masks is not None:
        c_np = raw_canopy_masks.cpu().numpy().squeeze()
        if c_np.ndim == 3:
            for m in c_np:
                canopy_mask |= m.astype(bool)
        elif c_np.ndim == 2:
            canopy_mask = c_np.astype(bool)

    trunk_mask = np.zeros((h, w), dtype=bool)
    if raw_trunk_masks is not None:
        t_np = raw_trunk_masks.cpu().numpy().squeeze()
        if t_np.ndim == 3:
            for m in t_np:
                trunk_mask |= m.astype(bool)
        elif t_np.ndim == 2:
            trunk_mask = t_np.astype(bool)

    print(f"[3/4] 執行拓撲閉環與 CIELAB 葉綠素分析...")
    results = extract_topological_foliage_and_gaps(
        img_bgr=img_bgr,
        canopy_raw_mask=canopy_mask,
        trunk_raw_mask=trunk_mask,
    )

    print(f"--------------------------------------------------")
    print(f"【拓撲閉環分析統計報告】")
    print(f"樹冠總幾何包絡 (Envelope): {results['envelope_pixels']:,} px")
    print(f"純樹葉實體面積 (Living Foliage): {results['foliage_pixels']:,} px")
    print(f"木質樹幹枝幹 (Woody Trunk):     {results['trunk_pixels']:,} px")
    print(f"真實穿透孔隙 (True Gaps):        {results['gap_pixels']:,} px")
    print(f"判定機制: {results['method_desc']}")
    print(f"▶ 實心樹葉密度 (Leaf Density):    {results['leaf_density_pct']:.2f}%")
    print(f"▶ 樹冠孔隙率 (Gap Fraction):      {results['gap_fraction_pct']:.2f}%")
    print(f"▶ 木質比率 (Wood Ratio):          {results['trunk_ratio_pct']:.2f}%")
    print(f"--------------------------------------------------")

    print(f"[4/4] 渲染多視圖診斷成果圖...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), facecolor="#121212")

    # Panel 1: 輸入原圖
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title(f"(A) Input: {Path(image_path).name}", color="white", fontsize=14, fontweight="bold")
    axes[0, 0].axis("off")

    # Panel 2: CIELAB a* 色度熱力圖 (綠色植物 vs 天空/木質)
    im2 = axes[0, 1].imshow(results["a_chan"], cmap="PRGn_r", vmin=95, vmax=140)
    axes[0, 1].set_title(f"(B) CIELAB a* Channel [{results['method_desc']}]", color="#50E3C2", fontsize=14, fontweight="bold")
    axes[0, 1].axis("off")
    cbar = fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cbar.set_label("a* Value (<128: Green Chlorophyll, >=128: Sky/Wood)", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    # Panel 3: 樹冠外包絡幾何基準與樹幹
    env_vis = img_rgb.copy()
    env_overlay = np.zeros_like(img_rgb)
    env_overlay[results["canopy_envelope"]] = [0, 160, 255]
    env_overlay[results["wood_trunk_mask"]] = [255, 140, 0]
    env_vis = cv2.addWeighted(env_vis, 0.65, env_overlay, 0.35, 0)
    cnts, _ = cv2.findContours(results["canopy_envelope"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(env_vis, cnts, -1, (255, 255, 255), 3)
    axes[1, 0].imshow(env_vis)
    axes[1, 0].set_title("(C) Geometric Envelope (White) & Trunk (Orange)", color="white", fontsize=14, fontweight="bold")
    axes[1, 0].axis("off")

    # Panel 4: 最終拓撲閉環三色無損覆蓋圖 (綠葉/天藍孔隙/橘幹)
    final_vis = img_rgb.copy()
    final_overlay = np.zeros_like(img_rgb)
    final_overlay[results["pure_foliage_mask"]] = [30, 240, 70]         # 鮮亮翠綠: 樹葉
    final_overlay[results["true_internal_gaps_mask"]] = [0, 220, 255]    # 電光青藍: 穿透孔洞
    final_overlay[results["wood_trunk_mask"]] = [255, 130, 0]           # 亮橘色: 樹幹
    
    # 疊加渲染
    final_vis = cv2.addWeighted(final_vis, 0.55, final_overlay, 0.45, 0)
    cv2.drawContours(final_vis, cnts, -1, (255, 255, 255), 2)
    axes[1, 1].imshow(final_vis)
    axes[1, 1].set_title(
        f"(D) Topological Output: Foliage {results['leaf_density_pct']:.1f}% (Green), Gaps {results['gap_fraction_pct']:.1f}% (Cyan), Trunk {results['trunk_ratio_pct']:.1f}% (Orange)",
        color="#00FFCC", fontsize=13, fontweight="bold"
    )
    axes[1, 1].axis("off")

    plt.tight_layout()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"成果圖已儲存至: {out_p}")


if __name__ == "__main__":
    test_img = "data/IMG_20260831_144604.jpg"
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
    run_test(image_path=test_img)
