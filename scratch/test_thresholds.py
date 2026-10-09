import cv2
import numpy as np

render_img = cv2.imread("samplingAnalysis/output/IMG_20260831_144604_cielab_otsu_render.png")
h, w = render_img.shape[:2]
p4 = render_img[int(h * 0.5):, int(w * 0.5):]
p4_rgb = cv2.cvtColor(p4, cv2.COLOR_BGR2RGB)

r, g, b = p4_rgb[:, :, 0].astype(float), p4_rgb[:, :, 1].astype(float), p4_rgb[:, :, 2].astype(float)
gap_mask = (b > r + 20) & (b > 120)

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(gap_mask.astype(np.uint8), connectivity=8)
areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
total_gap_px = sum(areas)

# 樹冠外包絡大約是 total_gap_px / 0.2608 (因為報告中孔隙率是 26.08%)
est_envelope_px = total_gap_px / 0.2608

print(f"孔洞總數: {len(areas)}, 估算總包絡像素: {est_envelope_px:.0f}")
print("==================================================")
for ratio in [0.035, 0.02, 0.01, 0.008, 0.006, 0.005, 0.003, 0.002, 0.001]:
    cutoff = est_envelope_px * ratio
    filtered = [a for a in areas if a >= cutoff]
    removed_px = sum(filtered)
    pct_removed = removed_px / total_gap_px * 100
    print(f"ratio={ratio:.3f} ({ratio*100:4.1f}%): 門檻={cutoff:6.0f} px | 命中孔洞數={len(filtered):3d} | 隔離像素={removed_px:6d} ({pct_removed:5.1f}% 的孔隙)")
