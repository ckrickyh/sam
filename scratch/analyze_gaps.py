import cv2
import numpy as np

# 讀取渲染圖
render_img = cv2.imread("samplingAnalysis/output/IMG_20260831_144604_cielab_otsu_render.png")
h, w = render_img.shape[:2]

# 取 Panel 4 (右下角四分之一)
panel4 = render_img[h//2:, w//2:]

# 找出天藍色孔隙 (在 BGR 下為 [255, 140, 0] 附近)
# RGB: [0, 140, 255] -> BGR: [255, 140, 0]
b, g, r = panel4[:, :, 0], panel4[:, :, 1], panel4[:, :, 2]
blue_gaps = (b > 200) & (g > 100) & (g < 180) & (r < 50)

# 連通元件分析
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(blue_gaps.astype(np.uint8), connectivity=8)

areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
total_gap_area = sum(areas)

print(f"孔洞總數量: {len(areas)}")
print(f"孔隙總像素: {total_gap_area}")
if areas:
    sorted_areas = sorted(areas, reverse=True)
    print("前 10 大孔洞面積 (px):", sorted_areas[:10])
    print("前 10 大孔洞佔總孔隙面積比例 (%):", [round(a / total_gap_area * 100, 2) for a in sorted_areas[:10]])
