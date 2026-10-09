import cv2
import numpy as np

# 讀取渲染圖 (16x12 inches, dpi=150)
render_img = cv2.imread("samplingAnalysis/output/IMG_20260831_144604_cielab_otsu_render.png")
h, w = render_img.shape[:2]

# 取 Panel 4: 右下角區域 (y: h//2 ~ h, x: w//2 ~ w)
p4 = render_img[int(h * 0.5):, int(w * 0.5):]
p4_rgb = cv2.cvtColor(p4, cv2.COLOR_BGR2RGB)

# 在 Panel 4 中，天藍色孔隙有明顯的藍色高值，綠色葉片有高 G 值，橘色木質有高 R 值
# 藉由色彩特徵提取天藍色覆蓋區
# 藍色 overlay: [0, 140, 255] RGB，與原圖以 0.42 : 0.58 混合
# 差異最大的特徵是 B 通道顯著高於 R 通道 (在天光下更明顯)
r = p4_rgb[:, :, 0].astype(float)
g = p4_rgb[:, :, 1].astype(float)
b = p4_rgb[:, :, 2].astype(float)

# 找出天空/孔隙區域 (天藍色標註區: B > R + 25 且 G > 100)
gap_mask = (b > r + 20) & (b > 120)

# 連通元件分析
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(gap_mask.astype(np.uint8), connectivity=8)

areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
total_pixels = sum(areas)

print(f"Panel 4 縮圖中偵測到的孔洞個數: {len(areas)}")
print(f"Panel 4 孔洞總像素: {total_pixels}")

if areas:
    sorted_areas = sorted(areas, reverse=True)
    print("前 10 大孔洞面積 (縮圖 px):", sorted_areas[:10])
    print("前 10 大孔洞佔總孔隙面積比例 (%):", [round(a / total_pixels * 100, 2) for a in sorted_areas[:10]])
    print("前 10 大孔洞佔整個 Panel 4 比例 (%):", [round(a / (p4.shape[0]*p4.shape[1]) * 100, 3) for a in sorted_areas[:10]])
