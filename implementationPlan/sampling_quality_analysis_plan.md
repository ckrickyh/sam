# 資料集採樣品質評估與統計視覺化實施計畫 (Sampling Quality & EDA Pipeline)

> 參考基準腳本：[segment_focusPrimary_subcanopy_withoutTrunk_leafPixel_nearbyCanopy_yolo_batch_fix.py](file:///Users/rickyho/Documents/github/sam/simpleSam/segment_focusPrimary_subcanopy_withoutTrunk_leafPixel_nearbyCanopy_yolo_batch_fix.py)

---

## 一、 核心目標與架構（Objective & Architecture）

本計畫旨在針對 SAM 3 批次轉換產出的 YOLO 資料集（`dataset/`），建立自動化的「採樣品質審查、分佈偏移檢定與視覺化圖表匯出系統」。
支援 **Jupyter Notebook 互動式探索** 與 **CLI 自動化批次分析** 雙模式，自動產出包括多樣性熱力圖、尺度分佈直方圖、置信度難易度箱形圖等專業 MLOps 資料品質診斷報告。

```
[ dataset/ (images/ & labels/) ] + [ SAM 3 (sam3.pt) ]
       │
       ▼ 【採樣品質審查引擎】
┌─────────────────────────────────────────────────────────────┐
│ 1. 幾何統計分析 (Geometric & Spatial Analytics)              │
│    - 遮罩面積分佈標準差 (σ_area) 與長寬比                     │
│    - 主體樹 (Class 0) vs 周邊樹 (Class 1) 空間中心點分佈       │
│ 2. 光譜與亮度分佈 (Spectral & Luminance Diversity)          │
│    - 樹冠平均亮度 (YUV/Luminance) 與超綠指數 (ExG) 分佈       │
│ 3. SAM 3 語意特徵餘弦相似度 (Feature Cosine Similarity)     │
│    - ViT Embedding 矩陣分析，抓出同樹連拍與高重複樣本         │
│ 4. Train vs. Val 分佈偏移檢定 (Covariate Shift / KS-Test)    │
└─────────────────────────────────────────────────────────────┘
       │
       ├─► 輸出互動分析筆記本：notebooks/dataset_sampling_eda.ipynb
       ├─► 輸出自動化分析腳本：scripts/export_sampling_analytics.py
       └─► 匯出高解析度圖表集：runs/dataset_analysis/*.png & *.json
```

---

## 二、 匯出統計圖表清單（Distribution Graphs & Artifacts）

系統將自動運算並匯出以下 5 大核心分佈圖表至 `runs/dataset_analysis/` 目錄：

| 圖表檔案名稱 | 視覺化形式 | 評估用途與決策依據 |
| :--- | :--- | :--- |
| **`1_area_distribution_kde.png`** | **核密度估計圖（KDE / Histogram）** | 對比 Train 與 Val 在「遮罩面積佔比」上的分佈，檢驗兩者標準差（$\sigma$）與均值（$\mu$）是否吻合。 |
| **`2_spatial_center_scatter.png`** | **2D 空間中心散佈圖（Spatial Scatter）** | 分析主體樹與周邊樹的質心座標分佈，診斷是否存在「中心偏置（Center Bias）」過重問題。 |
| **`3_feature_similarity_heatmap.png`** | **特徵餘弦相似度熱力圖（Heatmap）** | 利用 SAM 3 影像特徵矩陣，標示相似度 $> 92\%$ 的同樹連拍樣本組，提供隔離建議。 |
| **`4_confidence_difficulty_box.png`** | **SAM 3 置信度難易度箱形圖（Box Plot）** | 統計所有樣本的預測難度分佈，確保簡單樣本（70%）與硬樣本（30%）比例平衡。 |
| **`5_sampling_summary_report.json`** | **結構化品質數值報告檔** | 記錄樣本數、各維度標準差、KS 檢定 $p$-value、重複樣本清單。 |

---

## 三、 模組實施步驟（Implementation Steps）

### 模組 1：資料集幾何與光譜統計提取器
* **檔案**：`scripts/export_sampling_analytics.py`
* **功能**：
  1. 解析 `dataset/labels/train` 與 `dataset/labels/val` 所有 `.txt` 多邊形。
  2. 計算每個實例的多邊形頂點數、外接矩形、多邊形面積、外包絡面積。
  3. 讀取對應影像，計算樹冠區域的平均亮度與 ExG 指數。

### 模組 2：SAM 3 特徵相似度與重複審查
* **功能**：
  1. 單次載入 `sam3.pt` 與 `Sam3Processor`。
  2. 提取每張圖片的影像特徵向量（Embedding Vector）。
  3. 建立 $N \times N$ 相似度矩陣，自動標註跨 Train/Val 的資料洩漏組。

### 模組 3：Jupyter Notebook 互動式探索環境
* **檔案**：`notebooks/dataset_sampling_eda.ipynb`
* **內容架構**：
  * **Cell 1**：環境載入與 `dataset.yaml` 解析。
  * **Cell 2**：執行統計指標計算與 Kolmogorov-Smirnov (KS) 分佈偏移檢定。
  * **Cell 3**：繪製 Matplotlib / Seaborn 互動式圖表。
  * **Cell 4**：抽檢異常樣本（重複連拍、極小面積遮罩、低置信度難題）。

---

## 四、 驗證與驗收標準（Quality Gates）

1. **分佈一致性標準（No Covariate Shift）**：
   * Train 與 Val 的面積均值差距 $|\mu_{\text{train}} - \mu_{\text{val}}| \le 0.10$。
   * KS 檢定 $p\text{-value} > 0.05$（接受 Train 與 Val 來自同一分佈之假設）。
2. **無跨集合資料洩漏（No Data Leakage）**：
   * 餘弦相似度 $> 0.95$ 的連拍相片組，跨越 Train 與 Val 的數量為 $0$。
3. **可重現性驗證**：
   * 執行 `uv run python scripts/export_sampling_analytics.py` 能於 60 秒內無報錯產出全套 PNG 圖表與 JSON 報告。
