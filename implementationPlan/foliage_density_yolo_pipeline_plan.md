# 樹冠次冠層分割與樹葉密度計算實施計畫 (SAM 3 知識遷移至 YOLO + CIELAB 光譜後處理)

> **核心基準腳本**：[segment_focusPrimary_subcanopy_withoutTrunk_leafPixel_nearbyCanopy_yolo_batch_fix.py](file:///Users/rickyho/Documents/github/sam/simpleSam/segment_focusPrimary_subcanopy_withoutTrunk_leafPixel_nearbyCanopy_yolo_batch_fix.py)  
> 本計畫遵循 `@ai-ml` 與 `@ai-engineer` 規範，以 `segment_focusPrimary_subcanopy_withoutTrunk_leafPixel_nearbyCanopy_yolo_batch_fix.py` 為核心資料生成與階層過濾基準，建立從「SAM 3 離線大模型偽標籤生成」到「Ultralytics YOLO-seg 輕量模型訓練」，最後接續「CIELAB $a^*$ 通道與形態學外包絡像素級樹葉密度與內部孔洞（Gaps）計算」之完整端到端架構。

---

## 一、 系統整體架構（System Architecture）

```
[ 原始樹木影像集 (data/) ]
       │
       ▼ 【階段一：離線資料引擎 (Teacher Model)】
[ SAM 3 視覺分割與偽標籤生成流水線 (基準: segment_focusPrimary_..._batch_fix.py) ]
       ├─ 1. 正向提示詞：tree subcanopy (次冠層)
       ├─ 2. 負向提示詞：tree trunk, tree branch (排除樹幹與粗枝)
       ├─ 3. 綠色保護 (ExG & HSV) 與階層超遮罩過濾 (filter_hierarchical_super_masks)
       ├─ 4. 互斥雙類別分離：Class 0 (primary_subcanopy) 與 Class 1 (nearby_subcanopy)
       ├─ 5. 多邊形輪廓座標歸一化 (cv2.approxPolyDP 精簡頂點)
       └─ 6. 產出 YOLO-seg 標準鏡像資料集 (80% Train / 20% Val)
       │
       ▼ 【階段二：輕量模型微調 (Student Model)】
[ Ultralytics YOLO-seg (yolo11n-seg / yolo26s-seg) 訓練 ]
       ├─ 1. 骨幹網路遷移學習：freeze=10 (凍結前 10 層)
       ├─ 2. 防過擬合：close_mosaic=10, patience=30
       ├─ 3. 產出極速權重：best.pt (< 20 MB，推論延遲 < 10 ms)
       │
       ▼ 【階段三：推論後處理與樹葉密度分析 (Inference & Post-Processing)】
[ 樹葉密度（Foliage Density）與內部孔洞（Gaps）分析引擎 ]
       ├─ 1. 空間約束：由 YOLO 快速生成樹冠外包絡 (Canopy Envelope, 分母)
       ├─ 2. 光譜分離：原圖解析度下 CIELAB a* 通道 Otsu 自適應分割 (分子：純葉片)
       ├─ 3. 雜質剔除：自動將棕色樹枝、天空強光、後方建築劃分為「內部孔隙 (Gaps)」
       └─ 4. 數值與圖表輸出：計算樹葉密度 (%)、空隙率 (%) 並生成三色診斷圖
```

---

## 二、 數學定義與演算法邏輯（Mathematical Formulation）

### 1. 樹葉密度與空隙率定義
* **樹冠總包絡面積（Canopy Envelope, $S_{\text{envelope}}$）**：樹冠最外層緊密邊界的封閉像素總數（分母）。
* **純樹葉有效像素（Foliage Pixels, $S_{\text{leaf}}$）**：樹冠內部真正具備綠色植物反射特徵的像素總數（分子）。
* **內部透光孔隙（Internal Gaps, $S_{\text{gap}}$）**：樹冠內部非綠葉（透光天空、棕色枝幹、背景穿透物）的像素總數。

$$\text{Foliage Density} = \frac{S_{\text{leaf}}}{S_{\text{envelope}}}$$

$$\text{Gap Fraction (Porosity)} = \frac{S_{\text{gap}}}{S_{\text{envelope}}} = 1 - \text{Foliage Density}$$

### 2. CIELAB $a^*$ 光譜色度分離原理
在 CIELAB 色彩空間中，亮度（$L^*$）與色度（$a^*, b^*$）完全解耦：
* **$a^*$ 通道物理意義**：負值代表綠色（Green），正值代表洋紅/紅色/棕色（Magenta / Brown / Red）。
* **不論光照強弱（向陽或背光深陰影）**：綠葉在 $a^*$ 通道上皆具備顯著的低值特徵；棕色樹皮、灰白天空、紅磚背景在 $a^*$ 通道上皆為高值。
* **單像素級分離**：在 YOLO 框定的樹冠範圍內，利用 Otsu 自適應演算法計算最佳分割閾值 $T_{a}$，凡 $a^* < T_{a}$ 判定為綠葉，其餘一律劃入孔隙與雜質。

---

## 三、 模組實施計畫（Implementation Modules）

### 模組 1：SAM 3 批次偽標籤生成器（Data Pipeline）
* **實作依據**：直接呼叫並繼承 [segment_focusPrimary_subcanopy_withoutTrunk_leafPixel_nearbyCanopy_yolo_batch_fix.py](file:///Users/rickyho/Documents/github/sam/simpleSam/segment_focusPrimary_subcanopy_withoutTrunk_leafPixel_nearbyCanopy_yolo_batch_fix.py)
* **核心職責**：
  1. 批次處理 `data/` 下所有原始圖片。
  2. 執行超遮罩過濾（`filter_hierarchical_super_masks`）與地基主幹錨點評分（`score_primary_canopy`）。
  3. 產出 `0: primary_subcanopy` 與 `1: nearby_subcanopy` 互斥多邊形標註。
  4. 自動輸出 `dataset/data.yaml` 並切分 Train / Val。

### 模組 2：YOLO-seg 輕量模型微調（Training Pipeline）
* **實作路徑**：`ultralystic/train.py`
* **配置規範**：
  ```python
  from ultralytics import YOLO

  model = YOLO("yolo26s-seg.pt")
  model.train(
      data="dataset/dataset.yaml",
      epochs=100,
      imgsz=640,
      batch=16,
      freeze=10,
      close_mosaic=10,
      device="mps",  # Apple Silicon Mac
      project="runs/segment",
      name="tree_canopy_exp",
  )
  ```

### 模組 3：樹葉密度與內部孔隙後處理引擎（Analysis Engine）
* **實作路徑**：`ultralystic/density_analyzer.py`
* **功能清單**：
  1. 載入訓練好的 `best.pt` 進行即時推論，提取次冠層多邊形遮罩。
  2. 透過形態學閉運算（Closing）生成緊湊樹冠外包絡。
  3. 在原圖解析度下轉換 CIELAB 空間，執行 $a^*$ 通道自適應分割，精確提取非樹葉區域（棕色枝條、天空孔洞）。
  4. 計算密度指標並儲存三色視覺化診斷圖（綠色＝葉片、藍色＝內部空隙、紅色＝外緣輪廓）。

---

## 四、 驗證與品質標準（Verification & Quality Gates）

1. **標註品質校驗**：
   * 資料集多邊形座標嚴格歸一化於 $[0.0, 1.0]$，無超出邊界或 NaN 數值。
2. **訓練指標標準**：
   * 驗證集 **Mask mAP50 $\ge 85\%$**，**Mask mAP50-95 $\ge 60\%$**。
   * 單張推論時間在 Mac MPS 上 $\le 15\text{ ms}$（滿足即時分析需求）。
3. **密度計算誤差驗證**：
   * 抽樣 10 張測試圖，比對 YOLO + CIELAB 後處理之密度數值與 SAM 3 原生分割之基準值，整體相對誤差控制在 $\le \pm 3.0\%$。
