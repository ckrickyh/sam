# SAM 3 偽標籤至 Ultralytics YOLO26-seg 機器學習流水線實施計畫
> 遵循 `@ai-ml`、`@ai-engineer` 工業級規範與 Ultralytics 官方 Agent Skills 最佳實踐

本計畫採用 **AI/ML 生產級工程框架**，將 SAM 3（3.4 GB）高精度次冠層分割能力透過「偽標籤知識遷移（Dataset Distillation）」注入至最新一代輕量級 **Ultralytics YOLO26-seg（NMS-Free 端到端分割）** 模型中，建立可復現、高吞吐、具備完整 MLOps 效能圖表分析與極致壓縮的邊緣/雲端部署流水線。

---

## 一、 系統架構設計（Architecture & Data Flow）

```
[ 原始相片集 data/ ] (支援 .jpg, .jpeg, .png, .webp)
       │
       ▼ (Phase 1: Data Engineering & Pseudo-labeling)
[ SAM 3 視覺分割批次管線 (simpleSam/segment_focusPrimary_..._batch.py) ]
       │  ├─ 1. 雙向提示詞 (Positive: 'tree subcanopy' & Negative: 'tree trunk')
       │  ├─ 2. 歸一化超綠指數 (Norm-ExG) + HSV 綠色植被保護 (防止稀疏葉被挖空)
       │  ├─ 3. 階層超遮罩過濾 (filter_hierarchical_super_masks, 剃除多木聚合巨型遮罩)
       │  ├─ 4. 形態學外包絡空間 (Envelope Area) + 高斯空間中心衰減 + 地基主幹錨點
       │  └─ 5. 互斥雙類別分離：Class 0 (主體樹) 與 Class 1 (周邊樹)
       ▼
[ OpenCV 多邊形輪廓座標歸一化轉換器 (approxPolyDP 精簡頂點) ]
       │
       ▼ 產生 [ YOLO-seg 標準鏡像資料集 (80% Train / 20% Val) ]
[ dataset/ ]
├── dataset.yaml                         (絕對路徑與 0-based 類別定義)
├── images/train/, images/val/           (自動歸檔之影像集)
└── labels/train/, labels/val/           (1 圖 1 TXT 之多邊形座標標註)
       │
       ▼ (Phase 2: Model Training & Distillation)
[ Ultralytics YOLO26-seg 輕量模型微調流水線 ]
       │  ├─ 預訓練權重：yolo26n-seg.pt (約 6 MB) / yolo26s-seg.pt (約 18 MB)
       │  ├─ 端到端特性：原生 NMS-Free (無須後處理，速度極致)
       │  ├─ 損失函數：Box Loss + Mask BCE Loss + DFL
       │  ├─ 小樣本防過擬合：freeze=10 (凍結骨幹), close_mosaic=10 (保護中央幾何)
       │  └─ 光影增強：hsv_h=0.015, hsv_s=0.7, degrees=5.0
       ▼
[ 產出 1: 最佳蒸餾權重 runs/segment/train/weights/best.pt ]
[ 產出 2: MLOps 全套效能分析圖表 runs/segment/train/*.png (Loss, F1, PR, mAP) ]
       │
       ▼ (Phase 3: Model Export & Edge Deployment)
[ 跨平台邊緣模型 ] ──┬─> [ FP16 ONNX (opset=17) ] ──> 雲端低成本 CPU / GPU API (延遲 < 15 ms)
                     └─> [ CoreML (half=True) ]   ──> iOS / iPad 100% 離線巡檢 App
```

---

## 二、 開發與模組實施計畫（Proposed Modules）

### 模組 1：依賴環境與專案規範（Dependencies）
* **管理工具**：強制使用 Python 3.12 與 `uv`。
* **主要套件**：
  ```bash
  uv add ultralytics tensorboard
  ```

### 模組 2：自動化資料工程與批次標註器（Data Pipeline）
* **檔案**：`simpleSam/segment_focusPrimary_subcanopy_withoutTrunk_leafPixel_nearbyCanopy_yolo_batch.py`
* **設計職責**：
  1. 遍歷 `data/` 所有影像，單次載入 SAM 3 模型實例（避免反覆載入 3.4 GB 權重開銷）。
  2. 自動執行雙向提示、綠色保護、階層超遮罩過濾與強健主體評分。
  3. 產出互斥多邊形標註（避免重疊導致 NMS 遮罩互殺）：
     * `0: primary_subcanopy`（中央主體樹純葉）
     * `1: nearby_subcanopy`（周邊鄰居樹純葉）
  4. 使用 `cv2.approxPolyDP(epsilon_ratio=0.002)` 進行頂點精簡，避免座標檔案肥大。
  5. 依照 80% Train / 20% Val 自動切分輸出至 `dataset/`，並產出/更新 `dataset.yaml`。

### 模組 3：YOLO26-seg 微調訓練配置（Training Pipeline）
* **腳本**：`scripts/train_yolo26_seg.py` 或直接調用 CLI
* **關鍵訓練超參數（依據 YOLO Skills 調優）**：
  ```python
  from ultralytics import YOLO

  model = YOLO("yolo26n-seg.pt")  # 或 yolo26s-seg.pt
  results = model.train(
      data="dataset/dataset.yaml",
      epochs=100,
      imgsz=640,
      batch=16,
      freeze=10,          # 凍結前 10 層骨幹，小樣本遷移學習最佳實踐
      close_mosaic=10,    # 最後 10 輪關閉 Mosaic，保護主木居中空間特徵
      degrees=5.0,        # 適度小角度旋轉增強
      hsv_h=0.015,
      hsv_s=0.7,
      patience=30,        # 早停機制 (30 Epochs 無提升即提前停止)
      device="mps",       # Mac 專屬 Apple Silicon 加速 (若 Linux/CUDA 則設 "0")
      project="runs/segment",
      name="tree_distill_exp",
  )
  ```

### 模組 4：MLOps 訓練效能分析圖表與指標評估（Performance Analytics & Graphs）
訓練過程中與完成後，系統會於 `runs/segment/tree_distill_exp/` 自動產出完整視覺化分析圖表：

| 分析圖表檔案 | 涵蓋評估指標 | 分析用途與決策依據 |
| :--- | :--- | :--- |
| **`results.png`**<br>*(★ 總體指標圖)* | **Train/Val Loss, Precision, Recall, mAP50, mAP50-95** | 10 合 1 綜合折線圖，直觀檢視分割損失（Mask BCE/Seg Loss）是否收斂，有無過擬合。 |
| **`MaskF1_curve.png`** | **F1-Score 隨置信度曲線 (F1 vs. Conf)** | 綜合平衡查準率與查全率，直接標示**最佳推論置信度門檻值（Optimal Threshold）**。 |
| **`MaskPR_curve.png`** | **Precision-Recall (PR) 曲線** | 評估主體樹（Class 0）與周邊樹（Class 1）在不同召回率下的精確率（mAP 積分面積）。 |
| **`MaskP_curve.png`** | **Precision 曲線 (查準率 / 誤報分析)** | 分析高信心度下的樹葉預測純度，確保不會將背景誤判為樹葉。 |
| **`MaskR_curve.png`** | **Recall 曲線 (查全率 / 漏報分析)** | 分析模型能否完整抓出稀疏分枝的所有葉片。 |
| **`confusion_matrix.png`** | **混淆矩陣 (Confusion Matrix / Accuracy)** | 驗證 Class 0（主樹）與 Class 1（周邊樹）之間的分類準確度與邊界混淆狀況。 |
| **`val_batch0_pred.jpg`** | **驗證集真實預測視覺化圖** | 肉眼抽檢 AI 在未知驗證照片上的多邊形分割效果與 SAM 3 真值（GT）之對比。 |
| **`results.csv`** | **逐輪訓練數值數據檔** | 包含 100 輪所有損失與指標的精確浮點數，支援自訂報表與論文製圖。 |

* **即時監控面板（Live TensorBoard Dashboard）**：
  ```bash
  uv run tensorboard --logdir runs/segment/
  # 瀏覽器開啟 http://localhost:6006 觀看即時動態折線圖
  ```

### 模組 5：模型導出與邊緣部署驗證（Export & Benchmark）
* **腳本**：`scripts/export_and_benchmark.py`
* **功能**：
  1. **模型導出**：
     ```bash
     # 導出為極速 FP16 ONNX
     uv run yolo export model=runs/segment/tree_distill_exp/weights/best.pt format=onnx half=True opset=17
     # 導出為 Apple 原生 CoreML (iOS / iPad 離線巡檢)
     uv run yolo export model=runs/segment/tree_distill_exp/weights/best.pt format=coreml half=True
     ```
  2. **密度一致性檢驗**：比較 YOLO26-seg 實時推論計算之樹冠密度（Foliage Density）與 SAM 3 基準值誤差在 $\pm 3.0\%$ 以內。

---

## 三、 驗證與品質卡點（Quality Gates）

1. **資料集標註有效性**：
   * 符合「1 圖 1 TXT」鏡像規範，所有多邊形座標正規化於 $[0.0, 1.0]$。
   * 隨機抽取驗證集樣本進行邊界框與多邊形反向繪圖檢驗。
2. **訓練收斂與效能指標（依據圖表驗收）**：
   * `val/seg_loss` 於 50 Epochs 內平穩下降且無明顯回彈（無過擬合）。
   * 主體樹與周邊樹的 **Mask mAP50 達到 $85\%$ 以上**，最佳 **F1-Score 達到 $0.80$ 以上**。
3. **邊緣推論效能驗證**：
   * 導出之 FP16 ONNX / CoreML 模型體積 $< 15\text{ MB}$。
   * 單張推論耗時較 SAM 3（約 1500 ms）提升超過 **100 倍**（Mac M 系列晶片上 $< 15\text{ ms}$）。
