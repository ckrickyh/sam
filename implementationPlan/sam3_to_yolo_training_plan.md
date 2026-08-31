# SAM 3 偽標籤至 Ultralytics YOLO-seg 機器學習流水線實施計畫
> 遵循 `@ai-ml` 與 `@ai-engineer` 工業級生產架構規範

本計畫採用 **AI/ML 生產級工程框架**，將 SAM 3 高精度分割能力透過「偽標籤知識遷移（Dataset Distillation）」注入至輕量級 **Ultralytics YOLOv11-seg** 實例分割模型中，建立可復現、具備 MLOps 評估與極致壓縮的邊緣/雲端部署流水線。

---

## 一、 系統架構設計（Architecture & Data Flow）

```
[ 原始相片池 data/raw_images/ ]
       │
       ▼ (Phase 1: Data Engineering & Pseudo-labeling)
[ SAM 3 視覺分割引擎 (segment_canopy_leaves.py) ]
       │  ├─ 1. 中央單木拓撲連通隔離 (主幹錨點過濾兩側鄰居)
       │  ├─ 2. 歸一化超綠指數 (Norm-ExG) + HSV 色相雙通道檢驗
       │  └─ 3. 表面光學優先判定 (樹幹上有葉算葉、無葉算木質部)
       ▼
[ OpenCV 邊界多邊形座標歸一化轉換器 ] ──> 產生 [ YOLO-seg 標準資料集 (80/20 分割) ]
                                              │
       ┌──────────────────────────────────────┘
       ▼ (Phase 2: Model Training & Distillation)
[ Ultralytics YOLO11-seg 輕量模型訓練流水線 ]
       │  ├─ Backbone: C3k2 / SPPF 樹木紋理特徵提取
       │  ├─ Loss: Box Loss + Mask BCE Loss + DFL
       │  └─ Augmentation: Mosaic, Scale Jitter, HSV-H/S/V Perturbation
       ▼
[ 最佳權重 best.pt (約 6 ~ 15 MB) ]
       │
       ▼ (Phase 3: Model Export & Edge Deployment)
[ 跨平台邊緣模型 ] ──┬─> [ FP16 ONNX ]  ──> 雲端低成本 CPU / GPU API
                     └─> [ CoreML ]     ──> iOS / iPad 100% 離線巡檢 App
```

---

## 二、 開發與模組實施計畫（Proposed Changes）

### 模組 1：依賴環境與專案結構（Dependencies）
* **檔案**：`pyproject.toml`
* **規範**：強制使用 Python 3.12 與 `uv` 套件管理。
* **變更**：
  ```bash
  uv add ultralytics
  ```

### 模組 2：自動化資料工程與多邊形轉換器（Data Pipeline）
* **檔案**：`scripts/batch_sam3_dataset_generator.py`
* **設計職責**：
  1. 遍歷 `data/raw_images/` 所有影像。
  2. 調用 `segment_canopy_leaves.py` 產出高純度二值遮罩。
  3. 使用 `cv2.findContours(..., cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)` 提取平滑頂點並做解析度歸一化（$0.0 \sim 1.0$）。
  4. 類別對應：
     * `0: crown`（樹冠分母）
     * `1: leaf`（純綠葉分子）
     * `2: trunk`（木質樹幹）
  5. 依照 80% Train / 20% Val 自動切分輸出至 `data/dataset_yolo/`。

### 模組 3：資料集與訓練超參數配置（Training Configuration）
* **檔案**：`dataset/trees_seg.yaml` 與 `scripts/train_yolo_seg.py`
* **關鍵超參數**：
  * `model`: `yolo11n-seg.pt`（Nano，約 6 MB）或 `yolo11s-seg.pt`（Small，約 18 MB）
  * `imgsz`: 640（或 1024 高解析）
  * `epochs`: 100
  * `batch`: 16（依 Mac MPS / GPU 自動調整）
  * `device`: 優先 `mps`（Mac M4 專屬硬體加速）或 `cuda`
  * `augment`: 啟用樹葉光影適應性增強（`hsv_h=0.015`, `hsv_s=0.7`, `degrees=15.0`, `mosaic=1.0`）

### 模組 4：評估矩陣與模型導出（Evaluation & MLOps）
* **檔案**：`scripts/export_and_benchmark.py`
* **功能**：
  1. 計算測試集上的 **Mask mAP50** 與 **Mask mAP50-95**。
  2. 匯出為 **FP16 ONNX**（體積 < 10 MB）與 **CoreML**。
  3. 驗證 ONNX 推論延遲（目標：Mac M4 上單張推論 $< 15\text{ ms}$）。
  4. 驗證密度一致性：比較 YOLO-seg 與 SAM 3 計算出的 `Crown Porosity` 誤差在 $\pm 2.5\%$ 以內。

---

## 三、 驗證與品質卡點（Quality Gates）

1. **資料集多邊形有效性**：
   * 檢查生成的 `.txt` 檔案無空標註、座標皆介於 $[0.0, 1.0]$。
   * 隨機抽取 5 張訓練樣本進行座標反向繪圖驗證。
2. **訓練收斂指標**：
   * `val/mask_loss` 與 `val/box_loss` 於 50 Epochs 內顯著收斂。
   * 樹冠與葉片的 Mask mAP50 達到 $85\%$ 以上。
3. **推論效能驗證**：
   * ONNX 模型體積 $< 15\text{ MB}$。
   * 單張推論耗時較 SAM 3（約 1500 ms）提升超過 100 倍。
