# 專案執行備忘與架構決策紀錄 (Remarks & Architectural Decisions)

## 一、 常用執行指令

```bash
# 啟動自適應 CIELAB a* 樹葉密度與孔隙分析引擎 (Box Prompt 版)
uv run python samplingAnalysis/adaptive_cielab_otsu_foliage_extractor_fixBranchlet_order_box.py \
  --images data/IMG_20260901_141707.jpg \
  --output-dir samplingAnalysis/output

# 啟動 Web 視覺化互動介面
uv run python samplingAnalysis/app.py
```

---

## 二、 放棄 SAM 3 蒸餾至 YOLO 分割路線之原因

原規劃將 SAM 3 的分割c能力透過偽標籤（Pseudo-labeling）知識遷移至輕量化 YOLO-seg 模型，經實務驗證後決定終止該路線，並將 `ultralystic/`、`simpleSam/` 等相關模組移入 `.trash/` 封存。核心技術原因如下：

1. **神經網路特徵降採樣損失（Resolution Loss）**：
   * YOLO-seg 依賴原型遮罩（Prototype Masks）機制，在較低解析度特徵圖上組合預測。
   * 樹枝間的透光孔隙（Canopy Gaps）與葉片邊緣通常僅佔數個像素，經過卷積多次下採樣後，微觀高頻空間細節被平均模糊化，無法精確分割孔洞。
2. **多邊形近似（Polygon Approximation）的幾何限制**：
   * YOLO 資料集格式須將遮罩轉換為頂點座標（TXT Polygon）。樹冠內部具備大量「甜甜圈狀」複雜穿透孔隙，轉為多邊形時頂點數量膨脹且造成嚴重拓撲失真。
3. **標註雜訊與 Loss 震盪**：
   * 偽標籤邊緣像素的抖動在模型訓練時產生高度標註雜訊（Label Noise），導致邊界鋸齒化與實心密度評估誤差。

---

## 三、 現行架構替代方案：兩階段混合管線（Two-Stage Hybrid Pipeline）

轉為採用「宏觀幾何約束 + 微觀物理光學」分工：
* **宏觀空間約束**：由 SAM 3（Box Prompt）負責鎖定主角樹外包絡（Canopy Envelope），無需管孔隙細節。
* **微觀無損分析**：在原生解析度下，交由 `CIELAB a* + Otsu 自適應門檻 + OpenCV 拓撲補洞` 演算法分離葉片、木質結構與透光孔隙，達成 100% 物理守恆與無損量化。



---

## 四、 Hugging Face Space 部署與 BFloat16 型態陷阱排除 (Monkey-patch 策略)

在將模型部署至 Hugging Face Space (ZeroGPU 基礎設施) 時，為節省顯示卡記憶體 (VRAM) 避免 OOM (Out Of Memory)，我們將主模型透過 `.to(torch.bfloat16)` 轉型。此操作在轉換過程中引發了深度學習框架底層的型態衝突，紀錄與解決方案如下：

1. **核心報錯 (`mat1 and mat2 must have the same dtype, but got Float and BFloat16`)**：
   * **問題原因**：SAM 3 內部包含了諸多自訂組件（如 Text Prompt Grounding 生成的張量、動態宣告的 `dot_prod_scoring_head` 等），這些物件未被註冊為正規的 `nn.Module`，導致 `to(bfloat16)` 無法完整遞迴轉換。當原生 Float32 的特徵進入 BFloat16 的權重層時，PyTorch 即會報錯。在本地端 (CPU/MPS) 因為未觸發 BFloat16 轉換，故無此問題。
   * **全局防禦解法 (核彈級 Monkey-patch)**：單純攔截模組層級的 `nn.Linear` 仍無法阻擋底層 MultiheadAttention 直接呼叫 C++ 的 `F.linear`。最終解法為直接在 `app.py` 中全域攔截 `torch.nn.functional` 底層核心（`linear`, `conv2d`, `layer_norm`），只要運算前發現矩陣型態不一致，就強制將 `input` 對齊 `weight` 的型態。

2. **引數深拷貝遺失修改問題 (`KeyError: 'pred_boxes'`)**：
   * **問題原因**：實作參數轉型時，一開始使用了建立新字典的回傳方式（Deep Copy），導致 SAM 3 原始程式碼的 `_update_scores_and_boxes` 原地寫入 (In-place update) 結果（如 `"pred_boxes"`）被寫在分身字典上而遺失。
   * **解法**：修改遞迴轉型函式 `deep_cast_bf16`，針對 `dict` 與 `list` 強制採用**原地修改 (In-place mutation)**，以保證物件的記憶體參考位址不變。

3. **Numpy 型態不支援 (`TypeError: Got unsupported ScalarType BFloat16`)**：
   * **問題原因**：BFloat16 是針對神經網路加速設計的特殊格式，Numpy 函式庫本身並不支援。
   * **解法**：在影像後處理邏輯提取張量（如 bounding boxes, masks）時，於呼叫 `.numpy()` 前插入 `.float()` 進行轉型過濾。

---

## Testing
uv run samplingAnalysis/adaptive_cielab_otsu_foliage_extractor_fixBranchlet_order_box.py --images data/IMG_20260901_141418.jpg --output-dir samplingAnalysis/output

---
adaptive_cielab_otsu_foliage_extractor_fixBranchlet_order.py 為原型

adaptive_cielab_otsu_foliage_extractor_fixBranchlet_order_box.py 為連接 gradio (基於adaptive_cielab_otsu_foliage_extractor_fixBranchlet_order.py開發)

---
Scratch folder for study only

---
gpu ram 125gb
qwen 3.6 125b para 128gb
deepseek 70b para 128gb

50萬 （10萬裝機）

---
vendor 
香港 深圳

---
# 步驟一：專心做特徵萃取與產生標註（直接使用原腳本，不改寫）
uv run scripts/adaptive_cielab_otsu_foliage_extractor_fixBranchlet_order_maxGapRatio_ExG.py --data-dir data/
# 步驟二：專心做資料集切分與建立 data.yaml（使用上一篇提供的腳本）
uv run scripts/split_yolo_dataset.py