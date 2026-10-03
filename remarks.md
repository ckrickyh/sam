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

原規劃將 SAM 3 的分割能力透過偽標籤（Pseudo-labeling）知識遷移至輕量化 YOLO-seg 模型，經實務驗證後決定終止該路線，並將 `ultralystic/`、`simpleSam/` 等相關模組移入 `.trash/` 封存。核心技術原因如下：

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