---
title: Crown Porosity
emoji: 🌳
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
short_description: SAM 3 Box Prompt & CIELAB a* Foliage & Gap Analyzer
---

# 樹冠葉片密度與孔隙分析引擎 (Tree Canopy Foliage & Gap Analysis Engine)

本專案採用 **SAM 3 (Segment Anything Model 3)** 幾何約束與 **CIELAB a* 自適應色度演算法**，專為戶外林木樹冠提供高精度的葉片覆蓋率、內部透光孔隙率與木質結構無損量化分析。

---

## 系統核心特色

1. **空間幾何約束 (Box Prompt Driven Architecture)**：
   * 採用空間方框提示 (Box Prompt) 鎖定前景樹冠，強加空間邊界約束，徹底消除背景雜林篡位與語意擴散問題。
2. **前景主角樹優先評分排序 (Focus-Primary Priority Scoring)**：
   * 整合「高斯居中度 (45%)」、「外包絡規模 (25%)」、「地基主幹錨點 (20%)」與「模型信心度 (10%)」四大權重，自動精確鎖定畫面唯一「主角樹」，其餘自動降級為周邊樹。
3. **無損 CIELAB a* 物理雙峰色度演算法**：
   * 以國際 CIE 色度標準 ($a^* < 128$) 為綠色基準，結合 Otsu 雙峰自適應門檻與超綠指數保護盾 (Norm-ExG + HSV)，完美保留高光反光葉與陰影綠葉。
4. **木質結構修復與微觀逆光細枝條提取**：
   * 結合 OpenCV 外部輪廓填洞消除採樣黑斑，並在色度空間 ($L^* < 95, a^* \ge 126$) 自動分離逆光細枝條，確保拓撲閉環：
     $$\text{樹冠空間 (Canopy Envelope)} = \text{純綠葉} + \text{木質枝幹} + \text{透光孔隙}$$
5. **現代化 Web 視覺化互動介面**：
   * 內建 FastAPI + 高質感互動前端介面，支援單圖即時上傳診斷、動態繪製 Box Prompt 約束框與四面板診斷圖即時預覽。

---

## 系統環境與安裝

本專案使用 `uv` 作為唯一 Python 專案與依賴管理工具（基準版本 Python 3.12）。

### 1. 安裝環境與依賴
```bash
# 同步安裝專案虛擬環境與套件
uv sync
