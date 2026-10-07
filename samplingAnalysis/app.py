import json
from pathlib import Path
import shutil
import sys
import tempfile
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import uvicorn

# 確保專案根目錄在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from samplingAnalysis.adaptive_cielab_otsu_foliage_extractor_fixBranchlet_order_box2 import (
    build_sam3_image_model,
    get_optimal_device,
    process_image_cielab_adaptive,
    Sam3Processor,
)

import os
from huggingface_hub import hf_hub_download

DEFAULT_CHECKPOINT = "sam3.pt"

# 全域單例快取 SAM 3 模型與處理器
GLOBAL_PROCESSOR = None


def ensure_model_checkpoint(checkpoint_path: str = DEFAULT_CHECKPOINT) -> str:
    """確認本機是否有模型權重，若無則從 Hugging Face Hub 自動下載"""
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    repo_id = os.environ.get("HF_MODEL_REPO")
    if not repo_id:
        raise ValueError(
            f"本地未找到 {checkpoint_path}，且未設定環境變數 HF_MODEL_REPO，無法自動下載權重。"
        )

    print(f"[*] 本地未偵測到權重 {checkpoint_path}，正在從 Hugging Face ({repo_id}) 下載...")
    token = os.environ.get("HF_TOKEN_READ") or os.environ.get("HF_TOKEN")
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=checkpoint_path,
        local_dir=".",
        token=token,
    )
    print(f"[*] 權重下載完成：{downloaded_path}")
    return downloaded_path


def get_processor(checkpoint_path: str = DEFAULT_CHECKPOINT) -> Sam3Processor:
    global GLOBAL_PROCESSOR
    if GLOBAL_PROCESSOR is None:
        valid_checkpoint_path = ensure_model_checkpoint(checkpoint_path)
        device = get_optimal_device()
        print(f"[*] 正在初始化 SAM 3 視覺處理器 ({valid_checkpoint_path}) 於 {device}...")
        model = build_sam3_image_model(checkpoint_path=valid_checkpoint_path, device=device)
        model = model.to(device).float()
        GLOBAL_PROCESSOR = Sam3Processor(model, device=device)
        print("[*] SAM 3 視覺處理器載入完成。")
    return GLOBAL_PROCESSOR


def parse_box_str(box_text: str | None) -> list[float] | None:
    """將字串解析為 4 個數值的浮點數列表"""
    if not box_text or not box_text.strip():
        return None
    cleaned = box_text.replace(",", " ").strip().split()
    if len(cleaned) != 4:
        raise ValueError("方框座標必須包含 4 個數值：xmin ymin xmax ymax")
    return [float(val) for val in cleaned]


app = FastAPI(title="樹冠密度與孔隙分析引擎")

INDEX_HTML = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>樹冠密度與孔隙分析引擎 (幾何約束 Box Prompt 版)</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-primary: #0f172a;
      --bg-secondary: #1e293b;
      --bg-tertiary: #334155;
      --border-color: #475569;
      --text-primary: #f8fafc;
      --text-secondary: #94a3b8;
      --color-canopy: #eab308;
      --color-trunk: #f97316;
      --color-primary: #2563eb;
      --color-primary-hover: #1d4ed8;
      --color-success: #10b981;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    body {
      background-color: var(--bg-primary);
      color: var(--text-primary);
      padding: 24px;
      line-height: 1.5;
    }

    .header {
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border-color);
    }

    .header h1 {
      font-size: 1.75rem;
      font-weight: 700;
      color: #38bdf8;
      margin-bottom: 8px;
    }

    .header p {
      color: var(--text-secondary);
      font-size: 0.95rem;
    }

    .main-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
    }

    @media (max-width: 1024px) {
      .main-grid {
        grid-template-columns: 1fr;
      }
    }

    .card {
      background-color: var(--bg-secondary);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .card-title {
      font-size: 1.15rem;
      font-weight: 600;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    /* 樣本快速選擇按鈕 */
    .sample-bar {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }

    .sample-btn {
      background-color: var(--bg-tertiary);
      color: var(--text-primary);
      border: 1px solid var(--border-color);
      padding: 6px 12px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.85rem;
      transition: all 0.15s ease;
    }

    .sample-btn:hover {
      background-color: #475569;
    }

    /* 模式切換按鈕組 */
    .mode-switch-group {
      display: flex;
      gap: 10px;
    }

    .mode-btn {
      flex: 1;
      padding: 10px 14px;
      border-radius: 8px;
      font-weight: 600;
      font-size: 0.9rem;
      cursor: pointer;
      border: 2px solid transparent;
      transition: all 0.15s ease;
      text-align: center;
    }

    .mode-btn-canopy {
      background-color: rgba(234, 179, 8, 0.15);
      color: #fde047;
      border-color: rgba(234, 179, 8, 0.4);
    }

    .mode-btn-canopy.active {
      background-color: #eab308;
      color: #000;
      border-color: #ca8a04;
      box-shadow: 0 0 12px rgba(234, 179, 8, 0.4);
    }

    .mode-btn-trunk {
      background-color: rgba(249, 115, 22, 0.15);
      color: #fdba74;
      border-color: rgba(249, 115, 22, 0.4);
    }

    .mode-btn-trunk.active {
      background-color: #f97316;
      color: #fff;
      border-color: #ea580c;
      box-shadow: 0 0 12px rgba(249, 115, 22, 0.4);
    }

    /* 畫布容器 */
    .canvas-wrapper {
      position: relative;
      width: 100%;
      min-height: 380px;
      background-color: #0b0f19;
      border: 2px dashed var(--border-color);
      border-radius: 8px;
      display: flex;
      justify-content: center;
      align-items: center;
      overflow: hidden;
      user-select: none;
    }

    #annotation_canvas {
      display: block;
      max-width: 100%;
      height: auto;
      cursor: crosshair;
    }

    .canvas-placeholder {
      color: var(--text-secondary);
      text-align: center;
      pointer-events: none;
      padding: 20px;
    }

    /* 狀態提示列 */
    .status-alert {
      padding: 10px 14px;
      border-radius: 6px;
      font-size: 0.85rem;
      background-color: rgba(56, 189, 248, 0.1);
      color: #38bdf8;
      border: 1px solid rgba(56, 189, 248, 0.2);
    }

    /* 輸入表單與滑桿 */
    .form-group {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .form-label {
      font-size: 0.85rem;
      font-weight: 500;
      color: var(--text-secondary);
    }

    .form-input {
      background-color: var(--bg-tertiary);
      border: 1px solid var(--border-color);
      color: var(--text-primary);
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 0.9rem;
    }

    .form-input:focus {
      outline: none;
      border-color: #38bdf8;
    }

    .slider-row {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .slider-row input[type="range"] {
      flex: 1;
    }

    /* 動作按鈕 */
    .btn-action-row {
      display: flex;
      gap: 10px;
      margin-top: 8px;
    }

    .btn-primary {
      flex: 2;
      background-color: var(--color-primary);
      color: #fff;
      font-weight: 600;
      padding: 12px;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      font-size: 1rem;
      transition: background-color 0.15s ease;
    }

    .btn-primary:hover {
      background-color: var(--color-primary-hover);
    }

    .btn-primary:disabled {
      background-color: #475569;
      cursor: not-allowed;
    }

    .btn-secondary {
      flex: 1;
      background-color: var(--bg-tertiary);
      color: var(--text-primary);
      border: 1px solid var(--border-color);
      padding: 12px;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 500;
    }

    .btn-secondary:hover {
      background-color: #475569;
    }

    /* 右側結果展示 */
    .render-img-container {
      width: 100%;
      min-height: 420px;
      background-color: #0b0f19;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }

    .render-img-container img {
      width: 100%;
      height: auto;
      display: block;
    }

    /* 指標網格 */
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
    }

    .metric-card {
      background-color: var(--bg-tertiary);
      padding: 12px;
      border-radius: 8px;
      border-left: 4px solid #38bdf8;
    }

    .metric-card.green {
      border-left-color: #10b981;
    }

    .metric-card.orange {
      border-left-color: #f97316;
    }

    .metric-card.yellow {
      border-left-color: #eab308;
    }

    .metric-label {
      font-size: 0.8rem;
      color: var(--text-secondary);
    }

    .metric-value {
      font-size: 1.4rem;
      font-weight: 700;
      color: var(--text-primary);
      margin-top: 4px;
    }

    /* 詳細數據表格 */
    .metrics-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
    }

    .metrics-table th, .metrics-table td {
      padding: 8px 10px;
      text-align: left;
      border-bottom: 1px solid var(--border-color);
    }

    .metrics-table th {
      color: var(--text-secondary);
      font-weight: 500;
    }
  </style>
</head>
<body>

  <div class="header">
    <h1>🌲 樹冠密度與孔隙分析引擎 (幾何約束 Box Prompt 版)</h1>
    <p>獨立純前端 HTML5 Canvas 畫布：按住滑鼠左鍵直接在照片上拖曳長方形，放開後自動輸出真實像素坐標。</p>
  </div>

  <div class="main-grid">
    <!-- 左側面板：上傳與標註 -->
    <div class="card">
      <div class="card-title">
        <span>照片上傳與標註工作台</span>
      </div>

      <!-- 快速範例與上傳 -->
      <div class="sample-bar">
        <span style="font-size: 0.85rem; color: var(--text-secondary);">快速選擇：</span>
        <button class="sample-btn" onclick="loadSampleImage('test01.jpeg')">範例 1 (test01)</button>
        <button class="sample-btn" onclick="loadSampleImage('test02.jpeg')">範例 2 (test02)</button>
        <label class="sample-btn" style="background-color: #2563eb; cursor: pointer;">
          上傳自訂照片
          <input type="file" id="file_input" accept="image/*" style="display: none;" onchange="handleFileUpload(event)">
        </label>
      </div>

      <!-- 標註模式切換 -->
      <div class="mode-switch-group">
        <button id="btn_canopy" class="mode-btn mode-btn-canopy active" onclick="switchMode('canopy')">
          樹冠方框 (黃色)
        </button>
        <button id="btn_trunk" class="mode-btn mode-btn-trunk" onclick="switchMode('trunk')">
          樹幹方框 (橘色)
        </button>
      </div>

      <!-- 提示條 -->
      <div id="status_alert" class="status-alert">
        目前模式：【樹冠方框】。請在下方照片上按住滑鼠左鍵拖拉出樹冠空間約束框。
      </div>

      <!-- 核心獨立 Canvas 畫布 -->
      <div class="canvas-wrapper" id="canvas_container">
        <div id="canvas_placeholder" class="canvas-placeholder">
          <p>請點擊上方按鈕載入範例照片或上傳自訂照片</p>
        </div>
        <canvas id="annotation_canvas" style="display: none;"></canvas>
      </div>

      <!-- 方框坐標與參數設定 -->
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
        <div class="form-group">
          <label class="form-label">樹冠方框 (選填，未填預設使用 Text Prompt)</label>
          <input type="text" id="canopy_box_input" class="form-input" value="" placeholder="拖拉設定方框，或留空使用 tree subcanopy">
        </div>
        <div class="form-group">
          <label class="form-label">樹幹方框 (選填，未填使用語意提示)</label>
          <input type="text" id="trunk_box_input" class="form-input" value="" placeholder="拖拉或手動輸入">
        </div>
      </div>

      <!-- 滑桿門檻 -->
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
        <div class="form-group">
          <label class="form-label">樹冠偵測門檻：<span id="conf_val">0.25</span></label>
          <div class="slider-row">
            <input type="range" id="slider_conf" min="0.10" max="0.90" step="0.01" value="0.25" oninput="document.getElementById('conf_val').innerText = this.value">
          </div>
        </div>
        <div class="form-group">
          <label class="form-label">樹幹負向門檻：<span id="neg_val">0.155</span></label>
          <div class="slider-row">
            <input type="range" id="slider_neg" min="0.05" max="0.50" step="0.005" value="0.155" oninput="document.getElementById('neg_val').innerText = this.value">
          </div>
        </div>
      </div>

      <!-- 按鈕 -->
      <div class="btn-action-row">
        <button class="btn-secondary" onclick="resetBoxes()">重設方框</button>
        <button id="btn_analyze" class="btn-primary" onclick="runAnalysis()">開始執行自適應診斷分析</button>
      </div>
    </div>

    <!-- 右側面板：診斷成果圖與量化數據 -->
    <div class="card">
      <div class="card-title">
        <span>4 面板高對比診斷圖與量化報告</span>
      </div>

      <div class="render-img-container" id="result_img_wrapper">
        <div id="result_placeholder" style="color: var(--text-secondary); text-align: center; padding: 20px;">
          <p>尚未執行分析。點擊左側「開始執行自適應診斷分析」後此處將呈現高對比診斷圖。</p>
        </div>
        <img id="result_img" src="" alt="診斷圖" style="display: none;">
      </div>

      <!-- 4 大核心指標卡 -->
      <div class="metric-grid">
        <div class="metric-card green">
          <div class="metric-label">樹冠密度 (Canopy Density)</div>
          <div class="metric-value" id="val_canopy_density">-- %</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">內部孔隙率 (Canopy Porosity)</div>
          <div class="metric-value" id="val_canopy_porosity">-- %</div>
        </div>
        <div class="metric-card yellow">
          <div class="metric-label">淨葉覆蓋率 (Net Foliage Cover)</div>
          <div class="metric-value" id="val_foliage_cover">-- %</div>
        </div>
        <div class="metric-card orange">
          <div class="metric-label">木質結構佔比 (Wood Ratio)</div>
          <div class="metric-value" id="val_wood_ratio">-- %</div>
        </div>
      </div>

      <!-- 詳細量化統計表格 -->
      <table class="metrics-table">
        <thead>
          <tr>
            <th>統計維度</th>
            <th>數值</th>
          </tr>
        </thead>
        <tbody id="metrics_tbody">
          <tr><td>主角樹綜合評分 (Primary Score)</td><td id="val_primary_score">--</td></tr>
          <tr><td>周邊樹偵測數量 (Nearby Count)</td><td id="val_nearby_count">--</td></tr>
          <tr><td>純葉片像素 (Living Foliage)</td><td id="val_foliage_pixels">-- px</td></tr>
          <tr><td>木質結構像素 (Wood Structure)</td><td id="val_wood_pixels">-- px</td></tr>
          <tr><td>內部穿透孔隙 (Canopy Gaps)</td><td id="val_gaps_pixels">-- px</td></tr>
          <tr><td>樹冠外包絡面積 (Canopy Envelope)</td><td id="val_envelope_pixels">-- px</td></tr>
          <tr><td>色度判定機制 (Method Used)</td><td id="val_method_used">--</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <script>
    let currentMode = 'canopy'; // 'canopy' | 'trunk'
    let currentImage = null;
    let currentFile = null;
    let isDragging = false;
    let startX = 0, startY = 0;
    let currentX = 0, currentY = 0;

    let savedBoxes = {
      canopy: null, // [xmin, ymin, xmax, ymax]
      trunk: null
    };

    const canvas = document.getElementById('annotation_canvas');
    const ctx = canvas.getContext('2d');
    const container = document.getElementById('canvas_container');
    const placeholder = document.getElementById('canvas_placeholder');

    function switchMode(mode) {
      currentMode = mode;
      document.getElementById('btn_canopy').classList.toggle('active', mode === 'canopy');
      document.getElementById('btn_trunk').classList.toggle('active', mode === 'trunk');
      const statusEl = document.getElementById('status_alert');
      if (mode === 'canopy') {
        statusEl.style.color = '#fde047';
        statusEl.innerText = '目前模式：【樹冠方框】(黃色)。按住滑鼠拖拉以設定樹冠空間約束。';
      } else {
        statusEl.style.color = '#fdba74';
        statusEl.innerText = '目前模式：【樹幹方框】(橘色)。按住滑鼠拖拉以定位主要樹幹骨架。';
      }
    }

    function loadSampleImage(filename) {
      currentFile = null;
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = function() {
        initImageOnCanvas(img);
        document.getElementById('status_alert').innerText = `已載入範例照片：${filename}。請按住滑鼠左鍵拖拉方框。`;
      };
      img.src = `/data/${filename}`;
    }

    function handleFileUpload(event) {
      const file = event.target.files[0];
      if (!file) return;
      currentFile = file;
      const reader = new FileReader();
      reader.onload = function(e) {
        const img = new Image();
        img.onload = function() {
          initImageOnCanvas(img);
          document.getElementById('status_alert').innerText = `照片 ${file.name} 載入完成。請按住滑鼠左鍵拖拉方框。`;
        };
        img.src = e.target.result;
      };
      reader.readAsDataURL(file);
    }

    function initImageOnCanvas(img) {
      currentImage = img;
      placeholder.style.display = 'none';
      canvas.style.display = 'block';

      // 依容器寬度計算適合的顯示畫布比例
      const maxDisplayWidth = container.clientWidth || 600;
      const scale = Math.min(1, maxDisplayWidth / img.naturalWidth);
      canvas.width = Math.round(img.naturalWidth * scale);
      canvas.height = Math.round(img.naturalHeight * scale);

      savedBoxes.canopy = null;
      savedBoxes.trunk = null;
      document.getElementById('canopy_box_input').value = '';
      document.getElementById('trunk_box_input').value = '';

      renderCanvas();
    }

    function renderCanvas() {
      if (!currentImage) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);

      const scaleX = canvas.width / currentImage.naturalWidth;
      const scaleY = canvas.height / currentImage.naturalHeight;

      // 繪製樹冠方框 (黃色)
      if (savedBoxes.canopy) {
        const [xmin, ymin, xmax, ymax] = savedBoxes.canopy;
        const bx = xmin * scaleX, by = ymin * scaleY, bw = (xmax - xmin) * scaleX, bh = (ymax - ymin) * scaleY;
        ctx.setLineDash([]);
        ctx.strokeStyle = '#eab308';
        ctx.lineWidth = 3;
        ctx.strokeRect(bx, by, bw, bh);
        ctx.fillStyle = 'rgba(234, 179, 8, 0.2)';
        ctx.fillRect(bx, by, bw, bh);
        ctx.fillStyle = '#fde047';
        ctx.font = 'bold 13px Inter, sans-serif';
        ctx.fillText('樹冠 (Canopy)', bx + 6, Math.max(16, by + 16));
      }

      // 繪製樹幹方框 (橘色)
      if (savedBoxes.trunk) {
        const [xmin, ymin, xmax, ymax] = savedBoxes.trunk;
        const bx = xmin * scaleX, by = ymin * scaleY, bw = (xmax - xmin) * scaleX, bh = (ymax - ymin) * scaleY;
        ctx.setLineDash([]);
        ctx.strokeStyle = '#f97316';
        ctx.lineWidth = 3;
        ctx.strokeRect(bx, by, bw, bh);
        ctx.fillStyle = 'rgba(249, 115, 22, 0.25)';
        ctx.fillRect(bx, by, bw, bh);
        ctx.fillStyle = '#fdba74';
        ctx.font = 'bold 13px Inter, sans-serif';
        ctx.fillText('樹幹 (Trunk)', bx + 6, Math.max(16, by + 16));
      }

      // 繪製進行中的拖曳框
      if (isDragging) {
        const bx = Math.min(startX, currentX);
        const by = Math.min(startY, currentY);
        const bw = Math.abs(currentX - startX);
        const bh = Math.abs(currentY - startY);

        ctx.setLineDash([6, 4]);
        ctx.strokeStyle = currentMode === 'canopy' ? '#fde047' : '#fdba74';
        ctx.lineWidth = 2;
        ctx.strokeRect(bx, by, bw, bh);
        ctx.fillStyle = currentMode === 'canopy' ? 'rgba(234, 179, 8, 0.3)' : 'rgba(249, 115, 22, 0.3)';
        ctx.fillRect(bx, by, bw, bh);
      }
    }

    // 畫布滑鼠事件監聽
    canvas.addEventListener('mousedown', function(e) {
      if (!currentImage) return;
      isDragging = true;
      startX = e.offsetX;
      startY = e.offsetY;
      currentX = e.offsetX;
      currentY = e.offsetY;
    });

    canvas.addEventListener('mousemove', function(e) {
      if (!isDragging) return;
      currentX = e.offsetX;
      currentY = e.offsetY;
      renderCanvas();
    });

    window.addEventListener('mouseup', function(e) {
      if (!isDragging) return;
      isDragging = false;

      const rect = canvas.getBoundingClientRect();
      const endX = Math.max(0, Math.min(canvas.width, e.clientX - rect.left));
      const endY = Math.max(0, Math.min(canvas.height, e.clientY - rect.top));

      const bx = Math.min(startX, endX);
      const by = Math.min(startY, endY);
      const bw = Math.abs(endX - startX);
      const bh = Math.abs(endY - startY);

      if (bw < 6 || bh < 6) {
        renderCanvas();
        return;
      }

      const scaleX = currentImage.naturalWidth / canvas.width;
      const scaleY = currentImage.naturalHeight / canvas.height;

      const natXmin = Math.round(bx * scaleX);
      const natYmin = Math.round(by * scaleY);
      const natXmax = Math.round((bx + bw) * scaleX);
      const natYmax = Math.round((by + bh) * scaleY);

      const boxCoords = [natXmin, natYmin, natXmax, natYmax];
      const boxStr = `${natXmin} ${natYmin} ${natXmax} ${natYmax}`;

      if (currentMode === 'canopy') {
        savedBoxes.canopy = boxCoords;
        document.getElementById('canopy_box_input').value = boxStr;
        document.getElementById('status_alert').innerText = `已設定樹冠方框：[${boxStr}]`;
      } else {
        savedBoxes.trunk = boxCoords;
        document.getElementById('trunk_box_input').value = boxStr;
        document.getElementById('status_alert').innerText = `已設定樹幹方框：[${boxStr}]`;
      }

      renderCanvas();
    });

    function resetBoxes() {
      savedBoxes.canopy = null;
      savedBoxes.trunk = null;
      document.getElementById('canopy_box_input').value = '';
      document.getElementById('trunk_box_input').value = '';
      document.getElementById('status_alert').innerText = '方框已重設（預設使用 Text Prompt: "tree subcanopy"）。';
      renderCanvas();
    }

    async function runAnalysis() {
      if (!currentImage) {
        alert('請先載入範例照片或上傳照片！');
        return;
      }

      const btn = document.getElementById('btn_analyze');
      btn.disabled = true;
      btn.innerText = '正在調用 SAM 3 視覺引擎推論中...';

      const formData = new FormData();
      if (currentFile) {
        formData.append('file', currentFile);
      } else {
        // 從目前的 src 提取範例檔名
        const parts = currentImage.src.split('/');
        formData.append('sample_name', parts[parts.length - 1]);
      }

      formData.append('canopy_box', document.getElementById('canopy_box_input').value);
      formData.append('trunk_box', document.getElementById('trunk_box_input').value);
      formData.append('confidence_threshold', document.getElementById('slider_conf').value);
      formData.append('negative_threshold', document.getElementById('slider_neg').value);

      try {
        const resp = await fetch('/api/analyze', {
          method: 'POST',
          body: formData
        });

        const result = await resp.json();
        if (!result.success) {
          alert('分析失敗：' + result.error);
          return;
        }

        // 渲染成果圖
        const resultImg = document.getElementById('result_img');
        const placeholder = document.getElementById('result_placeholder');
        placeholder.style.display = 'none';
        resultImg.style.display = 'block';
        resultImg.src = result.image_url + '?t=' + new Date().getTime();

        // 填入 4 大核心指標
        const m = result.metrics;
        document.getElementById('val_canopy_density').innerText = m.canopy_density_pct.toFixed(2) + ' %';
        document.getElementById('val_canopy_porosity').innerText = m.canopy_porosity_pct.toFixed(2) + ' %';
        document.getElementById('val_foliage_cover').innerText = m.foliage_cover_pct.toFixed(2) + ' %';
        document.getElementById('val_wood_ratio').innerText = m.wood_ratio_pct.toFixed(2) + ' %';

        // 填入詳細數據
        document.getElementById('val_primary_score').innerText = m.primary_score.toFixed(3);
        document.getElementById('val_nearby_count').innerText = m.nearby_count + ' 株';
        document.getElementById('val_foliage_pixels').innerText = m.foliage_pixels.toLocaleString() + ' px';
        document.getElementById('val_wood_pixels').innerText = m.wood_pixels.toLocaleString() + ' px';
        document.getElementById('val_gaps_pixels').innerText = m.gaps_pixels.toLocaleString() + ' px';
        document.getElementById('val_envelope_pixels').innerText = m.envelope_pixels.toLocaleString() + ' px';
        document.getElementById('val_method_used').innerText = m.method_used;

      } catch (err) {
        alert('網路請求異常：' + err);
      } finally {
        btn.disabled = false;
        btn.innerText = '開始執行自適應診斷分析';
      }
    }

    // 預設頁面載入時自動載入範例 1
    window.onload = function() {
      loadSampleImage('test01.jpeg');
    };
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index_page():
    return HTMLResponse(content=INDEX_HTML)


@app.get("/data/{filename}")
def serve_data_image(filename: str):
    file_path = project_root / "data" / filename
    if file_path.exists():
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"error": "File not found"})


@app.get("/output/{filename}")
def serve_output_image(filename: str):
    file_path = project_root / "samplingAnalysis" / "output" / filename
    if file_path.exists():
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"error": "File not found"})


@app.post("/api/analyze")
async def api_analyze(
    file: UploadFile = File(None),
    sample_name: str = Form(None),
    canopy_box: str = Form(""),
    trunk_box: str = Form(""),
    confidence_threshold: float = Form(0.25),
    negative_threshold: float = Form(0.155),
):
    temp_img_path = None
    try:
        if file and file.filename:
            # 儲存上傳之檔案至暫存區
            suffix = Path(file.filename).suffix or ".jpg"
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_img_path = Path(temp_file.name)
            with open(temp_img_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            eff_image_path = temp_img_path
        elif sample_name:
            eff_image_path = project_root / "data" / sample_name
            if not eff_image_path.exists():
                return JSONResponse(status_code=400, content={"success": False, "error": f"找不到範例圖片：{sample_name}"})
        else:
            return JSONResponse(status_code=400, content={"success": False, "error": "未提供有效影像來源。"})

        processor = get_processor()
        canopy_parsed = parse_box_str(canopy_box)
        trunk_parsed = parse_box_str(trunk_box)

        out_dir = project_root / "samplingAnalysis" / "output"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 執行核心 CIELAB 自適應分析
        metrics = process_image_cielab_adaptive(
            image_path=eff_image_path,
            processor=processor,
            canopy_box=canopy_parsed,
            trunk_box=trunk_parsed,
            confidence_threshold=confidence_threshold,
            negative_threshold=negative_threshold,
            output_dir=out_dir,
        )

        stem = eff_image_path.stem
        render_filename = f"{stem}_cielab_otsu_render.png"

        return JSONResponse({
            "success": True,
            "image_url": f"/output/{render_filename}",
            "metrics": metrics,
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

    finally:
        if temp_img_path and temp_img_path.exists():
            temp_img_path.unlink(missing_ok=True)


if __name__ == "__main__":
    print("[*] 正在啟動樹冠密度與孔隙分析引擎 Web 平台...")
    print("[*] 存取網址：http://127.0.0.1:7860")
    uvicorn.run(app, host="0.0.0.0", port=7860)
