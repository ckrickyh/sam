"""Ultralytics YOLO-Seg Live Monitoring & Training Control Server

專為樹木次冠層微調訓練打造之即時動態監控與訓練控制伺服器：
1. 支援訓練流程全生命週期控制 (REST API)：
   - POST /api/train/start (啟動訓練)
   - POST /api/train/pause (暫停訓練 - SIGSTOP)
   - POST /api/train/resume (繼續訓練 - SIGCONT)
   - POST /api/train/stop (終止訓練 - SIGTERM/SIGKILL)
2. 自動監控 `runs/segment/` 目錄下的 `results.csv` 與圖檔變更。
3. 內嵌現代化響應式 Web 儀表板 (包含暗黑漸層平滑折線圖、即時控制按鈕、最新樹葉預測圖對照、即時 KPI 卡片)。
4. 預設監聽連接埠：http://localhost:8765
"""

import argparse
import csv
import json
import mimetypes
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# 確保專案根目錄在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class TrainingProcessManager:
    """管理 YOLO 訓練子行程 (啟動、暫停、繼續、終止)"""

    def __init__(self):
        self.process: subprocess.Popen | None = None
        self.status: str = "idle"  # idle | running | paused | completed | error
        self.start_time: float | None = None
        self.log_file = project_root / "runs" / "training_console.log"
        self.params: dict = {"epochs": 30, "batch": 4, "imgsz": 640, "freeze": 10}

    def start(self, epochs: int = 30, batch: int = 4, imgsz: int = 640, freeze: int = 10) -> dict:
        if self.status in ["running", "paused"]:
            return {"success": False, "message": "訓練已在進行中，請勿重複啟動。"}

        self.params = {"epochs": epochs, "batch": batch, "imgsz": imgsz, "freeze": freeze}
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(self.log_file, "w", encoding="utf-8")

        cmd = [
            sys.executable,
            str(project_root / "ultralystic" / "train.py"),
            "--data", "dataset/dataset.yaml",
            "--epochs", str(epochs),
            "--batch", str(batch),
            "--imgsz", str(imgsz),
            "--freeze", str(freeze),
        ]

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )
            self.status = "running"
            self.start_time = time.time()

            threading.Thread(target=self._monitor_worker, daemon=True).start()
            return {"success": True, "message": f"訓練已啟動 (PID: {self.process.pid})", "pid": self.process.pid}
        except Exception as e:
            self.status = "error"
            return {"success": False, "message": f"啟動失敗: {e}"}

    def _monitor_worker(self):
        if self.process:
            ret = self.process.wait()
            if self.status != "idle":
                self.status = "completed" if ret == 0 else "error"
            self.process = None

    def pause(self) -> dict:
        if not self.process or self.status != "running":
            return {"success": False, "message": "目前無正在運行的訓練任務可暫停。"}

        try:
            os.kill(self.process.pid, signal.SIGSTOP)
            self.status = "paused"
            return {"success": True, "message": "訓練已成功暫停 (SIGSTOP)。"}
        except Exception as e:
            return {"success": False, "message": f"暫停失敗: {e}"}

    def resume(self) -> dict:
        if not self.process or self.status != "paused":
            return {"success": False, "message": "目前無暫停中的訓練任務可繼續。"}

        try:
            os.kill(self.process.pid, signal.SIGCONT)
            self.status = "running"
            return {"success": True, "message": "訓練已成功繼續 (SIGCONT)。"}
        except Exception as e:
            return {"success": False, "message": f"繼續失敗: {e}"}

    def stop(self) -> dict:
        if not self.process:
            self.status = "idle"
            return {"success": True, "message": "目前無訓練任務運行中。"}

        try:
            if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            else:
                self.process.terminate()

            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                else:
                    self.process.kill()

            self.process = None
            self.status = "idle"
            return {"success": True, "message": "訓練任務已強制安全終止。"}
        except Exception as e:
            self.status = "idle"
            self.process = None
            return {"success": False, "message": f"終止時發生異常: {e}"}

    def get_status(self) -> dict:
        return {
            "status": self.status,
            "pid": self.process.pid if self.process else None,
            "params": self.params,
            "elapsed_seconds": int(time.time() - self.start_time) if self.start_time and self.status in ["running", "paused"] else 0,
        }


training_manager = TrainingProcessManager()


def find_latest_experiment_dir(base_dir: Path) -> Path | None:
    """尋找 runs/segment 下最新的實驗目錄 (包含巢狀結構)"""
    if not base_dir.exists():
        return None
    
    csv_candidates = list(base_dir.glob("**/results.csv"))
    if csv_candidates:
        csv_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return csv_candidates[0].parent

    subdirs = [p for p in base_dir.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return subdirs[0]


def parse_results_csv(csv_path: Path) -> dict:
    """解析 Ultralytics results.csv 檔案為結構化時序數據"""
    if not csv_path.exists():
        return {"epochs": [], "metrics": {}, "latest": {}}

    epochs = []
    train_total_loss = []
    val_total_loss = []
    train_seg_loss = []
    val_seg_loss = []
    train_box_loss = []
    val_box_loss = []
    train_cls_loss = []
    val_cls_loss = []
    metrics_map50 = []
    metrics_map50_95 = []
    metrics_precision = []
    metrics_recall = []

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = [h.strip() for h in next(reader, [])]
            if not header:
                return {"epochs": [], "metrics": {}, "latest": {}}

            col_map = {col.lower(): idx for idx, col in enumerate(header)}

            for row in reader:
                if not row or len(row) < len(header):
                    continue
                try:
                    def get_val(key_fragment, default=0.0):
                        for k, idx in col_map.items():
                            if key_fragment in k:
                                return float(row[idx].strip())
                        return default

                    epoch = int(float(row[0].strip()))
                    epochs.append(epoch)

                    # 收集各項子損失
                    t_seg = get_val("train/seg_loss", 0.0)
                    v_seg = get_val("val/seg_loss", 0.0)
                    t_box = get_val("train/box_loss", 0.0)
                    v_box = get_val("val/box_loss", 0.0)
                    t_cls = get_val("train/cls_loss", 0.0)
                    v_cls = get_val("val/cls_loss", 0.0)
                    t_sem = get_val("train/sem_loss", 0.0)
                    v_sem = get_val("val/sem_loss", 0.0)
                    t_dfl = get_val("train/dfl_loss", 0.0)
                    v_dfl = get_val("val/dfl_loss", 0.0)

                    # 計算加總的 Total Train Loss 與 Total Val Loss
                    t_total = t_seg + t_box + t_cls + t_sem + t_dfl
                    v_total = v_seg + v_box + v_cls + v_sem + v_dfl

                    train_total_loss.append(round(t_total, 4))
                    val_total_loss.append(round(v_total, 4))
                    train_seg_loss.append(t_seg)
                    val_seg_loss.append(v_seg)
                    train_box_loss.append(t_box)
                    val_box_loss.append(v_box)
                    train_cls_loss.append(t_cls)
                    val_cls_loss.append(v_cls)

                    metrics_map50.append(get_val("metrics/map50(m)", get_val("map50", 0.0)))
                    metrics_map50_95.append(get_val("metrics/map50-95(m)", get_val("map50-95", 0.0)))
                    metrics_precision.append(get_val("metrics/precision(m)", get_val("precision", 0.0)))
                    metrics_recall.append(get_val("metrics/recall(m)", get_val("recall", 0.0)))
                except Exception:
                    continue

        latest = {}
        if epochs:
            latest = {
                "epoch": epochs[-1],
                "train_loss": train_total_loss[-1],
                "val_loss": val_total_loss[-1],
                "train_seg_loss": train_seg_loss[-1],
                "val_seg_loss": val_seg_loss[-1],
                "train_box_loss": train_box_loss[-1],
                "val_box_loss": val_box_loss[-1],
                "train_cls_loss": train_cls_loss[-1],
                "val_cls_loss": val_cls_loss[-1],
                "map50": metrics_map50[-1],
                "map50_95": metrics_map50_95[-1],
                "precision": metrics_precision[-1],
                "recall": metrics_recall[-1],
            }

        return {
            "epochs": epochs,
            "train_loss": train_total_loss,
            "val_loss": val_total_loss,
            "train_seg_loss": train_seg_loss,
            "val_seg_loss": val_seg_loss,
            "train_box_loss": train_box_loss,
            "val_box_loss": val_box_loss,
            "map50": metrics_map50,
            "map50_95": metrics_map50_95,
            "precision": metrics_precision,
            "recall": metrics_recall,
            "latest": latest,
        }
    except Exception as e:
        print(f"[Error] 解析 results.csv 失敗: {e}")
        return {"epochs": [], "metrics": {}, "latest": {}}


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh-TW" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO26-seg 即時訓練動態監控中心</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
        .font-mono { font-family: 'JetBrains Mono', monospace; }
    </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col">
    <!-- 頂部導覽列 -->
    <header class="border-b border-slate-800 bg-slate-900/60 sticky top-0 z-50 backdrop-blur-md px-6 py-4 flex items-center justify-between">
        <div class="flex items-center space-x-3">
            <div class="w-10 h-10 rounded-xl bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center text-emerald-400 font-bold text-xl shadow-inner">
                🌲
            </div>
            <div>
                <div class="flex items-center space-x-2">
                    <h1 class="text-base font-bold text-white tracking-wide">YOLO26-seg 即時訓練控制與動態監控</h1>
                    <span class="px-2 py-0.5 rounded text-[10px] font-mono bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">React Engine</span>
                </div>
                <p class="text-xs text-slate-400">SAM 3 偽標籤至 YOLO26 實例分割全自動知識蒸餾</p>
            </div>
        </div>

        <div class="flex items-center space-x-4">
            <div id="status-badge" class="flex items-center space-x-2 px-3.5 py-1.5 rounded-full border text-xs font-mono bg-slate-800 border-slate-700 text-slate-300">
                <span id="status-dot" class="w-2.5 h-2.5 rounded-full bg-slate-400"></span>
                <span id="status-text" class="font-semibold">等待指令 (Idle)</span>
            </div>
            <div class="text-xs text-slate-400 font-mono hidden md:block" id="last-update">更新: --:--:--</div>
        </div>
    </header>

    <!-- 主體內容 -->
    <main class="max-w-7xl mx-auto px-6 py-6 flex-1 w-full space-y-6">
        
        <!-- Toast 訊息列 -->
        <div id="toast-banner" class="hidden p-3 rounded-xl text-xs font-medium border flex items-center justify-between transition-all duration-300">
            <span id="toast-text" class="flex items-center space-x-2"></span>
            <button onclick="hideToast()" class="text-slate-400 hover:text-white">✕</button>
        </div>

        <!-- 🎮 控制面板 (Action Control Bar) -->
        <div class="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-2xl backdrop-blur-xl">
            <div class="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
                
                <div class="flex items-center flex-wrap gap-3">
                    <button id="btn-start" onclick="startTraining()" class="px-5 py-2.5 rounded-xl font-semibold text-sm flex items-center space-x-2 shadow-lg transition-all duration-200 bg-emerald-600 hover:bg-emerald-500 active:scale-95 text-white shadow-emerald-900/30">
                        <span>▶️</span>
                        <span>開始訓練 (Start)</span>
                    </button>

                    <button id="btn-pause" onclick="togglePause()" class="px-5 py-2.5 rounded-xl font-semibold text-sm flex items-center space-x-2 shadow-lg transition-all duration-200 bg-amber-600 hover:bg-amber-500 active:scale-95 text-white shadow-amber-900/30">
                        <span id="icon-pause">⏸️</span>
                        <span id="text-pause">暫停 (Pause)</span>
                    </button>

                    <button id="btn-stop" onclick="stopTraining()" class="px-5 py-2.5 rounded-xl font-semibold text-sm flex items-center space-x-2 shadow-lg transition-all duration-200 bg-red-600 hover:bg-red-500 active:scale-95 text-white shadow-red-900/30">
                        <span>⏹️</span>
                        <span>終止訓練 (Stop)</span>
                    </button>
                </div>

                <div class="flex items-center flex-wrap gap-3 bg-slate-950/70 px-4 py-2.5 rounded-xl border border-slate-800 text-xs font-mono">
                    <div class="flex items-center space-x-1.5">
                        <span class="text-slate-400">輪數 (Epochs):</span>
                        <input id="input-epochs" type="number" value="30" class="w-14 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-white font-bold text-center focus:border-emerald-500 focus:outline-none">
                    </div>
                    <div class="flex items-center space-x-1.5">
                        <span class="text-slate-400">批次 (Batch):</span>
                        <input id="input-batch" type="number" value="4" class="w-12 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-white font-bold text-center focus:border-emerald-500 focus:outline-none">
                    </div>
                    <div class="flex items-center space-x-1.5">
                        <span class="text-slate-400">凍結骨幹 (Freeze):</span>
                        <input id="input-freeze" type="number" value="10" class="w-12 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-white font-bold text-center focus:border-emerald-500 focus:outline-none">
                    </div>
                </div>

            </div>
        </div>

        <!-- 🌟 核心：暗黑平滑漸層動態折線圖 (Train Loss vs Val Loss) -->
        <div class="bg-[#11161d] border border-slate-800/90 rounded-2xl p-6 shadow-2xl backdrop-blur-xl">
            <div class="flex items-center justify-between mb-2">
                <div class="flex items-center space-x-2">
                    <span class="w-2.5 h-2.5 rounded-full bg-sky-400"></span>
                    <h2 class="text-sm font-bold text-slate-100 tracking-wide">訓練與驗證損失收斂圖 (Loss vs. Epochs)</h2>
                </div>
                <span class="text-xs text-slate-400 font-mono" id="chart-epochs-count">實時動態渲染 • 0 Epochs</span>
            </div>

            <!-- 折線圖容器 -->
            <div class="w-full h-80 relative flex flex-col mt-2">
                <div class="flex items-center justify-center space-x-6 mb-2">
                    <div class="flex items-center space-x-2">
                        <span class="w-4 h-2.5 rounded-sm border border-sky-400 bg-sky-500/30 inline-block"></span>
                        <span class="text-xs font-semibold text-sky-300 tracking-wide">Train Loss</span>
                    </div>
                    <div class="flex items-center space-x-2">
                        <span class="w-4 h-2.5 rounded-sm border border-emerald-400 bg-emerald-500/30 inline-block"></span>
                        <span class="text-xs font-semibold text-emerald-300 tracking-wide">Val Loss</span>
                    </div>
                </div>
                <div class="flex-1 w-full h-full relative">
                    <canvas id="lossCanvas" class="w-full h-full block"></canvas>
                </div>
            </div>
        </div>

        <!-- KPI 數值卡片矩陣 -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-4 shadow-lg">
                <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider">當前輪數 (Epoch)</div>
                <div class="mt-2 flex items-baseline justify-between">
                    <div class="text-3xl font-black font-mono text-white" id="kpi-epoch">--</div>
                    <span class="text-xs font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20" id="kpi-progress">Ready</span>
                </div>
                <div class="mt-3 w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                    <div id="kpi-progressbar" class="bg-emerald-500 h-full rounded-full transition-all duration-500" style="width: 0%"></div>
                </div>
            </div>

            <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-4 shadow-lg">
                <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider">分割損失 (Seg Loss)</div>
                <div class="mt-2 flex items-baseline justify-between">
                    <div class="text-3xl font-black font-mono text-amber-400" id="kpi-train-loss">--</div>
                    <span class="text-xs font-mono text-slate-400" id="kpi-val-loss">Val: --</span>
                </div>
                <div class="mt-2 text-[11px] text-slate-400">📉 樹葉多邊形擬合誤差 (越小越準)</div>
            </div>

            <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-4 shadow-lg">
                <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider">遮罩精度 (Mask mAP50)</div>
                <div class="mt-2 flex items-baseline justify-between">
                    <div class="text-3xl font-black font-mono text-emerald-400" id="kpi-map50">--%</div>
                    <span class="text-xs font-mono text-slate-400" id="kpi-map95">mAP95: --%</span>
                </div>
                <div class="mt-2 text-[11px] text-emerald-400/80">🎯 主樹 / 周邊樹 綜合分割精度</div>
            </div>

            <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-4 shadow-lg">
                <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider">查準率 / 召回率 (P / R)</div>
                <div class="mt-2 flex items-baseline justify-between">
                    <div class="text-3xl font-black font-mono text-sky-400" id="kpi-pr">-- / --</div>
                </div>
                <div class="mt-2 text-[11px] text-slate-400">🔍 Precision vs Recall</div>
            </div>
        </div>

        <!-- 視覺化真值 vs 預測對照面板 -->
        <div class="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4">
            <div class="flex items-center justify-between border-b border-slate-800 pb-3">
                <div class="flex items-center space-x-3">
                    <h2 class="text-sm font-bold text-white flex items-center space-x-2">
                        <span class="w-2.5 h-2.5 rounded-full bg-sky-400"></span>
                        <span>即時樹葉分割預測比對 (Live Proof Panel)</span>
                    </h2>
                    <span class="text-xs bg-emerald-500/10 text-emerald-400 px-2 py-0.5 rounded border border-emerald-500/20 font-medium">綠色: 主體樹 (Class 0) | 藍色: 周邊樹 (Class 1)</span>
                </div>
                
                <button onclick="fetchData()" class="text-xs bg-slate-800 hover:bg-slate-700 text-slate-200 px-3 py-1.5 rounded-lg border border-slate-700 transition flex items-center space-x-1">
                    <span>🔄 即時刷新</span>
                </button>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div class="space-y-2">
                    <div class="text-xs font-semibold text-slate-300 flex items-center space-x-1.5">
                        <span class="w-2 h-2 rounded-full bg-emerald-400"></span>
                        <span>AI 實例分割預測 (val_batch0_pred.jpg)</span>
                    </div>
                    <div class="rounded-xl overflow-hidden bg-slate-950 border border-slate-800 aspect-video flex items-center justify-center p-1">
                        <img id="img-pred" src="/api/image/val_batch0_pred.jpg" alt="AI 預測圖" class="w-full h-full object-contain rounded-lg">
                    </div>
                </div>

                <div class="space-y-2">
                    <div class="text-xs font-semibold text-slate-300 flex items-center space-x-1.5">
                        <span class="w-2 h-2 rounded-full bg-sky-400"></span>
                        <span>SAM 3 偽標籤真值 (val_batch0_labels.jpg)</span>
                    </div>
                    <div class="rounded-xl overflow-hidden bg-slate-950 border border-slate-800 aspect-video flex items-center justify-center p-1">
                        <img id="img-labels" src="/api/image/val_batch0_labels.jpg" alt="標籤真值圖" class="w-full h-full object-contain rounded-lg">
                    </div>
                </div>
            </div>
        </div>

    </main>

    <!-- 底部 Footer -->
    <footer class="border-t border-slate-800 py-3 text-center text-xs text-slate-500 font-mono">
        Ultralytics YOLO26-seg Distillation Pipeline • Antigravity MLOps
    </footer>

    <!-- 即時繪圖與狀態邏輯腳本 -->
    <script>
        let currentStatus = 'idle';

        function showToast(text, type = 'info') {
            const banner = document.getElementById('toast-banner');
            const toastText = document.getElementById('toast-text');
            banner.className = `p-3 rounded-xl text-xs font-medium border flex items-center justify-between transition-all duration-300 ` + 
                (type === 'success' ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300' :
                 type === 'error' ? 'bg-red-500/10 border-red-500/30 text-red-300' : 'bg-sky-500/10 border-sky-500/30 text-sky-300');
            toastText.innerHTML = `<span>${type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️'}</span><span>${text}</span>`;
            banner.classList.remove('hidden');
            setTimeout(hideToast, 4000);
        }

        function hideToast() {
            document.getElementById('toast-banner').classList.add('hidden');
        }

        async function startTraining() {
            const epochs = parseInt(document.getElementById('input-epochs').value) || 30;
            const batch = parseInt(document.getElementById('input-batch').value) || 4;
            const freeze = parseInt(document.getElementById('input-freeze').value) || 10;
            showToast('正在發送啟動訓練請求...', 'info');
            try {
                const res = await fetch('/api/train/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ epochs, batch, freeze })
                });
                const data = await res.json();
                showToast(data.message, data.success ? 'success' : 'error');
                fetchData();
            } catch(e) {
                showToast('連線失敗: ' + e.message, 'error');
            }
        }

        async function togglePause() {
            const action = currentStatus === 'paused' ? 'resume' : 'pause';
            showToast(`正在發送${action === 'resume' ? '繼續' : '暫停'}請求...`, 'info');
            try {
                const res = await fetch(`/api/train/${action}`, { method: 'POST' });
                const data = await res.json();
                showToast(data.message, data.success ? 'success' : 'error');
                fetchData();
            } catch(e) {
                showToast('連線失敗: ' + e.message, 'error');
            }
        }

        async function stopTraining() {
            if (!confirm('確定要強制終止目前的訓練任務嗎？')) return;
            showToast('正在發送終止請求 (SIGTERM)...', 'info');
            try {
                const res = await fetch('/api/train/stop', { method: 'POST' });
                const data = await res.json();
                showToast(data.message, data.success ? 'success' : 'error');
                fetchData();
            } catch(e) {
                showToast('連線失敗: ' + e.message, 'error');
            }
        }

        function drawLossChart(epochs, trainLoss, valLoss) {
            const canvas = document.getElementById('lossCanvas');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            const width = canvas.width = canvas.parentElement.clientWidth;
            const height = canvas.height = canvas.parentElement.clientHeight || 300;

            ctx.clearRect(0, 0, width, height);

            if (!epochs || epochs.length === 0) {
                ctx.fillStyle = '#64748b';
                ctx.font = '13px Inter, sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText('等待訓練數據中 (Waiting for Epoch data)...', width / 2, height / 2);
                return;
            }

            const padLeft = 65, padRight = 30, padTop = 45, padBottom = 45;
            const plotW = width - padLeft - padRight;
            const plotH = height - padTop - padBottom;

            const allVals = [...trainLoss, ...valLoss].filter(v => typeof v === 'number' && !isNaN(v));
            const rawMax = allVals.length > 0 ? Math.max(...allVals) : 10;
            const maxVal = Math.ceil(rawMax * 1.15) || 5;
            const minVal = 0;

            // 繪製格線
            const yTicks = 6;
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
            ctx.lineWidth = 1;
            ctx.fillStyle = '#64748b';
            ctx.font = '11px "JetBrains Mono", monospace';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';

            for (let i = 0; i <= yTicks; i++) {
                const val = minVal + (maxVal - minVal) * (i / yTicks);
                const y = padTop + plotH - (i / yTicks) * plotH;

                ctx.beginPath();
                ctx.moveTo(padLeft, y);
                ctx.lineTo(width - padRight, y);
                ctx.stroke();

                ctx.fillText(val < 10 ? val.toFixed(2) : Math.round(val), padLeft - 12, y);
            }

            // Y 軸標題
            ctx.save();
            ctx.translate(18, padTop + plotH / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.fillStyle = '#94a3b8';
            ctx.font = '12px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Total Loss', 0, 0);
            ctx.restore();

            const nPoints = epochs.length;
            const getX = (idx) => nPoints === 1 ? padLeft + plotW / 2 : padLeft + (idx / (nPoints - 1)) * plotW;
            const getY = (v) => padTop + plotH - ((v - minVal) / (maxVal - minVal)) * plotH;

            // X 軸刻度文字 (E1, E2, E3...)
            ctx.fillStyle = '#94a3b8';
            ctx.font = '11px "JetBrains Mono", monospace';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';

            epochs.forEach((ep, idx) => {
                if (nPoints > 15 && idx % Math.ceil(nPoints / 10) !== 0 && idx !== nPoints - 1) return;
                const x = getX(idx);
                ctx.fillText(`E${ep}`, x, padTop + plotH + 12);
            });

            // X 軸標題
            ctx.fillStyle = '#64748b';
            ctx.font = '12px Inter, sans-serif';
            ctx.fillText('Epoch', padLeft + plotW / 2, height - 12);

            // 繪製漸層曲線函式
            const drawSpline = (data, strokeColor, fillColorStart, dotColor) => {
                if (!data || data.length === 0) return;
                const points = data.map((v, i) => ({ x: getX(i), y: getY(v) }));

                // 漸層區域
                ctx.save();
                ctx.beginPath();
                ctx.moveTo(points[0].x, padTop + plotH);
                ctx.lineTo(points[0].x, points[0].y);

                for (let i = 0; i < points.length - 1; i++) {
                    const curr = points[i];
                    const next = points[i + 1];
                    const cpX = (curr.x + next.x) / 2;
                    ctx.bezierCurveTo(cpX, curr.y, cpX, next.y, next.x, next.y);
                }

                ctx.lineTo(points[points.length - 1].x, padTop + plotH);
                ctx.closePath();

                const grad = ctx.createLinearGradient(0, padTop, 0, padTop + plotH);
                grad.addColorStop(0, fillColorStart);
                grad.addColorStop(1, 'rgba(0, 0, 0, 0.0)');
                ctx.fillStyle = grad;
                ctx.fill();
                ctx.restore();

                // 平滑線
                ctx.save();
                ctx.beginPath();
                ctx.moveTo(points[0].x, points[0].y);
                for (let i = 0; i < points.length - 1; i++) {
                    const curr = points[i];
                    const next = points[i + 1];
                    const cpX = (curr.x + next.x) / 2;
                    ctx.bezierCurveTo(cpX, curr.y, cpX, next.y, next.x, next.y);
                }
                ctx.strokeStyle = strokeColor;
                ctx.lineWidth = 2.5;
                ctx.stroke();
                ctx.restore();

                // 節點小圓圈
                points.forEach(p => {
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
                    ctx.fillStyle = '#0f172a';
                    ctx.fill();
                    ctx.strokeStyle = dotColor;
                    ctx.lineWidth = 2;
                    ctx.stroke();
                });
            };

            drawSpline(trainLoss, '#0284c7', 'rgba(2, 132, 199, 0.25)', '#38bdf8');
            drawSpline(valLoss, '#86efac', 'rgba(134, 239, 172, 0.18)', '#4ade80');
        }

        async function fetchData() {
            try {
                const res = await fetch('/api/metrics');
                const data = await res.json();

                document.getElementById('last-update').innerText = '更新: ' + new Date().toLocaleTimeString();

                const st = data.training_status || {};
                currentStatus = st.status || 'idle';
                const pid = st.pid;

                const badge = document.getElementById('status-badge');
                const dot = document.getElementById('status-dot');
                const stText = document.getElementById('status-text');

                if (currentStatus === 'running') {
                    badge.className = 'flex items-center space-x-2 px-3.5 py-1.5 rounded-full border text-xs font-mono bg-emerald-500/10 border-emerald-500/30 text-emerald-300';
                    dot.className = 'w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping';
                    stText.innerText = `訓練中 (PID: ${pid})`;
                    document.getElementById('icon-pause').innerText = '⏸️';
                    document.getElementById('text-pause').innerText = '暫停 (Pause)';
                } else if (currentStatus === 'paused') {
                    badge.className = 'flex items-center space-x-2 px-3.5 py-1.5 rounded-full border text-xs font-mono bg-amber-500/10 border-amber-500/30 text-amber-300';
                    dot.className = 'w-2.5 h-2.5 rounded-full bg-amber-400';
                    stText.innerText = `已暫停 (PID: ${pid})`;
                    document.getElementById('icon-pause').innerText = '⏯️';
                    document.getElementById('text-pause').innerText = '繼續 (Resume)';
                } else if (currentStatus === 'completed') {
                    badge.className = 'flex items-center space-x-2 px-3.5 py-1.5 rounded-full border text-xs font-mono bg-sky-500/10 border-sky-500/30 text-sky-300';
                    dot.className = 'w-2.5 h-2.5 rounded-full bg-sky-400';
                    stText.innerText = '訓練已完成 (Completed)';
                } else {
                    badge.className = 'flex items-center space-x-2 px-3.5 py-1.5 rounded-full border text-xs font-mono bg-slate-800 border-slate-700 text-slate-300';
                    dot.className = 'w-2.5 h-2.5 rounded-full bg-slate-400';
                    stText.innerText = '等待指令 (Idle)';
                }

                if (data.epochs && data.epochs.length > 0) {
                    const latest = data.latest || {};
                    const ep = latest.epoch || 0;
                    document.getElementById('kpi-epoch').innerText = ep;
                    document.getElementById('kpi-progress').innerText = `${ep} / 30`;
                    document.getElementById('kpi-progressbar').style.width = `${Math.min(100, (ep / 30) * 100)}%`;

                    document.getElementById('kpi-train-loss').innerText = (latest.train_loss || latest.train_seg_loss || 0).toFixed(4);
                    document.getElementById('kpi-val-loss').innerText = 'Val: ' + (latest.val_loss || latest.val_seg_loss || 0).toFixed(4);
                    document.getElementById('kpi-map50').innerText = ((latest.map50 || 0) * 100).toFixed(1) + '%';
                    document.getElementById('kpi-map95').innerText = 'mAP95: ' + ((latest.map50_95 || 0) * 100).toFixed(1) + '%';
                    document.getElementById('kpi-pr').innerText = ((latest.precision || 0) * 100).toFixed(0) + '% / ' + ((latest.recall || 0) * 100).toFixed(0) + '%';

                    document.getElementById('chart-epochs-count').innerText = `實時動態渲染 • ${data.epochs.length} Epochs`;

                    drawLossChart(data.epochs, data.train_loss || data.train_seg_loss, data.val_loss || data.val_seg_loss);

                    const t = new Date().getTime();
                    document.getElementById('img-pred').src = '/api/image/val_batch0_pred.jpg?t=' + t;
                    document.getElementById('img-labels').src = '/api/image/val_batch0_labels.jpg?t=' + t;
                }
            } catch (err) {
                console.error('Fetch error:', err);
            }
        }

        window.addEventListener('DOMContentLoaded', () => {
            fetchData();
            setInterval(fetchData, 1500);
            window.addEventListener('resize', () => {
                fetchData();
            });
        });
    </script>
</body>
</html>
"""


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    """自訂 HTTP Request Handler 處理控制 API、即時數據與前端靜態頁面"""

    def __init__(self, *args, experiment_base: Path = Path("runs/segment"), **kwargs):
        self.experiment_base = experiment_base
        super().__init__(*args, **kwargs)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body_data = {}
        if content_len > 0:
            try:
                body_str = self.rfile.read(content_len).decode("utf-8")
                body_data = json.loads(body_str) if body_str else {}
            except Exception:
                pass

        if self.path == "/api/train/start":
            epochs = int(body_data.get("epochs", 30))
            batch = int(body_data.get("batch", 4))
            imgsz = int(body_data.get("imgsz", 640))
            freeze = int(body_data.get("freeze", 10))
            res = training_manager.start(epochs=epochs, batch=batch, imgsz=imgsz, freeze=freeze)
            self._send_json(res)
            return

        if self.path == "/api/train/pause":
            res = training_manager.pause()
            self._send_json(res)
            return

        if self.path == "/api/train/resume":
            res = training_manager.resume()
            self._send_json(res)
            return

        if self.path == "/api/train/stop":
            res = training_manager.stop()
            self._send_json(res)
            return

        self.send_response(404)
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/api/status"):
            status_data = training_manager.get_status()
            self._send_json(status_data)
            return

        if self.path.startswith("/api/metrics"):
            latest_exp = find_latest_experiment_dir(self.experiment_base)
            if latest_exp:
                csv_file = latest_exp / "results.csv"
                data = parse_results_csv(csv_file)
                data["experiment_name"] = latest_exp.name
            else:
                data = {"epochs": [], "metrics": {}, "latest": {}, "experiment_name": "none"}

            data["training_status"] = training_manager.get_status()
            self._send_json(data)
            return

        if self.path.startswith("/api/image/"):
            filename = self.path.split("/api/image/")[1].split("?")[0]
            latest_exp = find_latest_experiment_dir(self.experiment_base)
            if latest_exp:
                target_file = latest_exp / filename
                if target_file.exists() and target_file.is_file():
                    mime, _ = mimetypes.guess_type(str(target_file))
                    self.send_response(200)
                    self.send_header("Content-Type", mime or "image/jpeg")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    with open(target_file, "rb") as f:
                        self.wfile.write(f.read())
                    return

            self.send_response(404)
            self.end_headers()
            return

        if self.path in ["/", "/index.html"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

    def _send_json(self, data: dict):
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))


def start_monitor_server(port: int = 8765, base_dir: str | Path = "runs/segment") -> HTTPServer:
    """啟動監控伺服器"""
    b_path = Path(base_dir).resolve()
    handler = lambda *args, **kwargs: DashboardRequestHandler(*args, experiment_base=b_path, **kwargs)
    httpd = HTTPServer(("0.0.0.0", port), handler)
    print(f"==================================================")
    print(f"🌲 [YOLO26-seg Live Dashboard & Controller] 監控與控制伺服器已啟動！")
    print(f"★ 儀表板網址：http://localhost:{port}")
    print(f"★ 支援指令  ：開始 (Start) | 暫停 (Pause) | 繼續 (Resume) | 終止 (Stop)")
    print(f"==================================================")
    return httpd


def main():
    parser = argparse.ArgumentParser(description="Ultralytics YOLO26-seg Live Training Monitor & Control Server")
    parser.add_argument("--port", type=int, default=8765, help="Web 儀表板連接埠 (預設 8765)")
    parser.add_argument("--runs-dir", type=str, default="runs/segment", help="監控的 runs 輸出目錄")
    args = parser.parse_args()

    server = start_monitor_server(port=args.port, base_dir=args.runs_dir)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Dashboard] 監控伺服器已安全停止。")
        server.server_close()


if __name__ == "__main__":
    main()
