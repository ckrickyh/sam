import React, { useState, useEffect, useRef } from 'react';

// 自訂與截圖 100% 相同視覺風格之動態平滑漸層折線圖組件
function LossLineChart({ epochs, trainLoss, valLoss, title = "Total Loss", xLabel = "Epoch" }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width = canvas.parentElement.clientWidth;
    const height = canvas.height = canvas.parentElement.clientHeight || 320;

    ctx.clearRect(0, 0, width, height);

    if (!epochs || epochs.length === 0) {
      ctx.fillStyle = '#64748b';
      ctx.font = '13px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('等待訓練數據中 (Waiting for Epoch data)...', width / 2, height / 2);
      return;
    }

    const padLeft = 65;
    const padRight = 30;
    const padTop = 45;
    const padBottom = 45;

    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;

    // 計算 Y 軸極值
    const allVals = [...trainLoss, ...valLoss].filter(v => typeof v === 'number' && !isNaN(v));
    const rawMax = allVals.length > 0 ? Math.max(...allVals) : 10;
    const maxVal = Math.ceil(rawMax * 1.15) || 5;
    const minVal = 0;

    // 繪製格線與 Y 軸刻度
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

    // Y 軸標題 (旋轉 90 度)
    ctx.save();
    ctx.translate(18, padTop + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = '#94a3b8';
    ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(title, 0, 0);
    ctx.restore();

    // X 軸座標點
    const nPoints = epochs.length;
    const getX = (idx) => nPoints === 1 ? padLeft + plotW / 2 : padLeft + (idx / (nPoints - 1)) * plotW;
    const getY = (v) => padTop + plotH - ((v - minVal) / (maxVal - minVal)) * plotH;

    // 繪製 X 軸刻度文字 (E1, E2, E3, E4...)
    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    epochs.forEach((ep, idx) => {
      // 若點數過多自動抽樣間隔顯示
      if (nPoints > 15 && idx % Math.ceil(nPoints / 10) !== 0 && idx !== nPoints - 1) return;
      const x = getX(idx);
      ctx.fillText(`E${ep}`, x, padTop + plotH + 12);
    });

    // X 軸標題
    ctx.fillStyle = '#64748b';
    ctx.font = '12px Inter, sans-serif';
    ctx.fillText(xLabel, padLeft + plotW / 2, height - 12);

    // 繪製漸層曲線函式 (平滑三次貝茲曲線 + 漸層填充)
    const drawSplineSeries = (data, strokeColor, fillColorStart, dotColor) => {
      if (!data || data.length === 0) return;

      const points = data.map((v, i) => ({ x: getX(i), y: getY(v) }));

      // 1. 繪製底部漸層區域
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

      // 2. 繪製平滑主折線
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

      // 3. 繪製發光節點小圓圈
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

    // 依序繪製 Train Loss (藍色) 與 Val Loss (綠色)
    drawSplineSeries(trainLoss, '#0284c7', 'rgba(2, 132, 199, 0.25)', '#38bdf8');
    drawSplineSeries(valLoss, '#86efac', 'rgba(134, 239, 172, 0.18)', '#4ade80');

  }, [epochs, trainLoss, valLoss, title, xLabel]);

  return (
    <div className="w-full h-80 relative flex flex-col">
      {/* 頂部圖例 (Legend) */}
      <div className="flex items-center justify-center space-x-6 mb-2">
        <div className="flex items-center space-x-2">
          <span className="w-4 h-2.5 rounded-sm border border-sky-400 bg-sky-500/30 inline-block"></span>
          <span className="text-xs font-semibold text-sky-300 tracking-wide">Train Loss</span>
        </div>
        <div className="flex items-center space-x-2">
          <span className="w-4 h-2.5 rounded-sm border border-emerald-400 bg-emerald-500/30 inline-block"></span>
          <span className="text-xs font-semibold text-emerald-300 tracking-wide">Val Loss</span>
        </div>
      </div>
      {/* 畫布容器 */}
      <div className="flex-1 w-full h-full relative">
        <canvas ref={canvasRef} className="w-full h-full block"></canvas>
      </div>
    </div>
  );
}


export default function App() {
  const [metricsData, setMetricsData] = useState({
    epochs: [],
    train_seg_loss: [],
    val_seg_loss: [],
    map50: [],
    map50_95: [],
    precision: [],
    recall: [],
    latest: {},
    training_status: { status: 'idle', pid: null, elapsed_seconds: 0 },
  });

  const [lastUpdate, setLastUpdate] = useState('--:--:--');
  const [imgTimestamp, setImgTimestamp] = useState(Date.now());
  const [actionMsg, setActionMsg] = useState({ text: '', type: 'info' });

  // 訓練設定參數
  const [trainConfig, setTrainConfig] = useState({
    epochs: 100,
    batch: 4,
    imgsz: 1024,
    freeze: 5,
  });

  // 輪詢 /api/metrics 接口
  const fetchData = async () => {
    try {
      const res = await fetch('/api/metrics');
      if (res.ok) {
        const data = await res.json();
        setMetricsData(data);
        setLastUpdate(new Date().toLocaleTimeString());
        setImgTimestamp(Date.now());
      }
    } catch (err) {
      console.error('Fetch error:', err);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 1500);
    return () => clearInterval(interval);
  }, []);

  const showToast = (text, type = 'info') => {
    setActionMsg({ text, type });
    setTimeout(() => setActionMsg({ text: '', type: 'info' }), 4000);
  };

  // 控制按鈕處理函式
  const handleStart = async () => {
    try {
      showToast('正在發送啟動訓練請求...', 'info');
      const res = await fetch('/api/train/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(trainConfig),
      });
      const data = await res.json();
      showToast(data.message, data.success ? 'success' : 'error');
      fetchData();
    } catch (err) {
      showToast(`連線失敗: ${err.message}`, 'error');
    }
  };

  const handlePause = async () => {
    try {
      showToast('正在發送暫停請求 (SIGSTOP)...', 'info');
      const res = await fetch('/api/train/pause', { method: 'POST' });
      const data = await res.json();
      showToast(data.message, data.success ? 'success' : 'error');
      fetchData();
    } catch (err) {
      showToast(`連線失敗: ${err.message}`, 'error');
    }
  };

  const handleResume = async () => {
    try {
      showToast('正在發送繼續請求 (SIGCONT)...', 'info');
      const res = await fetch('/api/train/resume', { method: 'POST' });
      const data = await res.json();
      showToast(data.message, data.success ? 'success' : 'error');
      fetchData();
    } catch (err) {
      showToast(`連線失敗: ${err.message}`, 'error');
    }
  };

  const handleStop = async () => {
    if (!window.confirm('確定要強制終止目前的訓練任務嗎？')) return;
    try {
      showToast('正在發送終止請求 (SIGTERM)...', 'info');
      const res = await fetch('/api/train/stop', { method: 'POST' });
      const data = await res.json();
      showToast(data.message, data.success ? 'success' : 'error');
      fetchData();
    } catch (err) {
      showToast(`連線失敗: ${err.message}`, 'error');
    }
  };

  const latest = metricsData.latest || {};
  const currentEpoch = latest.epoch || 0;
  const trainStatus = metricsData.training_status?.status || 'idle';
  const processPid = metricsData.training_status?.pid;

  const trainTotalLoss = latest.train_loss !== undefined ? latest.train_loss.toFixed(4) : (latest.train_seg_loss !== undefined ? latest.train_seg_loss.toFixed(4) : '--');
  const valTotalLoss = latest.val_loss !== undefined ? latest.val_loss.toFixed(4) : (latest.val_seg_loss !== undefined ? latest.val_seg_loss.toFixed(4) : '--');
  const maskMap50 = latest.map50 !== undefined ? (latest.map50 * 100).toFixed(1) + '%' : '--%';
  const maskMap95 = latest.map50_95 !== undefined ? (latest.map50_95 * 100).toFixed(1) + '%' : '--%';
  const precisionVal = latest.precision !== undefined ? (latest.precision * 100).toFixed(0) + '%' : '--%';
  const recallVal = latest.recall !== undefined ? (latest.recall * 100).toFixed(0) + '%' : '--%';

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* 頂部導覽列 */}
      <header className="border-b border-slate-800 bg-slate-900/60 sticky top-0 z-50 backdrop-blur-md px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center text-emerald-400 font-bold text-xl shadow-inner">
            🌲
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-base font-bold text-white tracking-wide">
                YOLO26-seg 即時訓練控制與動態監控
              </h1>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                React 18
              </span>
            </div>
            <p className="text-xs text-slate-400">
              SAM 3 偽標籤至 YOLO26 實例分割全自動知識蒸餾
            </p>
          </div>
        </div>

        {/* 狀態指示燈 */}
        <div className="flex items-center space-x-4">
          <div className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-full border text-xs font-mono ${
            trainStatus === 'running' 
              ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300' 
              : trainStatus === 'paused' 
              ? 'bg-amber-500/10 border-amber-500/30 text-amber-300' 
              : trainStatus === 'completed'
              ? 'bg-sky-500/10 border-sky-500/30 text-sky-300'
              : 'bg-slate-800 border-slate-700 text-slate-300'
          }`}>
            <span className={`w-2.5 h-2.5 rounded-full ${
              trainStatus === 'running' 
                ? 'bg-emerald-400 animate-ping' 
                : trainStatus === 'paused' 
                ? 'bg-amber-400' 
                : trainStatus === 'completed'
                ? 'bg-sky-400'
                : 'bg-slate-400'
            }`}></span>
            <span className="font-semibold capitalize">
              {trainStatus === 'running' && `訓練中 (PID: ${processPid})`}
              {trainStatus === 'paused' && `已暫停 (PID: ${processPid})`}
              {trainStatus === 'completed' && `訓練已完成`}
              {trainStatus === 'idle' && `等待指令 (Idle)`}
              {trainStatus === 'error' && `異常停止`}
            </span>
          </div>

          <div className="text-xs text-slate-400 font-mono hidden md:block">
            更新: {lastUpdate}
          </div>
        </div>
      </header>

      {/* 主體內容 */}
      <main className="max-w-7xl mx-auto px-6 py-6 flex-1 w-full space-y-6">
        
        {/* Toast 訊息列 */}
        {actionMsg.text && (
          <div className={`p-3 rounded-xl text-xs font-medium border flex items-center justify-between transition-all duration-300 ${
            actionMsg.type === 'success' 
              ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300' 
              : actionMsg.type === 'error'
              ? 'bg-red-500/10 border-red-500/30 text-red-300'
              : 'bg-sky-500/10 border-sky-500/30 text-sky-300'
          }`}>
            <span className="flex items-center space-x-2">
              <span>{actionMsg.type === 'success' ? '✅' : actionMsg.type === 'error' ? '❌' : 'ℹ️'}</span>
              <span>{actionMsg.text}</span>
            </span>
            <button onClick={() => setActionMsg({ text: '', type: 'info' })} className="text-slate-400 hover:text-white">✕</button>
          </div>
        )}

        {/* 🎮 控制面板 (Action Control Bar) */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-2xl backdrop-blur-xl">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
            
            {/* 左側：按鈕控制群組 */}
            <div className="flex items-center flex-wrap gap-3">
              {/* 1. 開始按鈕 */}
              <button
                onClick={handleStart}
                disabled={trainStatus === 'running' || trainStatus === 'paused'}
                className="px-5 py-2.5 rounded-xl font-semibold text-sm flex items-center space-x-2 shadow-lg transition-all duration-200 bg-emerald-600 hover:bg-emerald-500 active:scale-95 text-white disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-emerald-600 shadow-emerald-900/30"
              >
                <span>▶️</span>
                <span>開始訓練 (Start)</span>
              </button>

              {/* 2. 暫停 / 繼續按鈕 */}
              {trainStatus === 'paused' ? (
                <button
                  onClick={handleResume}
                  className="px-5 py-2.5 rounded-xl font-semibold text-sm flex items-center space-x-2 shadow-lg transition-all duration-200 bg-amber-600 hover:bg-amber-500 active:scale-95 text-white shadow-amber-900/30"
                >
                  <span>⏯️</span>
                  <span>繼續訓練 (Resume)</span>
                </button>
              ) : (
                <button
                  onClick={handlePause}
                  disabled={trainStatus !== 'running'}
                  className="px-5 py-2.5 rounded-xl font-semibold text-sm flex items-center space-x-2 shadow-lg transition-all duration-200 bg-amber-600/80 hover:bg-amber-500 active:scale-95 text-white disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-amber-600/80 shadow-amber-900/20"
                >
                  <span>⏸️</span>
                  <span>暫停 (Pause)</span>
                </button>
              )}

              {/* 3. 終止按鈕 */}
              <button
                onClick={handleStop}
                disabled={trainStatus === 'idle' || trainStatus === 'completed'}
                className="px-5 py-2.5 rounded-xl font-semibold text-sm flex items-center space-x-2 shadow-lg transition-all duration-200 bg-red-600 hover:bg-red-500 active:scale-95 text-white disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-red-600 shadow-red-900/30"
              >
                <span>⏹️</span>
                <span>終止訓練 (Stop)</span>
              </button>
            </div>

            {/* 右側：訓練參數輸入 */}
            <div className="flex items-center flex-wrap gap-3 bg-slate-950/70 px-4 py-2.5 rounded-xl border border-slate-800 text-xs font-mono">
              <div className="flex items-center space-x-1.5">
                <span className="text-slate-400">輪數 (Epochs):</span>
                <input
                  type="number"
                  value={trainConfig.epochs}
                  disabled={trainStatus === 'running' || trainStatus === 'paused'}
                  onChange={(e) => setTrainConfig({ ...trainConfig, epochs: parseInt(e.target.value) || 100 })}
                  className="w-14 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-white font-bold text-center focus:border-emerald-500 focus:outline-none disabled:opacity-50"
                />
              </div>

              <div className="flex items-center space-x-1.5">
                <span className="text-slate-400">尺寸 (ImgSz):</span>
                <input
                  type="number"
                  step="32"
                  value={trainConfig.imgsz}
                  disabled={trainStatus === 'running' || trainStatus === 'paused'}
                  onChange={(e) => setTrainConfig({ ...trainConfig, imgsz: parseInt(e.target.value) || 1024 })}
                  className="w-16 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-white font-bold text-center focus:border-emerald-500 focus:outline-none disabled:opacity-50"
                />
              </div>

              <div className="flex items-center space-x-1.5">
                <span className="text-slate-400">批次 (Batch):</span>
                <input
                  type="number"
                  value={trainConfig.batch}
                  disabled={trainStatus === 'running' || trainStatus === 'paused'}
                  onChange={(e) => setTrainConfig({ ...trainConfig, batch: parseInt(e.target.value) || 4 })}
                  className="w-12 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-white font-bold text-center focus:border-emerald-500 focus:outline-none disabled:opacity-50"
                />
              </div>

              <div className="flex items-center space-x-1.5">
                <span className="text-slate-400">凍結骨幹 (Freeze):</span>
                <input
                  type="number"
                  value={trainConfig.freeze}
                  disabled={trainStatus === 'running' || trainStatus === 'paused'}
                  onChange={(e) => setTrainConfig({ ...trainConfig, freeze: parseInt(e.target.value) || 5 })}
                  className="w-12 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-white font-bold text-center focus:border-emerald-500 focus:outline-none disabled:opacity-50"
                />
              </div>
            </div>

          </div>
        </div>

        {/* 🌟 核心：與截圖 100% 相同之暗黑平滑漸層動態折線圖 (Train/Val Loss vs. Epoch) */}
        <div className="bg-[#11161d] border border-slate-800/90 rounded-2xl p-6 shadow-2xl backdrop-blur-xl">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <span className="w-2.5 h-2.5 rounded-full bg-sky-400"></span>
              <h2 className="text-sm font-bold text-slate-100 tracking-wide">
                訓練與驗證損失收斂圖 (Loss vs. Epochs)
              </h2>
            </div>
            <span className="text-xs text-slate-400 font-mono">
              實時動態渲染 • {metricsData.epochs.length} Epochs
            </span>
          </div>

          <LossLineChart 
            epochs={metricsData.epochs}
            trainLoss={metricsData.train_loss || metricsData.train_seg_loss}
            valLoss={metricsData.val_loss || metricsData.val_seg_loss}
            title="Total Loss"
            xLabel="Epoch"
          />
        </div>

        {/* KPI 數值卡片矩陣 */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 shadow-lg">
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              當前輪數 (Epoch)
            </div>
            <div className="mt-2 flex items-baseline justify-between">
              <div className="text-3xl font-black font-mono text-white">
                {currentEpoch > 0 ? currentEpoch : '--'}
              </div>
              <span className="text-xs font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                {currentEpoch > 0 ? `${((currentEpoch / trainConfig.epochs) * 100).toFixed(0)}%` : 'Ready'}
              </span>
            </div>
            <div className="mt-3 w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
              <div 
                className="bg-emerald-500 h-full rounded-full transition-all duration-500"
                style={{ width: `${Math.min(100, (currentEpoch / trainConfig.epochs) * 100)}%` }}
              ></div>
            </div>
          </div>

          <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 shadow-lg">
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              總損失 (Total Loss)
            </div>
            <div className="mt-2 flex items-baseline justify-between">
              <div className="text-3xl font-black font-mono text-amber-400">
                {trainTotalLoss}
              </div>
              <span className="text-xs font-mono text-slate-400">
                Val: {valTotalLoss}
              </span>
            </div>
            <div className="mt-2 text-[11px] text-slate-400">
              📉 綜合訓練與驗證加總誤差 (越小越準)
            </div>
          </div>

          <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 shadow-lg">
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              遮罩精度 (Mask mAP50)
            </div>
            <div className="mt-2 flex items-baseline justify-between">
              <div className="text-3xl font-black font-mono text-emerald-400">
                {maskMap50}
              </div>
              <span className="text-xs font-mono text-slate-400">
                mAP95: {maskMap95}
              </span>
            </div>
            <div className="mt-2 text-[11px] text-emerald-400/80">
              🎯 主樹 / 周邊樹 綜合分割精度
            </div>
          </div>

          <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 shadow-lg">
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              查準率 / 召回率 (P / R)
            </div>
            <div className="mt-2 flex items-baseline justify-between">
              <div className="text-3xl font-black font-mono text-sky-400">
                {precisionVal} / {recallVal}
              </div>
            </div>
            <div className="mt-2 text-[11px] text-slate-400">
              🔍 Precision (防誤報) vs Recall (防漏報)
            </div>
          </div>
        </div>

        {/* 視覺化真值 vs 預測對照面板 */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center space-x-3">
              <h2 className="text-sm font-bold text-white flex items-center space-x-2">
                <span className="w-2.5 h-2.5 rounded-full bg-sky-400"></span>
                <span>即時樹葉分割預測比對 (Live Proof Panel)</span>
              </h2>
              <span className="text-xs bg-emerald-500/10 text-emerald-400 px-2 py-0.5 rounded border border-emerald-500/20 font-medium">
                綠色: 主體樹 (Class 0) | 藍色: 周邊樹 (Class 1)
              </span>
            </div>
            
            <button 
              onClick={fetchData}
              className="text-xs bg-slate-800 hover:bg-slate-700 text-slate-200 px-3 py-1.5 rounded-lg border border-slate-700 transition flex items-center space-x-1"
            >
              <span>🔄 即時刷新</span>
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* AI 預測圖 */}
            <div className="space-y-2">
              <div className="text-xs font-semibold text-slate-300 flex items-center space-x-1.5">
                <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
                <span>AI 實例分割預測 (val_batch0_pred.jpg)</span>
              </div>
              <div className="rounded-xl overflow-hidden bg-slate-950 border border-slate-800 aspect-video flex items-center justify-center p-1">
                <img 
                  src={`/api/image/val_batch0_pred.jpg?t=${imgTimestamp}`} 
                  alt="AI 預測圖" 
                  className="w-full h-full object-contain rounded-lg"
                />
              </div>
            </div>

            {/* SAM 3 真值圖 */}
            <div className="space-y-2">
              <div className="text-xs font-semibold text-slate-300 flex items-center space-x-1.5">
                <span className="w-2 h-2 rounded-full bg-sky-400"></span>
                <span>SAM 3 偽標籤真值 (val_batch0_labels.jpg)</span>
              </div>
              <div className="rounded-xl overflow-hidden bg-slate-950 border border-slate-800 aspect-video flex items-center justify-center p-1">
                <img 
                  src={`/api/image/val_batch0_labels.jpg?t=${imgTimestamp}`} 
                  alt="標籤真值圖" 
                  className="w-full h-full object-contain rounded-lg"
                />
              </div>
            </div>
          </div>
        </div>

      </main>

      {/* 底部 Footer */}
      <footer className="border-t border-slate-800 py-3 text-center text-xs text-slate-500 font-mono">
        Ultralytics YOLO26-seg Distillation Pipeline • Antigravity MLOps
      </footer>
    </div>
  );
}
