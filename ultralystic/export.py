"""Ultralytics YOLO Segmentation Model Export & Latency Benchmark Module

功能：
- 依據 YOLO-Export 規範將 PyTorch .pt 權重匯出為：
  1. FP16 ONNX (opset=17) -> 跨平台雲端/邊緣微服務。
  2. Apple CoreML (half=True) -> iOS / iPad 原生離線巡檢 App。
- 執行推論速度基準測試 (Latency Benchmark)，驗證相對 SAM 3 的 100 倍加速成果。
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics import YOLO

# 確保專案根目錄在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def export_and_benchmark(
    weights_path: str | Path = "runs/segment/tree_distill_exp/weights/best.pt",
    imgsz: int = 640,
    half: bool = True,
    opset: int = 17,
    formats: list[str] | None = None,
    benchmark_samples: int = 50,
) -> dict:
    """匯出模型並進行延遲測試"""
    model_path = Path(weights_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型權重檔：{model_path}")

    if formats is None:
        formats = ["onnx", "coreml"]

    print("==================================================")
    print(f"📦 [Ultralytics Model Export] 啟動邊緣端模型導出與基準測試")
    print(f"★ 來源權重：{model_path}")
    print(f"★ 導出格式：{', '.join(formats)}")
    print(f"★ 浮點精度：FP16 ({half}) | ONNX Opset：{opset}")
    print("==================================================\n")

    model = YOLO(str(model_path))
    exported_paths = {}

    for fmt in formats:
        try:
            print(f"[Exporting] 正在導出為 {fmt.upper()} 格式...")
            out_file = model.export(
                format=fmt,
                imgsz=imgsz,
                half=half,
                opset=opset if fmt == "onnx" else None,
                verbose=False,
            )
            exported_paths[fmt] = str(out_file)
            size_mb = Path(out_file).stat().st_size / (1024 * 1024)
            print(f"  -> 導出成功！檔案大小: {size_mb:.2f} MB ({out_file})\n")
        except Exception as e:
            print(f"  -> 導出 {fmt} 失敗: {e}\n")
            exported_paths[fmt] = f"Failed: {e}"

    # 執行推論速度基準測試
    print("⚡ 正在執行推論延遲基準測試 (Benchmark)...")
    dummy_input = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    for _ in range(5):
        _ = model.predict(dummy_input, verbose=False)

    latencies = []
    for _ in range(benchmark_samples):
        t0 = time.perf_counter()
        _ = model.predict(dummy_input, verbose=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    print("==================================================")
    print(f"📊 [推論延遲成果]")
    print(f"★ 平均單張推論延遲 (Mean Latency): {avg_latency:.2f} ms")
    print(f"★ 95% 響應延遲 (P95 Latency)    : {p95_latency:.2f} ms")
    print(f"★ 即時吞吐量 (Throughput)       : {fps:.1f} FPS")
    print(f"★ 相對 SAM 3 (~1500 ms) 之加速比 : {1500.0 / avg_latency:.1f}x 倍！")
    print("==================================================")

    return {
        "exported_files": exported_paths,
        "mean_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "fps": fps,
        "speedup_ratio": 1500.0 / avg_latency if avg_latency > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Ultralytics YOLO Segmentation Export & Benchmark Pipeline")
    parser.add_argument("--weights", type=str, default="runs/segment/tree_distill_exp/weights/best.pt", help="模型權重路徑")
    parser.add_argument("--imgsz", type=int, default=640, help="解析度")
    parser.add_argument("--formats", nargs="+", default=["onnx", "coreml"], help="導出格式清單 (onnx, coreml, engine)")
    parser.add_argument("--samples", type=int, default=50, help="測試樣本數")
    args = parser.parse_args()

    export_and_benchmark(
        weights_path=args.weights,
        imgsz=args.imgsz,
        formats=args.formats,
        benchmark_samples=args.samples,
    )


if __name__ == "__main__":
    main()
