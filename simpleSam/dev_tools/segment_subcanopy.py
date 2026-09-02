"""SAM 3 Subcanopy Segmentation Module

使用 SAM 3 (Segment Anything Model 3) 針對樹木影像進行次冠層 / 子樹冠 (Subcanopy) 辨識與語意分割。
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def get_optimal_device() -> str:
    """自動偵測運算裝置 (優先順序: CUDA -> CPU)"""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def segment_subcanopy(
    image_path: str | Path,
    checkpoint_path: str | Path = "sam3.pt",
    prompt: str = "tree subcanopy",
    confidence_threshold: float = 0.25,
    output_dir: str | Path = "simpleSam/output",
    device: str | None = None,
) -> dict:
    """使用 SAM 3 進行 Subcanopy 分割與視覺化輸出

    Args:
        image_path: 輸入影像路徑
        checkpoint_path: SAM 3 模型權重檔案路徑 (.pt)
        prompt: 文字提示詞 (預設: "tree subcanopy")
        confidence_threshold: 置信度門檻值
        output_dir: 結果輸出目錄
        device: 運算裝置 ('cuda' 或 'cpu')

    Returns:
        包含分割統計與輸出路徑的字典
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"找不到輸入影像：{image_path}")

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到模型權重：{checkpoint_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = get_optimal_device()

    print(f"[Device] 使用運算裝置：{device}")
    print(f"[Model] 正在載入 SAM 3 模型 ({ckpt_path})...")

    model = build_sam3_image_model(checkpoint_path=str(ckpt_path), device=device)
    model = model.to(device).float()
    processor = Sam3Processor(model, device=device, confidence_threshold=confidence_threshold)

    print(f"[Image] 讀取影像：{img_path}")
    raw_image = Image.open(img_path).convert("RGB")
    image_np = np.array(raw_image)
    h, w = image_np.shape[:2]
    total_pixels = h * w

    print(f"[Inference] 提示詞：'{prompt}' | 門檻值：{confidence_threshold}")
    state = processor.set_image(raw_image)
    result = processor.set_text_prompt(prompt=prompt, state=state)

    masks = result.get("masks", None)
    boxes = result.get("boxes", None)
    scores = result.get("scores", None)

    num_detections = len(scores) if scores is not None else 0
    print(f"[Detection] 檢測到 {num_detections} 個 Subcanopy 候選區域。")

    # 合併所有檢測到的 subcanopy 遮罩
    combined_mask = np.zeros((h, w), dtype=bool)
    mask_list = []

    if masks is not None and num_detections > 0:
        masks_np = masks.cpu().numpy().astype(bool)
        if masks_np.ndim == 4:
            masks_np = masks_np.squeeze(1)
        for i in range(num_detections):
            m = masks_np[i]
            mask_list.append(m)
            combined_mask |= m

    subcanopy_pixels = int(np.sum(combined_mask))
    coverage_percentage = (subcanopy_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    # 渲染視覺化圖層
    overlay_img = image_np.copy()

    # 隨機或固定色板標註各個個別 subcanopy 區域
    palette = [
        (30, 200, 80),   # 翠綠
        (50, 160, 240),  # 天藍
        (255, 180, 0),   # 金黃
        (230, 60, 120),  # 洋紅
        (160, 90, 240),  # 紫色
        (0, 230, 200),   # 青色
    ]

    for idx, m in enumerate(mask_list):
        color = palette[idx % len(palette)]
        overlay_img[m] = (overlay_img[m] * 0.4 + np.array(color) * 0.6).astype(np.uint8)

        # 繪製邊界
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_img, contours, -1, color=(255, 255, 255), thickness=2)

        # 標註分數
        if boxes is not None and idx < len(boxes):
            box = boxes[idx].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            score_val = float(scores[idx].item()) if scores is not None else 0.0
            label = f"Subcanopy #{idx + 1}: {score_val:.2f}"
            cv2.rectangle(overlay_img, (x1, y1), (x2, y2), color=color, thickness=2)
            cv2.putText(
                overlay_img,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    # 儲存輸出圖檔
    stem = img_path.stem
    overlay_output_path = out_dir / f"{stem}_subcanopy_overlay.png"
    mask_output_path = out_dir / f"{stem}_subcanopy_mask.png"

    Image.fromarray(overlay_img).save(overlay_output_path)
    Image.fromarray((combined_mask * 255).astype(np.uint8)).save(mask_output_path)

    print("========================================")
    print(f"影像解析度：{w} x {h} ({total_pixels:,} px)")
    print(f"Subcanopy 偵測數量：{num_detections}")
    print(f"Subcanopy 總覆蓋面積：{subcanopy_pixels:,} px ({coverage_percentage:.2f}%)")
    print(f"視覺化結果：{overlay_output_path}")
    print(f"純遮罩圖檔：{mask_output_path}")
    print("========================================")

    return {
        "image_shape": (h, w),
        "total_pixels": total_pixels,
        "detection_count": num_detections,
        "subcanopy_pixels": subcanopy_pixels,
        "coverage_percentage": coverage_percentage,
        "overlay_output_path": str(overlay_output_path),
        "mask_output_path": str(mask_output_path),
    }


def main():
    parser = argparse.ArgumentParser(description="使用 SAM 3 進行樹木次冠層 (Subcanopy) 分割")
    parser.add_argument("--image", type=str, default="data/test02.jpeg", help="輸入影像路徑")
    parser.add_argument("--checkpoint", type=str, default="sam3.pt", help="SAM 3 權重路徑")
    parser.add_argument("--prompt", type=str, default="tree subcanopy", help="文字提示詞")
    parser.add_argument("--threshold", type=float, default=0.25, help="置信度門檻值")
    parser.add_argument("--output-dir", type=str, default="simpleSam/output", help="輸出目錄")
    args = parser.parse_args()

    segment_subcanopy(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        confidence_threshold=args.threshold,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
