import spaces
import os
import sys
from pathlib import Path
import tempfile
import gradio as gr
from huggingface_hub import hf_hub_download

# 將專案根目錄加入路徑
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from samplingAnalysis.adaptive_cielab_otsu_foliage_extractor_fixBranchlet_order_box import (
    process_image_cielab_adaptive,
    get_optimal_device,
    build_sam3_image_model,
    Sam3Processor
)

DEFAULT_CHECKPOINT = "sam3.pt"
GLOBAL_PROCESSOR = None

def ensure_model_checkpoint(checkpoint_path: str = DEFAULT_CHECKPOINT) -> str:
    """確認本機是否有模型權重，若無則從 Hugging Face Hub 自動下載"""
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    repo_id = os.environ.get("HF_MODEL_REPO")
    if not repo_id:
        raise ValueError(f"未找到 {checkpoint_path}，且未設定 HF_MODEL_REPO")
    token = os.environ.get("HF_TOKEN_READ") or os.environ.get("HF_TOKEN")
    return hf_hub_download(repo_id=repo_id, filename=checkpoint_path, local_dir=".", token=token)

def get_processor(checkpoint_path: str = DEFAULT_CHECKPOINT) -> Sam3Processor:
    global GLOBAL_PROCESSOR
    if GLOBAL_PROCESSOR is None:
        valid_checkpoint_path = ensure_model_checkpoint(checkpoint_path)
        device = get_optimal_device()
        model = build_sam3_image_model(checkpoint_path=valid_checkpoint_path, device=device)
        GLOBAL_PROCESSOR = Sam3Processor(model)
    return GLOBAL_PROCESSOR

def parse_box_str(box_str):
    if not box_str or not box_str.strip():
        return None
    try:
        parts = [float(x.strip()) for x in box_str.split(',')]
        if len(parts) == 4:
            return parts
    except Exception:
        pass
    return None

@spaces.GPU
def run_analysis_gradio(img_filepath, canopy_box_str, trunk_box_str, conf_thresh, neg_thresh):
    if not img_filepath:
        return None, "請先上傳圖片"

    processor = get_processor()
    canopy_parsed = parse_box_str(canopy_box_str)
    trunk_parsed = parse_box_str(trunk_box_str)

    out_dir = root_dir / "samplingAnalysis" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = process_image_cielab_adaptive(
        image_path=img_filepath,
        processor=processor,
        canopy_box=canopy_parsed,
        trunk_box=trunk_parsed,
        confidence_threshold=conf_thresh,
        negative_threshold=neg_thresh,
        output_dir=out_dir,
    )

    stem = Path(img_filepath).stem
    result_img_path = out_dir / f"{stem}_cielab_otsu_render.png"

    import json
    return str(result_img_path), json.dumps(metrics, indent=2, ensure_ascii=False)

with gr.Blocks(title="樹冠密度與孔隙分析引擎") as demo:
    gr.Markdown("# 🌳 樹冠密度與孔隙分析引擎 (純 Gradio + ZeroGPU)")
    gr.Markdown("此版本完全拋棄 FastAPI 框架，改用 Gradio 直接銜接分析引擎核心，以保證與 Hugging Face ZeroGPU 的 100% 相容性。")
    
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="filepath", label="上傳樹木圖片")
            canopy_box = gr.Textbox(label="樹冠邊界框 (Canopy Box)", placeholder="例如: 100, 100, 500, 500 (可留空)")
            trunk_box = gr.Textbox(label="樹幹邊界框 (Trunk Box)", placeholder="例如: 200, 400, 300, 600 (可留空)")
            conf_thresh = gr.Slider(0.0, 1.0, value=0.25, label="SAM 3 信心門檻")
            neg_thresh = gr.Slider(0.0, 1.0, value=0.155, label="Negative Threshold")
            btn = gr.Button("開始分析 (ZeroGPU)", variant="primary")
        
        with gr.Column():
            img_out = gr.Image(label="分析結果四大面板")
            metrics_out = gr.Textbox(label="量化指標 (JSON)", lines=15)

    btn.click(
        fn=run_analysis_gradio,
        inputs=[img_in, canopy_box, trunk_box, conf_thresh, neg_thresh],
        outputs=[img_out, metrics_out]
    )

