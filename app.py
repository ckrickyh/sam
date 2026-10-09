try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func=None, *args, **kwargs):
            if func is not None and callable(func):
                return func
            def decorator(f):
                return f
            return decorator
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

from samplingAnalysis.adaptive_cielab_otsu_foliage_extractor_fixBranchlet_order_maxGapRatio_ExG_box import (
    process_image_cielab_adaptive,
    get_optimal_device,
    build_sam3_image_model,
    Sam3Processor
)

DEFAULT_CHECKPOINT = "sam3.pt"
DEFAULT_MODEL_REPO = "ckrickydevs/sam3-weights"
GLOBAL_PROCESSOR = None

def ensure_model_checkpoint(checkpoint_path: str = DEFAULT_CHECKPOINT) -> str:
    """確認本機是否有模型權重，若無則從 Hugging Face Hub 自動下載"""
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    repo_id = os.environ.get("HF_MODEL_REPO", DEFAULT_MODEL_REPO)
    token = os.environ.get("HF_TOKEN_READ") or os.environ.get("HF_TOKEN")
    print(f"正在從 Hugging Face Hub ({repo_id}) 下載權重 {checkpoint_path}...")
    return hf_hub_download(repo_id=repo_id, filename=checkpoint_path, local_dir=".", token=token)

# 服務啟動時預先確認/下載模型權重至本地快取，避免 ZeroGPU 請求超時
try:
    ensure_model_checkpoint()
except Exception as e:
    print(f"[Warning] 啟動預載入模型失敗，將於推論請求時重試: {e}")

def get_processor(checkpoint_path: str = DEFAULT_CHECKPOINT) -> Sam3Processor:
    global GLOBAL_PROCESSOR
    if GLOBAL_PROCESSOR is None:
        valid_checkpoint_path = ensure_model_checkpoint(checkpoint_path)
        device = get_optimal_device()
        print(f"正在初始化 SAM 3 模型，運算裝置: {device}...")
        model = build_sam3_image_model(checkpoint_path=valid_checkpoint_path, device=device)
        if device == "cuda":
            import torch
            model = model.to(torch.bfloat16)
        
        GLOBAL_PROCESSOR = Sam3Processor(model, device=device)
        
        if device == "cuda":
            import torch
            # Monkey-patch 1: 強制將輸入影像轉換為 bfloat16 避免 Conv2d 報錯
            orig_forward = GLOBAL_PROCESSOR.model.backbone.forward_image
            def forward_image_bf16(image, *args, **kwargs):
                return orig_forward(image.to(torch.bfloat16), *args, **kwargs)
            GLOBAL_PROCESSOR.model.backbone.forward_image = forward_image_bf16
            
            # Monkey-patch 2: 強制將 SAM1 幾何方框/點提示轉換為 bfloat16 避免 PromptEncoder 報錯
            if hasattr(GLOBAL_PROCESSOR.model, "inst_interactive_predictor") and GLOBAL_PROCESSOR.model.inst_interactive_predictor:
                orig_prompt = GLOBAL_PROCESSOR.model.inst_interactive_predictor.model.sam_prompt_encoder.forward
                def prompt_bf16(points, boxes, masks, *args, **kwargs):
                    if points is not None:
                        points = (points[0].to(torch.bfloat16), points[1])
                    if boxes is not None:
                        boxes = boxes.to(torch.bfloat16)
                    if masks is not None:
                        masks = masks.to(torch.bfloat16)
                    return orig_prompt(points, boxes, masks, *args, **kwargs)
                GLOBAL_PROCESSOR.model.inst_interactive_predictor.model.sam_prompt_encoder.forward = prompt_bf16
                
            # Monkey-patch 3: 強制將 SAM3 自身的 Geometry Encoder 提示轉換為 bfloat16
            if hasattr(GLOBAL_PROCESSOR.model, "geometry_encoder") and GLOBAL_PROCESSOR.model.geometry_encoder:
                orig_geo_forward = GLOBAL_PROCESSOR.model.geometry_encoder.forward
                def geo_forward_bf16(geo_prompt, *args, **kwargs):
                    if geo_prompt.point_embeddings is not None: geo_prompt.point_embeddings = geo_prompt.point_embeddings.to(torch.bfloat16)
                    if geo_prompt.box_embeddings is not None: geo_prompt.box_embeddings = geo_prompt.box_embeddings.to(torch.bfloat16)
                    if geo_prompt.mask_embeddings is not None: geo_prompt.mask_embeddings = geo_prompt.mask_embeddings.to(torch.bfloat16)
                    return orig_geo_forward(geo_prompt, *args, **kwargs)
                GLOBAL_PROCESSOR.model.geometry_encoder.forward = geo_forward_bf16
                
                # Monkey-patch 4: 強制將內部 Float32 的位置編碼轉回 bfloat16 避免後續 Linear 報錯
                if hasattr(GLOBAL_PROCESSOR.model.geometry_encoder, "points_pos_enc_project") and GLOBAL_PROCESSOR.model.geometry_encoder.points_pos_enc_project:
                    orig_pts_proj = GLOBAL_PROCESSOR.model.geometry_encoder.points_pos_enc_project.forward
                    def pts_proj_bf16(x, *args, **kwargs):
                        return orig_pts_proj(x.to(torch.bfloat16), *args, **kwargs)
                    GLOBAL_PROCESSOR.model.geometry_encoder.points_pos_enc_project.forward = pts_proj_bf16
                    
                if hasattr(GLOBAL_PROCESSOR.model.geometry_encoder, "boxes_pos_enc_project") and GLOBAL_PROCESSOR.model.geometry_encoder.boxes_pos_enc_project:
                    orig_box_proj = GLOBAL_PROCESSOR.model.geometry_encoder.boxes_pos_enc_project.forward
                    def box_proj_bf16(x, *args, **kwargs):
                        return orig_box_proj(x.to(torch.bfloat16), *args, **kwargs)
                    GLOBAL_PROCESSOR.model.geometry_encoder.boxes_pos_enc_project.forward = box_proj_bf16
                    
            # Monkey-patch 5: torchvision.ops.roi_align 內部寫死了 .float() 導致格式衝突，我們強制同步型態
            import torchvision
            if hasattr(torchvision.ops, "roi_align"):
                orig_roi_align = torchvision.ops.roi_align
                def roi_align_matched(input, rois, *args, **kwargs):
                    if isinstance(rois, torch.Tensor):
                        rois = rois.to(input.dtype)
                    elif isinstance(rois, (list, tuple)):
                        rois = type(rois)(r.to(input.dtype) if isinstance(r, torch.Tensor) else r for r in rois)
                    return orig_roi_align(input, rois, *args, **kwargs)
                torchvision.ops.roi_align = roi_align_matched
                
            # Monkey-patch 6: 攔截 SAM 3 主模型的 Transformer Encoder/Decoder，確保合併後的 prompt 被降轉回 bfloat16
            if hasattr(GLOBAL_PROCESSOR.model, "_run_encoder"):
                orig_run_enc = GLOBAL_PROCESSOR.model._run_encoder
                def run_enc_bf16(backbone_out, prompt, *args, **kwargs):
                    if isinstance(prompt, torch.Tensor): prompt = prompt.to(torch.bfloat16)
                    return orig_run_enc(backbone_out, prompt, *args, **kwargs)
                GLOBAL_PROCESSOR.model._run_encoder = run_enc_bf16
                
            if hasattr(GLOBAL_PROCESSOR.model, "_run_decoder"):
                orig_run_dec = GLOBAL_PROCESSOR.model._run_decoder
                def run_dec_bf16(backbone_out, encoder_out, prompt, *args, **kwargs):
                    if isinstance(prompt, torch.Tensor): prompt = prompt.to(torch.bfloat16)
                    return orig_run_dec(backbone_out, encoder_out, prompt, *args, **kwargs)
                GLOBAL_PROCESSOR.model._run_decoder = run_dec_bf16
                
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

@spaces.GPU(duration=120)
def run_analysis_gradio(img_filepath, canopy_box_str, trunk_box_str, conf_thresh, neg_thresh, max_gap_ratio, exg_thresh):
    if not img_filepath:
        return None, "請先上傳圖片"

    try:
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
            max_gap_ratio=max_gap_ratio,
            exg_threshold=exg_thresh,
            output_dir=out_dir,
        )

        stem = Path(img_filepath).stem
        result_img_path = out_dir / f"{stem}_cielab_otsu_render.png"
        img_out_val = str(result_img_path) if result_img_path.exists() else None

        import json
        return img_out_val, json.dumps(metrics, indent=2, ensure_ascii=False)
    except Exception as e:
        import traceback
        err_detail = f"分析執行失敗：{str(e)}\n\n詳細日誌：\n{traceback.format_exc()}"
        print(err_detail)
        return None, err_detail

with gr.Blocks(title="Crown Porosity") as demo:
    gr.Markdown("# 🌳 Crown Porosity")
    
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="filepath", label="上傳樹木圖片")
            canopy_box = gr.Textbox(label="樹冠邊界框 (Canopy Box)", placeholder="例如: 100, 100, 500, 500 (可留空)")
            trunk_box = gr.Textbox(label="樹幹邊界框 (Trunk Box)", placeholder="例如: 200, 400, 300, 600 (可留空)")
            conf_thresh = gr.Slider(0.0, 1.0, value=0.25, label="SAM 3 Threshold")
            neg_thresh = gr.Slider(0.0, 1.0, value=0.155, label="Negative Threshold")
            max_gap_ratio_slider = gr.Slider(0.0, 1.0, value=0.05, label="Max Gap Ratio inside Crown")
            exg_thresh_slider = gr.Slider(0.0, 0.05, step=0.001, value=0.015, label="Green Leaf ExG Threshold")
            btn = gr.Button("Start Analysis (ZeroGPU)", variant="primary")
        
        with gr.Column():
            img_out = gr.Image(label="Analysis Results 4 Panels")
            metrics_out = gr.Textbox(label="Quantitative indicators (JSON)", lines=15)

    btn.click(
        fn=run_analysis_gradio,
        inputs=[img_in, canopy_box, trunk_box, conf_thresh, neg_thresh, max_gap_ratio_slider, exg_thresh_slider],
        outputs=[img_out, metrics_out]
    )

if __name__ == "__main__":
    import os
    # 修正 Gradio 內部檢查 localhost 失敗的 bug
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
    
    # 必須呼叫 launch 才能觸發 ZeroGPU 的代理伺服器
    demo.launch(server_name="0.0.0.0")

