import spaces
import os
import sys
from pathlib import Path
import gradio as gr
import uvicorn

@spaces.GPU
def _dummy_gpu_function():
    pass

# 將專案根目錄加入路徑
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# 直接引入現有的 FastAPI app 作為進入點
from samplingAnalysis.app import app as fastapi_app

# 必須宣告為全域變數 demo，讓 Hugging Face ZeroGPU 的 AST / Registry 掃描能抓到
with gr.Blocks(title="樹冠密度與孔隙分析引擎") as demo:
    gr.Markdown("此為系統 API 與靜態頁面底層 (Gradio Wrapper for ZeroGPU)")
    btn = gr.Button("Dummy GPU Trigger", visible=False)
    # 綁定事件是通過 ZeroGPU 檢查的唯一條件
    btn.click(fn=_dummy_gpu_function, inputs=[], outputs=[])

# 將 FastAPI 與 Gradio 結合
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    # 啟動 Uvicorn 保持執行緒存活
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
