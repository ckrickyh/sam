import spaces
import os
import sys
from pathlib import Path
import uvicorn
import gradio as gr

@spaces.GPU
def _dummy_gpu():
    pass

# 將專案根目錄加入路徑
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# 引入現有的 FastAPI app
from samplingAnalysis.app import app as fastapi_app

with gr.Blocks() as demo:
    btn = gr.Button("GPU")
    btn.click(_dummy_gpu)

app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    # 啟動 Uvicorn，傳入字串讓 ASGI 正確解析
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
