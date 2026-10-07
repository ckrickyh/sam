import os
import sys
from pathlib import Path
import gradio as gr
import uvicorn

# 將專案根目錄加入路徑
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# 引入現有的 FastAPI app
from samplingAnalysis.app import app as fastapi_app

# 建立極簡 Gradio Blocks 包裝，確保通過 HF Space 的健康檢查
with gr.Blocks(title="樹冠密度與孔隙分析引擎") as demo:
    pass

# 將 FastAPI 與 Gradio 結合
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    # Space 預設讀取 7860 Port
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port)
