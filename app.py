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

def create_app():
    # 建立極簡 Gradio Blocks 包裝，隱藏於區域變數避免被 HF auto-launch
    with gr.Blocks(title="樹冠密度與孔隙分析引擎") as blocks:
        gr.Markdown("此為系統 API 與靜態頁面底層")
    
    # 將 FastAPI 與 Gradio 結合
    return gr.mount_gradio_app(fastapi_app, blocks, path="/gradio")

app = create_app()

if __name__ == "__main__":
    # 必須在此啟動 uvicorn，否則腳本執行到底就會 exit 0
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
