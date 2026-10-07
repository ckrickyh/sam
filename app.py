import spaces
import os
import sys
from pathlib import Path

# 將專案根目錄加入路徑
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# 直接引入現有的 FastAPI app 作為進入點
from samplingAnalysis.app import app

# 在最外層宣告一個假的 GPU 函式，確保 AST 解析器能看到
@spaces.GPU
def _dummy_gpu():
    pass

# 不要呼叫 uvicorn.run()，讓 Hugging Face Gradio SDK 自動尋找並啟動名為 app 的 FastAPI 實例
