import os
import sys
from pathlib import Path
import uvicorn

# 將專案根目錄加入路徑
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# 直接引入現有的 FastAPI app 作為進入點
from samplingAnalysis.app import app

if __name__ == "__main__":
    # 啟動 Uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
