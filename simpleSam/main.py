import sys
from pathlib import Path

# 確保專案根目錄在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from simpleSam import segment_subcanopy


def run():
    print("啟動 simpleSam 次冠層 (Subcanopy) 分割任務...")
    results = segment_subcanopy(
        image_path="data/test01.jpeg",
        checkpoint_path="sam3.pt",
        prompt="tree subcanopy",
        confidence_threshold=0.25,
        output_dir="simpleSam/output",
    )
    print("\n任務執行成功！輸出資訊：")
    print(f"- 檢測數量：{results['detection_count']}")
    print(f"- 覆蓋面積：{results['subcanopy_pixels']:,} px ({results['coverage_percentage']:.2f}%)")
    print(f"- 標記圖檔：{results['overlay_output_path']}")
    print(f"- 遮罩圖檔：{results['mask_output_path']}")


if __name__ == "__main__":
    run()
