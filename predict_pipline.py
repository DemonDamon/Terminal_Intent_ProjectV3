import sys
import os

# --- 路径修复块 (确保能导入src模块) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.pipeline.predict_common import run_prediction_pipeline


def run_prediction():
    """标准预测 Pipeline：生成营销名单"""
    search_paths = [
        'predict_data.csv',
        'data/raw/predict_data_10.csv',
        'data/raw/predict_data.csv',
    ]
    save_path = 'data/processed/marketing_list.csv'

    run_prediction_pipeline(search_paths, save_path)


if __name__ == "__main__":
    run_prediction()
