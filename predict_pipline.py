import sys
import os
import pandas as pd
import joblib
import numpy as np

# --- 1. 路径修复块 (确保能导入src模块) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.data.loader import DataLoader
from src.features.unified_fe import UnifiedFeatureEngineer
from config.feature_list import CAT_FEATURES

def load_optimal_threshold(model_obj):
    """
    智能阈值加载逻辑：
    1. 优先读取模型对象内部绑定的 best_threshold (最准确)
    2. 其次尝试读取外部配置文件 (兼容旧版)
    3. 最后使用业务默认值 0.5 (兜底)
    """
    # 策略 1: 模型内部属性
    if hasattr(model_obj, 'best_threshold'):
        print(f"[Smart Load] 从模型内部属性加载最佳阈值: {model_obj.best_threshold:.3f}")
        return model_obj.best_threshold

    # 策略 2: 外部文件
    threshold_path = 'models/optimal_threshold.pkl'
    if os.path.exists(threshold_path):
        try:
            threshold_config = joblib.load(threshold_path)
            if isinstance(threshold_config, dict):
                best_t = threshold_config.get('best_threshold', 0.5)
            else:
                best_t = float(threshold_config)
            print(f"[Smart Load] 模型内无阈值，从外部文件加载: {best_t:.3f}")
            return best_t
        except:
            pass
    
    # 策略 3: 兜底默认值
    print("[Smart Load] 未找到配置，使用业务默认阈值: 0.500")
    return 0.5

def run_prediction():
    # --- 1. 环境准备与模型加载 ---
    model_path = 'models/intent_v2.pkl'
    feature_names_path = 'models/feature_names.pkl'

    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}，请先运行 train_pipline.py")
        return

    print(">>> 正在加载集成模型及编码规则...")
    try:
        model_obj = joblib.load(model_path)
        feature_names = joblib.load(feature_names_path)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 加载阈值
    threshold = load_optimal_threshold(model_obj)

    # --- 2. 数据加载与清洗 ---
    input_file = 'predict_data.csv'  # 默认在根目录查找
    if not os.path.exists(input_file):
        input_file = 'data/raw/predict_data_10.csv' # 尝试查找前面生成的10月预测数据
        if not os.path.exists(input_file):
             # 再尝试找通用预测数据
            input_file = 'data/raw/predict_data.csv'
            if not os.path.exists(input_file):
                print(f"错误：找不到预测日志文件，请确认 data/raw/ 下有预测数据")
                return

    print(f"正在读取数据: {input_file}")
    loader = DataLoader()
    raw_df = loader.load_and_clean(input_file)
    
    if raw_df.empty:
        print("错误：加载的数据为空，请检查CSV格式。")
        return

    # --- 3. 执行特征工程 ---
    print(">>> 正在执行特征工程...")
    engineer = UnifiedFeatureEngineer()
    feature_df = engineer.execute(raw_df)

    if feature_df.empty:
        print("错误：特征工程后数据为空，可能是所有数据都被清洗规则过滤了。")
        return

    # --- 4. 特征对齐与格式转换 ---
    print(">>> 正在执行特征格式对齐...")
    
    # 强制对齐特征列，缺失的补0
    X_infer = feature_df.reindex(columns=feature_names).fillna(0)

    # 关键修复：直接转为 category 类型，与训练一致
    actual_cat = [c for c in CAT_FEATURES if c in X_infer.columns]
    for col in actual_cat:
        X_infer[col] = X_infer[col].astype(str).astype('category')

    # 非类别列确保是数值型
    num_cols = [c for c in X_infer.columns if c not in actual_cat]
    for col in num_cols:
        X_infer[col] = pd.to_numeric(X_infer[col], errors='coerce').fillna(0)

    # --- 5. 执行集成打分 (已修复报错) ---
    print(f">>> 正在调用集成模型进行意向探测 (判定阈值: {threshold:.3f})...")
    
    try:
        # 情况A: 如果是我们的自定义 IntentModel (通常已经封装好了 predict_proba 返回一维数组)
        if hasattr(model_obj, 'predict_proba'):
            # 注意：IntentModel.predict_proba 内部可能已经取了 [:, 1]
            probs = model_obj.predict_proba(X_infer)
            
            # 双重保险：如果万一返回的是二维数组 (N, 2)，我们再取一次
            if probs.ndim == 2:
                probs = probs[:, 1]
                
        # 情况B: 如果是原生的 LGBMClassifier 或 sklearn 风格对象
        elif hasattr(model_obj, 'classes_'):
            probs = model_obj.predict_proba(X_infer)[:, 1]
            
        # 情况C: 如果是原生的 Booster 对象 (只有 predict 方法)
        else:
            probs = model_obj.predict(X_infer)
            
    except Exception as e:
        print(f"预测过程出错: {e}")
        # 打印详细类型以便调试
        print(f"当前模型对象类型: {type(model_obj)}")
        return
    
    # 将结果写回 feature_df
    feature_df['intent_probability'] = probs
    feature_df['intent_label'] = np.where(probs >= threshold, '1', '2')

    # --- 6. 筛选并导出名单 ---
    # 逻辑：只要预测结果，不关心历史是否有Target（因为这是盲测/预测）
    final_output = feature_df.copy()

    # 只保留用户要求的三个核心字段
    output_cols = ['user_id', 'intent_label', 'intent_probability']
    
    # 确保列存在 (防止某些特殊情况 user_id 丢失)
    output_cols = [c for c in output_cols if c in final_output.columns]
    output = final_output[output_cols]

    # 保存结果
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed', exist_ok=True)
    
    save_path = 'data/processed/marketing_list.csv'
    # 按概率倒序排列，高潜用户排前面
    output.sort_values(by='intent_probability', ascending=False).to_csv(
        save_path, index=False, encoding='utf-8-sig'
    )

    # --- 7. 预测结果统计汇总 ---
    intent_1_count = (final_output['intent_label'] == '1').sum()
    total = len(final_output)
    
    print(f"\n" + "="*50)
    print("          终端商机挖掘 - 预测任务完成")
    print(f"="*50)
    print(f"1. 最终判定阈值 : {threshold:.3f}")
    print(f"2. 最高意向得分 : {probs.max():.4f}")
    print(f"3. 识别商机数量 : {intent_1_count:,} 名")
    if total > 0:
        print(f"4. 商机占比     : {intent_1_count/total:.1%}")
    print(f"5. 名单保存路径 : {save_path}")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_prediction()