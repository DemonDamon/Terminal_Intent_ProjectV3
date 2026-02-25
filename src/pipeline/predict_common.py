"""
预测 Pipeline 公共模块
=====================
抽取 predict_pipline.py 和 predict_pipline_backtest.py 的共享逻辑，
消除 95%+ 的代码重复。两个脚本只需提供差异化配置即可。
"""

import os
import pandas as pd
import joblib
import numpy as np

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
        except (ValueError, TypeError, EOFError):
            pass

    # 策略 3: 兜底默认值
    print("[Smart Load] 未找到配置，使用业务默认阈值: 0.500")
    return 0.5


def load_model(model_path='models/intent_v2.pkl', feature_names_path='models/feature_names.pkl'):
    """
    加载模型和特征名列表。
    返回 (model_obj, feature_names) 或在失败时返回 (None, None)。
    """
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}，请先运行 train_pipline.py")
        return None, None

    print(">>> 正在加载集成模型及编码规则...")
    try:
        model_obj = joblib.load(model_path)
        feature_names = joblib.load(feature_names_path)
        return model_obj, feature_names
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None, None


def find_input_file(search_paths):
    """
    按优先级搜索输入数据文件。
    search_paths: 按优先级排列的文件路径列表。
    返回找到的文件路径，或 None。
    """
    for path in search_paths:
        if os.path.exists(path):
            return path
    return None


def run_feature_engineering(input_file):
    """
    数据加载 → 清洗 → 特征工程。
    返回 feature_df 或 None。
    """
    print(f"正在读取数据: {input_file}")
    loader = DataLoader()
    raw_df = loader.load_and_clean(input_file)

    if raw_df.empty:
        print("错误：加载的数据为空，请检查CSV格式。")
        return None

    print(">>> 正在执行特征工程...")
    engineer = UnifiedFeatureEngineer()
    feature_df = engineer.execute(raw_df)

    if feature_df.empty:
        print("错误：特征工程后数据为空，可能是所有数据都被清洗规则过滤了。")
        return None

    return feature_df


def align_features(feature_df, feature_names):
    """
    特征对齐与格式转换：补缺失列、类别编码、数值转换。
    """
    print(">>> 正在执行特征格式对齐...")

    # 强制对齐特征列，缺失的补0
    X_infer = feature_df.reindex(columns=feature_names).fillna(0)

    # 类别特征转换
    actual_cat = [c for c in CAT_FEATURES if c in X_infer.columns]
    for col in actual_cat:
        X_infer[col] = X_infer[col].astype(str).astype('category')

    # 数值特征转换
    num_cols = [c for c in X_infer.columns if c not in actual_cat]
    for col in num_cols:
        X_infer[col] = pd.to_numeric(X_infer[col], errors='coerce').fillna(0)

    return X_infer


def run_model_predict(model_obj, X_infer):
    """
    执行模型预测，兼容多种模型类型。
    返回概率数组或 None。
    """
    try:
        if hasattr(model_obj, 'predict_proba'):
            probs = model_obj.predict_proba(X_infer)
            if probs.ndim == 2:
                probs = probs[:, 1]
        elif hasattr(model_obj, 'classes_'):
            probs = model_obj.predict_proba(X_infer)[:, 1]
        else:
            probs = model_obj.predict(X_infer)
        return probs
    except Exception as e:
        print(f"预测过程出错: {e}")
        print(f"当前模型对象类型: {type(model_obj)}")
        return None


def save_predictions(feature_df, probs, threshold, save_path):
    """
    将预测结果写入 feature_df 并导出 CSV。
    """
    feature_df['intent_probability'] = probs
    feature_df['intent_label'] = np.where(probs >= threshold, '1', '2')

    final_output = feature_df.copy()
    output_cols = [c for c in ['user_id', 'intent_label', 'intent_probability'] if c in final_output.columns]
    output = final_output[output_cols]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    output.sort_values(by='intent_probability', ascending=False).to_csv(
        save_path, index=False, encoding='utf-8-sig'
    )

    return feature_df


def print_prediction_summary(feature_df, probs, threshold, save_path):
    """
    打印预测结果统计汇总。
    """
    intent_1_count = (feature_df['intent_label'] == '1').sum()
    total = len(feature_df)

    print(f"\n" + "=" * 50)
    print("          终端商机挖掘 - 预测任务完成")
    print(f"=" * 50)
    print(f"1. 最终判定阈值 : {threshold:.3f}")
    print(f"2. 最高意向得分 : {probs.max():.4f}")
    print(f"3. 识别商机数量 : {intent_1_count:,} 名")
    if total > 0:
        print(f"4. 商机占比     : {intent_1_count / total:.1%}")
    print(f"5. 名单保存路径 : {save_path}")
    print("=" * 50 + "\n")


def run_prediction_pipeline(search_paths, save_path, post_predict_fn=None):
    """
    通用预测 Pipeline 入口。

    参数:
        search_paths: 输入文件搜索路径列表（按优先级排列）
        save_path: 输出 CSV 文件路径
        post_predict_fn: 可选的后处理函数，签名为 fn(feature_df, threshold)
    """
    # 1. 加载模型
    model_obj, feature_names = load_model()
    if model_obj is None:
        return

    # 2. 加载阈值
    threshold = load_optimal_threshold(model_obj)

    # 3. 查找输入文件
    input_file = find_input_file(search_paths)
    if input_file is None:
        print("错误：找不到预测日志文件，请确认 data/raw/ 下有预测数据")
        return

    # 4. 特征工程
    feature_df = run_feature_engineering(input_file)
    if feature_df is None:
        return

    # 5. 特征对齐
    X_infer = align_features(feature_df, feature_names)

    # 6. 模型预测
    print(f">>> 正在调用集成模型进行意向探测 (判定阈值: {threshold:.3f})...")
    probs = run_model_predict(model_obj, X_infer)
    if probs is None:
        return

    # 7. 保存结果
    feature_df = save_predictions(feature_df, probs, threshold, save_path)

    # 8. 打印统计
    print_prediction_summary(feature_df, probs, threshold, save_path)

    # 9. 可选的后处理（如回测分析）
    if post_predict_fn:
        post_predict_fn(feature_df, threshold)
