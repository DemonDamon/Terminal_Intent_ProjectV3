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
        input_file = 'data/raw/predict_data_10_backtest.csv' # 尝试查找前面生成的10月预测数据
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
    
    save_path = 'data/processed/marketing_list_backtest.csv'
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

def analyze_backtest_performance(raw_df, pred_df, threshold):
    """
    回测分析核心函数：对比预测结果与真实标签
    (已优化混淆矩阵展示效果)
    """
    print(f"\n" + "="*60)
    print("              回测效果详细分析报告")
    print(f"="*60)

    # 1. 数据合并与准备
    df_analysis = raw_df.copy()
    df_analysis['pred_prob'] = pred_df['intent_probability'].values
    
    # 尝试寻找真实标签列
    target_col = None
    for col in ['target', 'label', 'is_buy', 'Target']:
        if col in df_analysis.columns:
            target_col = col
            break
    
    # --- 情况 A: 无真实标签 ---
    if target_col is None:
        print("[提示] 数据中未检测到 'target' 列，无法进行回测对比。")
        bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
        dist = pd.cut(df_analysis['pred_prob'], bins=bins).value_counts().sort_index(ascending=False)
        print("\n[预测得分分布]")
        print(dist)
        return

    # --- 情况 B: 有真实标签 ---
    df_analysis['target'] = pd.to_numeric(df_analysis[target_col], errors='coerce').fillna(0).astype(int)
    
    total_samples = len(df_analysis)
    total_positives = df_analysis['target'].sum() # 实际商机总数 (TP + FN)
    
    if total_positives == 0:
        print("[警告] 真实标签中没有正样本 (Target全为0)，无法计算命中率。")
        return

    global_ctr = total_positives / total_samples
    print(f"数据概览: 总样本 {total_samples} 条, 实际商机总数 {total_positives} 条, 自然转化率 {global_ctr:.2%}")

    # 2. 核心指标计算
    pred_1_mask = df_analysis['pred_prob'] >= threshold
    pred_0_mask = df_analysis['pred_prob'] < threshold
    true_1_mask = df_analysis['target'] == 1
    true_0_mask = df_analysis['target'] == 0
    
    # 四维数据
    tp = len(df_analysis[pred_1_mask & true_1_mask]) # 命中 (True Positive)
    fp = len(df_analysis[pred_1_mask & true_0_mask]) # 误报 (False Positive)
    fn = len(df_analysis[pred_0_mask & true_1_mask]) # 漏报 (False Negative)
    tn = len(df_analysis[pred_0_mask & true_0_mask]) # 排除 (True Negative)
    
    pred_positives_count = tp + fp # 预测出的商机数 (名单人数)

    precision = tp / pred_positives_count if pred_positives_count > 0 else 0
    recall = tp / total_positives if total_positives > 0 else 0
    lift = precision / global_ctr if global_ctr > 0 else 0
    
    # 3. 输出优化后的混淆矩阵 (带边框和注释)
    # 定义列宽
    c1_w, c2_w, c3_w, c4_w = 14, 22, 22, 12
    
    print(f"\n[混淆矩阵]")
    print("+" + "-"*(c1_w+2) + "+" + "-"*(c2_w+2) + "+" + "-"*(c3_w+2) + "+" + "-"*(c4_w+2) + "+")
    # 表头
    print(f"| {'':<{c1_w}} | {' 实: 购买 (Target=1)':<{c2_w}} | {' 实: 未购 (Target=0)':<{c3_w}} | {' 合计':<{c4_w}} |")
    print("+" + "-"*(c1_w+2) + "+" + "-"*(c2_w+2) + "+" + "-"*(c3_w+2) + "+" + "-"*(c4_w+2) + "+")
    
    # 第一行：预测购买
    row1_title = " 预: 购买(1)"
    cell_tp = f" {tp} (TP 命中)"
    cell_fp = f" {fp} (FP 误报)"
    row1_sum = f" {tp+fp}"
    print(f"| {row1_title:<{c1_w}} | {cell_tp:<{c2_w}} | {cell_fp:<{c3_w}} | {row1_sum:<{c4_w}} |")
    
    # 分隔线
    print("|" + "-"*(c1_w+2) + "+" + "-"*(c2_w+2) + "+" + "-"*(c3_w+2) + "+" + "-"*(c4_w+2) + "|")
    
    # 第二行：预测未购
    row2_title = " 预: 未购(0)"
    cell_fn = f" {fn} (FN 漏报)"
    cell_tn = f" {tn} (TN 排除)"
    row2_sum = f" {fn+tn}"
    print(f"| {row2_title:<{c1_w}} | {cell_fn:<{c2_w}} | {cell_tn:<{c3_w}} | {row2_sum:<{c4_w}} |")
    
    print("+" + "-"*(c1_w+2) + "+" + "-"*(c2_w+2) + "+" + "-"*(c3_w+2) + "+" + "-"*(c4_w+2) + "+")
    # 底部合计
    print(f"| {' 合计':<{c1_w}} | {f' {tp+fn} (实际商机)':<{c2_w}} | {f' {fp+tn} (实际未购)':<{c3_w}} | {f' {total_samples}':<{c4_w}} |")
    print("+" + "-"*(c1_w+2) + "+" + "-"*(c2_w+2) + "+" + "-"*(c3_w+2) + "+" + "-"*(c4_w+2) + "+")
    
    # 4. 阈值效果指标
    print(f"\n[阈值评估 (Threshold={threshold:.3f})]")
    # 修改点：显示 预测数 / 总商机数
    print(f"  - 预测商机数 : {pred_positives_count} / {total_positives} (预测名单数 / 实际商机总数)")
    print(f"  - 准确率 (Precision) : {precision:.2%} (预测名单中 {int(precision*100)}% 是真买家)")
    print(f"  - 召回率 (Recall)    : {recall:.2%} (捕获了 {recall*100:.1f}% 的实际商机)")
    print(f"  - 提升度 (Lift)      : {lift:.2f}倍 (对比随机投放)")


    # 5. Top-N 效果分析
    print(f"\n[Top-N 高潜名单截断效果]")
    df_sorted = df_analysis.sort_values(by='pred_prob', ascending=False)
    
    print(f"{'Top N':<8} | {'命中数(TP)':<10} | {'准确率(Precision)':<18} | {'提升度(Lift)':<10}")
    print("-" * 55)
    
    for top_n in [100, 500, 1000, 5000]:
        if top_n > total_samples:
            continue
            
        top_subset = df_sorted.head(top_n)
        hit_count = top_subset['target'].sum()
        top_precision = hit_count / top_n
        top_lift = top_precision / global_ctr if global_ctr > 0 else 0
        
        print(f"{top_n:<8} | {hit_count:<10} | {top_precision:.2%}{'':<12} | {top_lift:.1f} 倍")

    print("="*60 + "\n")
    



# ==========================================
#  修改 run_prediction 函数以调用分析逻辑
# ==========================================

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
    # 优先查找回测专用数据
    input_file = 'data/raw/predict_data_backtest.csv' 
    if not os.path.exists(input_file):
        input_file = 'data/raw/predict_data_10_backtest.csv'
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
    
    # 强制对齐特征列
    X_infer = feature_df.reindex(columns=feature_names).fillna(0)

    # 类型转换
    actual_cat = [c for c in CAT_FEATURES if c in X_infer.columns]
    for col in actual_cat:
        X_infer[col] = X_infer[col].astype(str).astype('category')

    num_cols = [c for c in X_infer.columns if c not in actual_cat]
    for col in num_cols:
        X_infer[col] = pd.to_numeric(X_infer[col], errors='coerce').fillna(0)

    # --- 5. 执行集成打分 ---
    print(f">>> 正在调用集成模型进行意向探测 (判定阈值: {threshold:.3f})...")
    
    try:
        if hasattr(model_obj, 'predict_proba'):
            probs = model_obj.predict_proba(X_infer)
            if probs.ndim == 2:
                probs = probs[:, 1]
        elif hasattr(model_obj, 'classes_'):
            probs = model_obj.predict_proba(X_infer)[:, 1]
        else:
            probs = model_obj.predict(X_infer)
            
    except Exception as e:
        print(f"预测过程出错: {e}")
        return
    
    # 结果回写
    feature_df['intent_probability'] = probs
    feature_df['intent_label'] = np.where(probs >= threshold, '1', '2')

    # --- 6. 筛选并导出名单 ---
    final_output = feature_df.copy()
    output_cols = ['user_id', 'intent_label', 'intent_probability']
    output_cols = [c for c in output_cols if c in final_output.columns]
    output = final_output[output_cols]

    if not os.path.exists('data/processed'):
        os.makedirs('data/processed', exist_ok=True)
    
    save_path = 'data/processed/marketing_list_backtest.csv'
    output.sort_values(by='intent_probability', ascending=False).to_csv(
        save_path, index=False, encoding='utf-8-sig'
    )
    
    print(f"预测名单已保存: {save_path}")

    # --- 7. 调用结果分析 (新增) ---
    # 注意：我们需要将原始数据(包含target)和预测结果一起传入
    # feature_df 经过了特征工程，行数可能少于 raw_df (如果清洗过滤了)，但通常 DataLoader 清洗后是对齐的
    # 为了保险，我们直接用 feature_df，因为它包含了 target (如果 DataLoader 加载了的话) 和 预测列
    analyze_backtest_performance(feature_df, feature_df, threshold)

if __name__ == "__main__":
    run_prediction()