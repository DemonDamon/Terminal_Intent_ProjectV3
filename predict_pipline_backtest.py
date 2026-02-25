import sys
import os
import pandas as pd
import numpy as np

# --- 路径修复块 (确保能导入src模块) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.pipeline.predict_common import run_prediction_pipeline


def analyze_backtest_performance(feature_df, threshold):
    """
    回测分析核心函数：对比预测结果与真实标签
    (已优化混淆矩阵展示效果)
    """
    print(f"\n" + "="*60)
    print("              回测效果详细分析报告")
    print(f"="*60)

    df_analysis = feature_df.copy()
    df_analysis['pred_prob'] = feature_df['intent_probability']

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
    total_positives = df_analysis['target'].sum()

    if total_positives == 0:
        print("[警告] 真实标签中没有正样本 (Target全为0)，无法计算命中率。")
        return

    global_ctr = total_positives / total_samples
    print(f"数据概览: 总样本 {total_samples} 条, 实际商机总数 {total_positives} 条, 自然转化率 {global_ctr:.2%}")

    # 核心指标计算
    pred_1_mask = df_analysis['pred_prob'] >= threshold
    pred_0_mask = df_analysis['pred_prob'] < threshold
    true_1_mask = df_analysis['target'] == 1
    true_0_mask = df_analysis['target'] == 0

    tp = len(df_analysis[pred_1_mask & true_1_mask])
    fp = len(df_analysis[pred_1_mask & true_0_mask])
    fn = len(df_analysis[pred_0_mask & true_1_mask])
    tn = len(df_analysis[pred_0_mask & true_0_mask])

    pred_positives_count = tp + fp

    precision = tp / pred_positives_count if pred_positives_count > 0 else 0
    recall = tp / total_positives if total_positives > 0 else 0
    lift = precision / global_ctr if global_ctr > 0 else 0

    # 输出混淆矩阵
    c1_w, c2_w, c3_w, c4_w = 14, 22, 22, 12

    print(f"\n[混淆矩阵]")
    print("+" + "-"*(c1_w+2) + "+" + "-"*(c2_w+2) + "+" + "-"*(c3_w+2) + "+" + "-"*(c4_w+2) + "+")
    print(f"| {'':<{c1_w}} | {' 实: 购买 (Target=1)':<{c2_w}} | {' 实: 未购 (Target=0)':<{c3_w}} | {' 合计':<{c4_w}} |")
    print("+" + "-"*(c1_w+2) + "+" + "-"*(c2_w+2) + "+" + "-"*(c3_w+2) + "+" + "-"*(c4_w+2) + "+")

    row1_title = " 预: 购买(1)"
    cell_tp = f" {tp} (TP 命中)"
    cell_fp = f" {fp} (FP 误报)"
    row1_sum = f" {tp+fp}"
    print(f"| {row1_title:<{c1_w}} | {cell_tp:<{c2_w}} | {cell_fp:<{c3_w}} | {row1_sum:<{c4_w}} |")

    print("|" + "-"*(c1_w+2) + "+" + "-"*(c2_w+2) + "+" + "-"*(c3_w+2) + "+" + "-"*(c4_w+2) + "|")

    row2_title = " 预: 未购(0)"
    cell_fn = f" {fn} (FN 漏报)"
    cell_tn = f" {tn} (TN 排除)"
    row2_sum = f" {fn+tn}"
    print(f"| {row2_title:<{c1_w}} | {cell_fn:<{c2_w}} | {cell_tn:<{c3_w}} | {row2_sum:<{c4_w}} |")

    print("+" + "-"*(c1_w+2) + "+" + "-"*(c2_w+2) + "+" + "-"*(c3_w+2) + "+" + "-"*(c4_w+2) + "+")
    print(f"| {' 合计':<{c1_w}} | {f' {tp+fn} (实际商机)':<{c2_w}} | {f' {fp+tn} (实际未购)':<{c3_w}} | {f' {total_samples}':<{c4_w}} |")
    print("+" + "-"*(c1_w+2) + "+" + "-"*(c2_w+2) + "+" + "-"*(c3_w+2) + "+" + "-"*(c4_w+2) + "+")

    # 阈值效果指标
    print(f"\n[阈值评估 (Threshold={threshold:.3f})]")
    print(f"  - 预测商机数 : {pred_positives_count} / {total_positives} (预测名单数 / 实际商机总数)")
    print(f"  - 准确率 (Precision) : {precision:.2%} (预测名单中 {int(precision*100)}% 是真买家)")
    print(f"  - 召回率 (Recall)    : {recall:.2%} (捕获了 {recall*100:.1f}% 的实际商机)")
    print(f"  - 提升度 (Lift)      : {lift:.2f}倍 (对比随机投放)")

    # Top-N 效果分析
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


def run_prediction():
    """回测预测 Pipeline：生成回测名单 + 效果分析"""
    search_paths = [
        'data/raw/predict_data_backtest.csv',
        'data/raw/predict_data_10_backtest.csv',
        'data/raw/predict_data.csv',
    ]
    save_path = 'data/processed/marketing_list_backtest.csv'

    run_prediction_pipeline(search_paths, save_path, post_predict_fn=analyze_backtest_performance)


if __name__ == "__main__":
    run_prediction()
