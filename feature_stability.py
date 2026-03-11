"""
feature_stability.py — PSI 特征稳定性诊断脚本

量化每个特征在参考集(8-9月) vs 测试集(10月)之间的分布漂移程度(PSI)，
输出排序报告供人工审查，将不稳定特征加入排除列表。
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import DataLoader
from src.features.unified_fe import UnifiedFeatureEngineer
from config.feature_list import CAT_FEATURES, RULE_LAYER_FEATURES

# ==================== 配置区 ====================
DATA_PATH_89 = 'data/raw/train_data_89.csv'
DATA_PATH_89_FALLBACK = 'data/raw/train_data.csv'
DATA_PATH_10 = 'data/raw/predict_data_10.csv'

CUTOFF_TEST = pd.Timestamp('2024-10-01')  # 8-9月 vs 10月

PSI_THRESHOLD = 0.25   # PSI >= 0.25 判定为不稳定
PSI_EPSILON = 0.0001   # 占比为0时的替代值

OUTPUT_CSV = 'data/processed/feature_psi_report.csv'
# ================================================


def calculate_psi(reference, test, bins=10):
    """连续特征 PSI：按参考集分位数分桶"""
    ref = reference.dropna().values
    tst = test.dropna().values

    if len(ref) == 0 or len(tst) == 0:
        return np.nan

    # 按参考集的分位数切桶边界
    quantiles = np.linspace(0, 100, bins + 1)
    boundaries = np.percentile(ref, quantiles)
    boundaries = np.unique(boundaries)  # 去重（值集中时分位数可能重复）

    if len(boundaries) <= 1:
        # 参考集几乎全是同一个值，无法分桶
        return 0.0

    # 统计每个桶的占比
    ref_counts = np.histogram(ref, bins=boundaries)[0]
    tst_counts = np.histogram(tst, bins=boundaries)[0]

    ref_pct = ref_counts / len(ref)
    tst_pct = tst_counts / len(tst)

    # 零值替代
    ref_pct = np.where(ref_pct == 0, PSI_EPSILON, ref_pct)
    tst_pct = np.where(tst_pct == 0, PSI_EPSILON, tst_pct)

    psi = np.sum((tst_pct - ref_pct) * np.log(tst_pct / ref_pct))
    return float(psi)


def calculate_psi_categorical(reference, test):
    """类别特征 PSI：按类别值频率对比"""
    ref = reference.dropna().astype(str)
    tst = test.dropna().astype(str)

    if len(ref) == 0 or len(tst) == 0:
        return np.nan

    # 合并所有类别值
    all_categories = set(ref.unique()) | set(tst.unique())

    ref_counts = ref.value_counts()
    tst_counts = tst.value_counts()

    psi = 0.0
    for cat in all_categories:
        ref_pct = ref_counts.get(cat, 0) / len(ref)
        tst_pct = tst_counts.get(cat, 0) / len(tst)

        ref_pct = max(ref_pct, PSI_EPSILON)
        tst_pct = max(tst_pct, PSI_EPSILON)

        psi += (tst_pct - ref_pct) * np.log(tst_pct / ref_pct)

    return float(psi)


def run_stability_analysis():
    print("=" * 70)
    print("  PSI 特征稳定性诊断")
    print("=" * 70)

    # --- 1. 加载数据 ---
    loader = DataLoader()

    data_path = DATA_PATH_89
    if not os.path.exists(data_path):
        data_path = DATA_PATH_89_FALLBACK
    if not os.path.exists(data_path):
        print(f"[错误] 找不到训练数据: {DATA_PATH_89} 或 {DATA_PATH_89_FALLBACK}")
        return

    print(f"加载8-9月数据: {data_path}")
    raw_df = loader.load_and_clean(data_path)

    if os.path.exists(DATA_PATH_10):
        print(f"加载10月数据: {DATA_PATH_10}")
        oct_df = loader.load_and_clean(DATA_PATH_10)
        raw_df = pd.concat([raw_df, oct_df], ignore_index=True)
        print(f"合并后总数据量: {len(raw_df)} 条")
    else:
        print(f"[错误] 找不到10月数据: {DATA_PATH_10}")
        return

    # --- 2. 特征工程 ---
    print("执行特征工程...")
    engineer = UnifiedFeatureEngineer()
    feature_df = engineer.execute(raw_df)

    if 'last_action_date' not in feature_df.columns:
        print("[错误] 特征表中缺少 last_action_date 列")
        return

    # --- 3. 时间切分 (与 train_pipline.py 一致) ---
    ref_df = feature_df[feature_df['last_action_date'] < CUTOFF_TEST]
    tst_df = feature_df[feature_df['last_action_date'] >= CUTOFF_TEST]

    print(f"\n参考集 (8-9月): {len(ref_df)} 样本")
    print(f"测试集 (10月):  {len(tst_df)} 样本")

    if len(ref_df) == 0 or len(tst_df) == 0:
        print("[错误] 参考集或测试集为空，请检查数据时间范围")
        return

    # --- 4. 确定要分析的特征列 ---
    drop_cols = {'user_id', 'target', 'last_action_date'}
    feature_cols = [c for c in feature_df.columns if c not in drop_cols]

    # --- 5. 逐特征计算 PSI ---
    results = []
    for col in feature_cols:
        is_cat = col in CAT_FEATURES

        if is_cat:
            psi = calculate_psi_categorical(ref_df[col], tst_df[col])
            ref_mean = '—'
            tst_mean = '—'
        else:
            ref_series = pd.to_numeric(ref_df[col], errors='coerce')
            tst_series = pd.to_numeric(tst_df[col], errors='coerce')
            psi = calculate_psi(ref_series, tst_series)
            ref_mean = f"{ref_series.mean():.4f}" if not ref_series.isna().all() else '—'
            tst_mean = f"{tst_series.mean():.4f}" if not tst_series.isna().all() else '—'

        verdict = '不稳定' if (psi is not None and not np.isnan(psi) and psi >= PSI_THRESHOLD) else '稳定'

        results.append({
            'feature': col,
            'ref_mean': ref_mean,
            'test_mean': tst_mean,
            'psi': psi if psi is not None and not np.isnan(psi) else 0.0,
            'verdict': verdict,
        })

    # 按 PSI 降序排列
    results.sort(key=lambda x: x['psi'], reverse=True)

    # --- 6. 控制台报告 ---
    print(f"\n{'=' * 80}")
    print(f"  特征稳定性报告 (参考集: 8-9月, 测试集: 10月)")
    print(f"  参考集样本数: {len(ref_df)}, 测试集样本数: {len(tst_df)}")
    print(f"{'=' * 80}")
    print(f"{'排名':<6s}{'特征名':<28s}{'参考均值':>10s}{'测试均值':>10s}{'PSI':>10s}  {'判定'}")
    print(f"{'-' * 80}")

    for i, r in enumerate(results, 1):
        print(f"{i:<6d}{r['feature']:<28s}{r['ref_mean']:>10s}{r['test_mean']:>10s}{r['psi']:>10.4f}  {r['verdict']}")

    # 不稳定特征汇总
    unstable = [r['feature'] for r in results if r['verdict'] == '不稳定']
    print(f"\n{'=' * 80}")
    print(f"建议排除的特征 (PSI >= {PSI_THRESHOLD}):")
    if unstable:
        print(f"UNSTABLE_FEATURES = {unstable}")
    else:
        print("所有特征均稳定，无需排除。")
    print(f"{'=' * 80}")

    # --- 7. 输出 CSV ---
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    report_df = pd.DataFrame(results)
    report_df.index = range(1, len(report_df) + 1)
    report_df.index.name = 'rank'
    report_df.to_csv(OUTPUT_CSV, encoding='utf-8-sig')
    print(f"\nCSV 报告已保存: {OUTPUT_CSV}")


if __name__ == '__main__':
    run_stability_analysis()
