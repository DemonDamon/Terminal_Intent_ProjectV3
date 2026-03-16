import sys
import os
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report, f1_score, precision_score, recall_score

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import DataLoader
from src.features.unified_fe import UnifiedFeatureEngineer
from src.models.model_lgb import IntentModel
from config.feature_list import CAT_FEATURES, RULE_LAYER_FEATURES, RULE_LAYER_CONFIG, FEATURE_WEAKENING_CONFIG, RANKING_CONFIG

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_logger():
    return logging.getLogger(__name__)

def analyze_probability_distribution(y_true, probs, label=""):
    """
    【P0】概率分布分析：验证阈值搜索假设，检查正负样本在各概率区间的重叠情况。
    重点关注 E8 新覆盖的 [0.10, 0.30) 区间。
    """
    y_true = np.array(y_true)
    pos_probs = probs[y_true == 1]
    neg_probs = probs[y_true == 0]

    print(f"\n{'='*60}")
    print(f"  【P0】概率分布分析 {label}")
    print(f"{'='*60}")
    print(f"  样本数: 正={len(pos_probs)}, 负={len(neg_probs)}")
    print(f"  概率均值:   正={pos_probs.mean():.4f}, 负={neg_probs.mean():.4f}")
    print(f"  概率中位数: 正={np.median(pos_probs):.4f}, 负={np.median(neg_probs):.4f}")
    print(f"  概率标准差: 正={pos_probs.std():.4f}, 负={neg_probs.std():.4f}")

    # 分区间统计
    bins = [(0.00, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.25),
            (0.25, 0.30), (0.30, 0.50), (0.50, 0.70), (0.70, 1.01)]

    print(f"\n  {'区间':<14s} {'正样本':>6s} {'负样本':>6s} {'正占比':>7s} {'区间精度':>8s}")
    print(f"  {'-'*48}")

    for low, high in bins:
        pos_in = int(((pos_probs >= low) & (pos_probs < high)).sum())
        neg_in = int(((neg_probs >= low) & (neg_probs < high)).sum())
        total_in = pos_in + neg_in
        prec = pos_in / total_in if total_in > 0 else 0
        pos_pct = pos_in / len(pos_probs) * 100 if len(pos_probs) > 0 else 0
        mark = " *" if 0.10 <= low < 0.30 else ""
        print(f"  [{low:.2f}, {high:.2f}) {pos_in:>6d} {neg_in:>6d} {pos_pct:>6.1f}% {prec:>7.1%}{mark}")

    # E8 关键区间汇总
    e8_pos = int(((pos_probs >= 0.10) & (pos_probs < 0.30)).sum())
    e8_neg = int(((neg_probs >= 0.10) & (neg_probs < 0.30)).sum())
    e8_total = e8_pos + e8_neg
    print(f"\n  E8 关键区间 [0.10, 0.30) 汇总:")
    print(f"    正样本: {e8_pos} ({e8_pos/len(pos_probs)*100:.1f}% 的全部正样本)" if len(pos_probs) > 0 else "")
    print(f"    负样本: {e8_neg}")
    if e8_total > 0:
        print(f"    区间精度: {e8_pos/e8_total:.1%} (高于50%说明E8扩展有效)")
    print(f"{'='*60}")

def print_evaluation_report(y_true, probs, threshold, dataset_name):
    """评估报告：计算并打印 ROC-AUC、PR-AUC、KS、Precision、Recall、F1"""
    y_true = np.array(y_true)
    y_pred = (probs >= threshold).astype(int)

    try:
        roc_val = roc_auc_score(y_true, probs)
    except ValueError:
        roc_val = np.nan

    prec_arr, rec_arr, _ = precision_recall_curve(y_true, probs)
    pr_auc_val = auc(rec_arr, prec_arr)

    # KS 值
    df_ks = pd.DataFrame({'label': y_true, 'prob': probs})
    df_ks['good'] = 1 - df_ks['label']
    df_ks = df_ks.sort_values(by='prob', ascending=False)
    df_ks['cum_bad'] = df_ks['label'].cumsum() / df_ks['label'].sum()
    df_ks['cum_good'] = df_ks['good'].cumsum() / df_ks['good'].sum()
    ks_value = (df_ks['cum_bad'] - df_ks['cum_good']).abs().max()

    print(f"\n==================================================")
    print(f" {dataset_name}")
    print(f"==================================================")
    print(f"1. ROC-AUC :       {roc_val:.4f}")
    print(f"2. PR-AUC  :       {pr_auc_val:.4f}")
    print(f"3. KS值    :       {ks_value:.4f}")
    print(f"4. Precision :   {precision_score(y_true, y_pred):.2%}")
    print(f"5. Recall    :   {recall_score(y_true, y_pred):.2%}")
    print(f"6. F1-Score  :      {f1_score(y_true, y_pred):.4f}")
    print(f"--------------------------------------------------")
    print(f"判定阈值设定为: {threshold:.3f}")
    print(f"==================================================")


def print_topk_evaluation(y_true, probs, dataset_name, ranking_config=None):
    """
    【R1】Top-K% 排名制分层评估报告
    输出各 K% 级别的 Precision / Recall / 人数，不依赖固定阈值。
    """
    if ranking_config is None:
        ranking_config = RANKING_CONFIG

    y_true = np.array(y_true)
    probs = np.array(probs)
    n_total = len(probs)
    n_pos = int(y_true.sum())

    # 按概率降序排序
    sorted_indices = np.argsort(probs)[::-1]
    sorted_labels = y_true[sorted_indices]
    sorted_probs = probs[sorted_indices]

    # 评估多个 K% 级别
    eval_pcts = [0.0010, 0.0025, 0.0050, 0.0075, 0.0100]
    tiers_config = ranking_config.get('tiers', {})
    min_prob = ranking_config.get('min_prob', 0.30)

    print(f"\n{'='*65}")
    print(f"  【R1】Top-K% 排名制评估 — {dataset_name}")
    print(f"{'='*65}")
    print(f"  总用户: {n_total:,}  |  正样本: {n_pos:,}  |  正样本率: {n_pos/n_total:.2%}")
    print(f"  安全门槛: min_prob={min_prob}")
    print(f"{'─'*65}")
    print(f"  {'Top-K%':>8s} {'人数':>7s} {'Precision':>10s} {'Recall':>8s} {'最低概率':>9s} {'等级':>4s}")
    print(f"  {'─'*58}")

    for pct in eval_pcts:
        k = min(int(n_total * pct), n_total)
        if k == 0:
            continue

        # 安全门槛截断
        actual_k = k
        for i in range(k):
            if sorted_probs[i] < min_prob:
                actual_k = i
                break

        top_labels = sorted_labels[:actual_k]
        tp = int(top_labels.sum())
        prec = tp / actual_k if actual_k > 0 else 0
        rec = tp / n_pos if n_pos > 0 else 0
        min_p = sorted_probs[actual_k - 1] if actual_k > 0 else 0

        # 确定对应的置信等级
        tier = '-'
        for tier_name in ['S', 'A', 'B']:
            if tier_name in tiers_config and pct <= tiers_config[tier_name]['pct_upper']:
                tier = tier_name
                break

        truncated = f" (截断:{k}→{actual_k})" if actual_k < k else ""
        print(f"  {pct:>7.2%} {actual_k:>7,d} {prec:>9.1%} {rec:>7.1%} {min_p:>9.4f}  {tier:>3s}{truncated}")

    # 推荐配置下的结果
    rec_k_raw = int(n_total * ranking_config['top_k_pct'])
    rec_k = max(ranking_config['min_k'], min(ranking_config['max_k'], rec_k_raw))
    rec_k = min(rec_k, n_total)
    # 安全门槛截断
    actual_rec_k = rec_k
    for i in range(rec_k):
        if sorted_probs[i] < min_prob:
            actual_rec_k = i
            break
    rec_labels = sorted_labels[:actual_rec_k]
    rec_tp = int(rec_labels.sum())
    rec_prec = rec_tp / actual_rec_k if actual_rec_k > 0 else 0
    rec_rec = rec_tp / n_pos if n_pos > 0 else 0

    print(f"  {'─'*58}")
    print(f"  推荐配置: Top {ranking_config['top_k_pct']:.2%} → {actual_rec_k:,} 人")
    print(f"  Precision={rec_prec:.1%}  Recall={rec_rec:.1%}  clamp=[{ranking_config['min_k']}, {ranking_config['max_k']}]")
    print(f"{'='*65}")

def run_training():
    logger = get_logger()
    logger.info(">>> 启动[集成模型训练]流程 <<<")

    # --- 1. 自动清理旧环境 (防止读取残留配置) ---
    files_to_clean = [
        'models/intent_v2.pkl',
        'models/optimal_threshold.pkl',
        'models/feature_names.pkl'
    ]
    print("-" * 30)
    for f in files_to_clean:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"已自动清理旧文件: {f}")
            except:
                pass
    print("-" * 30)

    # --- 2. 数据加载 ---
    loader = DataLoader()
    # 优先加载清洗后的序列采样数据，如果不存在则加载原始数据
    data_path = 'data/raw/train_data_89.csv'
    if not os.path.exists(data_path):
        data_path = 'data/raw/train_data.csv'

    if not os.path.exists(data_path):
        logger.error(f"找不到训练数据: {data_path}")
        return

    raw_df = loader.load_and_clean(data_path)

    # 10月数据由 data_extra_withdata.py 单独输出，不在 train_data_89 中
    # 需要独立加载并合并，供后续时间切分作为测试集
    oct_data_path = 'data/raw/predict_data_10.csv'
    if os.path.exists(oct_data_path):
        logger.info(f"单独加载10月数据: {oct_data_path}")
        oct_df = loader.load_and_clean(oct_data_path)
        raw_df = pd.concat([raw_df, oct_df], ignore_index=True)
        logger.info(f"合并后总数据量: {len(raw_df)} 条 (含10月 {len(oct_df)} 条)")
    else:
        logger.warning(f"未找到10月数据文件: {oct_data_path}，将跳过10月测试评估")
    
    # 标签分布检查
    if 'target' in raw_df.columns:
        counts = raw_df['target'].value_counts()
        if len(counts) < 2:
            logger.error(" 错误：训练数据中只包含一种标签！请检查数据采样逻辑。")
            print(counts)
            return

    # --- 3. 特征工程 ---
    engineer = UnifiedFeatureEngineer()
    feature_df = engineer.execute(raw_df)

    # --- 4. 数据拆分（时间切分，解决瓶颈1） ---
    # 确保 target 存在
    if 'target' not in feature_df.columns:
        logger.error("特征表中缺少 target 列")
        return

    # 确保 last_action_date 存在
    if 'last_action_date' not in feature_df.columns:
        logger.error("特征表中缺少 last_action_date 列，无法执行时间切分")
        return

    # 【P2b】三层时间切分：8月训练 / 9月调参(阈值+校准) / 10月测试(只看不调)
    # 按月边界严格切分，模拟真实场景（用本月数据预测下月）
    cutoff_tune = pd.Timestamp('2024-09-01')
    cutoff_test = pd.Timestamp('2024-10-01')

    train_df = feature_df[feature_df['last_action_date'] < cutoff_tune]
    val_df   = feature_df[(feature_df['last_action_date'] >= cutoff_tune) &
                          (feature_df['last_action_date'] < cutoff_test)]
    test_df  = feature_df[feature_df['last_action_date'] >= cutoff_test]

    logger.info(f"三层切分: 训练集 {len(train_df)} 条 (8月), 调参集 {len(val_df)} 条 (9月), 测试集 {len(test_df)} 条 (10月)")
    logger.info(f"训练集正样本率: {train_df['target'].mean():.4%}, 调参集正样本率: {val_df['target'].mean():.4%}")
    if len(test_df) > 0:
        logger.info(f"测试集正样本率: {test_df['target'].mean():.4%}")

    # 安全检查：训练集和调参集必须有正负样本
    for name, subset in [('训练集', train_df), ('调参集', val_df)]:
        if len(subset) == 0:
            logger.error(f"{name}为空，请检查数据时间范围")
            return
        if subset['target'].nunique() < 2:
            logger.error(f"{name}只有一种标签，无法训练")
            return

    # --- 5. 准备训练输入 ---
    # last_action_date 仅用于时间切分，不参与模型训练
    drop_cols = ['user_id', 'target', 'last_action_date']
    X_train = train_df.drop(drop_cols, axis=1, errors='ignore')
    y_train = train_df['target']

    X_val = val_df.drop(drop_cols, axis=1, errors='ignore')
    y_val = val_df['target']

    # 【方案2】排除规则层特征，不参与模型训练
    rule_cols_in_train = [c for c in RULE_LAYER_FEATURES if c in X_train.columns]
    if rule_cols_in_train:
        print(f">>> 方案2：从训练特征中排除规则层特征: {rule_cols_in_train}")
        X_train = X_train.drop(columns=rule_cols_in_train)

    # 转换类别特征为 category 类型 (LightGBM 原生支持)
    actual_cat = [c for c in CAT_FEATURES if c in X_train.columns]
    for c in actual_cat:
        X_train[c] = X_train[c].astype(str).astype('category')

    # --- 6. 训练集成模型 ---
    model_runner = IntentModel()

    # 【E12】正样本质量降权：对行为极少的正样本降权
    # target=1 但 total_actions < log1p(3) 的样本，权重降为 0.3
    noisy_mask = (y_train == 1) & (X_train['total_actions'] < np.log1p(3))
    sample_weight = np.where(noisy_mask, 0.3, 1.0)
    logger.info(f"【E12】低质正样本数量: {noisy_mask.sum()} (已降权至 0.3)")

    # 【P1】定义单调性约束特征（这些特征与购机意向应为单调递增关系）
    MONOTONE_FEATURES = ['click_to_process_rate', 'view_detail_cnt',
                         'high_intent_ratio', 'cnt_bussProcessing']

    # 启动自动调优训练
    # 【E11】n_trials=30 + 【E12】sample_weight + 【P1】monotone_features
    model_runner.auto_train(X_train, y_train, cat_features=actual_cat, n_trials=15,
                            sample_weight=sample_weight, monotone_features=MONOTONE_FEATURES)

    # --- 7. 验证集评估与阈值优化 ---
    print("\n>>> 正在执行验证集评估 <<<")

    # A. 准备验证数据 (对齐列名，填充缺失)
    # 关键：确保验证集列顺序和数量与训练集完全一致
    X_val_aligned = X_val.reindex(columns=X_train.columns, fill_value=0)

    # B. 转换验证集格式 (必须与训练集一致)
    for c in actual_cat:
        X_val_aligned[c] = X_val_aligned[c].astype(str).astype('category')

    # 确保数值列格式正确
    num_cols = [c for c in X_val_aligned.columns if c not in actual_cat]
    for col in num_cols:
        X_val_aligned[col] = pd.to_numeric(X_val_aligned[col], errors='coerce').fillna(0)

    # C. 预测
    print(" 正在计算验证集指标")
    y_pred_prob = model_runner.predict_proba(X_val_aligned)

    # 【P0】概率分布分析
    analyze_probability_distribution(y_val, y_pred_prob, label="")

    # 【E3+E8】在验证集上搜索最佳阈值
    print(">>> 【E3】在验证集上搜索最佳阈值...")
    model_runner.optimize_by_metric(y_val, y_pred_prob)

    # 使用调参集优化后的阈值
    best_threshold = model_runner.best_threshold
    y_pred = (y_pred_prob >= best_threshold).astype(int)

    # D. 调参集评估报告
    print_evaluation_report(y_val, y_pred_prob, best_threshold, "调参集评估报告 (9月数据)")

    # 【R1】调参集 Top-K% 分层评估
    print_topk_evaluation(y_val, y_pred_prob, "调参集 (9月数据)", RANKING_CONFIG)

    # --- 8. 存档（在阈值优化之后保存，确保阈值正确） ---
    if not os.path.exists('models'):
        os.makedirs('models')

    # 保存模型、特征名列表
    model_runner.save(
        'models/intent_v2.pkl',
        'models/feature_names.pkl',
        X_train.columns.tolist()
    )

    # # --- 9. 特征重要性分析 (购机相关特征弱化效果) ---
    # print("\n>>> 特征重要性分析 <<<")
    # importance = model_runner.model.feature_importance(importance_type='gain')
    # feat_names = model_runner.model.feature_name()
    # feat_imp_df = pd.DataFrame({
    #     'feature': feat_names,
    #     'importance': importance,
    #     'is_purchase_related': ['是' if f in PURCHASE_DIRECT_FEATURES else '否' for f in feat_names]
    # }).sort_values('importance', ascending=False)

    # print("\n--- 全部特征重要性排名 ---")
    # for i, row in feat_imp_df.iterrows():
    #     marker = " [购机相关-已弱化]" if row['is_purchase_related'] == '是' else ""
    #     print(f"  {row['feature']:30s}  重要性: {row['importance']:10.1f}{marker}")

    # # 汇总购机相关 vs 非购机相关的重要性占比
    # total_imp = feat_imp_df['importance'].sum()
    # purchase_imp = feat_imp_df[feat_imp_df['is_purchase_related'] == '是']['importance'].sum()
    # other_imp = total_imp - purchase_imp

    # print(f"\n--- 特征重要性占比 ---")
    # print(f"  购机相关特征占比: {purchase_imp/total_imp:.1%}")
    # print(f"  其他特征占比:     {other_imp/total_imp:.1%}")

    if FEATURE_WEAKENING_CONFIG.get('enabled', False):
        print(f"\n  弱化方法: {FEATURE_WEAKENING_CONFIG.get('method', 'N/A')}")
        print(f"  权重衰减因子: {FEATURE_WEAKENING_CONFIG.get('weight_decay_factor', 'N/A')}")
    print(f"==================================================")

    # --- 9. 测试集评估 (10月数据，只看不调) ---
    if len(test_df) > 0 and 'target' in test_df.columns and test_df['target'].nunique() >= 2:
        print("\n>>> 正在执行测试集评估 (10月数据 — 无偏估计) <<<")

        X_test = test_df.drop(drop_cols, axis=1, errors='ignore')
        y_test = test_df['target']

        # 特征对齐（与训练集列一致）
        X_test_aligned = X_test.reindex(columns=X_train.columns, fill_value=0)
        for c in actual_cat:
            X_test_aligned[c] = X_test_aligned[c].astype(str).astype('category')
        for col in [c for c in X_test_aligned.columns if c not in actual_cat]:
            X_test_aligned[col] = pd.to_numeric(X_test_aligned[col], errors='coerce').fillna(0)

        # 预测
        y_test_prob = model_runner.predict_proba(X_test_aligned)

        # P0 概率分布分析
        analyze_probability_distribution(y_test, y_test_prob, label="(测试集-10月)")

        # 测试集评估报告（使用调参集确定的阈值，不做任何调整）
        print_evaluation_report(y_test, y_test_prob, best_threshold, "测试集评估报告 (10月数据 — 无偏估计)")

        # 【R1】测试集 Top-K% 分层评估（核心验收指标）
        print_topk_evaluation(y_test, y_test_prob, "测试集 (10月数据 — 无偏估计)", RANKING_CONFIG)
    else:
        logger.warning("测试集 (10月) 为空或只有一种标签，跳过最终测试评估")

    logger.info("整个模型训练流程执行成功。")

if __name__ == "__main__":
    run_training()