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
from config.feature_list import CAT_FEATURES, RULE_LAYER_FEATURES, RULE_LAYER_CONFIG, FEATURE_WEAKENING_CONFIG

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_logger():
    return logging.getLogger(__name__)

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

    # 【E1】按时间切分：8月训练 / 9月验证 / 10月由 predict_pipline_backtest.py 独立回测
    cutoff_train = pd.Timestamp('2024-09-01')
    cutoff_val   = pd.Timestamp('2024-10-01')

    train_df = feature_df[feature_df['last_action_date'] < cutoff_train]
    val_df   = feature_df[(feature_df['last_action_date'] >= cutoff_train) &
                          (feature_df['last_action_date'] < cutoff_val)]

    logger.info(f"时间切分完成: 训练集 {len(train_df)} 条 (8月前), 验证集 {len(val_df)} 条 (9月)")
    logger.info(f"训练集正样本率: {train_df['target'].mean():.4%}, 验证集正样本率: {val_df['target'].mean():.4%}")

    # 安全检查：确保两个集合都有正负样本
    for name, subset in [('训练集', train_df), ('验证集', val_df)]:
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
    
    # 启动自动调优训练
    # 【E2】n_trials 保持 15，配合 scale_pos_weight 扩大到 1~100(log) 更高效
    model_runner.auto_train(X_train, y_train, cat_features=actual_cat, n_trials=15)

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

    # 【E3】在验证集上搜索最佳阈值（而非训练集）
    print(">>> 【E3】在验证集上搜索最佳阈值...")
    model_runner.optimize_by_metric(y_val, y_pred_prob)

    # 使用验证集优化后的阈值
    best_threshold = model_runner.best_threshold
    y_pred = (y_pred_prob >= best_threshold).astype(int)

    # D. 计算指标
    try:
        roc_val = roc_auc_score(y_val, y_pred_prob)
    except ValueError:
        roc_val = np.nan

    precision, recall, _ = precision_recall_curve(y_val, y_pred_prob)
    pr_auc = auc(recall, precision)

    # KS值计算
    df_ks = pd.DataFrame({'label': y_val, 'prob': y_pred_prob})
    df_ks['good'] = 1 - df_ks['label']
    df_ks = df_ks.sort_values(by='prob', ascending=False)
    df_ks['cum_bad'] = df_ks['label'].cumsum() / df_ks['label'].sum()
    df_ks['cum_good'] = df_ks['good'].cumsum() / df_ks['good'].sum()
    ks_value = (df_ks['cum_bad'] - df_ks['cum_good']).abs().max()

    print(f"\n==================================================")
    print(f" 验证集评估报告 (9月数据)")
    print(f"==================================================")
    print(f"1. ROC-AUC :       {roc_val:.4f}")
    print(f"2. PR-AUC  :       {pr_auc:.4f}")
    print(f"3. KS值    :       {ks_value:.4f}")
    print(f"4. Precision :   {precision_score(y_val, y_pred):.2%}")
    print(f"5. Recall    :   {recall_score(y_val, y_pred):.2%}")
    print(f"6. F1-Score  :      {f1_score(y_val, y_pred):.4f}")
    print(f"--------------------------------------------------")
    print(f"判定阈值设定为: {best_threshold:.3f}")
    print(f"==================================================")

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

    logger.info("整个模型训练流程执行成功。")

if __name__ == "__main__":
    run_training()