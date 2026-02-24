import sys
import os
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report, f1_score, precision_score, recall_score

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import DataLoader
from src.features.unified_fe import UnifiedFeatureEngineer
from src.models.model_lgb import IntentModel
from config.feature_list import CAT_FEATURES, PURCHASE_DIRECT_FEATURES, FEATURE_WEAKENING_CONFIG

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

    # --- 4. 数据拆分 ---
    # 确保 target 存在
    if 'target' not in feature_df.columns:
        logger.error("特征表中缺少 target 列")
        return

    train_df, test_df = train_test_split(
        feature_df, test_size=0.2, random_state=42, stratify=feature_df['target']
    )

    # --- 5. 准备训练输入 ---
    X_train = train_df.drop(['user_id', 'target'], axis=1, errors='ignore')
    y_train = train_df['target']

    # 转换类别特征为 category 类型 (LightGBM 原生支持)
    actual_cat = [c for c in CAT_FEATURES if c in X_train.columns]
    for c in actual_cat:
        X_train[c] = X_train[c].astype(str).astype('category')

    # --- 6. 训练集成模型 ---
    model_runner = IntentModel()
    
    # 启动自动调优训练
    # n_trials 可以根据时间调整，建议 15-30 次
    model_runner.auto_train(X_train, y_train, cat_features=actual_cat, n_trials=8)

    # --- 7. 存档 ---
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # 保存模型、特征名列表
    model_runner.save(
        'models/intent_v2.pkl', 
        'models/feature_names.pkl', 
        X_train.columns.tolist()
    )

    # --- 8. 最终回测评估 (修复部分) ---
    print("\n>>> 正在执行最终回测评估 <<<")

    # A. 准备测试数据 (对齐列名，填充缺失)
    X_test_raw = test_df.drop(['user_id', 'target'], axis=1, errors='ignore')
    # 关键：确保测试集列顺序和数量与训练集完全一致
    X_test = X_test_raw.reindex(columns=X_train.columns, fill_value=0)
    y_test = test_df['target']

    # B. 转换测试集格式 (必须与训练集一致)
    for c in actual_cat:
        X_test[c] = X_test[c].astype(str).astype('category')
    
    # 确保数值列格式正确
    num_cols = [c for c in X_test.columns if c not in actual_cat]
    for col in num_cols:
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)

    # C. 预测
    print(" 正在计算回测指标")
    # 直接调用 predict_proba，不再需要 transform_features
    y_pred_prob = model_runner.predict_proba(X_test)
    
    # 使用自动优化的阈值
    best_threshold = model_runner.best_threshold
    y_pred = (y_pred_prob >= best_threshold).astype(int)

    # D. 计算指标
    try:
        roc_val = roc_auc_score(y_test, y_pred_prob)
    except:
        roc_val = np.nan
        
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recall, precision)
    
    # KS值计算
    df_ks = pd.DataFrame({'label': y_test, 'prob': y_pred_prob})
    df_ks['good'] = 1 - df_ks['label']
    df_ks = df_ks.sort_values(by='prob', ascending=False)
    df_ks['cum_bad'] = df_ks['label'].cumsum() / df_ks['label'].sum()
    df_ks['cum_good'] = df_ks['good'].cumsum() / df_ks['good'].sum()
    ks_value = (df_ks['cum_bad'] - df_ks['cum_good']).abs().max()

    print(f"\n==================================================")
    print(f" 最终回测报告")
    print(f"==================================================")
    print(f"1. ROC-AUC :       {roc_val:.4f}")
    print(f"2. PR-AUC  :       {pr_auc:.4f}")
    print(f"3. KS值    :       {ks_value:.4f}")
    print(f"4. Precision :   {precision_score(y_test, y_pred):.2%}")
    print(f"5. Recall    :   {recall_score(y_test, y_pred):.2%}")
    print(f"6. F1-Score  :      {f1_score(y_test, y_pred):.4f}")
    print(f"--------------------------------------------------")
    print(f"判定阈值设定为: {best_threshold:.3f}")
    print(f"==================================================")

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