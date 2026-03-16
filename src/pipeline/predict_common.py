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
from config.feature_list import CAT_FEATURES, RULE_LAYER_FEATURES, RULE_LAYER_CONFIG, RANKING_CONFIG


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
        except Exception:
            # 覆盖 joblib.load 反序列化异常等，确保异常时回退默认阈值不中断
            print("[Smart Load] 外部阈值文件读取异常，使用业务默认阈值: 0.500")
            return 0.5

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


def apply_rule_layer(feature_df, rule_config=None):
    """
    【方案2 Step2】规则层叠加：基于购买行为特征为用户打标签。

    根据 step_sms_input / step_sms_submit / click_buy_now_cnt 的值，
    按优先级从高到低匹配规则，给每个用户分配确定性等级标签。

    标签含义：
      - 高确定性：已进入购买流程（验证码输入+提交）
      - 较高确定性：已输入验证码但未提交
      - 有购买动作：点击了购买按钮
      - 纯意向：仅有浏览/意向信号，无直接购买行为
    """
    if rule_config is None:
        rule_config = RULE_LAYER_CONFIG

    if not rule_config.get('enabled', False):
        return feature_df

    default_tag = rule_config.get('default_tag', '纯意向')
    rules = rule_config.get('rules', [])

    def evaluate_user(row):
        for rule in rules:
            conditions = rule.get('conditions', {})
            matched = True
            for key, expected in conditions.items():
                # 支持 _gte 后缀（>=判断），如 click_buy_now_cnt_gte: 1
                if key.endswith('_gte'):
                    col_name = key[:-4]
                    val = row.get(col_name, 0)
                    if pd.isna(val) or val < expected:
                        matched = False
                        break
                else:
                    val = row.get(key, 0)
                    if pd.isna(val) or val != expected:
                        matched = False
                        break
            if matched:
                return rule['tag']
        return default_tag

    feature_df['certainty_tag'] = feature_df.apply(evaluate_user, axis=1)

    # 打印规则层统计
    tag_counts = feature_df['certainty_tag'].value_counts()
    print("\n>>> 规则层标签分布:")
    for tag, cnt in tag_counts.items():
        print(f"    {tag}: {cnt} 名 ({cnt / len(feature_df):.1%})")

    return feature_df


def save_predictions(feature_df, probs, threshold, save_path, ranking_config=None):
    """
    将预测结果写入 feature_df 并导出 CSV。
    【R1】支持 Top-K% 排名制输出：含排名、置信等级、规则层标签。
    兼容模式：同时保留阈值制标签。
    """
    if ranking_config is None:
        ranking_config = RANKING_CONFIG

    feature_df['intent_probability'] = probs

    # --- 阈值制标签（兼容保留） ---
    feature_df['intent_label'] = np.where(probs >= threshold, '1', '2')

    # --- 【R1】排名制输出 ---
    if ranking_config.get('enabled', False):
        n_total = len(probs)
        sorted_indices = np.argsort(probs)[::-1]

        # 计算 K：clamp(总用户数 × K%, 下限, 上限)
        k_raw = int(n_total * ranking_config['top_k_pct'])
        k = max(ranking_config['min_k'], min(ranking_config['max_k'], k_raw))
        k = min(k, n_total)

        # 安全门槛截断
        min_prob = ranking_config.get('min_prob', 0.30)
        actual_k = k
        sorted_probs = probs[sorted_indices]
        for i in range(k):
            if sorted_probs[i] < min_prob:
                actual_k = i
                break

        # 分配排名和置信等级
        tiers_config = ranking_config.get('tiers', {})
        rank_col = np.full(n_total, 0, dtype=int)
        tier_col = np.full(n_total, '', dtype=object)

        for rank_pos, idx in enumerate(sorted_indices[:actual_k]):
            rank_col[idx] = rank_pos + 1
            rank_pct = (rank_pos + 1) / n_total
            tier = 'B'
            for tier_name in ['S', 'A', 'B']:
                if tier_name in tiers_config and rank_pct <= tiers_config[tier_name]['pct_upper']:
                    tier = tier_name
                    break
            tier_col[idx] = tier

        feature_df['rank'] = rank_col
        feature_df['confidence_tier'] = tier_col
        # 排名制标签：入选 Top-K% 的标为 '1'
        feature_df['intent_label_topk'] = np.where(rank_col > 0, '1', '2')

    # 【方案2 Step2】叠加规则层标签
    feature_df = apply_rule_layer(feature_df)

    # 导出
    final_output = feature_df.copy()
    output_cols = [c for c in ['rank', 'user_id', 'intent_probability', 'confidence_tier',
                                'certainty_tag', 'intent_label', 'intent_label_topk']
                   if c in final_output.columns]
    output = final_output[output_cols]

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    output.sort_values(by='intent_probability', ascending=False).to_csv(
        save_path, index=False, encoding='utf-8-sig'
    )

    return feature_df


def print_prediction_summary(feature_df, probs, threshold, save_path):
    """
    打印预测结果统计汇总。
    【R1】新增排名制统计。
    """
    total = len(feature_df)

    print(f"\n" + "=" * 50)
    print("          终端商机挖掘 - 预测任务完成")
    print(f"=" * 50)

    # 排名制统计（优先展示）
    if 'rank' in feature_df.columns and 'confidence_tier' in feature_df.columns:
        topk_count = (feature_df['rank'] > 0).sum()
        print(f"\n--- 【R1】Top-K% 排名制输出 ---")
        print(f"1. 高潜名单人数 : {topk_count:,} 名")
        if total > 0:
            print(f"2. 占比         : {topk_count / total:.2%}")
        # 分层统计
        tier_counts = feature_df[feature_df['rank'] > 0]['confidence_tier'].value_counts()
        for tier in ['S', 'A', 'B']:
            cnt = tier_counts.get(tier, 0)
            if cnt > 0:
                print(f"   {tier} 级: {cnt:,} 名")
        print(f"3. 最高意向得分 : {probs.max():.4f}")
        if topk_count > 0:
            min_topk_prob = feature_df[feature_df['rank'] > 0]['intent_probability'].min()
            print(f"4. 名单最低概率 : {min_topk_prob:.4f}")

    # 阈值制统计（兼容模式）
    intent_1_count = (feature_df['intent_label'] == '1').sum()
    print(f"\n--- 阈值制参考 (兼容) ---")
    print(f"1. 判定阈值     : {threshold:.3f}")
    print(f"2. 识别商机数量 : {intent_1_count:,} 名")
    if total > 0:
        print(f"3. 商机占比     : {intent_1_count / total:.1%}")
    print(f"4. 名单保存路径 : {save_path}")

    # 规则层标签统计
    if 'certainty_tag' in feature_df.columns:
        # 以排名制入选名单为基准（如果启用）
        if 'rank' in feature_df.columns:
            high_intent = feature_df[feature_df['rank'] > 0]
            label_desc = "高潜名单(Top-K%)"
        else:
            high_intent = feature_df[feature_df['intent_label'] == '1']
            label_desc = "高意向(阈值制)"
        print(f"\n--- 营销策略分层 ({label_desc}) ---")
        if len(high_intent) > 0:
            tag_dist = high_intent['certainty_tag'].value_counts()
            for tag, cnt in tag_dist.items():
                print(f"  {tag}: {cnt} 名")

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
