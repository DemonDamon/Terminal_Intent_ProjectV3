import pandas as pd
import numpy as np
from config.business_rules import INTENT_RANK, BRANDS, ACTION_KEYWORDS
from config.feature_list import PURCHASE_DIRECT_FEATURES, FEATURE_WEAKENING_CONFIG

class UnifiedFeatureEngineer:
    def execute(self, df):
        """
        执行特征工程
        修复说明：增加了“特征泄露”过滤逻辑，剔除下单成功事件，仅保留下单前的行为特征。
        """
        # --- 1. 数据清洗与防泄露处理 ---
        # 提取所有用户的真实标签（从原始数据获取，确保即使删除了下单行，Target=1 依然保留）
        # 假设 DataLoader 已经将 target 填充在用户的每一行，这里取 max 即可获取该用户是否为正样本
        if 'target' in df.columns:
            user_targets = df.groupby('user_id')['target'].max()
        else:
            user_targets = None

        # 定义必须剔除的“结果型”事件（特征泄露源）
        # 包含：orderSuccess, server_orderSuccess (来自 INTENT_RANK 等级 5 的事件)
        LEAKY_EVENTS = ['orderSuccess', 'server_orderSuccess', '支付成功', '办理成功']
        
        # 创建用于特征计算的“纯净”数据集（只保留发生转化之前的行为）
        df_features = df[~df['标识符'].isin(LEAKY_EVENTS)].copy()
        
        # 如果某个用户只有“下单成功”这一条日志（极端异常数据），他会在 df_features 中消失
        # 但这符合逻辑，因为没有前序行为我们无法预测。
        
        # 映射意图等级 (注意：此时最高等级应该只有 4，不会出现 5)
        df_features['intent_level'] = df_features['标识符'].map(INTENT_RANK).fillna(0)
        
        # 排序，确保时序正确
        df_features = df_features.sort_values(['user_id', 'trigger_time'])

        # --- 2. 向量化计算行为间隔 (高性能版) ---
        df_features['diff_time'] = df_features.groupby('user_id')['trigger_time'].diff().dt.total_seconds()
        
        # 使用过滤后的数据建立分组对象
        grouped = df_features.groupby('user_id')

        # 【风险防御】拼写一致性检查：bussinessProcessing 必须与数据中的标识符一致
        known_identifiers = df_features['标识符'].unique()
        if 'bussinessProcessing' not in known_identifiers:
            print("[Warning] 数据中未出现 'bussinessProcessing' 事件，cnt_bussProcessing 将全部为 0")
            print(f"  数据中实际标识符: {sorted(known_identifiers)[:10]}...")

        # 初始化特征表
        fe = pd.DataFrame(index=grouped.groups.keys())
        fe.index.name = 'user_id'

        # --- 维度一：基础统计 ---
        fe['cnt_eventClick'] = grouped['标识符'].apply(lambda x: (x == 'eventClick').sum())
        fe['cnt_bussProcessing'] = grouped['标识符'].apply(lambda x: (x == 'bussinessProcessing').sum())
        fe['total_actions'] = fe['cnt_eventClick'] + fe['cnt_bussProcessing']
        
        # 关键词匹配 (buy, spec, fail)
        fe['click_buy_now_cnt'] = grouped['办理步骤'].apply(lambda x: x.str.contains(ACTION_KEYWORDS['buy'], na=False).sum())
        fe['click_specs_cnt'] = grouped['区域名称'].apply(lambda x: x.str.contains(ACTION_KEYWORDS['spec'], na=False).sum())
        # 重点：办理失败/异常是强烈的购买受阻信号
        fe['process_fail_count'] = grouped['办理步骤'].apply(lambda x: x.str.contains(ACTION_KEYWORDS['fail'], na=False).sum())
        fe['view_detail_cnt'] = grouped['页面名称'].apply(lambda x: x.str.contains('详情', na=False).sum())

        # 【E4】view_detail_cnt 压缩：打破单特征垄断（原始值独占 59.40% 重要性）
        # 先用原始值计算比例特征，再做 log1p 压缩
        fe['detail_view_ratio'] = fe['view_detail_cnt'] / (fe['total_actions'] + 1)
        fe['view_detail_cnt'] = np.log1p(fe['view_detail_cnt'])

        # --- 维度二：转化深度与质量 ---
        fe['max_intent_level'] = grouped['intent_level'].max()
        fe['last_intent_level'] = grouped['intent_level'].last()
        fe['avg_intent_level'] = grouped['intent_level'].mean()
        
        # 意图密度
        fe['intent_density'] = fe['max_intent_level'] / (fe['total_actions'] + 1)
        # 高意向行为占比 (Level >= 3: 登录、试算)
        fe['high_intent_ratio'] = grouped['intent_level'].apply(lambda x: (x >= 3).sum() / (len(x) + 0.1))
        # 转化漏斗：点击到试算的转化率
        fe['click_to_process_rate'] = np.sqrt(fe['cnt_bussProcessing'] / (fe['cnt_eventClick'] + 1))

        # --- 维度三：时序特征 ---
        fe['avg_action_interval'] = grouped['diff_time'].mean().fillna(0)
        fe['max_action_interval'] = grouped['diff_time'].max().fillna(0)
        fe['std_action_interval'] = grouped['diff_time'].std().fillna(0)
        fe['action_frequency'] = fe['total_actions'] / (fe['max_action_interval'] + 1)

        # 【E6】意向趋势：直接计算，不再保留冗余的中间变量 recent_3/5_avg_intent
        fe['intent_trend'] = grouped['intent_level'].apply(
            lambda x: x.tail(3).mean() - x.tail(5).mean()
        )
        # 【E6】意图落差：最高意图 vs 最后意图，提取 max/last 之间的差异信息
        fe['intent_level_gap'] = fe['max_intent_level'] - fe['last_intent_level']

        # --- 维度四：价格能力 (工业级降权版) ---
        # 仅基于有价格记录的行计算
        price_g = df_features[df_features['商品价格'] > 0].groupby('user_id')['商品价格']
        # 注意：这里可能部分用户没有价格记录，reindex 会在最后填充 0
        p_max = price_g.max()

        def get_price_level(p):
            if pd.isna(p): return 0
            if p >= 8000: return 4
            if p >= 5000: return 3
            if p >= 2000: return 2
            return 1 if p > 0 else 0

        # 这里需要先对齐索引，因为 price_g 可能比 fe 少
        fe['price_level'] = p_max.reindex(fe.index).apply(get_price_level)

        # --- 维度五：业务标签与序列 ---
        def get_top_brand(names):
            name_str = " ".join(names.astype(str))
            for b in BRANDS:
                if b in name_str: return b
            return 'Other'

        fe['preferred_brand'] = grouped['商品名称'].apply(get_top_brand)
        fe['last_3_actions_seq'] = grouped['标识符'].apply(lambda x: "_".join(x.tail(3).astype(str).tolist()))
        fe['last_login_type'] = grouped['登录方式'].last()
        fe['last_platform'] = grouped['平台类型'].last()
        
        # 登录位置特征
        fe['has_login'] = grouped['标识符'].apply(lambda x: 1 if 'zdscLogin' in x.values else 0)

        # --- 维度六：针对新数据的高级特征扩充 ---
        # 精细化办理步骤
        fe['step_sms_input'] = grouped['办理步骤'].apply(lambda x: x.str.contains('输入验证码', na=False).any().astype(int))
        fe['step_sms_submit'] = grouped['办理步骤'].apply(lambda x: x.str.contains('提交', na=False).any().astype(int))

        # 区分【手机终端】和【福袋业务】
        phone_keywords = 'iPhone|华为Mate|Pura|OPPO|vivo|荣耀|小米|REDMI|机价立减'
        fe['view_phone_cnt'] = grouped['商品名称'].apply(lambda x: x.str.contains(phone_keywords, case=False, na=False).sum())
        fe['is_pure_gift_user'] = grouped['商品名称'].apply(lambda x: x.str.contains('福袋|年包|券', na=False).all().astype(int))

        # 登录偏好
        fe['login_sms_cnt'] = grouped['登录方式'].apply(lambda x: (x == '短信认证').sum())
        fe['login_quick_cnt'] = grouped['登录方式'].apply(lambda x: (x == '一键登录').sum())

        # 【E5】新增意图前兆特征 — 补充用户犹豫/决策过程信号
        # 跨天回访：多天反复来看 = 在认真考虑
        fe['active_days'] = grouped['trigger_time'].apply(lambda x: x.dt.date.nunique())
        fe['is_multi_day_visitor'] = (fe['active_days'] >= 2).astype(int)
        # 访问时间跨度
        fe['visit_span_days'] = grouped['trigger_time'].apply(
            lambda x: (x.max() - x.min()).total_seconds() / 86400.0
        )
        # 浏览商品多样性：看了很多不同手机 = 在比价
        fe['unique_items_viewed'] = grouped['商品名称'].nunique()
        # 重复浏览率：总浏览/去重商品，越高 = 反复看同一款（注意用 log1p 之前的原始值已不可取，此处用 exp 还原再算）
        fe['repeat_view_ratio'] = np.expm1(fe['view_detail_cnt']) / (fe['unique_items_viewed'] + 1)
        # 【E9】repeat_view_ratio 压缩：当前占 13.87%，与 view_detail_cnt 同等 log1p 待遇
        fe['repeat_view_ratio'] = np.log1p(fe['repeat_view_ratio'])
        # 会话数：30 分钟无操作视为新会话
        SESSION_GAP = 1800
        fe['session_count'] = grouped['diff_time'].apply(lambda x: (x > SESSION_GAP).sum() + 1)
        fe['actions_per_session'] = fe['total_actions'] / (fe['session_count'] + 0.1)
        # 意图加速度：最后 3 步中意图升高的次数，越高 = 接近决策
        fe['intent_acceleration'] = grouped['intent_level'].apply(
            lambda x: (x.diff().tail(3) > 0).sum() if len(x) >= 3 else 0
        )

        # 【E10】交叉特征：提升中层特征的联合信号强度

        # 1. 跨天反复犹豫 — active_days 和 repeat_view_ratio 的交互
        fe['revisit_intensity'] = fe['active_days'] * fe['repeat_view_ratio']

        # 3. 看机型占比 — 区分"随便逛"和"认真选机"
        fe['phone_browse_ratio'] = fe['view_phone_cnt'] / (fe['total_actions'] + 1)

        # 4. 意向趋势与落差交互 — 同时捕获方向和幅度
        fe['intent_momentum'] = fe['intent_trend'] * fe['intent_level_gap']

        # 【E6】删除 conversion_efficiency（与 click_to_process_rate 公式完全一致，属冗余特征）

        # --- 3. 最终收尾 ---
        # 【E1】保留时间信息用于时间切分（训练管道需要此列做 8月/9月 切分）
        fe['last_action_date'] = grouped['trigger_time'].max()

        fe = fe.reset_index()
        
        # 填充 Target：将之前提取的 user_targets 映射回来
        if user_targets is not None:
            # 注意：map 需要用 set_index 后的 user_id，或者直接 map
            fe['target'] = fe['user_id'].map(user_targets).fillna(0)

        # 类别特征处理
        cat_cols = ['preferred_brand', 'last_3_actions_seq', 'last_login_type', 'last_platform']
        for col in cat_cols:
            if col in fe.columns:
                fe[col] = fe[col].fillna('unknown').astype(str)

        # --- 4. 购机相关特征弱化处理 ---
        fe = self._weaken_purchase_features(fe)

        return fe.fillna(0)

    def _weaken_purchase_features(self, fe):
        """
        弱化与购机行为直接相关的特征
        目的：降低验证码、点击购买等强信号特征的影响，让模型更关注其他间接特征
        """
        if not FEATURE_WEAKENING_CONFIG.get('enabled', False):
            return fe

        method = FEATURE_WEAKENING_CONFIG.get('method', 'log_transform')
        features_to_weaken = [f for f in PURCHASE_DIRECT_FEATURES if f in fe.columns]

        if not features_to_weaken:
            return fe

        print(f">>> 正在弱化购机相关特征 (方法: {method}): {features_to_weaken}")

        for col in features_to_weaken:
            if col not in fe.columns:
                continue

            if method == 'log_transform':
                # 对数变换：降低大值的影响，保留趋势
                base = FEATURE_WEAKENING_CONFIG.get('log_base', 2)
                fe[col] = np.log1p(fe[col]) / np.log(base)

            elif method == 'sqrt_transform':
                # 平方根变换：更温和的压缩
                fe[col] = np.sqrt(np.abs(fe[col])) * np.sign(fe[col])

            elif method == 'cap_percentile':
                # 百分位截断：限制极端值
                cap_pct = FEATURE_WEAKENING_CONFIG.get('cap_percentile', 90)
                cap_value = np.percentile(fe[col].dropna(), cap_pct)
                fe[col] = fe[col].clip(upper=cap_value)

            elif method == 'min_max_compress':
                # Min-Max 压缩到 [0, 1] 范围
                min_val = fe[col].min()
                max_val = fe[col].max()
                if max_val > min_val:
                    fe[col] = (fe[col] - min_val) / (max_val - min_val)

        return fe