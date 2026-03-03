
CAT_FEATURES = [
    'last_login_type',
    'last_platform',
    'preferred_brand',
    'last_3_actions_seq'
]

# 购机行为直接相关特征（需要弱化的特征）
# 这些特征在已购机用户中表现过强，导致标签为2的人员预测分数偏低

# 二值特征（0/1）：值变换无效，仅通过 CEGB 重罚在模型层弱化
PURCHASE_BINARY_FEATURES = [
    'step_sms_input',         # 输入验证码步骤 (0/1) — 重要性58.75%，需最强惩罚
    'step_sms_submit',        # 提交验证码步骤 (0/1) — 重要性8%，中等惩罚
]

# 计数型特征：通过值变换 + CEGB 轻罚双重弱化
PURCHASE_COUNT_FEATURES = [
    'click_buy_now_cnt',      # 点击"立即购买"等按钮次数 — 重要性14%
]

# 合并列表（向后兼容）
PURCHASE_DIRECT_FEATURES = PURCHASE_BINARY_FEATURES + PURCHASE_COUNT_FEATURES

# 特征弱化配置
FEATURE_WEAKENING_CONFIG = {
    # 方法选择: 'log_transform' | 'sqrt_transform' | 'cap_percentile' | 'weight_decay'
    # 注意：仅对计数型特征生效，二值特征自动跳过值变换
    'method': 'log_transform',

    # 对数变换参数 (method='log_transform')
    'log_base': 5,  # 使用 log5，降低增长速度

    # 百分位截断参数 (method='cap_percentile')
    'cap_percentile': 90,  # 在90百分位截断

    # CEGB 特征惩罚参数 (用于 LightGBM cegb_penalty_feature_coupled)
    # 逐特征惩罚：根据原始重要性占比差异化设置
    'feature_penalties': {
        'step_sms_input': 20.0,       # 重要性58.75%，需最强惩罚
        'step_sms_submit': 5.0,       # 重要性8%，中等惩罚
        'click_buy_now_cnt': 3.0,     # 重要性14%，计数型多次分裂累积罚金
    },
    'weight_decay_factor': 0.6,           # 未列入字典的弱化特征的默认惩罚

    # 每个节点分裂时的特征采样比例，进一步限制弱化特征被选中的概率
    'feature_fraction_bynode': 0.8,

    # 是否启用特征弱化
    'enabled': True,
}
