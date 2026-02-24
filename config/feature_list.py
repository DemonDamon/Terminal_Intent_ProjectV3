
CAT_FEATURES = [
    'last_login_type',
    'last_platform',
    'preferred_brand',
    'last_3_actions_seq'
]

# 购机行为直接相关特征（需要弱化的特征）
# 这些特征在已购机用户中表现过强，导致标签为2的人员预测分数偏低
PURCHASE_DIRECT_FEATURES = [
    'click_buy_now_cnt',      # 点击"立即购买"等按钮次数
    'step_sms_input',         # 输入验证码步骤
    'step_sms_submit',        # 提交验证码步骤
    'cnt_bussProcessing',     # 业务处理/办理次数
    'process_fail_count',     # 办理失败次数
    'max_intent_level',       # 最大意图等级
    'last_intent_level',      # 最后意图等级
]

# 特征弱化配置
FEATURE_WEAKENING_CONFIG = {
    # 方法选择: 'log_transform' | 'sqrt_transform' | 'cap_percentile' | 'weight_decay'
    'method': 'log_transform',

    # 对数变换参数 (method='log_transform')
    'log_base': 2,  # 使用 log2，降低增长速度

    # 百分位截断参数 (method='cap_percentile')
    'cap_percentile': 90,  # 在90百分位截断

    # 权重衰减参数 (用于 LightGBM feature_contribs)
    'weight_decay_factor': 0.5,  # 这些特征的重要性降低50%

    # 是否启用特征弱化
    'enabled': True,
}