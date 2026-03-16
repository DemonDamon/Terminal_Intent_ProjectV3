
CAT_FEATURES = [
    'last_login_type',
    'last_platform',
    'preferred_brand',
    'last_3_actions_seq'
]

# 规则层特征：从模型训练中排除，但保留用于规则标签判定
# 这些特征过强会导致模型过度依赖，改为在预测后通过规则层叠加
RULE_LAYER_FEATURES = [
    'step_sms_input',         # 输入验证码步骤 (0/1)
    'step_sms_submit',        # 提交验证码步骤 (0/1)
    'click_buy_now_cnt',      # 点击"立即购买"等按钮次数
]

# 规则层标签配置
RULE_LAYER_CONFIG = {
    'enabled': True,
    # 确定性等级判定规则（按优先级从高到低）
    'rules': [
        {
            'tag': '高确定性',           # 已进入购买流程（验证码+提交）
            'conditions': {'step_sms_input': 1, 'step_sms_submit': 1},
        },
        {
            'tag': '较高确定性',         # 已输入验证码但未提交
            'conditions': {'step_sms_input': 1},
        },
        {
            'tag': '有购买动作',         # 点击了购买按钮
            'conditions': {'click_buy_now_cnt_gte': 1},
        },
    ],
    'default_tag': '纯意向',             # 无购买行为信号
}

# 保留向后兼容（其他文件可能引用）
PURCHASE_DIRECT_FEATURES = RULE_LAYER_FEATURES

# 特征弱化配置（方案2下关闭，因为特征直接从训练中排除）
FEATURE_WEAKENING_CONFIG = {
    'method': 'log_transform',
    'log_base': 2,
    'cap_percentile': 90,
    'weight_decay_factor': 0.5,
    'enabled': False,   # 方案2：关闭弱化，改用排除+规则层
}

# 【R1】Top-K% 排名制输出配置
# 替代固定阈值制，利用模型排序能力（ROC-AUC 0.99+），对分布偏移天然免疫
RANKING_CONFIG = {
    'enabled': True,
    'top_k_pct': 0.0025,      # Top 0.25%
    'min_k': 500,              # 下限：保证最少工作量
    'max_k': 3000,             # 上限：超出跟进能力无意义
    'min_prob': 0.30,          # 最低概率门槛（安全护栏）
    'tiers': {
        'S': {'pct_upper': 0.0010, 'desc': '最高置信'},   # Top 0.10%
        'A': {'pct_upper': 0.0025, 'desc': '高置信'},     # Top 0.25%
        'B': {'pct_upper': 0.0050, 'desc': '中置信'},     # Top 0.50%
    },
}
