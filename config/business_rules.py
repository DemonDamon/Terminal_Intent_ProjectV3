# 广东移动终端业务规则映射
INTENT_RANK = {
    'zdscH5_PageView': 1, 'eventClick': 2, 'zdscLogin': 3, 
    'bussinessProcessing': 4, 'orderSuccess': 5, 'server_orderSuccess': 5
}

BRANDS = ['Apple', '苹果', '华为', 'HUAWEI', '荣耀', '小米', 'OPPO', 'VIVO', '小天才']

ACTION_KEYWORDS = {
    'buy': '立即购买|立即申请|去结算|立即下单|提交订单',
    'sms': '短信二次确认|短信验证|获取验证码',
    'fail': '失败|错误|异常|未通过|校验失败',
    'spec': '颜色|内存|容量|规格|配置|选择产品'
}