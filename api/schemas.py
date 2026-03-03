"""
API 数据模型定义 (Pydantic Schemas)
===================================
定义所有API请求和响应的数据结构。

遵循规范:
- 使用Pydantic v2进行数据验证
- 所有字段都有类型注解和说明
- 提供示例值便于文档生成
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


# ============================================================
# 枚举类型定义
# ============================================================

class IntentLabel(str, Enum):
    """意向标签枚举"""
    HIGH = "1"   # 高购买意向
    LOW = "2"    # 低购买意向


class HealthStatus(str, Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


# ============================================================
# 用户行为日志数据模型 (输入)
# ============================================================

class UserBehaviorLog(BaseModel):
    """
    单条用户行为日志数据
    对应原始CSV的24列数据结构
    """
    user_id: str = Field(..., description="用户唯一标识 (base64编码或明文)", example="USER_001")
    target: Optional[int] = Field(2, description="标签(预测时可不传): 1=已购买, 2=未购买", example=2)
    identifier: str = Field(..., description="事件标识符", example="eventClick")
    trigger_time: str = Field(..., description="触发时间 (格式: YYYYMMDDHHMMSS)", example="20250801100001")
    page_name: Optional[str] = Field("", description="页面名称", example="广东移动终端商城")
    page_url: Optional[str] = Field("", description="页面URL", example="URL_A")
    platform_type: Optional[str] = Field("", description="平台类型", example="APP")
    area_name: Optional[str] = Field("", description="区域名称", example="宫格区")
    sub_area_name: Optional[str] = Field("", description="子区域名称", example="-")
    traffic_slot_name: Optional[str] = Field("", description="流量位名称", example="华为")
    sub_traffic_slot_name: Optional[str] = Field("", description="子流量位名称", example="P001")
    current_item_name: Optional[str] = Field("", description="当前流量位商品名称", example="华为Mate70")
    current_item_price: Optional[float] = Field(0, description="当前商品价格", example=7999.0)
    p_code: Optional[str] = Field("", description="产品编码", example="P001")
    item_price: Optional[float] = Field(0, description="商品价格", example=7999.0)
    item_name: Optional[str] = Field("", description="商品名称", example="华为Mate70")
    coupon_name: Optional[str] = Field("", description="使用优惠券名称", example="-")
    process_step: Optional[str] = Field("", description="办理步骤", example="立即购买")
    interface_name: Optional[str] = Field("", description="接口名称", example="INF_01")
    source_page: Optional[str] = Field("", description="来源页面", example="HOME")
    source_area: Optional[str] = Field("", description="来源区域", example="ZONE_1")
    cart_count: Optional[int] = Field(0, description="加购数量", example=1)
    login_type: Optional[str] = Field("", description="登录方式", example="一键登录")
    phone_seq: Optional[str] = Field("", description="手机序列号", example="1001")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "USER_001",
                "target": 2,
                "identifier": "eventClick",
                "trigger_time": "20250801100001",
                "page_name": "广东移动终端商城",
                "page_url": "URL_A",
                "platform_type": "APP",
                "area_name": "宫格区",
                "sub_area_name": "-",
                "traffic_slot_name": "华为",
                "sub_traffic_slot_name": "P001",
                "current_item_name": "华为Mate70",
                "current_item_price": 7999.0,
                "p_code": "P001",
                "item_price": 7999.0,
                "item_name": "华为Mate70",
                "coupon_name": "-",
                "process_step": "立即购买",
                "interface_name": "INF_01",
                "source_page": "HOME",
                "source_area": "ZONE_1",
                "cart_count": 1,
                "login_type": "一键登录",
                "phone_seq": "1001"
            }
        }


# ============================================================
# 预测请求模型
# ============================================================

class PredictRequest(BaseModel):
    """
    预测请求模型
    支持单用户多条行为记录或多用户批量预测
    """
    user_actions: List[UserBehaviorLog] = Field(
        ...,
        min_length=1,
        description="用户行为数据列表(至少1条，建议按时间排序)"
    )
    threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="自定义判定阈值 (0-1)，不传则使用模型默认阈值"
    )
    return_features: Optional[bool] = Field(
        False,
        description="是否返回特征详情 (调试用)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "user_actions": [
                    {
                        "user_id": "USER_001",
                        "identifier": "eventClick",
                        "trigger_time": "20250801100001",
                        "page_name": "广东移动终端商城",
                        "platform_type": "APP",
                        "item_name": "华为Mate70",
                        "item_price": 7999,
                        "process_step": "立即购买",
                        "login_type": "一键登录"
                    },
                    {
                        "user_id": "USER_001",
                        "identifier": "bussinessProcessing",
                        "trigger_time": "20250801100030",
                        "page_name": "广东移动终端商城",
                        "platform_type": "APP",
                        "item_name": "华为Mate70",
                        "item_price": 7999,
                        "process_step": "输入验证码",
                        "login_type": "短信认证"
                    }
                ],
                "threshold": 0.5,
                "return_features": False
            }
        }


# ============================================================
# 预测结果模型
# ============================================================

class UserPrediction(BaseModel):
    """单个用户的预测结果"""
    user_id: str = Field(..., description="用户唯一标识")
    intent_label: IntentLabel = Field(..., description="意向标签: 1=高意向, 2=低意向")
    intent_probability: float = Field(..., ge=0.0, le=1.0, description="购买意向概率 (0-1)")
    certainty_tag: Optional[str] = Field(
        '纯意向',
        description="确定性等级标签: 高确定性/较高确定性/有购买动作/纯意向"
    )
    features: Optional[Dict[str, Any]] = Field(None, description="特征详情 (仅当return_features=true时返回)")


class PredictResponse(BaseModel):
    """
    预测响应模型
    """
    success: bool = Field(..., description="请求是否成功")
    message: str = Field(..., description="响应消息")
    threshold_used: float = Field(..., description="本次预测使用的阈值")
    total_users: int = Field(..., description="预测用户数量")
    high_intent_count: int = Field(..., description="高意向用户数量")
    predictions: List[UserPrediction] = Field(..., description="预测结果列表 (按概率降序)")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "预测成功",
                "threshold_used": 0.5,
                "total_users": 2,
                "high_intent_count": 1,
                "predictions": [
                    {
                        "user_id": "USER_001",
                        "intent_label": "1",
                        "intent_probability": 0.8765,
                        "certainty_tag": "高确定性"
                    },
                    {
                        "user_id": "USER_002",
                        "intent_label": "2",
                        "intent_probability": 0.3421,
                        "certainty_tag": "纯意向"
                    }
                ]
            }
        }


# ============================================================
# 健康检查模型
# ============================================================

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: HealthStatus = Field(..., description="服务状态")
    version: str = Field(..., description="服务版本")
    model_loaded: bool = Field(..., description="模型是否已加载")
    threshold: float = Field(..., description="当前使用的阈值")
    timestamp: datetime = Field(..., description="检查时间")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "model_loaded": True,
                "threshold": 0.5,
                "timestamp": "2025-08-01T10:00:00"
            }
        }


# ============================================================
# 模型信息模型
# ============================================================

class ModelInfo(BaseModel):
    """模型详细信息"""
    model_type: str = Field(..., description="模型类型")
    feature_count: int = Field(..., description="特征数量")
    feature_names: List[str] = Field(..., description="特征名称列表")
    default_threshold: float = Field(..., description="默认阈值")
    model_path: str = Field(..., description="模型文件路径")

    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "LightGBM",
                "feature_count": 35,
                "feature_names": ["step_sms_input", "step_sms_submit", "..."],
                "default_threshold": 0.5,
                "model_path": "models/intent_v2.pkl"
            }
        }


# ============================================================
# 错误响应模型
# ============================================================

class ErrorDetail(BaseModel):
    """错误详情"""
    code: str = Field(..., description="错误代码")
    message: str = Field(..., description="错误信息")
    detail: Optional[str] = Field(None, description="详细说明")


class ErrorResponse(BaseModel):
    """
    标准错误响应格式
    """
    success: bool = Field(False, description="请求是否成功")
    error: ErrorDetail = Field(..., description="错误详情")

    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": {
                    "code": "INVALID_DATA",
                    "message": "数据验证失败",
                    "detail": "user_actions字段不能为空"
                }
            }
        }
