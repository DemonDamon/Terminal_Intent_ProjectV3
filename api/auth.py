"""
API 认证模块
============
实现基于 API Key 的认证机制。

认证方式:
- 通过 HTTP Header 传递 API Key
- Header名称: X-API-Key (可配置)
- 生产环境应通过环境变量设置强密钥

使用示例:
    curl -H "X-API-Key: your-api-key" http://localhost:8000/api/v1/predict
"""

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from .config import settings

# 创建 API Key Header 安全方案
api_key_header = APIKeyHeader(
    name=settings.API_KEY_HEADER,
    auto_error=True,
    description="API认证密钥，通过Header传递"
)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    验证 API Key 的依赖函数

    Args:
        api_key: 从Header中提取的API Key

    Returns:
        str: 验证通过返回API Key

    Raises:
        HTTPException: 401 认证失败
    """
    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "UNAUTHORIZED",
                "message": "API Key认证失败",
                "detail": "请在请求Header中提供有效的X-API-Key"
            }
        )
    return api_key


def get_api_key_optional(api_key: str = Security(api_key_header)) -> str:
    """
    可选的 API Key 验证（用于某些不强制认证的端点）
    """
    return api_key
