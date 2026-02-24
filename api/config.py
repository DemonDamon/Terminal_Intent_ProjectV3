"""
API 配置模块
=============
集中管理所有配置项，支持环境变量覆盖。

配置优先级 (从高到低):
1. 环境变量 (export API_KEY=xxx)
2. .env 文件
3. 代码默认值

环境变量说明:
- API_KEY: API认证密钥 (生产环境必须设置)
- MODEL_PATH: 模型文件路径
- FEATURE_NAMES_PATH: 特征名称文件路径
- LOG_LEVEL: 日志级别 (DEBUG, INFO, WARNING, ERROR)
- WORKERS: uvicorn工作进程数
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator # 导入 field_validator

class Settings(BaseSettings):
    """API配置类"""

    # ============ 基础配置 ============
    APP_NAME: str = "终端商机挖掘预测服务"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "基于用户行为数据预测购买意向的机器学习服务"
    DEBUG: bool = False

    # ============ 认证配置 ============
    # 生产环境应通过环境变量 API_KEY 设置强密钥
    API_KEY: str = "dev-api-key-change-in-production"
    @field_validator('API_KEY', mode='before')
    @classmethod
    def clean_api_key(cls, v: str) -> str:
        if v:
            # 去除首尾空白
            v = v.strip()
            # 去除首尾可能的引号
            if (v.startswith('"') and v.endswith('"')) or \
               (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]
        return v
    API_KEY_HEADER: str = "X-API-Key"

    # ============ 模型配置 ============
    # 获取项目根目录（api目录的上一级）
    PROJECT_ROOT: Path = Path(__file__).parent.parent.resolve()
    MODEL_PATH: Path = PROJECT_ROOT / "models" / "intent_v2.pkl"
    FEATURE_NAMES_PATH: Path = PROJECT_ROOT / "models" / "feature_names.pkl"
    THRESHOLD_PATH: Path = PROJECT_ROOT / "models" / "optimal_threshold.pkl"

    # ============ 预测配置 ============
    DEFAULT_THRESHOLD: float = 0.5
    MAX_BATCH_SIZE: int = 10000  # 单次批量预测最大记录数

    # ============ 服务配置 ============
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    LOG_LEVEL: str = "INFO"

    # ============ CORS 配置 ============
    CORS_ORIGINS: list = ["*"]  # 生产环境应限制具体域名

    # pydantic-settings v2 配置
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # 去除环境变量值两端的空白字符
        env_ignore_empty=True,
    )


# 全局配置单例
settings = Settings()
