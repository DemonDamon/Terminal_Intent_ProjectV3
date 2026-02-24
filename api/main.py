"""
FastAPI 主入口模块
==================
终端商机挖掘预测服务 API

功能:
- POST /api/v1/predict      预测用户购买意向
- GET  /api/v1/health       健康检查
- GET  /api/v1/model/info   模型信息查询

启动方式:
    python -m api.main
    或
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import logging
import time
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .schemas import (
    PredictRequest,
    PredictResponse,
    UserPrediction,
    HealthResponse,
    HealthStatus,
    ModelInfo,
    ErrorResponse,
)
from .auth import verify_api_key
from .predict import prediction_service


# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================
# 应用生命周期管理
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    启动时加载模型，关闭时清理资源
    """
    # === 启动阶段 ===
    logger.info("=" * 60)
    masked_key = settings.API_KEY[:5] + "***" if len(settings.API_KEY) > 5 else "***"
    logger.info(f"[DIAGNOSTIC] 服务端加载的 API_KEY: {masked_key}")
    
    logger.info("  终端商机挖掘预测服务启动中...")
    logger.info("=" * 60)

    success = prediction_service.load_model()
    if success:
        logger.info("模型加载成功")
    else:
        logger.error("模型加载失败，服务将以降级模式运行")

    logger.info(f"API版本: {settings.APP_VERSION}")
    logger.info(f"监听地址: {settings.HOST}:{settings.PORT}")
    logger.info(f"日志级别: {settings.LOG_LEVEL}")
    logger.info("=" * 60)

    yield

    # === 关闭阶段 ===
    logger.info("服务正在关闭...")


# ============================================================
# FastAPI 应用初始化
# ============================================================
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    responses={
        401: {"model": ErrorResponse, "description": "认证失败"},
        422: {"model": ErrorResponse, "description": "数据验证错误"},
        500: {"model": ErrorResponse, "description": "服务端错误"},
    }
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 请求日志中间件
# ============================================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录请求日志和耗时"""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} "
        f"-> {response.status_code} ({duration:.3f}s)"
    )
    return response


# ============================================================
# 全局异常处理
# ============================================================
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """处理数据验证错误"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": {
                "code": "INVALID_DATA",
                "message": "数据验证失败",
                "detail": str(exc)
            }
        }
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    """处理运行时错误"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "服务内部错误",
                "detail": str(exc)
            }
        }
    )


# ============================================================
# API 路由
# ============================================================

@app.post(
    "/api/v1/predict",
    response_model=PredictResponse,
    summary="预测用户购买意向",
    description="根据用户行为日志数据，预测用户的购买意向概率和分类标签",
    tags=["预测服务"],
    responses={
        200: {"description": "预测成功", "model": PredictResponse},
        401: {"description": "认证失败", "model": ErrorResponse},
        422: {"description": "数据格式错误", "model": ErrorResponse},
        500: {"description": "服务端错误", "model": ErrorResponse},
    }
)
async def predict(
    request: PredictRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    ## 预测用户购买意向

    接受用户行为日志数据，返回每个用户的购买意向概率和分类标签。

    ### 输入说明
    - **user_actions**: 用户行为数据列表，每条数据包含用户的一次行为记录
    - **threshold**: 自定义判定阈值 (可选，默认使用模型内置阈值)
    - **return_features**: 是否返回特征详情 (可选，默认false)

    ### 输出说明
    - **intent_label**: "1"=高购买意向, "2"=低购买意向
    - **intent_probability**: 购买意向概率，范围 [0, 1]
    - 结果按购买概率降序排列
    """
    # 检查模型状态
    if not prediction_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "MODEL_NOT_LOADED",
                "message": "模型未加载",
                "detail": "服务正在初始化中或模型文件缺失，请稍后重试"
            }
        )

    # 检查批量大小限制
    if len(request.user_actions) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "BATCH_TOO_LARGE",
                "message": f"请求数据过多",
                "detail": f"单次请求最多支持 {settings.MAX_BATCH_SIZE} 条记录，当前为 {len(request.user_actions)} 条"
            }
        )

    # 转换请求数据
    actions_data = [action.model_dump() for action in request.user_actions]

    # 执行预测
    results, used_threshold = prediction_service.predict(
        user_actions=actions_data,
        threshold=request.threshold,
        return_features=request.return_features
    )

    # 构建响应
    predictions = [
        UserPrediction(
            user_id=r['user_id'],
            intent_label=r['intent_label'],
            intent_probability=round(r['intent_probability'], 6),
            features=r.get('features')
        )
        for r in results
    ]

    high_intent_count = sum(1 for p in predictions if p.intent_label == "1")

    return PredictResponse(
        success=True,
        message="预测成功",
        threshold_used=round(used_threshold, 4),
        total_users=len(predictions),
        high_intent_count=high_intent_count,
        predictions=predictions
    )


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    summary="健康检查",
    description="检查服务运行状态和模型加载状态",
    tags=["系统管理"],
)
async def health_check():
    """
    ## 健康检查

    返回服务的运行状态，用于负载均衡器和监控系统。

    - **healthy**: 服务正常运行，模型已加载
    - **degraded**: 服务运行中但模型未加载
    - **unhealthy**: 服务异常
    """
    model_loaded = prediction_service.is_loaded

    if model_loaded:
        health_status = HealthStatus.HEALTHY
    else:
        health_status = HealthStatus.DEGRADED

    return HealthResponse(
        status=health_status,
        version=settings.APP_VERSION,
        model_loaded=model_loaded,
        threshold=prediction_service.threshold,
        timestamp=datetime.now()
    )


@app.get(
    "/api/v1/model/info",
    response_model=ModelInfo,
    summary="模型信息查询",
    description="查询当前加载模型的详细信息",
    tags=["系统管理"],
    dependencies=[Depends(verify_api_key)]
)
async def model_info():
    """
    ## 模型信息查询

    返回当前模型的类型、特征数量、特征列表等信息。
    需要API Key认证。
    """
    info = prediction_service.get_model_info()
    return ModelInfo(**info)


# ============================================================
# 根路由 (无需认证)
# ============================================================
@app.get("/", tags=["系统管理"], summary="服务信息")
async def root():
    """返回服务基本信息"""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health"
    }


# ============================================================
# 直接运行入口
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
        reload=settings.DEBUG
    )
