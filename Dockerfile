# ======================================================
# Terminal Intent API Dockerfile
# 基于 Miniconda 离线安装 & 项目结构定制
# ======================================================

# ==================== 阶段 1: 构建环境 (Builder) ====================
FROM ubuntu:22.04 AS builder

# 避免交互式前端报错
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /build

# 1. 复制构建所需文件
# 注意：确保 Miniconda 安装脚本在项目根目录
COPY Miniconda3-py310_23.1.0-1-Linux-x86_64.sh .
# 复制离线 wheel 包目录
COPY packages_final_310 /wheels/
# 复制 API 依赖文件 (根据您的项目结构)
COPY requirements-api.txt .

# 2. 安装 Miniconda (对应手册 1.1)
# 安装到 /opt/my_env
RUN bash Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b -p /opt/my_env

# 3. 离线安装 Python 依赖 (对应手册 1.2)
# 使用 /opt/my_env 下的 pip
RUN /opt/my_env/bin/pip install --no-cache-dir --no-index --find-links=/wheels -r requirements-api.txt


# ==================== 阶段 2: 生产环境 (Production) ====================
FROM ubuntu:22.04 AS production

# 设置环境变量
# PYTHONPATH=/app 确保 src.xx 和 config.xx 可以被正确导入
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/my_env/bin:$PATH" \
    PYTHONPATH=/app \
    TZ=Asia/Shanghai

WORKDIR /app

# # 1. 复制本地的 deb 包到容器临时目录
# COPY offline_debs/libgomp1_14.2.0-4ubuntu2~24.04_amd64.deb /tmp/libgomp1.deb

# # 2. 使用 dpkg 安装并清理
# # 注意：ubuntu:22.04 基础镜像通常自带 tzdata 的基本配置，可以直接设置时区
# RUN dpkg -i /tmp/libgomp1.deb

# 在 Dockerfile 生产阶段设置
ENV PATH="/opt/my_env/bin:$PATH" \
    LD_LIBRARY_PATH="/opt/my_env/lib:$LD_LIBRARY_PATH" \
    PYTHONPATH=/app
    
# 2. 从构建阶段复制完整的 Python 环境
COPY --from=builder /opt/my_env /opt/my_env

# 3. 创建非 root 用户 (安全最佳实践)
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# 4. 创建目录结构 (对应项目结构和数据产出)
# 预先创建 data/raw 和 data/processed 以便挂载或写入
RUN mkdir -p /app/data/raw /app/data/processed /app/models /app/logs && \
    chown -R appuser:appgroup /app

# 5. 复制项目代码 (根据您的 tree 结构)
COPY config/ /app/config/
COPY src/ /app/src/
COPY api/ /app/api/
# 复制已有的模型文件 (如果有的话，没有则为空文件夹)
COPY models/ /app/models/
# 复制环境变量模板 (作为默认配置)
COPY .env.example /app/.env

# 6. 权限修正
RUN chown -R appuser:appgroup /app

# 切换到应用用户
USER appuser

# 7. 端口暴露
EXPOSE 8000

# 8. 健康检查 (对应手册 6.1)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; \
    try: \
        urllib.request.urlopen('http://localhost:8000/api/v1/health').read() \
    except Exception: \
        exit(1)"

# 9. 启动命令 (对应手册 5.1 & 7.1)
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]