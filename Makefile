# ======================================================
# 终端商机挖掘项目 - Docker Makefile
# Terminal Intent Prediction - Docker Makefile
# ======================================================
#
# 使用: make <命令>
# 帮助: make help
#
# ======================================================

# 配置变量
IMAGE_NAME := terminal-intent-api
IMAGE_TAG := 1.0.0
CONTAINER_NAME := terminal-intent-api
HOST_PORT ?= 8000
API_KEY ?= dev-api-key-change-in-production
# --- 修改点 1: 自动从 .env 提取 API_KEY ---
# 如果 .env 文件存在，尝试从中提取 API_KEY
ifneq ("$(wildcard .env)","")
    # 使用 grep 提取 API_KEY 行，cut 分割等号，sed 去除可能的引号和回车符(Windows兼容)
    DOTENV_KEY := $(shell grep '^API_KEY=' .env | head -1 | cut -d '=' -f2- | sed 's/["'\'']//g' | sed 's/\r//')
endif

# 优先级逻辑：
# 1. 如果执行命令时传了 API_KEY=xxx，则使用传入的值
# 2. 否则如果 .env 里有值，则使用 .env 的值
# 3. 最后才使用默认的 dev 值
ifeq ($(API_KEY),dev-api-key-change-in-production)
    ifneq ($(DOTENV_KEY),)
        API_KEY := $(DOTENV_KEY)
    endif
endif
IMAGE_TAR := $(IMAGE_NAME).tar

# 默认目标
.DEFAULT_GOAL := help

# 伪目标声明
.PHONY: build start start-prod stop restart logs status test test-batch generate-key save load shell clean help check-docker check-models

# ==================== 检查函数 ====================

check-docker:
	@docker info > /dev/null 2>&1 || (echo "[ERROR] Docker 未运行"; exit 1)

check-models:
	@test -f models/intent_v2.pkl || (echo "[WARN] 模型不存在，请先运行: python train_pipline.py"; exit 1)

# ==================== 构建命令 ====================

## 构建 Docker 镜像
build: check-docker
	@echo "[INFO] 构建镜像: $(IMAGE_NAME):$(IMAGE_TAG)"
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .
	@echo "[OK] 构建完成"
	@docker images | grep $(IMAGE_NAME)

# ==================== 服务管理 ====================

## 启动服务 (开发环境)
start: check-docker check-models
	@if docker ps -q -f name=$(CONTAINER_NAME) | grep -q .; then \
		echo "[WARN] 容器已在运行"; \
		docker ps -f name=$(CONTAINER_NAME); \
	else \
		if ! docker images -q $(IMAGE_NAME):$(IMAGE_TAG) | grep -q .; then \
			echo "[WARN] 镜像不存在，先构建..."; \
			$(MAKE) build; \
		fi; \
		echo "[INFO] 启动服务..."; \
		docker-compose up -d; \
		echo "[OK] 服务已启动: http://localhost:$(HOST_PORT)"; \
		echo "[OK] API 文档:   http://localhost:$(HOST_PORT)/docs"; \
	fi

## 启动服务 (生产环境)
start-prod: check-docker check-models
	@if [ "$(API_KEY)" = "dev-api-key-change-in-production" ] && [ ! -f .env ]; then \
		echo "[ERROR] 生产环境必须设置 API_KEY"; \
		echo "[INFO] 请先: cp .env.example .env 并修改 API_KEY"; \
		exit 1; \
	fi
	@echo "[INFO] 启动生产环境..."
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
	@echo "[OK] 生产环境已启动"

## 停止服务
stop: check-docker
	@echo "[INFO] 停止服务..."
	docker-compose down
	@echo "[OK] 服务已停止"

## 重启服务
restart: check-docker
	@echo "[INFO] 重启服务..."
	docker-compose restart
	@echo "[OK] 服务已重启"

## 查看日志 (默认100行，可用 make logs LINES=50)
LINES ?= 100
logs: check-docker
	docker logs $(CONTAINER_NAME) --tail $(LINES) -f

# ==================== 状态查看 ====================

## 查看服务状态
status: check-docker
	@echo "[INFO] 容器状态:"
	@docker ps -a -f name=$(CONTAINER_NAME)
	@echo ""
	@echo "[INFO] 健康检查:"
	@curl -s http://localhost:$(HOST_PORT)/api/v1/health | python3 -m json.tool 2>/dev/null || echo "[WARN] 服务未就绪"
	@echo ""
	@echo "[INFO] 镜像信息:"
	@docker images | grep $(IMAGE_NAME) || echo "[WARN] 镜像不存在"

# --- 修改点 2: 增强 test 目标的调试输出 ---
test: check-docker
	@echo "[INFO] 使用 API_KEY: $(shell echo $(API_KEY) | cut -c 1-6)...******"
	@echo "[INFO] 1. 健康检查"
	@curl -s http://localhost:$(HOST_PORT)/api/v1/health | python3 -m json.tool || echo "请求失败"
	@echo ""
	@echo "[INFO] 2. 单条预测测试"
	@curl -s -X POST "http://localhost:$(HOST_PORT)/api/v1/predict" \
		-H "Content-Type: application/json" \
		-H "X-API-Key: $(API_KEY)" \
		-d '{"user_actions":[{"user_id":"TEST","identifier":"eventClick","trigger_time":"20250801100001","platform_type":"APP","process_step":"立即购买"}]}' \
		| python3 -m json.tool || echo "请求失败"
	@echo ""
	@echo "[OK] 基础测试完成"

## 批量测试 (从CSV文件)
test-batch:
	@echo "[INFO] 执行批量预测测试..."
	@if [ -z "$(API_KEY)" ] || [ "$(API_KEY)" = "dev-api-key-change-in-production" ]; then \
		if [ -f .env ]; then \
			export $$(grep -v '^#' .env | xargs) && python3 api_pipeline.py; \
		else \
			API_KEY=$(API_KEY) python3 api_pipeline.py; \
		fi \
	else \
		API_KEY=$(API_KEY) python3 api_pipeline.py; \
	fi
	@echo "[OK] 批量测试完成"

## 生成API密钥
generate-key:
	@python3 generate_api_key.py

# ==================== 镜像导入导出 ====================

## 导出镜像为 tar 文件 (用于传输到服务器)
save: check-docker
	@if ! docker images -q $(IMAGE_NAME):$(IMAGE_TAG) | grep -q .; then \
		echo "[WARN] 镜像不存在，先构建..."; \
		$(MAKE) build; \
	fi
	@echo "[INFO] 导出镜像: $(IMAGE_TAR)"
	docker save $(IMAGE_NAME):$(IMAGE_TAG) -o $(IMAGE_TAR)
	@echo "[OK] 导出完成: $$(ls -lh $(IMAGE_TAR) | awk '{print $$5}')"

## 从 tar 文件导入镜像
load: check-docker
	@test -f $(IMAGE_TAR) || (echo "[ERROR] 镜像文件不存在: $(IMAGE_TAR)"; exit 1)
	@echo "[INFO] 导入镜像: $(IMAGE_TAR)"
	docker load -i $(IMAGE_TAR)
	@echo "[OK] 导入完成"
	@docker images | grep $(IMAGE_NAME)

# ==================== 其他命令 ====================

## 进入容器 shell
shell: check-docker
	@docker exec -it $(CONTAINER_NAME) /bin/bash || docker exec -it $(CONTAINER_NAME) /bin/sh

## 清理 Docker 资源
clean: check-docker
	@echo "即将清理: 容器、镜像、网络"
	@read -p "确认清理? (y/N): " confirm && [ "$$confirm" = "y" ] || (echo "取消清理"; exit 1)
	@echo "[INFO] 停止容器..."
	-docker-compose down --remove-orphans
	@echo "[INFO] 删除镜像..."
	-docker rmi $(IMAGE_NAME):$(IMAGE_TAG)
	@echo "[OK] 清理完成"

# ==================== 帮助信息 ====================

## 显示帮助
help:
	@echo ""
	@echo "终端商机挖掘项目 - Docker Makefile"
	@echo ""
	@echo "使用: make <命令>"
	@echo ""
	@echo "构建命令:"
	@echo "  build         构建 Docker 镜像"
	@echo ""
	@echo "服务管理:"
	@echo "  start         启动服务 (开发环境)"
	@echo "  start-prod    启动服务 (生产环境)"
	@echo "  stop          停止服务"
	@echo "  restart       重启服务"
	@echo "  logs          查看日志 (可用 LINES=50 指定行数)"
	@echo ""
	@echo "测试命令:"
	@echo "  test          基础API测试 (健康检查+单条预测)"
	@echo "  test-batch    批量测试 (从CSV文件预测)"
	@echo ""
	@echo "安全管理:"
	@echo "  generate-key  生成API密钥"
	@echo ""
	@echo "镜像管理:"
	@echo "  save          导出镜像为 tar 文件"
	@echo "  load          从 tar 文件导入镜像"
	@echo ""
	@echo "其他:"
	@echo "  status        查看服务状态"
	@echo "  shell         进入容器 shell"
	@echo "  clean         清理 Docker 资源"
	@echo "  help          显示帮助"
	@echo ""
	@echo "示例:"
	@echo "  make build"
	@echo "  make start"
	@echo "  make test"
	@echo "  make test-batch"
	@echo "  make generate-key"
	@echo "  make logs LINES=50"
	@echo "  API_KEY=xxx make start-prod"
	@echo ""
