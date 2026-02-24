# 终端商机挖掘预测系统

## 项目概述

本项目是一个基于机器学习的**终端用户购买意向预测系统**，通过分析广东移动用户在终端商城的行为数据，预测用户的购买意向强度。系统提供完整的训练、预测、API服务三种使用方式，支持本地部署和Docker容器化部署。

## 核心特性

- **多维度特征工程**：构建了包括行为、转化深度、价格能力、时序特征等5个维度的特征体系
- **智能降权机制**：通过特征惩罚避免价格主导，更关注用户行为模式
- **RESTful API服务**：基于FastAPI的高性能预测接口，支持批量预测
- **企业级安全**：API Key认证、密钥生成工具、符合运维安全规范
- **多种部署方式**：支持本地Python、Docker容器、Docker Compose编排

## 项目版本

| 版本 | 说明 | 操作指南 |
|------|------|----------|
| V1 基础版 | 本地训练和预测 | [DEPLOYMENT_GUIDE_DEV.md](docs/DEPLOYMENT_GUIDE_DEV.md) |
| V2 API版 | 新增API服务 | [DEPLOYMENT_GUIDE_DEV.md](docs/DEPLOYMENT_GUIDE_DEV.md) |
| V3 Docker版 | Docker容器化部署 | [DEPLOYMENT_GUIDE_PROD.md](docs/DEPLOYMENT_GUIDE_PROD.md) |

## 项目结构

```
Terminal_Intent_ProjectV2/
├── api/                          # API服务模块
│   ├── main.py                  # FastAPI主入口
│   ├── config.py                # 配置管理
│   ├── schemas.py               # 数据模型定义
│   ├── auth.py                  # 认证模块
│   └── predict.py               # 预测核心逻辑
├── config/                       # 业务配置
│   ├── business_rules.py        # 业务规则映射
│   └── feature_list.py          # 特征列表配置
├── src/                          # 核心源码
│   ├── data/loader.py           # 数据加载器
│   ├── features/unified_fe.py   # 统一特征工程
│   └── models/model_lgb.py      # 模型定义
├── data/                         # 数据目录
│   ├── raw/                     # 原始数据 (train_data.csv, api_batch_test.csv)
│   └── processed/               # 处理后数据 (由脚本生成)
├── models/                       # 模型文件 (由 train_pipline.py 生成)
├── docs/                         # 文档
│   ├── API_DOCUMENTATION.md     # API文档
│   ├── DEPLOYMENT_GUIDE.md      # 部署指南
│   ├── DEPLOYMENT_GUIDE_DEV.md  # 开发环境部署
│   └── DEPLOYMENT_GUIDE_PROD.md # 生产环境部署
├── packages_final_310/           # 离线安装包 (44个whl文件)
├── train_pipline.py              # 训练流水线
├── predict_pipline.py            # 本地预测流水线
├── api_pipeline.py               # API批量预测脚本
├── generate_api_key.py           # API密钥生成工具
├── data_extra_withdata.py        # 数据分流处理
├── data_extra_train.py           # 数据采样处理
├── Dockerfile                    # Docker构建文件
├── docker-compose.yml            # Docker Compose配置
├── docker-compose.prod.yml       # 生产环境配置
├── Makefile                      # 构建脚本
├── requirements-api.txt          # API服务依赖
├── .env.example                  # 环境变量模板
└── .gitignore                    # Git忽略文件
```

## 快速开始

### 1. 环境准备

```bash
# 安装 Miniconda (离线环境)
bash Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b -p ./my_env
source ./my_env/bin/activate

# 安装依赖
pip install --no-index --find-links=./packages_final_310 -r requirements-api.txt
```

### 2. 模型训练

```bash
python train_pipline.py
```

### 3. 本地预测

```bash
python predict_pipline.py
```

输出文件: `data/processed/marketing_list.csv`

### 4. 启动API服务

```bash
# 前台启动
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# 后台启动
nohup python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 > intent-api-output-001 2>&1 &
```

### 5. API批量预测

```bash
# 设置API密钥
export API_KEY=$(grep '^API_KEY=' .env | cut -d '=' -f2)

# 执行批量预测 (从CSV文件)
python api_pipeline.py --input data/raw/api_batch_test.csv
```

输出文件: `data/processed/marketing_list.csv` (与本地预测格式一致)

## Docker 部署

### 构建和启动

```bash
# 构建镜像
make build

# 启动服务 (开发环境)
make start

# 启动服务 (生产环境)
make start-prod
```

### 常用命令

```bash
make status       # 查看状态
make test         # 基础测试
make test-batch   # 批量测试
make logs         # 查看日志
make stop         # 停止服务
make generate-key # 生成API密钥
```

## API密钥管理

### 生成生产环境密钥

```bash
# 生成密钥并显示
python generate_api_key.py

# 生成密钥并自动更新.env
python generate_api_key.py --update

# 生成密钥并备份旧配置
python generate_api_key.py --update --backup
```

### 密钥安全规范

- 生产环境必须使用强随机密钥 (至少32字符)
- 禁止在代码中硬编码密钥
- 禁止将 `.env` 文件提交到版本控制
- 建议定期轮换密钥

## API 接口

| 方法 | 端点 | 说明 | 需要认证 |
|------|------|------|----------|
| POST | `/api/v1/predict` | 预测用户购买意向 | 是 |
| GET | `/api/v1/health` | 健康检查 | 否 |
| GET | `/api/v1/model/info` | 模型信息 | 是 |

### 预测请求示例

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "user_actions": [
      {
        "user_id": "USER_001",
        "identifier": "eventClick",
        "trigger_time": "20250801100001",
        "item_name": "华为Mate70",
        "item_price": 7999,
        "process_step": "立即购买"
      }
    ]
  }'
```

详细API文档: [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md)

## 输出格式

本地预测 (`predict_pipline.py`) 和 API批量预测 (`api_pipeline.py`) 输出格式一致:

```csv
user_id,intent_label,intent_probability
USER_001,1,0.8765
USER_002,2,0.3421
```

| 字段 | 说明 |
|------|------|
| user_id | 用户唯一标识 |
| intent_label | 意向标签: 1=高意向, 2=低意向 |
| intent_probability | 购买概率 (0-1)，按降序排列 |

## 特征工程

### 维度一：交互行为特征
- `cnt_eventClick`: 点击事件次数
- `cnt_bussProcessing`: 业务处理次数
- `click_buy_now_cnt`: 点击"立即购买"次数

### 维度二：转化深度特征
- `max_intent_level`: 最高意向等级 (1-5级)
- `last_intent_level`: 最后一个行为的意向等级
- `avg_intent_level`: 平均意向等级

### 维度三：价格能力特征
- `price_level`: 价格档位 (0-4级)

### 维度四：时序特征
- `avg_action_interval`: 平均行为间隔时间
- `action_frequency`: 行为频率

### 维度五：用户历史统计
- `recent_3_avg_intent`: 最近3个行为的平均意向
- `intent_trend`: 意向趋势
- `has_login`: 是否有登录行为

## 文档索引

| 文档 | 说明 |
|------|------|
| [API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md) | API接口完整文档 |
| [DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) | 通用部署指南 |
| [DEPLOYMENT_GUIDE_DEV.md](docs/DEPLOYMENT_GUIDE_DEV.md) | 开发环境部署 |
| [DEPLOYMENT_GUIDE_PROD.md](docs/DEPLOYMENT_GUIDE_PROD.md) | 生产环境部署 |

## 版本历史

### v3.0 (2026-02)
- 新增 Docker 容器化部署
- 新增 API密钥生成工具 (`generate_api_key.py`)
- 新增 API批量预测脚本 (`api_pipeline.py`)
- API参数 `logs` 重命名为 `user_actions`
- 文件命名符合运维安全规范

### v2.0 (2026-01)
- 新增 FastAPI RESTful 接口
- 新增 API认证机制
- 支持批量预测

### v1.0 (2025-12)
- 初始版本发布
- 基础特征工程
- LightGBM模型训练和预测

## 安全规范

本项目遵循《电子渠道系统日志运维安全规范》:

- 输出文件不使用 `.log` 后缀
- 文件命名采用"业务模块-文件类型-标识"格式
- 日志查询使用 `tail`、`less`、`grep` 等只读命令
- 禁止使用 `vi`/`vim` 打开日志文件


---

**注意事项**：
1. 本系统仅用于内部业务预测
2. 模型需要定期重训以适应业务变化
3. 生产环境必须配置强API密钥
