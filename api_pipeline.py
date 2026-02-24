#!/usr/bin/env python3
"""
终端商机挖掘预测服务 - API批量预测脚本
======================================
从CSV文件读取数据，调用API进行批量预测，输出结果到文件。

输出格式与 predict_pipline.py 一致:
    user_id,intent_label,intent_probability
    按 intent_probability 降序排列
    默认保存到: data/processed/marketing_list.csv

使用方式:
    python api_pipeline.py                                    # 使用默认配置
    python api_pipeline.py --input data/raw/api_batch_test.csv  # 指定输入文件
    python api_pipeline.py --output results.csv               # 指定输出文件
    python api_pipeline.py --batch-size 1000                  # 指定批次大小

环境变量:
    API_KEY     - API认证密钥 (必须设置)
    API_URL     - API服务地址 (默认: http://localhost:8000)

配置文件:
    自动读取 .env 文件中的配置
"""

import os
import sys
import csv
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# 尝试导入第三方库
try:
    import requests
except ImportError:
    print("[ERROR] 请安装 requests 库: pip install requests")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    pd = None
    print("[WARN] pandas 未安装，将使用 CSV 模块处理数据")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
    print("[WARN] tqdm 未安装，将不显示进度条")

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass


# ============================================================
# 配置
# ============================================================

@dataclass
class PipelineConfig:
    """批量预测配置"""
    # API 配置
    api_url: str = "http://localhost:8000"
    api_key: str = ""
    api_endpoint: str = "/api/v1/predict"
    health_endpoint: str = "/api/v1/health"
    timeout: int = 60

    # 数据配置
    input_file: str = "data/raw/api_batch_test.csv"
    output_file: str = "data/processed/marketing_list.csv"  # 与 predict_pipline.py 一致
    batch_size: int = 1000
    max_retries: int = 3

    # 预测配置
    threshold: Optional[float] = None
    return_features: bool = False

    # 日志配置
    log_level: str = "INFO"


# ============================================================
# 日志配置
# ============================================================

def setup_logging(level: str = "INFO") -> logging.Logger:
    """配置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)


# ============================================================
# API 客户端
# ============================================================

class IntentAPIClient:
    """
    终端商机挖掘预测 API 客户端
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "X-API-Key": config.api_key
        })
        self.logger = logging.getLogger(__name__)

    def health_check(self) -> Tuple[bool, Dict[str, Any]]:
        """
        健康检查

        Returns:
            Tuple[bool, Dict]: (是否健康, 响应内容)
        """
        try:
            url = f"{self.config.api_url}{self.config.health_endpoint}"
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            data = response.json()
            is_healthy = data.get("status") == "healthy"
            return is_healthy, data
        except Exception as e:
            self.logger.error(f"健康检查失败: {e}")
            return False, {"error": str(e)}

    def predict(
        self,
        user_actions: List[Dict[str, Any]],
        threshold: Optional[float] = None,
        return_features: bool = False
    ) -> Dict[str, Any]:
        """
        调用预测接口

        Args:
            user_actions: 用户行为数据列表
            threshold: 自定义阈值
            return_features: 是否返回特征

        Returns:
            Dict: API响应
        """
        url = f"{self.config.api_url}{self.config.api_endpoint}"
        payload = {"user_actions": user_actions}

        if threshold is not None:
            payload["threshold"] = threshold
        if return_features:
            payload["return_features"] = return_features

        for attempt in range(self.config.max_retries):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.Timeout:
                self.logger.warning(f"请求超时，重试 {attempt + 1}/{self.config.max_retries}")
            except requests.exceptions.HTTPError as e:
                if response.status_code == 401:
                    self.logger.error("API认证失败，请检查API_KEY")
                    raise
                elif response.status_code == 422:
                    self.logger.error(f"数据验证失败: {response.text}")
                    raise
                else:
                    self.logger.warning(f"请求失败 ({response.status_code})，重试 {attempt + 1}/{self.config.max_retries}")
            except Exception as e:
                self.logger.warning(f"请求异常: {e}，重试 {attempt + 1}/{self.config.max_retries}")

            if attempt < self.config.max_retries - 1:
                import time
                time.sleep(2 ** attempt)  # 指数退避

        raise RuntimeError(f"API请求失败，已重试 {self.config.max_retries} 次")


# ============================================================
# 数据处理
# ============================================================

class DataProcessor:
    """数据处理器"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def load_csv(self, filepath: str) -> List[Dict[str, Any]]:
        """
        加载CSV文件

        Args:
            filepath: 文件路径

        Returns:
            List[Dict]: 数据列表
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

        if pd is not None:
            # 使用 pandas 读取
            df = pd.read_csv(filepath, dtype=str)
            df = df.fillna("")
            records = df.to_dict(orient="records")
        else:
            # 使用 csv 模块读取
            records = []
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append({k: (v if v else "") for k, v in row.items()})

        self.logger.info(f"加载数据: {len(records)} 条记录")
        return records

    def preprocess_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        预处理数据记录

        Args:
            records: 原始数据记录

        Returns:
            List[Dict]: 处理后的数据
        """
        processed = []
        for record in records:
            # 清理 trigger_time 格式 (移除小数点)
            if 'trigger_time' in record:
                time_val = str(record['trigger_time']).split('.')[0]
                record['trigger_time'] = time_val

            # 转换数值字段
            for field in ['item_price', 'current_item_price', 'cart_count']:
                if field in record and record[field]:
                    try:
                        record[field] = float(str(record[field]).split('.')[0] or 0)
                    except:
                        record[field] = 0

            # 移除空值字段 (减少传输量)
            cleaned = {k: v for k, v in record.items() if v != "" and v is not None}
            processed.append(cleaned)

        return processed

    def split_batches(self, records: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
        """
        分批处理数据

        Args:
            records: 数据记录
            batch_size: 批次大小

        Returns:
            List[List[Dict]]: 分批后的数据
        """
        batches = []
        for i in range(0, len(records), batch_size):
            batches.append(records[i:i + batch_size])
        return batches

    def save_results(
        self,
        results: List[Dict[str, Any]],
        filepath: str,
        threshold_used: float
    ) -> None:
        """
        保存预测结果 (与 predict_pipline.py 输出格式一致)

        Args:
            results: 预测结果
            filepath: 输出文件路径
            threshold_used: 使用的阈值

        输出格式:
            user_id,intent_label,intent_probability
            按 intent_probability 降序排列
        """
        # 确保目录存在
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # 统一输出字段顺序 (与 predict_pipline.py 一致)
        output_columns = ['user_id', 'intent_label', 'intent_probability']

        if pd is not None:
            df = pd.DataFrame(results)
            # 确保列顺序一致，过滤掉 features 等额外字段
            df = df[[c for c in output_columns if c in df.columns]]
            # 按概率降序排列 (API已排序，但双重保险)
            df = df.sort_values(by='intent_probability', ascending=False)
            # 使用 utf-8-sig 编码，支持中文 Excel 打开
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
        else:
            if results:
                # 按概率降序排列
                sorted_results = sorted(results, key=lambda x: x.get('intent_probability', 0), reverse=True)
                with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=output_columns, extrasaction='ignore')
                    writer.writeheader()
                    writer.writerows(sorted_results)

        self.logger.info(f"结果已保存: {filepath}")


# ============================================================
# 批量预测流水线
# ============================================================

class PredictionPipeline:
    """批量预测流水线"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = setup_logging(config.log_level)
        self.client = IntentAPIClient(config)
        self.processor = DataProcessor(config)

    def run(self) -> Dict[str, Any]:
        """
        执行批量预测流水线

        Returns:
            Dict: 执行统计信息
        """
        start_time = datetime.now()
        self.logger.info("=" * 60)
        self.logger.info("  终端商机挖掘预测服务 - 批量预测流水线")
        self.logger.info("=" * 60)

        # 1. 健康检查
        self.logger.info("[1/5] 检查API服务状态...")
        is_healthy, health_info = self.client.health_check()
        if not is_healthy:
            self.logger.error(f"API服务不可用: {health_info}")
            raise RuntimeError("API服务不可用，请先启动服务")
        self.logger.info(f"      服务状态: {health_info.get('status')}, 模型已加载: {health_info.get('model_loaded')}")

        # 2. 加载数据
        self.logger.info(f"[2/5] 加载数据文件: {self.config.input_file}")
        records = self.processor.load_csv(self.config.input_file)
        records = self.processor.preprocess_records(records)

        # 3. 分批处理
        self.logger.info(f"[3/5] 分批处理 (批次大小: {self.config.batch_size})")
        batches = self.processor.split_batches(records, self.config.batch_size)
        self.logger.info(f"      共 {len(batches)} 个批次")

        # 4. 执行预测
        self.logger.info("[4/5] 执行批量预测...")
        all_predictions = []
        threshold_used = self.config.threshold

        iterator = tqdm(batches, desc="预测进度") if tqdm else batches
        for batch in iterator:
            try:
                result = self.client.predict(
                    user_actions=batch,
                    threshold=self.config.threshold,
                    return_features=self.config.return_features
                )

                if result.get("success"):
                    all_predictions.extend(result.get("predictions", []))
                    if threshold_used is None:
                        threshold_used = result.get("threshold_used", 0.5)
                else:
                    self.logger.warning(f"批次预测失败: {result.get('message')}")

            except Exception as e:
                self.logger.error(f"批次预测异常: {e}")
                continue

        # 5. 保存结果
        self.logger.info(f"[5/5] 保存预测结果: {self.config.output_file}")
        self.processor.save_results(all_predictions, self.config.output_file, threshold_used or 0.5)

        # 统计信息
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        high_intent_count = sum(1 for p in all_predictions if p.get("intent_label") == "1")
        low_intent_count = len(all_predictions) - high_intent_count

        stats = {
            "input_file": self.config.input_file,
            "output_file": self.config.output_file,
            "total_records": len(records),
            "total_predictions": len(all_predictions),
            "high_intent_count": high_intent_count,
            "low_intent_count": low_intent_count,
            "high_intent_ratio": f"{high_intent_count / len(all_predictions) * 100:.2f}%" if all_predictions else "N/A",
            "threshold_used": threshold_used,
            "duration_seconds": round(duration, 2),
            "records_per_second": round(len(records) / duration, 2) if duration > 0 else 0
        }

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("  执行完成")
        self.logger.info("=" * 60)
        self.logger.info(f"  输入文件:     {stats['input_file']}")
        self.logger.info(f"  输出文件:     {stats['output_file']}")
        self.logger.info(f"  总记录数:     {stats['total_records']}")
        self.logger.info(f"  预测用户数:   {stats['total_predictions']}")
        self.logger.info(f"  高意向用户:   {stats['high_intent_count']} ({stats['high_intent_ratio']})")
        self.logger.info(f"  低意向用户:   {stats['low_intent_count']}")
        self.logger.info(f"  使用阈值:     {stats['threshold_used']}")
        self.logger.info(f"  执行耗时:     {stats['duration_seconds']} 秒")
        self.logger.info(f"  处理速度:     {stats['records_per_second']} 条/秒")
        self.logger.info("=" * 60)

        return stats


# ============================================================
# 命令行入口
# ============================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="终端商机挖掘预测服务 - API批量预测脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python api_pipeline.py
  python api_pipeline.py --input data/raw/api_batch_test.csv
  python api_pipeline.py --output results.csv --batch-size 500
  python api_pipeline.py --threshold 0.6

环境变量:
  API_KEY   - API认证密钥 (必须设置)
  API_URL   - API服务地址 (默认: http://localhost:8000)
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        default="data/raw/api_batch_test.csv",
        help="输入CSV文件路径 (默认: data/raw/api_batch_test.csv)"
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default="data/processed/marketing_list.csv",
        help="输出CSV文件路径 (默认: data/processed/marketing_list.csv)"
    )

    parser.add_argument(
        '--api-url',
        type=str,
        default=os.getenv("API_URL", "http://localhost:8000"),
        help="API服务地址 (默认: http://localhost:8000)"
    )

    parser.add_argument(
        '--api-key',
        type=str,
        default=os.getenv("API_KEY", ""),
        help="API认证密钥 (默认从环境变量API_KEY读取)"
    )

    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=1000,
        help="每批次处理的记录数 (默认: 1000)"
    )

    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=None,
        help="自定义判定阈值 (默认: 使用模型默认值)"
    )

    parser.add_argument(
        '--return-features',
        action='store_true',
        help="返回特征详情 (调试用)"
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=60,
        help="API请求超时时间(秒) (默认: 60)"
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="日志级别 (默认: INFO)"
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="仅检查配置，不执行预测"
    )

    return parser.parse_args()


def clean_env_value(value: str) -> str:
    """
    清理环境变量值中的隐藏字符

    处理常见问题:
    - Windows换行符 (\\r)
    - 首尾空白字符
    - 引号包裹
    """
    if not value:
        return value
    # 去除首尾空白和换行符
    value = value.strip().strip('\r\n')
    # 去除可能的引号包裹
    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        value = value[1:-1]
    return value


def main():
    """主函数"""
    args = parse_args()

    # 清理API_KEY中的隐藏字符
    api_key = clean_env_value(args.api_key)

    # 检查API_KEY
    if not api_key:
        print("[ERROR] 未设置 API_KEY")
        print("[INFO]  请设置环境变量: export API_KEY=<your-api-key>")
        print("[INFO]  或使用参数: --api-key <your-api-key>")
        print("[INFO]  或在 .env 文件中配置: API_KEY=<your-api-key>")
        sys.exit(1)

    # 创建配置
    config = PipelineConfig(
        api_url=args.api_url,
        api_key=api_key,
        input_file=args.input,
        output_file=args.output,
        batch_size=args.batch_size,
        threshold=args.threshold,
        return_features=args.return_features,
        timeout=args.timeout,
        log_level=args.log_level
    )

    # 显示配置
    if args.dry_run:
        print("\n[配置预览]")
        print(f"  API地址:     {config.api_url}")
        print(f"  API密钥:     {config.api_key[:8]}... (长度: {len(config.api_key)})")
        print(f"  输入文件:    {config.input_file}")
        print(f"  输出文件:    {config.output_file}")
        print(f"  批次大小:    {config.batch_size}")
        print(f"  判定阈值:    {config.threshold or '使用默认值'}")
        print("\n[INFO] 配置检查完成，去掉 --dry-run 参数执行预测")
        sys.exit(0)

    # 执行流水线
    try:
        pipeline = PredictionPipeline(config)
        stats = pipeline.run()
        sys.exit(0)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断执行")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERROR] 执行异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
