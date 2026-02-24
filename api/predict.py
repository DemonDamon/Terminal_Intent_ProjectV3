"""
预测核心逻辑模块
================
封装模型加载、数据处理、预测执行的完整流程。

此模块与原始的 predict_pipline.py 保持一致的处理逻辑，
但适配了API服务的数据输入格式。
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

# 设置日志
logger = logging.getLogger(__name__)

# 添加项目根目录到路径，以便导入原有模块
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入原有模块
from src.features.unified_fe import UnifiedFeatureEngineer
from config.feature_list import CAT_FEATURES
from config.business_rules import INTENT_RANK, BRANDS, ACTION_KEYWORDS

from .config import settings


class PredictionService:
    """
    预测服务类
    负责模型管理和预测执行
    """

    def __init__(self):
        """初始化预测服务"""
        self.model = None
        self.feature_names: List[str] = []
        self.threshold: float = settings.DEFAULT_THRESHOLD
        self.feature_engineer = UnifiedFeatureEngineer()
        self._loaded = False

    def load_model(self) -> bool:
        """
        加载模型和相关配置

        Returns:
            bool: 加载是否成功
        """
        try:
            # 加载模型
            if not settings.MODEL_PATH.exists():
                logger.error(f"模型文件不存在: {settings.MODEL_PATH}")
                return False

            logger.info(f"正在加载模型: {settings.MODEL_PATH}")
            self.model = joblib.load(settings.MODEL_PATH)

            # 加载特征名称
            if settings.FEATURE_NAMES_PATH.exists():
                self.feature_names = joblib.load(settings.FEATURE_NAMES_PATH)
                logger.info(f"特征名称加载成功，共 {len(self.feature_names)} 个特征")
            else:
                logger.warning(f"特征名称文件不存在: {settings.FEATURE_NAMES_PATH}")

            # 加载阈值
            self.threshold = self._load_threshold()
            logger.info(f"使用阈值: {self.threshold:.4f}")

            self._loaded = True
            return True

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False

    def _load_threshold(self) -> float:
        """
        智能阈值加载
        优先级：模型内置 > 外部文件 > 默认值
        """
        # 策略1：从模型对象获取
        if hasattr(self.model, 'best_threshold'):
            logger.info(f"从模型内部属性加载阈值: {self.model.best_threshold:.4f}")
            return self.model.best_threshold

        # 策略2：从外部文件获取
        if settings.THRESHOLD_PATH.exists():
            try:
                threshold_config = joblib.load(settings.THRESHOLD_PATH)
                if isinstance(threshold_config, dict):
                    threshold = threshold_config.get('best_threshold', settings.DEFAULT_THRESHOLD)
                else:
                    threshold = float(threshold_config)
                logger.info(f"从外部文件加载阈值: {threshold:.4f}")
                return threshold
            except Exception as e:
                logger.warning(f"阈值文件读取失败: {e}")

        # 策略3：使用默认值
        logger.info(f"使用默认阈值: {settings.DEFAULT_THRESHOLD}")
        return settings.DEFAULT_THRESHOLD

    @property
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._loaded and self.model is not None

    def _convert_actions_to_dataframe(self, user_actions: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        将API请求的用户行为数据列表转换为DataFrame

        Args:
            user_actions: 用户行为数据列表

        Returns:
            pd.DataFrame: 格式化后的数据框
        """
        # 创建DataFrame
        df = pd.DataFrame(user_actions)

        # 确保必需列存在
        required_cols = ['user_id', 'identifier', 'trigger_time']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"缺少必需字段: {col}")

        # 处理target字段（预测时默认为2）
        if 'target' not in df.columns:
            df['target'] = 2
        df['target'] = df['target'].fillna(2).apply(
            lambda x: 1 if str(x).strip().split('.')[0] == '1' else 0
        )

        # 时间格式转换
        def parse_time(v):
            v_s = str(v).strip()
            try:
                if 'E+' in v_s or 'e+' in v_s:
                    v_s = "{:.0f}".format(float(v_s))
                return v_s.split('.')[0]
            except:
                return v_s

        df['t_fix'] = df['trigger_time'].apply(parse_time)
        df['trigger_time'] = pd.to_datetime(df['t_fix'], format='%Y%m%d%H%M%S', errors='coerce')

        # 删除无效时间行
        df = df.dropna(subset=['trigger_time'])

        if df.empty:
            raise ValueError("时间格式转换后数据为空，请检查trigger_time格式（应为YYYYMMDDHHMMSS）")

        # 数值列处理
        for col in ['item_price', 'cart_count', 'current_item_price']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        # 创建中文列映射（特征工程依赖这些列名）
        col_mapping = {
            'login_type': '登录方式',
            'page_name': '页面名称',
            'item_price': '商品价格',
            'process_step': '办理步骤',
            'cart_count': '加购数量',
            'identifier': '标识符',
            'area_name': '区域名称',
            'interface_name': '接口名称',
            'item_name': '商品名称',
            'platform_type': '平台类型',
            'traffic_slot_name': '流量位名称',
            'current_item_name': '当前流量位商品名称',
            'source_page': '来源页面',
            'coupon_name': '使用优惠券名称'
        }

        for eng, chn in col_mapping.items():
            if eng in df.columns:
                df[chn] = df[eng]

        # 排序
        df = df.sort_values(['user_id', 'trigger_time']).reset_index(drop=True)

        return df

    def predict(
        self,
        user_actions: List[Dict[str, Any]],
        threshold: Optional[float] = None,
        return_features: bool = False
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        执行预测

        Args:
            user_actions: 用户行为数据列表
            threshold: 自定义阈值（可选）
            return_features: 是否返回特征详情

        Returns:
            Tuple[List[Dict], float]: (预测结果列表, 使用的阈值)
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        # 确定使用的阈值
        used_threshold = threshold if threshold is not None else self.threshold

        # 转换数据格式
        logger.info(f"接收到 {len(user_actions)} 条行为记录")
        df = self._convert_actions_to_dataframe(user_actions)
        logger.info(f"数据清洗后剩余 {len(df)} 条记录")

        # 执行特征工程
        logger.info("执行特征工程...")
        feature_df = self.feature_engineer.execute(df)

        if feature_df.empty:
            raise ValueError("特征工程后数据为空，可能所有数据都被规则过滤")

        logger.info(f"特征工程完成，共 {len(feature_df)} 个用户")

        # 特征对齐
        X_infer = feature_df.reindex(columns=self.feature_names).fillna(0)

        # 类型转换
        actual_cat = [c for c in CAT_FEATURES if c in X_infer.columns]
        for col in actual_cat:
            X_infer[col] = X_infer[col].astype(str).astype('category')

        num_cols = [c for c in X_infer.columns if c not in actual_cat]
        for col in num_cols:
            X_infer[col] = pd.to_numeric(X_infer[col], errors='coerce').fillna(0)

        # 执行预测
        logger.info(f"执行预测，阈值: {used_threshold:.4f}")

        try:
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(X_infer)
                if hasattr(probs, 'ndim') and probs.ndim == 2:
                    probs = probs[:, 1]
            elif hasattr(self.model, 'classes_'):
                probs = self.model.predict_proba(X_infer)[:, 1]
            else:
                probs = self.model.predict(X_infer)
        except Exception as e:
            logger.error(f"预测执行失败: {e}")
            raise RuntimeError(f"模型预测失败: {e}")

        # 构建结果
        feature_df['intent_probability'] = probs
        feature_df['intent_label'] = np.where(probs >= used_threshold, '1', '2')

        # 格式化输出
        results = []
        for idx, row in feature_df.iterrows():
            result = {
                'user_id': str(row['user_id']),
                'intent_label': row['intent_label'],
                'intent_probability': float(row['intent_probability'])
            }

            # 如果需要返回特征
            if return_features:
                features = {}
                for fname in self.feature_names:
                    if fname in X_infer.columns:
                        val = X_infer.loc[idx, fname] if idx in X_infer.index else 0
                        # 转换为Python原生类型
                        if pd.isna(val):
                            features[fname] = 0
                        elif isinstance(val, (np.integer, np.floating)):
                            features[fname] = float(val)
                        else:
                            features[fname] = str(val)
                result['features'] = features

            results.append(result)

        # 按概率降序排序
        results.sort(key=lambda x: x['intent_probability'], reverse=True)

        logger.info(f"预测完成，高意向用户: {sum(1 for r in results if r['intent_label'] == '1')}/{len(results)}")

        return results, used_threshold

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            Dict: 模型详细信息
        """
        if not self.is_loaded:
            return {
                'model_type': 'Unknown',
                'feature_count': 0,
                'feature_names': [],
                'default_threshold': self.threshold,
                'model_path': str(settings.MODEL_PATH)
            }

        model_type = 'LightGBM (IntentModel)'
        if hasattr(self.model, 'model'):
            model_type = f"LightGBM ({type(self.model.model).__name__})"

        return {
            'model_type': model_type,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'default_threshold': self.threshold,
            'model_path': str(settings.MODEL_PATH)
        }


# 创建全局预测服务实例
prediction_service = PredictionService()
