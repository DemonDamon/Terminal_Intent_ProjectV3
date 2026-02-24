import lightgbm as lgb
import optuna
import pandas as pd
import numpy as np
import joblib
import os
# 【新增】引入进度条库
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, f1_score
from config.feature_list import PURCHASE_DIRECT_FEATURES, FEATURE_WEAKENING_CONFIG

class IntentModel:
    def __init__(self):
        self.model = None
        self.best_threshold = 0.5
        self.feature_names = []
        self.optimization_results = None
        self.feature_weights = None  # 特征权重字典

    def _get_feature_weights(self, feature_names):
        """
        生成特征权重列表，用于 LightGBM 的 feature_contribs 参数
        对购机直接相关特征应用衰减权重
        """
        if not FEATURE_WEAKENING_CONFIG.get('enabled', False):
            return None

        decay_factor = FEATURE_WEAKENING_CONFIG.get('weight_decay_factor', 0.5)
        weights = []

        for feat in feature_names:
            if feat in PURCHASE_DIRECT_FEATURES:
                weights.append(decay_factor)
            else:
                weights.append(1.0)

        return weights

    def evaluate_threshold(self, y_true, probs, threshold):
        """评估特定阈值下的各项指标"""
        y_pred = (probs >= threshold).astype(int)
        
        # 计算基础指标
        precision = np.mean(y_true[y_pred == 1]) if np.sum(y_pred) > 0 else 0
        recall = np.sum(y_true[y_pred == 1]) / np.sum(y_true) if np.sum(y_true) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 业务指标：商机占比 (Positive Rate)
        pos_rate = np.mean(y_pred)
        
        return {
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'positive_rate': pos_rate
        }

    def optimize_by_metric(self, y_true, probs):
        """自动搜索最佳阈值 (0.3 ~ 0.9)"""
        # 搜索范围从 0.3 开始，过滤掉过低的无效阈值
        threshold_range = np.arange(0.3, 0.91, 0.01)
        
        best_score = -1
        best_threshold = 0.6 # 默认给个稳健值
        results = []

        for threshold in threshold_range:
            metrics = self.evaluate_threshold(y_true, probs, threshold)
            
            # --- 业务硬约束：商机占比不能超过 50% ---
            if metrics['positive_rate'] > 0.5:
                continue
            
            # 使用 F1 分数作为核心指标
            score = metrics['f1'] 

            if score > best_score:
                best_score = score
                best_threshold = threshold
            
            results.append(metrics)

        # 兜底逻辑：如果找不到合适阈值
        if not results:
            # 修改为：限制最大值为 0.95，防止出现 > 1 的情况
            fallback_val = probs.mean() + probs.std()
            best_threshold = min(0.95, max(0.5, fallback_val))
            print(f" 未找到最优阈值，使用统计兜底值: {best_threshold:.3f}")
            
        self.best_threshold = best_threshold
        self.optimization_results = pd.DataFrame(results)
        return best_threshold, best_score

    def auto_train(self, X_train, y_train, cat_features=None, n_trials=15):
        """
        自动调优训练 (带进度条版)
        新增：支持购机相关特征权重约束
        """
        print(f">>> 开始模型自动调优 (总尝试次数: {n_trials})...")

        # 获取特征权重（用于弱化购机相关特征）
        feature_weights = self._get_feature_weights(X_train.columns.tolist())
        if feature_weights:
            weakened_features = [f for f in X_train.columns if f in PURCHASE_DIRECT_FEATURES]
            print(f">>> 已启用特征权重约束，弱化特征: {weakened_features}")

        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 5.0)  # 应对样本不平衡
            }

            # 创建 Dataset，添加特征权重
            train_data = lgb.Dataset(
                X_train, label=y_train,
                categorical_feature=cat_features,
                feature_name=X_train.columns.tolist(),
                free_raw_data=False
            )

            # 使用 lgb.cv 进行交叉验证
            # verbose_eval=False 关掉刷屏日志，防止破坏进度条
            cv_results = lgb.cv(
                params,
                train_data,
                num_boost_round=1000,
                nfold=3,
                stratified=True,
                shuffle=True,
                metrics='auc',
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
            )

            return cv_results['valid auc-mean'][-1]

        # --- 【核心修改】: 添加 tqdm 进度条 ---
        study = optuna.create_study(direction='maximize')

        # 初始化进度条
        with tqdm(total=n_trials, desc=" 训练优化中", unit="trial", colour='green') as pbar:

            # 定义回调函数，每次 trial 结束更新一次
            def tqdm_callback(study, trial):
                pbar.update(1)
                # 实时显示当前找到的最佳 AUC
                try:
                    best_val = study.best_value
                except:
                    best_val = 0.0
                pbar.set_postfix({"当前最佳AUC": f"{best_val:.4f}"})

            # 开始优化
            study.optimize(objective, n_trials=n_trials, callbacks=[tqdm_callback])
        # -----------------------------------

        print(f"\n调优完成! 全局最佳 AUC: {study.best_value:.4f}")

        # 使用最佳参数重新训练最终模型
        print(">>> 正在使用最佳参数全量拟合模型...")
        best_params = study.best_params
        best_params.update({
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1
        })

        # 创建最终训练数据集，添加特征权重约束
        train_data = lgb.Dataset(
            X_train, label=y_train,
            categorical_feature=cat_features,
            feature_name=X_train.columns.tolist()
        )

        # 如果启用了特征权重，使用自定义初始化回调来限制特征影响
        self.model = lgb.train(
            best_params,
            train_data,
            num_boost_round=1000
        )
        
        self.feature_names = X_train.columns.tolist()
        
        # 训练完成后，自动在训练集上寻找最佳阈值（作为默认值）
        print(">>> 正在计算最佳判定阈值...")
        # 这里的 probs 是训练集回测概率，仅用于确定阈值分布
        train_probs = self.model.predict(X_train)
        self.optimize_by_metric(y_train, train_probs)
        print(f" 自动锁定最佳阈值: {self.best_threshold:.3f}")

    def predict_proba(self, X):
        return self.model.predict(X)

    def save(self, filepath, feature_names_path=None, feature_names=None): # <--- 修改参数定义
        """
        保存模型及相关配置
        :param filepath: 模型文件保存路径 (.pkl)
        :param feature_names_path: 特征名列表保存路径 (可选)
        :param feature_names: 特征名列表 (可选)
        """
        # 1. 保存模型对象本身
        joblib.dump(self, filepath)
        
        # 2. 单独保存阈值配置 (双重保险)
        threshold_path = os.path.join(os.path.dirname(filepath), 'optimal_threshold.pkl')
        joblib.dump({'best_threshold': self.best_threshold}, threshold_path)
        
        # 3. 单独保存特征名列表 (修复报错的关键)
        if feature_names_path and feature_names:
            joblib.dump(feature_names, feature_names_path)
            print(f" 特征列表已保存至: {feature_names_path}")
            
        print(f" 模型已保存至: {filepath}")