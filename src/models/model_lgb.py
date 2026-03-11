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

    def _get_feature_penalties(self, feature_names):
        """
        生成 CEGB 特征惩罚列表，用于 LightGBM 的 cegb_penalty_feature_coupled 参数
        对购机直接相关特征施加分裂代价，降低模型对这些特征的依赖
        """
        if not FEATURE_WEAKENING_CONFIG.get('enabled', False):
            return None

        penalty = FEATURE_WEAKENING_CONFIG.get('weight_decay_factor', 0.5)
        penalties = []

        for feat in feature_names:
            if feat in PURCHASE_DIRECT_FEATURES:
                penalties.append(penalty)
            else:
                penalties.append(0.0)

        return penalties

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

    def optimize_by_metric(self, y_true, probs, min_recall=0.60):
        """
        自动搜索最佳阈值
        - 搜索范围: 0.10 ~ 0.90（E8：下界从 0.3 扩展到 0.1）
        - 优化目标: F1-Score
        - 硬约束1: positive_rate <= 50%
        - 硬约束2: recall >= min_recall（E8：新增护栏）
        """
        # 【E8】搜索下界 0.3 → 0.10，解除 F1 峰值搜索盲区
        threshold_range = np.arange(0.10, 0.91, 0.01)

        best_score = -1
        best_threshold = 0.5
        results = []

        for threshold in threshold_range:
            metrics = self.evaluate_threshold(y_true, probs, threshold)

            # 原有约束：商机占比不超过 50%
            if metrics['positive_rate'] > 0.5:
                continue

            # 【E8】新增 recall 下限护栏
            if metrics['recall'] < min_recall:
                continue

            # 优化目标: F1
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

    def auto_train(self, X_train, y_train, cat_features=None, n_trials=15, sample_weight=None, monotone_features=None):
        """
        自动调优训练 (带进度条版)
        新增：支持购机相关特征权重约束
        新增：支持 sample_weight（E12 正样本质量降权）
        新增：支持 monotone_features（P1 单调性约束）
        """
        print(f">>> 开始模型自动调优 (总尝试次数: {n_trials})...")

        # 【P1】构建单调性约束列表
        monotone_constraints = None
        if monotone_features:
            feature_list = X_train.columns.tolist()
            monotone_constraints = [1 if f in monotone_features else 0 for f in feature_list]
            constrained = [f for f in feature_list if f in monotone_features]
            print(f">>> 【P1】已启用单调性约束 (单调递增): {constrained}")

        # 获取 CEGB 特征惩罚（用于弱化购机相关特征）
        feature_penalties = self._get_feature_penalties(X_train.columns.tolist())
        if feature_penalties:
            weakened_features = [f for f in X_train.columns if f in PURCHASE_DIRECT_FEATURES]
            print(f">>> 已启用 CEGB 特征惩罚，弱化特征: {weakened_features}")

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
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 100.0, log=True)  # 【E2】扩大范围，正样本率~1%，理论最优≈100
            }

            # 添加 CEGB 特征惩罚参数（实际弱化购机特征的分裂概率）
            if feature_penalties:
                params['cegb_tradeoff'] = trial.suggest_float('cegb_tradeoff', 0.01, 5.0, log=True)
                params['cegb_penalty_feature_coupled'] = feature_penalties

            # 【P1】添加单调性约束
            if monotone_constraints:
                params['monotone_constraints'] = monotone_constraints

            # 创建 Dataset，添加特征权重
            train_data = lgb.Dataset(
                X_train, label=y_train,
                categorical_feature=cat_features,
                feature_name=X_train.columns.tolist(),
                weight=sample_weight,  # 【E12】传入样本权重
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

        # 创建最终训练数据集
        train_data = lgb.Dataset(
            X_train, label=y_train,
            categorical_feature=cat_features,
            feature_name=X_train.columns.tolist(),
            weight=sample_weight  # 【E12】传入样本权重
        )

        # 添加 CEGB 特征惩罚到最终模型参数
        if feature_penalties:
            best_params['cegb_penalty_feature_coupled'] = feature_penalties

        # 【P1】添加单调性约束到最终模型参数
        if monotone_constraints:
            best_params['monotone_constraints'] = monotone_constraints

        self.model = lgb.train(
            best_params,
            train_data,
            num_boost_round=1000
        )

        self.feature_names = X_train.columns.tolist()

        # 【E3】不再在训练集上搜阈值，由 train_pipline.py 在验证集上调用 optimize_by_metric
        print(">>> 模型训练完成，阈值将在验证集上搜索")

    def predict_proba(self, X):
        """预测概率（直接返回模型原始输出）"""
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
