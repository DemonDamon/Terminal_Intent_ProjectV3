# v0.5.1 Code Review — 阈值搜索瓶颈分析与特征集中度评估

> 审查时间：2026-03-09
> 基线版本：v0.5（E4/E5/E6 特征优化后）
> 审查重点：阈值搜索机制缺陷、单特征依赖是否解决、分布漂移敏感性

---

## 一、v0.4 vs v0.5 验证集指标对比

| 指标 | v0.4 | v0.5 | 变化 | 判定 |
|------|------|------|------|------|
| ROC-AUC | 0.9897 | 0.9905 | +0.0008 | 微升 |
| PR-AUC | 0.8909 | 0.9036 | +0.0127 | 改善 |
| KS | 0.9187 | 0.9271 | +0.0084 | 改善 |
| Precision | 87.90% | 89.87% | +1.97pp | 升高 |
| **Recall** | **68.60%** | **66.07%** | **-2.53pp** | **下降** |
| **F1** | **0.7706** | **0.7616** | **-0.009** | **下降** |
| 阈值 | 0.300 | 0.300 | 不变 | 均撞下界 |
| 特征数 | 30 | 37 | +7 | E5 新增 |

**总结：** 模型排序能力提升（AUC/KS 均涨），但实际业务表现（F1）微降。模型变得更保守——Precision 升高但 Recall 下降，漏掉更多高意向用户。

---

## 二、特征重要性对比

### 2.1 v0.4 特征重要性 Top-20（30 个特征）

| 排名 | 特征 | 贡献度% |
|------|------|--------|
| 1 | view_detail_cnt | **35.19** |
| 2 | avg_intent_level | 10.57 |
| 3 | cnt_bussProcessing | 10.21 |
| 4 | view_phone_cnt | 9.11 |
| 5 | click_to_process_rate | 6.86 |
| 6 | avg_action_interval | 5.17 |
| 7 | max_action_interval | 2.75 |
| 8 | cnt_eventClick | 2.51 |
| 9 | price_level | 2.34 |
| 10 | high_intent_ratio | 2.24 |
| 11 | last_3_actions_seq | 1.58 |
| 12 | total_actions | 1.51 |
| 13 | intent_trend | 1.21 |
| 14 | action_frequency | 1.20 |
| 15 | conversion_efficiency | 1.13 |
| 16 | std_action_interval | 1.12 |
| 17 | click_specs_cnt | 1.07 |
| 18 | recent_5_avg_intent | 0.83 |
| 19 | process_fail_count | 0.83 |
| 20 | recent_3_avg_intent | 0.69 |

### 2.2 v0.5 特征重要性 Top-20（37 个特征）

| 排名 | 特征 | 贡献度% |
|------|------|--------|
| 1 | click_to_process_rate | **26.05** |
| 2 | view_detail_cnt | **22.24** |
| 3 | repeat_view_ratio | **13.87** |
| 4 | cnt_bussProcessing | 4.52 |
| 5 | max_action_interval | 4.18 |
| 6 | high_intent_ratio | 3.14 |
| 7 | total_actions | 2.96 |
| 8 | detail_view_ratio | 2.46 |
| 9 | actions_per_session | 1.93 |
| 10 | avg_intent_level | 1.75 |
| 11 | last_3_actions_seq | 1.72 |
| 12 | view_phone_cnt | 1.48 |
| 13 | price_level | 1.42 |
| 14 | avg_action_interval | 1.23 |
| 15 | intent_trend | 1.19 |
| 16 | visit_span_days | 1.14 |
| 17 | action_frequency | 1.03 |
| 18 | active_days | 1.02 |
| 19 | click_specs_cnt | 1.01 |
| 20 | session_count | 0.99 |

### 2.3 关键变化分析

| 特征 | v0.4 | v0.5 | 变化 |
|------|------|------|------|
| view_detail_cnt | 35.19% | 22.24% | -12.95pp（E4 log 压缩生效） |
| click_to_process_rate | 6.86% | 26.05% | +19.19pp（接替成为新主导） |
| repeat_view_ratio | 不存在 | 13.87% | E5 新特征，直接进入 Top-3 |
| avg_intent_level | 10.57% | 1.75% | -8.82pp（被新特征挤压） |
| view_phone_cnt | 9.11% | 1.48% | -7.63pp（被新特征挤压） |
| cnt_bussProcessing | 10.21% | 4.52% | -5.69pp |

---

## 三、核心问题：单特征依赖是否解决？

### 3.1 集中度量化

| 集中度指标 | v0.4 | v0.5 | 判定 |
|-----------|------|------|------|
| Top-1 占比 | 35.19% | 26.05% | 改善（-9.14pp） |
| **Top-3 占比** | **55.97%** | **62.16%** | **恶化（+6.19pp）** |
| Top-5 占比 | 71.94% | 70.86% | 持平 |
| >5% 的特征数 | 6 个 | 3 个 | 恶化 |
| #3→#4 落差 | 10.21%→9.11% | 13.87%→4.52% | 断崖加剧 |

### 3.2 结论

**部分解决，但产生了新的结构性问题。**

- **已解决：** `view_detail_cnt` 的独裁被打破（35.19% → 22.24%），E4 的 log 压缩 + ratio 特征生效，不再是"单特征模型"。
- **未解决（恶化）：** 依赖从"单寡头"转变为"三寡头"，Top-3 集中度从 55.97% 升至 62.16%，且 #3 到 #4 出现 13.87%→4.52% 的断崖。模型本质上依赖三个特征的投票，其余 34 个特征近乎陪跑。

### 3.3 分布漂移敏感性评估

| 漂移场景 | v0.4 风险 | v0.5 风险 |
|---------|----------|----------|
| 商城 UI 改版（浏览行为变化） | 失去 35% 决策依据 | 失去 ~25%（更安全） |
| 办理流程变更（转化路径变化） | 仅失去 7% | **失去 26%（更危险）** |
| 用户群体变化（新客/老客比例） | 多特征分散缓冲 | repeat_view_ratio 失去 14% |
| 多特征联动漂移（大促/节假日） | 6 个 >5% 特征缓冲 | **仅 3 个 >5%，缓冲层缺失** |

**风险提示：** `click_to_process_rate`、`repeat_view_ratio`、`view_detail_cnt` 三者的底层数据源高度重叠（均来自用户浏览-点击行为链），存在共线漂移风险——一旦用户行为模式整体变化，三个特征会同时失效。

---

## 四、Recall 下降根因分析

v0.5 Recall 比 v0.4 低 2.53pp（68.60% → 66.07%），三个原因：

1. **repeat_view_ratio 只对多次访问用户有效** — 单次深度访问即转化的用户该特征值为 0，模型更难识别这类样本。
2. **E4 压缩降低了 view_detail_cnt 的高值区分力** — `log1p` 压缩了高值区间的差异，导致中等意向用户的预测概率被拉低。
3. **click_to_process_rate 跃升至 26% 产生"门槛效应"** — 未进入办理流程的用户（rate=0）被模型直接压低分数，即使其他行为信号很强。

---

## 五、发现的关键 Bug：阈值搜索范围过窄

### 5.1 问题定位

`src/models/model_lgb.py:60-62`：

```python
def optimize_by_metric(self, y_true, probs):
    """自动搜索最佳阈值 (0.3 ~ 0.9)"""
    threshold_range = np.arange(0.3, 0.91, 0.01)
```

v0.4 和 v0.5 的最优阈值**都卡在 0.300**——恰好是搜索范围的下界。这说明 F1 在 0.30~0.90 范围内单调递减，真正的 F1 峰值位于 0.300 左侧（约 0.15~0.28 区间），但搜索器无法触达。

```
F1
 │    .
 │   . .        ← 真正的 F1 峰值（搜索盲区）
 │  .   .
 │ .     . . .
 │.       . . . . .
 ├──┬─────────────→ threshold
  0.15  0.30  0.50  0.70  0.90
        ↑
        当前搜索起点（每次都撞墙返回下界值）
```

### 5.2 影响

- 两个版本的阈值优化**均未找到真正最优点**，F1 被人为压低。
- Recall 被高阈值压制，大量概率在 0.15~0.30 之间的正样本被错误判为负样本。
- 阈值搜索流程形同虚设——每次都返回下界 0.300。

### 5.3 方案讨论：加 Recall 下限约束 vs 扩展搜索范围

**方案 A：仅加 Recall 下限约束（如 recall >= 70%）**

在当前搜索范围 0.3~0.9 下**会直接失败**：

| 版本 | 阈值 0.300 时 Recall | recall>=70% 约束 | 结果 |
|------|---------------------|----------------|------|
| v0.4 | 68.60% | 不满足 | 全部候选被过滤，触发兜底逻辑 |
| v0.5 | 66.07% | 不满足 | 全部候选被过滤，触发兜底逻辑 |

搜索范围内没有任何阈值能满足 recall>=70%（recall 随阈值升高只会更低），加约束但不扩范围 = 搜索必然失败。

**方案 B：扩展搜索范围至 0.10**

根因修复——让搜索器覆盖 F1 峰值所在的 0.10~0.30 区间。

**最终结论：B 先行 + A 作护栏**

| 方案 | 定位 | 单独使用 |
|------|------|---------|
| B（扩范围） | 修路——让搜索器到达正确区间 | 可独立生效 |
| A（recall 约束） | 装护栏——防止极端情况 | 不可单独使用（会失败） |
| **B + A 组合** | **最优方案** | **推荐** |

---

## 六、v0.5.1 修复方案：阈值搜索优化（E8）

### 6.1 修改目标

`src/models/model_lgb.py` 的 `optimize_by_metric` 方法，三处改动。

### 6.2 具体改动

```python
def optimize_by_metric(self, y_true, probs, min_recall=0.60):
    """
    自动搜索最佳阈值
    - 搜索范围: 0.10 ~ 0.90（修复：下界从 0.3 扩展到 0.1）
    - 优化目标: F1-Score
    - 硬约束1: positive_rate <= 50%
    - 硬约束2: recall >= min_recall（新增护栏）
    """
    # 改动1: 搜索下界 0.3 → 0.10
    threshold_range = np.arange(0.10, 0.91, 0.01)

    best_score = -1
    best_threshold = 0.5
    results = []

    for threshold in threshold_range:
        metrics = self.evaluate_threshold(y_true, probs, threshold)

        # 原有约束：商机占比不超过 50%
        if metrics['positive_rate'] > 0.5:
            continue

        # 改动2: 新增 recall 下限护栏
        if metrics['recall'] < min_recall:
            continue

        # 优化目标: F1
        score = metrics['f1']

        if score > best_score:
            best_score = score
            best_threshold = threshold

        results.append(metrics)

    # 兜底逻辑（不变）
    if not results:
        fallback_val = probs.mean() + probs.std()
        best_threshold = min(0.95, max(0.5, fallback_val))
        print(f" 未找到最优阈值，使用统计兜底值: {best_threshold:.3f}")

    self.best_threshold = best_threshold
    self.optimization_results = pd.DataFrame(results)
    return best_threshold, best_score
```

### 6.3 改动说明

| # | 改动点 | 原值 | 新值 | 原因 |
|---|--------|------|------|------|
| 1 | 搜索下界 | 0.30 | 0.10 | 解除 F1 峰值搜索盲区 |
| 2 | recall 护栏 | 无 | >= 0.60（默认） | 防止阈值过低导致 precision 崩塌 |
| 3 | 方法签名 | `(self, y_true, probs)` | `(self, y_true, probs, min_recall=0.60)` | 支持业务侧按需调整 recall 下限 |

### 6.4 预期效果

- 阈值预计从 0.300 下移至 **0.18~0.25** 区间
- Recall 预计提升至 **73~78%**（+7~12pp）
- Precision 预计下降至 **80~85%**（仍在业务可接受范围）
- F1 预计提升至 **0.79~0.82**（+0.03~0.06）

### 6.5 调用方无需改动

`train_pipline.py:152` 的调用保持兼容：

```python
# 现有调用无需修改（min_recall 使用默认值 0.60）
model_runner.optimize_by_metric(y_val, y_pred_prob)
```

---

## 七、后续优化方案（v0.5.2+）

### 7.1 E9：打破 Top-3 三寡头集中（P1 优先级）

**问题：** Top-3 特征（click_to_process_rate 26.05% + view_detail_cnt 22.24% + repeat_view_ratio 13.87%）合占 62.16%，其余 34 个特征均 <5%。

**方案：对 Top-3 中尚未压缩的两个特征施加变换**

修改文件：`src/features/unified_fe.py`

```python
# --- 在维度二 click_to_process_rate 计算之后 ---

# 【E9】click_to_process_rate 压缩：当前占 26.05%，用 sqrt 压缩 [0,1] 区间极端值
# sqrt(0) = 0, sqrt(0.5) = 0.71, sqrt(1) = 1 —— 拉升低值区区分度，压缩高值区差异
fe['click_to_process_rate'] = np.sqrt(fe['click_to_process_rate'])

# --- 在 E5 repeat_view_ratio 计算之后 ---

# 【E9】repeat_view_ratio 压缩：当前占 13.87%，与 view_detail_cnt 同等 log1p 待遇
fe['repeat_view_ratio'] = np.log1p(fe['repeat_view_ratio'])
```

**设计依据：**

| 特征 | 当前占比 | 压缩方式 | 选择原因 |
|------|---------|---------|---------|
| view_detail_cnt | 22.24% | log1p（E4 已做） | 计数型特征，log 天然适配 |
| click_to_process_rate | 26.05% | sqrt | 比率型特征（值域 [0,1]），log1p 效果弱，sqrt 在 0 附近有更强拉伸 |
| repeat_view_ratio | 13.87% | log1p | 计数比值，分布右偏，与 view_detail_cnt 同构 |

**预期效果：** Top-3 集中度从 62.16% 降至 45~50%，中层特征（cnt_bussProcessing、max_action_interval 等）占比提升。

### 7.2 E10：增加交叉特征（P2 优先级）

**问题：** 第 4~20 名特征贡献太碎（均 <5%），单独信号太弱，需要通过交叉组合形成"联合信号"。

**方案：新增 4 个交叉特征**

修改文件：`src/features/unified_fe.py`，在 E5 特征块之后追加：

```python
# 【E10】交叉特征：提升中层特征的联合信号强度

# 1. 深度浏览但未转化 —— 捕获"安静浏览型"正样本（click_to_process_rate=0 的漏网之鱼）
#    view_detail_cnt 高但 click_to_process_rate 低 → 在犹豫
fe['browse_depth_no_click'] = np.expm1(fe['view_detail_cnt']) * (1 - fe['click_to_process_rate'])

# 2. 跨天反复犹豫 —— active_days 和 repeat_view_ratio 的交互
#    多天来 + 反复看同一款 = 强犹豫信号
fe['revisit_intensity'] = fe['active_days'] * fe['repeat_view_ratio']

# 3. 看机型占比 —— 区分"随便逛"和"认真选机"
#    view_phone_cnt / total_actions，纯浏览手机的占比越高 → 越接近购机意向
fe['phone_browse_ratio'] = fe['view_phone_cnt'] / (fe['total_actions'] + 1)

# 4. 意向趋势与落差交互 —— 同时捕获方向和幅度
#    trend 正 + gap 大 → 意向正在快速回升（强信号）
#    trend 负 + gap 大 → 意向正在下滑（弱信号）
fe['intent_momentum'] = fe['intent_trend'] * fe['intent_level_gap']
```

**特征设计意图：**

| 交叉特征 | 捕获画像 | 弥补的缺口 |
|---------|---------|----------|
| browse_depth_no_click | 深度浏览但未走办理流程的用户 | click_to_process_rate=0 导致的"门槛效应" |
| revisit_intensity | 跨天反复查看同一商品的犹豫用户 | 单次访问正样本无法被 repeat_view_ratio 识别 |
| phone_browse_ratio | 专注浏览手机（vs 浏览福袋/券）的用户 | view_phone_cnt 单独占比太低（1.48%）无法发挥 |
| intent_momentum | 意向正在快速回升的用户 | intent_trend 和 intent_level_gap 各自 ~1%，组合后捕获方向+幅度 |

### 7.3 E11：Optuna 搜索力度提升（P1 优先级）

**问题：** 当前 `n_trials=15`，在 scale_pos_weight 范围 1~100（log 尺度）下覆盖不足。

**方案：** 修改 `train_pipline.py:123`

```python
# 原值
model_runner.auto_train(X_train, y_train, cat_features=actual_cat, n_trials=15)

# 修改为
model_runner.auto_train(X_train, y_train, cat_features=actual_cat, n_trials=30)
```

**依据：** Optuna 在 9 维超参空间（含 log 尺度的 scale_pos_weight、lambda_l1、lambda_l2）中，15 次试验的覆盖率约 15/9^2 ≈ 18.5%。提升到 30 次后覆盖率翻倍至 37%，且 Optuna 的 TPE 采样器在 25+ 次后才能有效建模参数分布。

### 7.4 E12：正样本质量过滤（P2 优先级）

**问题：** 部分正样本行为极少（total_actions < 3）就完成转化，特征矩阵近乎全零，是模型的噪声源。这些"低质正样本"拉低了模型对正常正样本的判别边界。

**方案 A：软降权（推荐）**

修改文件：`train_pipline.py`，在模型训练前添加 sample_weight：

```python
# 【E12】正样本质量过滤：对行为极少的正样本降权
# 逻辑：target=1 但 total_actions<3 的样本，权重降为 0.3
noisy_mask = (y_train == 1) & (X_train['total_actions'] < np.log1p(3))  # 注意 E4 已做 log1p
sample_weight = np.where(noisy_mask, 0.3, 1.0)
logger.info(f"低质正样本数量: {noisy_mask.sum()} (已降权至 0.3)")

# 传入 auto_train（需要扩展 auto_train 接口支持 sample_weight）
model_runner.auto_train(X_train, y_train, cat_features=actual_cat,
                        n_trials=30, sample_weight=sample_weight)
```

对应修改 `src/models/model_lgb.py` 的 `auto_train` 方法：

```python
def auto_train(self, X_train, y_train, cat_features=None, n_trials=15, sample_weight=None):
    # ... 在 lgb.Dataset 创建时传入 weight
    train_data = lgb.Dataset(
        X_train, label=y_train,
        categorical_feature=cat_features,
        feature_name=X_train.columns.tolist(),
        weight=sample_weight,      # 新增
        free_raw_data=False
    )
```

**方案 B：硬过滤（备选，更激进）**

```python
# 直接移除 total_actions < 3 的正样本
noisy_idx = (y_train == 1) & (X_train['total_actions'] < np.log1p(3))
X_train = X_train[~noisy_idx]
y_train = y_train[~noisy_idx]
logger.info(f"已移除 {noisy_idx.sum()} 条低质正样本")
```

**推荐方案 A**，原因：降权保留了样本的边际信息，硬过滤可能丢弃少数真实但行为稀疏的用户。

### 7.5 E13：二阶段模型（P3 优先级）

**问题：** 单一模型难以同时处理"高确定性用户"和"犹豫区间用户"，两类用户的特征模式差异大。

**方案：高置信模型 + 边界模型**

```
                 ┌──────────────┐
用户特征向量 ──→ │  Stage 1     │──→ prob >= 0.50 ──→ 直接判定"高意向"
                 │  主模型(当前) │
                 └──────┬───────┘
                        │
                        ↓ prob ∈ [0.15, 0.50)
                 ┌──────────────┐
                 │  Stage 2     │──→ 二次判定
                 │  边界模型     │     prob2 >= threshold2 → "中意向"
                 └──────────────┘     prob2 <  threshold2 → "低意向"
```

**Stage 2 模型特点：**
- 训练数据：仅使用 Stage 1 预测概率在 [0.15, 0.50) 区间的样本
- 特征重点：更多使用行为序列特征（last_3_actions_seq、intent_acceleration、session_count）
- 目标：在 Stage 1 的"犹豫区间"内进一步细分

**实施建议：** 仅当 E8~E12 优化后 F1 仍未达到 0.82 时启动。二阶段模型增加了系统复杂度（两套模型维护），需要 Recall 收益足够大才值得引入。

---

## 八、实施路线

```
v0.5.1  ──→  E8:  阈值搜索范围扩展 + recall 护栏
         │   E9:  Top-3 特征压缩（sqrt / log1p）
         │   E10: 4 个交叉特征
         │   E11: n_trials 15→30
         │   E12: 正样本质量降权
         │        目标：F1 >= 0.82，Recall >= 78%，Top-3 集中度 < 50%
         │
v0.5.2  ──→  E13: 二阶段模型（仅在 E8~E12 不达预期时启动）
              目标：犹豫区间用户捕获率提升，Recall +3~5pp
```
