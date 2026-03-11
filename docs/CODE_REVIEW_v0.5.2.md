# v0.5.2 Code Review — 概率校准、单调约束与泛化增强

> 审查时间：2026-03-10
> 基线版本：v0.5.1（E8-E12 阈值/特征/管道优化后）
> 审查重点：概率分布验证、校准稳健性、模型泛化能力、E9/E10 交互缺陷

---

## 一、v0.5.1 遗留问题回顾

v0.5.1 实施了 E8-E12 优化，核心改动是阈值搜索范围从 [0.3, 0.9] 扩展到 [0.1, 0.9]。但以下问题未解决：

| # | 问题 | 来源 | 风险等级 |
|---|------|------|---------|
| 1 | E8 扩展后 [0.10, 0.30) 区间的正负样本重叠程度未知 | 无数据验证 | P0 |
| 2 | 原始概率未校准，阈值跨月稳定性存疑 | 模型输出 | P1 |
| 3 | 核心特征无单调性约束，小样本区间可能学出反直觉分裂 | 模型结构 | P1 |
| 4 | E9 的 sqrt 压缩改变了 E10 交叉特征的语义 | 特征交互 | P2 |
| 5 | 验证集同时用于调参和评估，9 月指标可能偏乐观 | 验证策略 | P2 |
| 6 | 训练样本等权，早期数据与近期数据价值相同 | 数据策略 | P3 |
| 7 | last_3_actions_seq 高基数类别特征可能过拟合 | 特征质量 | P3 |

---

## 二、P0：概率分布分析

### 2.1 问题

E8 将阈值搜索下界从 0.30 扩展到 0.10，隐含假设是 **[0.10, 0.30) 区间内存在大量可回收的正样本**。但 CODE_REVIEW_v0.5.1 中未提供该区间的正负样本分布数据，假设未经验证。

如果该区间正负样本比接近 1:1，扩展阈值反而会让 Precision 崩塌，F1 不升反降。

### 2.2 方案

在 `train_pipline.py` 中新增 `analyze_probability_distribution()` 函数，训练后自动输出概率分布诊断报告。

### 2.3 具体改动

修改文件：`train_pipline.py`

在 `run_training()` 之前新增函数：

```python
def analyze_probability_distribution(y_true, probs, label=""):
    """
    【P0】概率分布分析：验证阈值搜索假设，检查正负样本在各概率区间的重叠情况。
    重点关注 E8 新覆盖的 [0.10, 0.30) 区间。
    """
    y_true = np.array(y_true)
    pos_probs = probs[y_true == 1]
    neg_probs = probs[y_true == 0]

    print(f"\n{'='*60}")
    print(f"  【P0】概率分布分析 {label}")
    print(f"{'='*60}")
    print(f"  样本数: 正={len(pos_probs)}, 负={len(neg_probs)}")
    print(f"  概率均值:   正={pos_probs.mean():.4f}, 负={neg_probs.mean():.4f}")
    print(f"  概率中位数: 正={np.median(pos_probs):.4f}, 负={np.median(neg_probs):.4f}")
    print(f"  概率标准差: 正={pos_probs.std():.4f}, 负={neg_probs.std():.4f}")

    # 分区间统计
    bins = [(0.00, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.25),
            (0.25, 0.30), (0.30, 0.50), (0.50, 0.70), (0.70, 1.01)]

    print(f"\n  {'区间':<14s} {'正样本':>6s} {'负样本':>6s} {'正占比':>7s} {'区间精度':>8s}")
    print(f"  {'-'*48}")

    for low, high in bins:
        pos_in = int(((pos_probs >= low) & (pos_probs < high)).sum())
        neg_in = int(((neg_probs >= low) & (neg_probs < high)).sum())
        total_in = pos_in + neg_in
        prec = pos_in / total_in if total_in > 0 else 0
        pos_pct = pos_in / len(pos_probs) * 100 if len(pos_probs) > 0 else 0
        mark = " *" if 0.10 <= low < 0.30 else ""
        print(f"  [{low:.2f}, {high:.2f}) {pos_in:>6d} {neg_in:>6d} {pos_pct:>6.1f}% {prec:>7.1%}{mark}")

    # E8 关键区间汇总
    e8_pos = int(((pos_probs >= 0.10) & (pos_probs < 0.30)).sum())
    e8_neg = int(((neg_probs >= 0.10) & (neg_probs < 0.30)).sum())
    e8_total = e8_pos + e8_neg
    print(f"\n  E8 关键区间 [0.10, 0.30) 汇总:")
    print(f"    正样本: {e8_pos} ({e8_pos/len(pos_probs)*100:.1f}% 的全部正样本)"
          if len(pos_probs) > 0 else "")
    print(f"    负样本: {e8_neg}")
    if e8_total > 0:
        print(f"    区间精度: {e8_pos/e8_total:.1%} (高于50%说明E8扩展有效)")
    print(f"{'='*60}")
```

在验证阶段调用两次（校准前 + 校准后）：

```python
    # C. 预测（此时 calibrator 尚未设置，返回原始概率）
    y_pred_prob_raw = model_runner.predict_proba(X_val_aligned)

    # 【P0】原始概率分布分析
    analyze_probability_distribution(y_val, y_pred_prob_raw, label="(原始概率)")

    # 【P1】概率校准
    y_pred_prob = model_runner.calibrate(y_val, y_pred_prob_raw)

    # 【P0】校准后概率分布分析
    analyze_probability_distribution(y_val, y_pred_prob, label="(校准后)")
```

### 2.4 输出示例（预期格式）

```
============================================================
  【P0】概率分布分析 (原始概率)
============================================================
  样本数: 正=XXX, 负=XXXX
  概率均值:   正=0.XXXX, 负=0.XXXX
  概率中位数: 正=0.XXXX, 负=0.XXXX
  概率标准差: 正=0.XXXX, 负=0.XXXX

  区间             正样本   负样本    正占比   区间精度
  ------------------------------------------------
  [0.00, 0.10)      XXX   XXXX   XX.X%   XX.X%
  [0.10, 0.15)      XXX    XXX   XX.X%   XX.X% *
  [0.15, 0.20)      XXX    XXX   XX.X%   XX.X% *
  [0.20, 0.25)      XXX    XXX   XX.X%   XX.X% *
  [0.25, 0.30)      XXX    XXX   XX.X%   XX.X% *
  [0.30, 0.50)      XXX    XXX   XX.X%   XX.X%
  [0.50, 0.70)      XXX      X   XX.X%   XX.X%
  [0.70, 1.01)      XXX      X   XX.X%   XX.X%

  E8 关键区间 [0.10, 0.30) 汇总:
    正样本: XX (XX.X% 的全部正样本)
    负样本: XXX
    区间精度: XX.X% (高于50%说明E8扩展有效)
============================================================
```

### 2.5 判读标准

| 区间精度 | 判定 | 后续动作 |
|---------|------|---------|
| > 70% | E8 扩展高度有效 | 正常使用扩展后阈值 |
| 50% ~ 70% | E8 扩展有效但有噪声 | 可配合 P1 校准提升 |
| < 50% | E8 扩展无效，正负混杂 | 需上调搜索下界或增强特征 |

---

## 三、P1：概率校准 + 单调性约束

### 3.1 P1a：概率校准（Isotonic Regression）

#### 问题

LightGBM 输出的原始概率未经校准，可能与真实正样本比例存在系统性偏差。如果概率分布集中在 0.20~0.35 的窄区间，阈值稍有偏移，Precision/Recall 就会剧烈变化。跨月部署时，数据分布漂移会放大这种不稳定性。

#### 方案

在验证集上使用 Isotonic Regression（保序回归）校准概率。校准器随模型一起序列化，预测时自动应用。

#### 具体改动

修改文件：`src/models/model_lgb.py`

1）新增导入和属性：

```python
from sklearn.isotonic import IsotonicRegression

class IntentModel:
    def __init__(self):
        ...
        self.calibrator = None       # 【P1】概率校准器
```

2）新增 `calibrate` 方法：

```python
    def calibrate(self, y_true, raw_probs):
        """
        【P1】概率校准：使用 Isotonic Regression 将原始概率映射到真实概率尺度。
        校准后的概率更接近真实的正样本比例，阈值选择更稳健。
        校准器会被序列化到模型文件中，预测时自动应用。
        """
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(raw_probs, y_true)
        calibrated = self.calibrator.transform(raw_probs)
        print(f">>> 【P1】概率校准完成 (Isotonic Regression)")
        print(f"    原始概率范围: [{raw_probs.min():.4f}, {raw_probs.max():.4f}]")
        print(f"    校准后范围:   [{calibrated.min():.4f}, {calibrated.max():.4f}]")
        return calibrated
```

3）修改 `predict_proba` 自动应用校准：

```python
    def predict_proba(self, X):
        """预测概率，自动应用校准（如果已校准）"""
        raw_probs = self.model.predict(X)
        calibrator = getattr(self, 'calibrator', None)
        if calibrator is not None:
            return calibrator.transform(raw_probs)
        return raw_probs
```

#### 设计说明

- **Isotonic Regression 选择原因**：相比 Platt Scaling（假设 sigmoid 关系），Isotonic Regression 是非参数方法，不预设概率映射的函数形式，对 LightGBM 输出的任意分布形态均有效。
- **`out_of_bounds='clip'`**：推理时遇到超出训练范围的概率值时，裁剪到 [0, 1] 而非报错。
- **`getattr(self, 'calibrator', None)`**：向后兼容——旧版本序列化的模型对象不含 `calibrator` 属性，`getattr` 安全返回 `None`。
- **校准器随 `joblib.dump(self)` 自动保存**：无需额外存储文件，预测管道无感知。

#### 训练流程变化

```
原始概率 → 【P0 分析】 → 【P1 校准】 → 【P0 分析】 → 【E8 阈值搜索（在校准后概率上）】
```

阈值搜索在校准后概率上进行，预测时 `predict_proba` 也返回校准后概率，两者一致。

---

### 3.2 P1b：单调性约束（Monotonic Constraints）

#### 问题

`click_to_process_rate`、`view_detail_cnt` 等特征与购机意向应为单调递增关系（值越大→意向越高）。但 LightGBM 的决策树可能在小样本区间学出非单调分裂（如"浏览 20 次以上反而降低预测概率"），这种反直觉的分裂在训练集上降低损失，但在新数据上容易失效。

#### 方案

在 `auto_train` 中新增 `monotone_features` 参数，对指定特征施加单调递增约束。

#### 具体改动

修改文件：`src/models/model_lgb.py` + `train_pipline.py`

1）`auto_train` 新增参数和约束构建：

```python
    def auto_train(self, X_train, y_train, cat_features=None, n_trials=15,
                   sample_weight=None, monotone_features=None):
        ...
        # 【P1】构建单调性约束列表
        monotone_constraints = None
        if monotone_features:
            feature_list = X_train.columns.tolist()
            monotone_constraints = [1 if f in monotone_features else 0 for f in feature_list]
            constrained = [f for f in feature_list if f in monotone_features]
            print(f">>> 【P1】已启用单调性约束 (单调递增): {constrained}")
```

2）在 Optuna objective 和最终训练中添加约束：

```python
            # 【P1】添加单调性约束
            if monotone_constraints:
                params['monotone_constraints'] = monotone_constraints
```

3）`train_pipline.py` 中定义约束特征并传入：

```python
    # 【P1】定义单调性约束特征（这些特征与购机意向应为单调递增关系）
    MONOTONE_FEATURES = ['click_to_process_rate', 'view_detail_cnt',
                         'high_intent_ratio', 'cnt_bussProcessing']

    model_runner.auto_train(X_train, y_train, cat_features=actual_cat, n_trials=30,
                            sample_weight=sample_weight, monotone_features=MONOTONE_FEATURES)
```

#### 约束特征选择依据

| 特征 | 约束方向 | 业务含义 | v0.5 重要性排名 |
|------|---------|---------|----------------|
| click_to_process_rate | 单调递增 | 办理转化率越高 → 意向越高 | #1 (26.05%) |
| view_detail_cnt | 单调递增 | 详情浏览越多 → 意向越高 | #2 (22.24%) |
| high_intent_ratio | 单调递增 | 高意图行为占比越高 → 意向越高 | #6 (3.14%) |
| cnt_bussProcessing | 单调递增 | 办理次数越多 → 意向越高 | #4 (4.52%) |

**未约束的特征**：`avg_action_interval`（间隔长可能代表犹豫也可能代表放弃，非单调）、`price_level`（高价不一定意味高意向）、所有类别特征（不支持单调约束）。

#### 预期效果

- 消除小样本区间的反直觉分裂，提升跨月泛化能力。
- AUC 可能微降 0.001~0.003（约束限制了模型自由度），但 F1 稳定性提升。
- 漂移场景下衰减更平缓——单调模型不会因少量噪声翻转决策方向。

---

## 四、P2：E9/E10 交互修复 + 验证策略增强

### 4.1 P2a：E9/E10 交互缺陷修复

#### 问题

E9 在 E10 之前对 `click_to_process_rate` 做了 sqrt 压缩，导致 E10 的 `browse_depth_no_click` 交叉特征语义改变：

```python
# E9 之后 click_to_process_rate 已经是 sqrt(原始值)
# E10 使用的是压缩后的值
fe['browse_depth_no_click'] = np.expm1(fe['view_detail_cnt']) * (1 - fe['click_to_process_rate'])
```

对犹豫用户（原始 rate=0.10~0.25）的影响：

| 原始 rate | sqrt(rate) | 1 - sqrt(rate) | 原始 1 - rate | 信号损失 |
|-----------|-----------|----------------|---------------|---------|
| 0.00 | 0.00 | 1.00 | 1.00 | 无 |
| 0.10 | 0.316 | **0.684** | 0.90 | **-24%** |
| 0.25 | 0.500 | **0.500** | 0.75 | **-33%** |
| 0.50 | 0.707 | **0.293** | 0.50 | **-41%** |

`browse_depth_no_click` 设计目的是捕获"深度浏览但未转化"的犹豫用户，但 E9 的 sqrt 让这类用户的信号被削弱了 24%~41%。

#### 方案

在计算 `browse_depth_no_click` 时使用 sqrt 压缩前的原始 rate：

修改文件：`src/features/unified_fe.py`

```python
        # --- 维度二：转化深度与质量 ---
        ...
        # 转化漏斗：点击到试算的转化率（原始值，用于 E10 交叉特征）
        click_to_process_rate_raw = fe['cnt_bussProcessing'] / (fe['cnt_eventClick'] + 1)
        # 【E9】click_to_process_rate 压缩：用 sqrt 压缩极端值
        fe['click_to_process_rate'] = np.sqrt(click_to_process_rate_raw)

        ...

        # 【E10】交叉特征
        # 修复：使用 sqrt 前的原始 rate，避免信号损失
        fe['browse_depth_no_click'] = np.expm1(fe['view_detail_cnt']) * (1 - click_to_process_rate_raw)
```

#### 预期效果

- 犹豫用户（rate=0.10~0.25）的 `browse_depth_no_click` 信号恢复至设计值。
- Recall 预计提升 1~2pp（该交叉特征专门捕获 click_to_process_rate=0 附近的正样本）。
- 不影响 `click_to_process_rate` 本身的压缩效果（模型训练中使用的仍是 sqrt 后的值）。

---

### 4.2 P2b：验证策略增强

#### 问题

当前验证集（9 月数据）同时承担两个角色：
1. 阈值搜索的调参集（`optimize_by_metric` 在验证集上搜索最优 F1 阈值）
2. 模型评估的测试集（最终报告的 Precision/Recall/F1 来自同一数据）

这导致 9 月指标可能偏乐观——阈值是"挑"出来最好的，不代表 10 月也能达到同样效果。

#### 方案

将 8 月训练数据再做一次时间子切分，形成三层结构：

```
8月前半段 (e.g. 8月1日-8月20日) → 训练模型
8月后半段 (e.g. 8月21日-8月31日) → 搜索阈值 + 校准概率
9月                              → 最终评估（只看不调，作为无偏估计）
```

修改文件：`train_pipline.py`

```python
    # 【P2b】三层切分：训练 / 调参 / 评估
    cutoff_train = pd.Timestamp('2024-08-21')   # 训练集截止
    cutoff_tune  = pd.Timestamp('2024-09-01')   # 调参集截止
    cutoff_val   = pd.Timestamp('2024-10-01')   # 评估集截止

    train_df = feature_df[feature_df['last_action_date'] < cutoff_train]
    tune_df  = feature_df[(feature_df['last_action_date'] >= cutoff_train) &
                           (feature_df['last_action_date'] < cutoff_tune)]
    val_df   = feature_df[(feature_df['last_action_date'] >= cutoff_tune) &
                           (feature_df['last_action_date'] < cutoff_val)]

    logger.info(f"三层切分: 训练 {len(train_df)} 条, 调参 {len(tune_df)} 条, 评估 {len(val_df)} 条")
```

然后：
- 在 `tune_df` 上做概率校准 + 阈值搜索
- 在 `val_df` 上做最终评估（不调任何参数）

#### 风险

- 训练集减少约 1/3（8 月后半段移出），可能导致模型拟合不足。
- 如果 8 月后半段数据量太少（正样本 < 50 条），阈值搜索不稳定。

#### 建议

先统计 8 月后半段的数据量和正样本数，再决定切分点。如果正样本不足，可退回当前方案（9 月同时调参+评估），接受指标轻微偏乐观。

---

## 五、P3：时间衰减加权 + 高基数特征降维

### 5.1 P3a：时间衰减加权（Recency Weighting）

#### 问题

当前 8 月训练数据中所有样本等权。但用户行为模式会随时间变化（促销活动、UI 改版、季节性），早期数据的分布可能已过时。等权训练会让模型记住已过时的模式。

#### 方案

在 E12 的 `sample_weight` 基础上叠加时间衰减，越近期的样本权重越高。

修改文件：`train_pipline.py`

```python
    # 【P3a】时间衰减加权：近期样本权重更高（指数衰减，半衰期 15 天）
    days_to_cutoff = (cutoff_train - train_df['last_action_date']).dt.days
    time_weight = np.exp(-0.693 * days_to_cutoff / 15)  # 半衰期 15 天
    sample_weight = sample_weight * time_weight.values    # 与 E12 降权叠加
    logger.info(f"【P3a】时间衰减加权: 最旧样本权重={time_weight.min():.3f}, 最新={time_weight.max():.3f}")
```

#### 衰减效果示例（假设 cutoff = 9月1日）

| 样本日期 | 距 cutoff 天数 | 权重 |
|---------|--------------|------|
| 8月31日 | 1 天 | 0.955 |
| 8月25日 | 7 天 | 0.722 |
| 8月15日 | 17 天 | 0.454 |
| 8月1日 | 31 天 | 0.242 |
| 7月15日 | 48 天 | 0.107 |

#### 预期效果

- 模型更贴近近期用户行为模式，跨月泛化时衰减更平缓。
- 如果近期数据与 9 月验证集分布更接近，F1 可能提升 0.01~0.02。

#### 风险

- 如果早期数据中包含稀有但重要的正样本模式，降权可能导致这些模式被遗忘。
- 半衰期 15 天是经验值，可能需要根据实际数据分布调整。

---

### 5.2 P3b：last_3_actions_seq 高基数降维

#### 问题

`last_3_actions_seq` 是用户最后 3 个事件拼接而成的字符串特征（如 `eventClick_bussinessProcessing_zdscLogin`），v0.5 中排名 #11（1.72%）。

该特征可能有数百个唯一值。LightGBM 处理高基数类别特征时会产生大量稀疏分裂，在少数叶子节点上过拟合。虽然 v0.5 中占比不高（1.72%），但它可能在特定数据子集上产生过拟合的分裂路径，影响泛化。

#### 方案

将原始序列字符串替换为结构化编码特征：

修改文件：`src/features/unified_fe.py`

```python
        # 【P3b】last_3_actions_seq 降基数：提取结构化特征替代原始字符串
        last_3 = grouped['标识符'].apply(lambda x: x.tail(3).tolist())

        # 最后一步是否为 bussinessProcessing（强信号）
        fe['last_action_is_process'] = last_3.apply(
            lambda x: 1 if x and x[-1] == 'bussinessProcessing' else 0
        )

        # 最后 3 步中是否包含登录（中等信号）
        fe['last_3_has_login'] = last_3.apply(
            lambda x: 1 if 'zdscLogin' in x else 0
        )

        # 最后 3 步的意图等级序列模式：升/降/平
        fe['last_3_intent_direction'] = grouped['intent_level'].apply(
            lambda x: 1 if len(x) >= 2 and x.iloc[-1] > x.iloc[-2]
            else (-1 if len(x) >= 2 and x.iloc[-1] < x.iloc[-2] else 0)
        )

        # 保留原始 last_3_actions_seq 用于规则层（但从模型训练特征中移除）
        fe['last_3_actions_seq'] = grouped['标识符'].apply(
            lambda x: "_".join(x.tail(3).astype(str).tolist())
        )
```

同时在 `config/feature_list.py` 中将 `last_3_actions_seq` 从 `CAT_FEATURES` 移除（新特征均为数值型）：

```python
CAT_FEATURES = [
    'last_login_type',
    'last_platform',
    'preferred_brand',
    # 'last_3_actions_seq',  # P3b: 替换为结构化数值特征
]
```

#### 预期效果

- 消除高基数类别特征的过拟合风险。
- 3 个新数值特征提供与原始序列等价的信号（最后动作类型、是否含登录、意图方向）。
- 训练速度微升（LightGBM 无需处理高基数 one-hot 分裂）。

---

## 六、实施优先级与依赖关系

```
P0（概率分布分析）─── 无依赖，诊断工具
  │
  ├─→ P1a（概率校准）── 依赖 P0 的分析结果确认校准必要性
  │
  ├─→ P1b（单调性约束）── 无依赖，独立于 P1a
  │
  ├─→ P2a（E9/E10 交互修复）── 无依赖
  │
  ├─→ P2b（验证策略三层分离）── 需先确认数据量是否足够
  │
  ├─→ P3a（时间衰减加权）── 依赖 E12 的 sample_weight 基础
  │
  └─→ P3b（last_3_actions_seq 降维）── 无依赖
```

### 推荐实施顺序

| 批次 | 改动 | 理由 |
|------|------|------|
| **第一批** | P0 + P1a + P1b + P2a | 收益确定，风险低，代码改动小 |
| **第二批** | P3a + P3b | 增量优化，需要训练验证 |
| **第三批** | P2b | 需要先统计数据量，决定切分点 |

---

## 七、改动文件汇总

| 优化项 | 文件 | 改动类型 |
|--------|------|---------|
| P0 | `train_pipline.py` | 新增函数 + 调用 |
| P1a | `src/models/model_lgb.py` | 新增导入/属性/方法 + 修改 predict_proba |
| P1a | `train_pipline.py` | 新增校准调用 |
| P1b | `src/models/model_lgb.py` | auto_train 新增参数 + 约束逻辑 |
| P1b | `train_pipline.py` | 定义 MONOTONE_FEATURES + 传参 |
| P2a | `src/features/unified_fe.py` | 保存原始 rate + 修改交叉特征计算 |
| P2b | `train_pipline.py` | 修改时间切分逻辑 |
| P3a | `train_pipline.py` | 新增时间衰减权重计算 |
| P3b | `src/features/unified_fe.py` | 新增 3 个结构化特征 |
| P3b | `config/feature_list.py` | CAT_FEATURES 移除 last_3_actions_seq |

---

## 八、v0.5.1 → v0.5.2 预期目标

| 指标 | v0.5 基线 | v0.5.1 预期 | v0.5.2 目标 |
|------|----------|------------|------------|
| ROC-AUC | 0.9905 | 0.990+ | 0.990+ |
| F1 | 0.7616 | 0.79~0.82 | 0.82~0.85 |
| Recall | 66.07% | 73~78% | 75~80% |
| Precision | 89.87% | 80~85% | 80~85% |
| Top-3 集中度 | 62.16% | 45~50% | 45~50% |
| 阈值稳定性 | 撞下界 | 在 0.18~0.25 | 校准后更稳健 |

---

*报告结束*
