# Code Review 报告：v0.5 分支（E4/E5/E6 特征优化）

## 基本信息
- **审查模式**：分支模式（v0.5 vs v0.4 增量）
- **目标**：`v0.5`（commit `bff4c17`）
- **基准**：`v0.4`（commit `fb014a9`）
- **变更类型**：feature [用户指定 - 特征优化]
- **提交人**：atwhitewhole <at.white.whole@gmail.com>
- **审查文件**：1 个（M `src/features/unified_fe.py`）
- **变更量**：+33 / -8

---

## 需求对齐（对照技术方案文档 v4 瓶颈 4-6）

### 功能性需求覆盖

| # | 功能点 | 状态 | 对应实现 | 备注 |
|---|--------|------|---------|------|
| E4 | `view_detail_cnt` 做 log1p 压缩 | ✅ 已实现 | `unified_fe.py:65` | 先算 ratio 再压缩，顺序正确 |
| E4 | 新增 `detail_view_ratio` | ✅ 已实现 | `unified_fe.py:64` | 公式正确 |
| E5 | `active_days` 跨天回访 | ✅ 已实现 | `unified_fe.py:138` | `.dt.date.nunique()` 正确 |
| E5 | `is_multi_day_visitor` | ✅ 已实现 | `unified_fe.py:139` | 方案中无此特征，为额外加分项 |
| E5 | `visit_span_days` | ✅ 已实现 | `unified_fe.py:141` | 方案中无此特征，有业务意义的补充 |
| E5 | `unique_items_viewed` | ✅ 已实现 | `unified_fe.py:144` | 用 `.nunique()` 正确 |
| E5 | `repeat_view_ratio` | ⚠️ 实现有隐患 | `unified_fe.py:146` | 用 `np.expm1` 反向还原 log1p，脆弱耦合 |
| E5 | `session_count` | ✅ 已实现 | `unified_fe.py:149` | 30分钟间隔，逻辑正确 |
| E5 | `actions_per_session` | ✅ 已实现 | `unified_fe.py:150` | 方案中无此特征，合理补充 |
| E5 | `intent_acceleration` | ✅ 已实现 | `unified_fe.py:152` | 与方案一致 |
| E6 | 删除 `conversion_efficiency` | ✅ 已实现 | `unified_fe.py:156` | 冗余删除正确 |
| E6 | 删除 `recent_3/5_avg_intent`，直接算 `intent_trend` | ✅ 已实现 | `unified_fe.py:85-87` | 简化正确，结果等价 |
| E6 | 新增 `intent_level_gap` | ✅ 已实现 | `unified_fe.py:89` | `max - last`，有效提取意图落差 |
| 风险修复 | `original_values = fe[col].copy()` 死代码清理 | ✅ 已实现 | `unified_fe.py:202` | 对应方案 🟢 风险项 |

---

## 发现（按严重级别）

### 🔴 高风险（必须修复）

#### 1. [unified_fe.py:146] `repeat_view_ratio` 通过 `np.expm1` 逆向还原 log1p 值，形成脆弱隐式耦合

**问题描述**：
```python
fe['repeat_view_ratio'] = np.expm1(fe['view_detail_cnt']) / (fe['unique_items_viewed'] + 1)
```

`view_detail_cnt` 在第 65 行被 `np.log1p()` 覆写后，第 146 行用 `np.expm1()` 逆向还原原始值。这构成了一个**隐式依赖**：
- 如果有人将 E4 的压缩方法改为 `np.log10()` 或分位数截断，此处会**静默产生错误结果**
- 如果有人调整代码顺序将 E5 特征移到 E4 之前，也会静默出错
- 代码中虽有注释说明原因，但这属于 defensive programming 的反面

**影响**：
- 代码脆弱，依赖隐式假设（E4 必须用 log1p）
- 违反单一职责原则（E5 特征计算依赖 E4 的具体实现细节）
- 可维护性差，未来修改 E4 压缩方式时容易遗漏此处

**建议修复**：
在 log1p 之前保存原始值，用于后续计算：
```python
# E4 部分
raw_view_detail_cnt = fe['view_detail_cnt'].copy()
fe['detail_view_ratio'] = raw_view_detail_cnt / (fe['total_actions'] + 1)
fe['view_detail_cnt'] = np.log1p(raw_view_detail_cnt)

# E5 部分（后续）
fe['repeat_view_ratio'] = raw_view_detail_cnt / (fe['unique_items_viewed'] + 1)
```

---

### 🟡 中风险（建议修复）

#### 2. [unified_fe.py:152-153] `intent_acceleration` 在 `len(x)==3` 时最大值被截断

**问题描述**：
```python
fe['intent_acceleration'] = grouped['intent_level'].apply(
    lambda x: (x.diff().tail(3) > 0).sum() if len(x) >= 3 else 0
)
```

`x.diff()` 首元素为 NaN，当 `len(x)==3` 时 `.tail(3)` 包含这个 NaN（`NaN > 0` → `False`），导致最大返回值为 2 而非 3。对只有 3 次行为的用户，加速度信号被低估。

**影响**：
- 对短序列用户（3 次行为）的意图加速度信号被低估
- 虽然影响面小（只有 3 次行为的用户占比可能不高），但逻辑不严谨

**建议修复**：
```python
lambda x: (x.diff().dropna().tail(3) > 0).sum() if len(x) >= 3 else 0
```

#### 3. [unified_fe.py:150] `actions_per_session` 分母偏差

**问题描述**：
```python
fe['actions_per_session'] = fe['total_actions'] / (fe['session_count'] + 0.1)
```

`session_count` 最小值为 1（`(x > SESSION_GAP).sum() + 1`），所以分母最小为 1.1，导致单会话用户的每会话行为数被系统性低估约 9%。

**影响**：
- 特征值系统性偏差，虽然幅度不大（~9%），但逻辑不必要
- 可能影响模型对单会话用户的判断

**建议**：
改为 `+ 1` 或直接不加 epsilon（因为 `session_count >= 1` 已保证不为零）：
```python
fe['actions_per_session'] = fe['total_actions'] / fe['session_count']
```

---

### 🟢 低风险 / 改善建议

#### 4. [unified_fe.py:144] `grouped['商品名称'].nunique()` 对 NaN 的处理

**问题描述**：
`nunique()` 默认排除 NaN。如果部分行为没有商品名称（浏览首页等），这些行为不计入多样性。业务上合理，但建议加注释说明意图。

**建议**：
添加注释说明：
```python
# 浏览商品多样性：看了很多不同手机 = 在比价（注意：nunique() 自动排除 NaN，首页浏览等无商品名的行为不计入）
fe['unique_items_viewed'] = grouped['商品名称'].nunique()
```

#### 5. [unified_fe.py:156] 删除 `conversion_efficiency` 只留了注释，无代码

**问题描述**：
删除方式是将赋值行替换为注释。代码可读，但长期来看空注释行应该清理。

**建议**：
如果确定不再需要，可以删除注释行；如果保留是为了说明删除原因，建议注释更详细：
```python
# 【E6】删除 conversion_efficiency（与 click_to_process_rate 公式完全一致，属冗余特征）
# 原代码：fe['conversion_efficiency'] = fe['cnt_bussProcessing'] / (fe['cnt_eventClick'] + 1)
```

#### 6. 方案外新增的 3 个特征未在方案文档中标注

**问题描述**：
`is_multi_day_visitor`、`visit_span_days`、`actions_per_session` 是方案之外额外新增的特征。特征本身合理，但建议更新方案文档或在 commit message 中注明。

**建议**：
- 更新 `docs/技术方案文档-v4-0303.md`，在 E5 节补充这 3 个特征
- 或在 commit message 中明确说明："E5 新增 8 个特征（方案 4 个 + 额外 3 个：is_multi_day_visitor, visit_span_days, actions_per_session）"

---

## 正面评价

- ✅ **E4 执行顺序严格正确**：先用原始 `view_detail_cnt` 算比例，再做 log1p 压缩
- ✅ **E5 特征设计有清晰的业务语义**：每个特征都有注释说明业务含义（跨天回访=在认真考虑、多样性=在比价等）
- ✅ **E6 `intent_trend` 简化优雅**：从 3 行（2 个中间变量 + 1 个差值）缩减为 1 行直接计算，结果等价
- ✅ **E6 `intent_level_gap` 是一个有价值的新信号**：max vs last 意图差异可以捕捉"曾经很感兴趣但现在冷却"的用户
- ✅ **死代码清理干净**：`original_values = fe[col].copy()` 已移除
- ✅ **新增特征超出方案**（8 个 vs 方案的 4 个），额外特征（`visit_span_days`、`is_multi_day_visitor` 等）有业务价值

---

## 测试 / 验证缺口

- ❌ **无新增特征的单元测试**：NaN 处理、边界 case、单行为用户等
- ❌ **缺少新特征与 `predict_common.py` 的端到端验证**（预测管道通过 `reindex` 自动对齐，但新特征是否出现在 `feature_names.pkl` 取决于训练执行结果）
- ❌ **建议在 v0.5 训练后对比特征重要性**，验证 `view_detail_cnt` 重要性是否从 59.40% 降至目标 ≤25%

---

## 总结

v0.5 分支实现了方案要求的 E4/E5/E6 三大特征优化，代码质量整体良好，特征设计有业务价值，但存在以下关键问题：

1. **🔴 必须修复**：`repeat_view_ratio` 通过 `expm1` 反向还原 log1p，形成脆弱隐式耦合（违反单一职责原则）
2. **🟡 建议修复**：`intent_acceleration` 的 NaN 边界处理、`actions_per_session` 分母偏差
3. **🟢 可优化**：注释完善、方案文档更新

建议优先修复 🔴 高风险问题（保存原始值再计算），再考虑 🟡 中风险改进。
