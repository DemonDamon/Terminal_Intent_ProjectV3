# Code Review 报告：v0.4 分支（E1/E2/E3 验证体系修正）

## 基本信息
- **审查模式**：分支模式
- **目标**：`v0.4`（commit `fb014a9`）
- **基准**：`main`（增量 commit）
- **变更类型**：bugfix [用户指定 - 修复验证体系瓶颈]
- **提交人**：atwhitewhole <at.white.whole@gmail.com>
- **审查文件**：3 个（M `src/features/unified_fe.py` / M `src/models/model_lgb.py` / M `train_pipline.py`）
- **变更量**：+85 / -54

---

## 需求对齐（对照技术方案文档 v4 瓶颈 1-3）

### 功能性需求覆盖

| # | 功能点 | 状态 | 对应实现 | 备注 |
|---|--------|------|---------|------|
| E1 | 随机切分 → 时间切分（8月训练/9月验证） | ✅ 已实现 | `train_pipline.py:78-88` | 切分逻辑正确，安全检查完备 |
| E1 | 特征表保留 `last_action_date` | ✅ 已实现 | `unified_fe.py:134` | 用了 `.max()` 而非方案的 `.max().dt.date`，功能等价 |
| E1 | `last_action_date` 不参与模型训练 | ✅ 已实现 | `train_pipline.py:100` | 通过 `drop_cols` 排除 |
| E2 | `scale_pos_weight` 扩大到 1~100 (log) | ✅ 已实现 | `model_lgb.py:122` | 加了 `log=True`，符合方案 |
| E2 | `n_trials` 从 8 提高到 30 | ⚠️ 部分实现 | `train_pipline.py:120` | **实际设为 15，方案要求 30** |
| E3 | 阈值搜索从训练集移到验证集 | ✅ 已实现 | `train_pipline.py:129-131` | 正确，模型训练后不再自动搜阈值 |
| E3 | 模型保存移至阈值优化之后 | ✅ 已实现 | `train_pipline.py:153-159` | 确保阈值正确保存 |
| 风险修复 | `except:` → `except ValueError:` | ✅ 已实现 | `train_pipline.py:137` | 对应方案 🟢 风险项 |
| 风险修复 | `bussinessProcessing` 拼写检查 | ⚠️ 部分实现 | `unified_fe.py:42-44` | 已加检查但用 `print` 而非 `logger` |
| 风险修复 | `X_test.reindex` 静默丢弃规则特征 | ❌ 未修复 | `train_pipline.py:123` | 方案 🔴 风险项，仍在 `X_val.reindex` 中静默丢弃 |

---

## 发现（按严重级别）

### 🔴 高风险（必须修复）

#### 1. [train_pipline.py:123] 规则层特征在验证集上被 `reindex` 静默丢弃

**问题描述**：
`X_val` 未经显式移除规则层特征，而是被 `X_val.reindex(columns=X_train.columns, fill_value=0)` 静默丢弃。方案文档将此列为 🔴 风险（"E1 一起改"），但实际未修复。如果后续有人给 `X_val` 添加特征但忘了对齐 `X_train`，会导致静默数据错位。

**影响**：
- 规则层特征在验证集评估时被静默填充为 0，可能影响评估准确性
- 代码可读性差，依赖隐式行为而非显式逻辑
- 与方案文档的修复要求不符

**建议修复**：
在 `reindex` 前显式移除规则层特征：
```python
rule_cols_in_val = [c for c in RULE_LAYER_FEATURES if c in X_val.columns]
if rule_cols_in_val:
    X_val = X_val.drop(columns=rule_cols_in_val)
X_val_aligned = X_val.reindex(columns=X_train.columns, fill_value=0)
```

---

### 🟡 中风险（建议修复）

#### 2. [train_pipline.py:120] `n_trials=15`，方案要求 30

**问题描述**：
方案明确写了 `n_trials=30`，实际只用了 15。10+ 维超参空间中 15 次尝试搜索效率有限（虽然比原来的 8 好多了），特别是 `scale_pos_weight` 范围扩大到 1~100 后搜索空间更大了。

**影响**：
- 超参搜索不充分，可能错过更优参数组合
- 与方案文档不一致

**建议**：
改为 `n_trials=30` 或至少 25，与方案一致。如果时间受限，请在 commit message 中注明偏差原因。

#### 3. [train_pipline.py:78-79] 时间切分硬编码日期

**问题描述**：
`cutoff_train = pd.Timestamp('2024-09-01')` 和 `cutoff_val = pd.Timestamp('2024-10-01')` 写死在代码中。当数据周期变化（如用 2025 年数据重训）时需要改代码。

**影响**：
- 代码可维护性差，需要手动修改才能适配新数据周期
- 容易出错（忘记修改导致数据切分错误）

**建议**：
提取到配置或从数据自动推断：
```python
CUTOFF_TRAIN = pd.Timestamp('2024-09-01')
CUTOFF_VAL = pd.Timestamp('2024-10-01')
```

---

### 🟢 低风险 / 改善建议

#### 4. [unified_fe.py:42-44] 拼写检查用 `print` 而非 `logger.warning`

**问题描述**：
`train_pipline.py` 使用 `logger`，但 `unified_fe.py` 中新增的拼写检查用 `print`。日志体系不统一。

**建议**：
统一使用 `logger.warning()` 或 `logger.info()`，保持代码风格一致。

#### 5. [unified_fe.py:134] `last_action_date` 存的是 Timestamp 而非 date

**问题描述**：
方案写的是 `.max().dt.date`，实际用 `.max()`。功能等价（比较结果一致），但列名与类型不匹配（名为"date"却存"datetime"），可能导致后续使用者困惑。

**建议**：
要么改为 `.max().dt.date` 与方案一致，要么将列名改为 `last_action_datetime` 更准确。

---

## 正面评价

- ✅ **E1 时间切分实现干净**，安全检查做得好（空集、单标签检测）
- ✅ **日志信息质量高**：切分后打印训练集/验证集样本量和正样本率，方便 debug
- ✅ **E2 `scale_pos_weight` 改用 `log=True` 采样**，比线性搜索更高效覆盖 1~100 范围
- ✅ **E3 执行顺序正确**：训练 → 验证集搜阈值 → 保存模型，避免了阈值未包含在模型中的问题
- ✅ **裸 `except:` 改为 `except ValueError:`** 是正确的健壮性改进
- ✅ **移除了 `from sklearn.model_selection import train_test_split`** 的导入，保持代码整洁

---

## 测试 / 验证缺口

- ❌ **缺少自动化测试**：时间切分逻辑无单元测试（边界日期、空月份等）
- ❌ **未验证训练数据 `train_data_89.csv` 是否确实覆盖 8-9 月**（文件名暗示是，但无断言）
- ❌ **回测管道 `predict_pipline_backtest.py` 未做对应修改**（10 月数据是否需要排除 `last_action_date`？该列在预测时通过 `reindex` 自动丢弃，但不显式）

---

## 总结

v0.4 分支整体实现了方案要求的 E1/E2/E3 三大验证体系修正，代码质量良好，但存在以下关键问题：

1. **🔴 必须修复**：规则层特征在验证集上被静默丢弃（方案标记的 🔴 风险未修复）
2. **🟡 建议修复**：`n_trials` 与方案不一致（15 vs 30），时间切分日期硬编码
3. **🟢 可优化**：日志体系统一、列名类型匹配

建议优先修复 🔴 高风险问题，再考虑 🟡 中风险改进。
