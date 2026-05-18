# 实现计划：基于 BDDL 谓词的 primitive/skill segment 评测（BEHAVIOR-1K）

## 目标与边界

### 目标

在 BEHAVIOR-1K / OmniGibson 的评测链路中新增"segment（primitive / skill）级别"的评测能力，复用 BDDL 谓词系统来定义与计算：

- segment 级 subgoal 的成功（boolean）
- segment 级进度（progress / delta）
- 可诊断输出（grounding 选择、哪些谓词被满足/新增满足、失败原因等）

该能力主要用于策略诊断与训练迭代，不改变现有 challenge leaderboard 的 task-level success / q_score 口径。

### 非目标（明确不做）

- 不修改 challenge 的任务终止条件与总体评分定义（仍以 task success / q_score 为主）
- 不引入任何自动下载数据/资产流程（遵守项目规则，避免误触发 dataset 下载）
- 不跨模块硬耦合（仿真与评测在 OmniGibson，谓词语言与解析在 bddl3）

---

## 现状调研结果

### 关键代码位置

| 组件 | 路径 | 用途 |
|------|------|------|
| 单段 primitive 评测 | `OmniGibson/omnigibson/learning/eval_primitive.py` | restore + rollout 单段 primitive |
| SubTaskEvaluator | `OmniGibson/omnigibson/learning/eval_subtask_reset.py` | 多段 primitive 评测框架、restore 逻辑 |
| TaskMetric | `OmniGibson/omnigibson/metrics/task_metric.py` | task-level q_score 计算（参考） |
| BDDL 条件评估 | `bddl3/bddl/condition_evaluation.py` | `evaluate_state()` / `get_ground_state_options()` |
| BehaviorTask | `OmniGibson/omnigibson/tasks/behavior_task.py` | `ground_goal_state_options` 生成位置 |
| BDDL 解析 | `bddl3/bddl/parsing.py` | 谓词可读化参考 |
| 配置 | `OmniGibson/omnigibson/learning/configs/eval_primitive_config.yaml` | 评测配置参考 |

### 关键接口

1. **`env.task.ground_goal_state_options`**: List[List[HEAD]] — 每个外层元素是一个 grounding option，每个内层元素是一个谓词 HEAD
2. **`HEAD.evaluate()`**: 返回当前状态下的布尔值
3. **`evaluate_state(compiled_state)`**: 返回 (all_satisfied: bool, results: {satisfied: [], unsatisfied: []})
4. **annotation 结构**: `primitive_annotation` / `skill_annotation` 包含 `frame_duration` 和描述字段

---

## 设计要点

### 1）segment 的 subgoal predicates 定义策略

对同一 demo 的 segment 起止帧做两次 restore（start_frame / end_frame），对每个 grounding option 计算 predicate truth 向量 `S_start / S_end`：

```
subgoal = { i | S_start[i] == False 且 S_end[i] == True }
```

仅将该段在 demo 中"新增满足"的谓词作为该段的 subgoal。

### 2）grounding option 选择策略

默认选择 `end_satisfied_ratio` 更高者；相近时选 `delta_count` 更大者。输出 top-K candidates 供复核。

### 3）segment success vs q_score 分离

- **segment success**: subgoal predicates 全部满足 → 二值成功
- **q_score_delta**: 仅作为辅助诊断输出，不用于二值判定

---

## 具体实施步骤

### Step 0：接口与数据源梳理（不改行为、只确认可用性）

**操作**：
- 确认 `primitive_annotation` 与 `skill_annotation` 的字段结构
- 确认 `SubTaskEvaluator._try_restore_to_frame()` 可稳定执行
- 确认 `env.task.ground_goal_state_options` 在 restore 后可用

**交付物**：
- `.trae/documents/bddl_predicate_segment_eval_interface.md` — 字段说明与示例结构

---

### Step 1：实现 `predicate_utils.py` — 谓词工具函数

**新建文件**：`OmniGibson/omnigibson/learning/utils/predicate_utils.py`

**函数**：
| 函数 | 签名 | 用途 |
|------|------|------|
| `format_bddl_expr` | `(expr_body, backend) -> str` | 将 HEAD/Expression 的 body 格式化为可读字符串 |
| `eval_ground_option` | `(option: List[HEAD]) -> List[bool]` | 计算某 grounding option 的谓词真值向量 |
| `diff_subgoal` | `(start_truth, end_truth) -> List[int]` | 计算新增满足的谓词索引 |
| `rank_groundings` | `(ground_options, s_start, s_end, topk) -> List[Dict]` | 排序 grounding 选项 |
| `compute_q_score_delta` | `(ground_option, s_start, s_end) -> float` | 计算 q_score 增量 |

**交付物**：
- `predicate_utils.py` 及其单元测试占位（注释说明如何测试）

---

### Step 2：实现 `eval_segment.py` — predicate subgoal 评测脚本

**新建文件**：`OmniGibson/omnigibson/learning/eval_segment.py`

**核心逻辑**：
1. 加载 demo annotations（支持 `primitive` / `skill` 级别）
2. 对选定 segment 在 start/end 两次 restore
3. 提取 `ground_goal_state_options` 并计算每个 grounding 的 S_start/S_end
4. 根据启发式选择 grounding，计算 subgoal predicates
5. Rollout 过程中每步评估 subgoal 进度
6. 输出 JSON（包含元信息、grounding 选择、subgoal 列表、progress、q_score 辅助）

**配置字段**：
| 字段 | 类型 | 说明 |
|------|------|------|
| `segment_level` | `primitive \| skill` | 评测粒度 |
| `segment_idx` | int | segment 索引（按 start_frame 排序后） |
| `success_mode` | `predicate_subgoal \| predicate_progress \| state_match` | 评测模式 |
| `grounding_topk` | int | 输出 top-K grounding 候选 |

**交付物**：
- `eval_segment.py` 可运行脚本，产出 JSON

---

### Step 3：新增 Hydra 配置

**新建文件**：`OmniGibson/omnigibson/learning/configs/eval_segment_config.yaml`

**内容**（基于 `eval_primitive_config.yaml` 扩展）：
```yaml
# Segment 评测配置
segment_level: primitive  # primitive | skill
segment_idx: ???          # 必填
success_mode: predicate_subgoal  # predicate_subgoal | predicate_progress | state_match
grounding_topk: 3

# 复用 eval_primitive 的 restore 逻辑
demo_data_path: ???
rawdata_path: null
primitive_state_cache_dir: null

# 输出控制
write_video: true
log_path: ./eval_logs/segment_eval
```

---

### Step 4：最小验证闭环

**操作**：
1. 选择一个 task + demo_id + segment_idx
2. 先跑 `predicate_progress` 模式（只做 subgoal 提取，不跑 policy）
3. 再跑 `predicate_subgoal` 模式（用 websocket policy）
4. 对比 state_match vs predicate_subgoal 成功差异

**交付物**：
- `.trae/documents/bddl_predicate_segment_eval_usage.md` — 运行示例与输出 schema 说明

---

### Step 5：生成 Skill 文档

**新建文件**：`.trae/skills/bddl_predicate_segment_eval.md`

**内容**：
- Skill 用途说明
- 关键参数解释
- 命令行示例（primitive & skill 各一条）
- 注意事项（避免 dataset 下载等）

---

## 风险与应对

| 风险 | 应对 |
|------|------|
| subgoal 为空（导航/对齐段） | predicate_progress 模式允许 subgoal 为空；后续可引入距离阈值等 domain-specific 规则 |
| grounding 选错 | 输出 top-K candidates + 可读化谓词列表，供人工复核 |
| 评测成本高 | 提供 `--dry_run` 模式（只提取 subgoal，不做 rollout） |

---

## 文件变更清单

| 操作 | 文件路径 |
|------|----------|
| 新建 | `OmniGibson/omnigibson/learning/utils/predicate_utils.py` |
| 新建 | `OmniGibson/omnigibson/learning/eval_segment.py` |
| 新建 | `OmniGibson/omnigibson/learning/configs/eval_segment_config.yaml` |
| 新建 | `.trae/documents/bddl_predicate_segment_eval_interface.md` |
| 新建 | `.trae/documents/bddl_predicate_segment_eval_usage.md` |
| 新建 | `.trae/skills/bddl_predicate_segment_eval.md` |

---

## 开发自检

1. **环境确认**：`conda activate behavior`
2. **最小运行**：`python -m omnigibson.learning.eval_segment ...`（先跑 dry_run 模式）
3. **输出验证**：检查 JSON 字段完整性
4. **不触发下载**：全程不运行 `setup.sh --dataset` 或直接调用下载函数
