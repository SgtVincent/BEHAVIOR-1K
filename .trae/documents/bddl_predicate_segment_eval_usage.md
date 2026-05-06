# BDDL Predicate Segment Eval 使用指南

## 快速开始

### 环境确认

```bash
conda activate behavior
```

### 基本用法

#### 1. Dry Run 模式（仅提取 subgoal，不运行 policy）

```bash
python OmniGibson/omnigibson/learning/eval_segment.py \
  policy=websocket \
  task.name=turning_on_radio \
  demo_data_path=/path/to/2025-challenge-demos \
  demo_id=00000010 \
  segment_level=primitive \
  segment_idx=0 \
  success_mode=predicate_progress \
  dry_run=true \
  log_path=./eval_logs/segment_eval
```

#### 2. Predicate Subgoal 模式（使用 BDDL 谓词作为成功标准）

```bash
python OmniGibson/omnigibson/learning/eval_segment.py \
  policy=websocket \
  task.name=turning_on_radio \
  env_wrapper._target_=omnigibson.learning.wrappers.wbvima_wrapper.WBVIMAWrapper \
  demo_data_path=/path/to/2025-challenge-demos \
  rawdata_path=/path/to/2025-challenge-rawdata \
  demo_id=00000010 \
  segment_level=primitive \
  segment_idx=0 \
  success_mode=predicate_subgoal \
  log_path=./eval_logs/segment_eval
```

#### 3. Skill 级别评测

```bash
python OmniGibson/omnigibson/learning/eval_segment.py \
  policy=websocket \
  task.name=turning_on_radio \
  demo_data_path=/path/to/2025-challenge-demos \
  demo_id=00000010 \
  segment_level=skill \
  segment_idx=0 \
  success_mode=predicate_subgoal \
  log_path=./eval_logs/segment_eval
```

---

## 输出 JSON Schema

### 顶层字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `demo_id` | string | Demo ID |
| `task_name` | string | 任务名称 |
| `segment_level` | string | "primitive" 或 "skill" |
| `segment_idx` | int | Segment 索引 |
| `segment_desc` | string | Segment 描述 |
| `frame_duration` | [int, int] | 起始帧和结束帧 |
| `success` | bool | 是否成功（二值） |
| `result_type` | string | 结果类型 |

### restore 字段

```json
"restore": {
  "start": {"restored": true, "method": "rawdata"},
  "end": {"restored": true, "method": "rawdata"}
}
```

- `method`: 恢复方法（rawdata / cache / robot / none）

### grounding 字段

```json
"grounding": {
  "chosen_option_idx": 0,
  "topk_candidates": [
    {
      "option_idx": 0,
      "end_satisfied_ratio": 0.85,
      "delta_count": 3,
      "subgoal_size": 3,
      "subgoal_indices": [2, 5, 8]
    }
  ],
  "n_options": 4
}
```

### subgoal 字段

```json
"subgoal": {
  "indices": [2, 5, 8],
  "predicates": [
    {
      "index": 2,
      "predicate_name": "ontop",
      "args": ["obj1", "table"],
      "readable": "(on top of obj1 table)"
    }
  ],
  "size": 3
}
```

### q_score 字段

```json
"q_score": {
  "start": 0.0,
  "end": 0.45,
  "delta": 0.45
}
```

### rollout 字段

```json
"rollout": {
  "max_steps": 300,
  "final_step": 300,
  "best_progress": 0.67,
  "final_progress": 0.67
}
```

---

## 参数说明

### 必需参数

| 参数 | 说明 |
|------|------|
| `demo_data_path` | 2025-challenge-demos 目录路径 |
| `segment_idx` | Segment 索引（按 start_frame 排序后） |

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `segment_level` | "primitive" | 评测粒度：primitive 或 skill |
| `success_mode` | "predicate_subgoal" | 评测模式 |
| `grounding_topk` | 3 | 输出的 top grounding 候选数 |
| `dry_run` | false | 是否仅提取 subgoal |
| `demo_id` | null | 指定 demo ID，为 null 时自动选择第一个 |
| `segment_max_steps` | null | 最大步数覆盖 |
| `rawdata_path` | null | raw HDF5 目录（用于完整状态恢复） |
| `write_video` | true | 是否写视频 |

### success_mode 选项

| 模式 | 说明 |
|------|------|
| `predicate_subgoal` | 所有 subgoal predicates 满足则成功 |
| `predicate_progress` | 仅记录 progress，无二值成功 |
| `state_match` | 回退到 state-match（与 eval_primitive 相同） |

---

## 注意事项

1. **不触发 dataset 下载**：运行前确保数据已准备好，不要运行 `setup.sh --dataset`
2. **GPU 设置**：如遇渲染问题，设置 `export OMNIGIBSON_GPU_ID=<可用GPU编号>`
3. **首次 import 慢**：首次 `import omnigibson` 可能需要数分钟初始化，属于正常行为
