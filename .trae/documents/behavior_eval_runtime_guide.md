# BEHAVIOR Eval Runtime Guide

> 本文档放在 `BEHAVIOR-1K/.trae/documents/`，用于被 `BEHAVIOR-1K` 仓库直接纳入版本管理。
>
> 配套文档：
>
> - `openpi-comet/.trae/documents/behavior_eval_orchestration_guide.md`
> - `openpi-comet/.trae/documents/eval-quick-reference.md`
> - `BEHAVIOR-1K/.trae/documents/rlinf-style-eval-optimization.md`

## 1. 范围

本文档只讲 **behavior / simulator runtime 侧**：

- `OmniGibson/omnigibson/learning/eval.py`
- `OmniGibson/omnigibson/learning/eval_segment.py`
- `OmniGibson/omnigibson/learning/configs/base_config.yaml`
- `OmniGibson/omnigibson/learning/configs/eval_segment_config.yaml`
- `OmniGibson/omnigibson/learning/configs/openpi-comet.yaml`
- `OmniGibson/omnigibson/learning/configs/policy/websocket.yaml`

## 2. 关键入口

### full-task eval

```bash
python OmniGibson/omnigibson/learning/eval.py \
  policy=websocket \
  task.name=<TASK_NAME> \
  log_path=<LOG_PATH>
```

### segment eval

```bash
python OmniGibson/omnigibson/learning/eval_segment.py \
  policy=websocket \
  task.name=<TASK_NAME> \
  demo_data_path=<DEMO_DATA_PATH> \
  rawdata_path=<RAWDATA_PATH> \
  demo_id=<DEMO_ID> \
  segment_level=skill \
  segment_idx=<SEGMENT_IDX> \
  log_path=<LOG_PATH>
```

## 3. runtime 侧关键 flags

### `headless`

- 默认 `true`，见 `BEHAVIOR-1K/OmniGibson/omnigibson/learning/configs/base_config.yaml:10`
- 正式 eval 基本都建议 headless

### `render_viewer_camera`

- 默认 `false`，见 `BEHAVIOR-1K/OmniGibson/omnigibson/learning/configs/base_config.yaml:11`
- 开启会增加 viewer 渲染开销

### `gui_viewport_only`

- 默认 `false`，见 `BEHAVIOR-1K/OmniGibson/omnigibson/learning/configs/base_config.yaml:12`
- 在更激进的 headless / persistent 路径中可能会被设成更收敛的值

### `partial_scene_load`

- 只加载任务相关房间，见 `BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py:168`
- `openpi-comet.yaml` 中默认设为 `false`，见 `BEHAVIOR-1K/OmniGibson/omnigibson/learning/configs/openpi-comet.yaml:14`
- 当前默认建议是不把 `partial_scene_load` 当成通用默认加速开关；只有在确认瓶颈主要是 NAS 冷加载时再显式开启

### `max_steps`

- 若为 `null`，则用人类 demo 平均长度的 2 倍作为 timeout
- 实现见 `BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py:184`

### `skip_intermediate_obs_in_chunk`

- 如果 policy 还有 cached action，则 chunk 内部步骤跳过 observation 刷新
- 实现见 `BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py:234`
- `openpi-comet.yaml` 默认设为 `true`，见 `BEHAVIOR-1K/OmniGibson/omnigibson/learning/configs/openpi-comet.yaml:32`

### `segment_max_steps`

- segment eval 的每段上限
- 定义见 `BEHAVIOR-1K/OmniGibson/omnigibson/learning/configs/eval_segment_config.yaml:38`

### `success_mode`

- 典型值：`predicate_subgoal`、`predicate_progress`、`state_match`、`segment_predicates`
- 定义见 `BEHAVIOR-1K/OmniGibson/omnigibson/learning/configs/eval_segment_config.yaml:28`

### `segment_predicate_min_success_steps`

- 过早命中的 predicate 不直接判成稳定成功
- 实现见 `BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_segment.py:671`

### `review_video_min_meaningful_frames`

- 视频太短时做尾帧 padding，保证 review 可读性
- 实现见 `BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_segment.py:1263`

## 4. websocket policy 与 server identity

当前 websocket policy config 支持：

- `expected_task_name`
- `expected_task_prompt_sha256`
- `expected_server_run_id`
- `expected_server_token`

定义见 `BEHAVIOR-1K/OmniGibson/omnigibson/learning/configs/policy/websocket.yaml:5`。

用途：

- 不只检查 healthz；
- 还要确保 evaluator 真的连到了预期 server；
- 防止多 task / 多 ckpt / 旧端口残留导致串线。

## 5. 特别注意

### 5.1 上游公开文档里的端口说明偏旧

`BEHAVIOR-1K/docs/challenge/evaluation.md:53` 仍写 evaluator 会 listen 在 `0.0.0.0:80`，但当前 websocket config 默认端口是 `8000`，见 `BEHAVIOR-1K/OmniGibson/omnigibson/learning/configs/policy/websocket.yaml:8`。以当前代码配置为准。

### 5.2 不要只靠 healthz 判断 server 正确

排查串线时一定要看 `expected_server_*` 四元组。

### 5.3 RLinf-style 运行时优化请看专门文档

例如：

- `renderer_texturestreaming_memory_budget`
- `skip_intermediate_obs_in_chunk`
- persistent worker 复用 / soft restart

请看 `BEHAVIOR-1K/.trae/documents/rlinf-style-eval-optimization.md`。
