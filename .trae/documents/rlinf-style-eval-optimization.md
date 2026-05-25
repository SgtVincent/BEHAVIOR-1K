# RLinf-style Eval Optimization

> 本文档放在 `BEHAVIOR-1K/.trae/documents/`，用于被 `BEHAVIOR-1K` 仓库直接纳入版本管理。

这里的 “RLinf-style” 指当前仓库里已经明确标注并实现的一类优化思路：

> 把 setup、render、observation、以及长寿命进程稳定性问题分层处理，只在真正必要的边界上付费。

## 1. 当前仓库里明确存在的 RLinf-style 标记

- `renderer_texturestreaming_memory_budget: 0.1  # RLinf-style RTX texture streaming budget override`，见 `BEHAVIOR-1K/OmniGibson/omnigibson/learning/configs/base_config.yaml:17`
- `skip_intermediate_obs_in_chunk: true  # RLinf-style SkipObs: keep only chunk-boundary observations`，见 `BEHAVIOR-1K/OmniGibson/omnigibson/learning/configs/openpi-comet.yaml:32`
- `_apply_runtime_renderer_settings()` 注释中直接写了 RLinf-style，见 `BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py:68`

## 2. 三层结构

### 2.1 生命周期层：尽量不要每段都重启 Isaac

最贵的通常不是 rollout，而是：

- import OmniGibson
- 拉起 Isaac Sim
- 加载 scene / task instance
- 起 policy server

`openpi-comet/scripts/persistent_skill_eval_worker.py:4` 的设计目标就是把这部分 setup 从“按 segment 支付”改成“按 task 支付”。

### 2.2 观测层：只在边界刷新 obs

`skip_intermediate_obs_in_chunk=true` 时：

- cached action 尚未用完时，不重新请求完整 obs；
- 只在 chunk 边界获取新 obs / 新 action。

实现见 `BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py:234`。

### 2.3 稳定性层：长寿命进程必须配 watchdog

如果只复用不清理，会积累：

- 显存碎片
- stage / cache 残留
- server 串线

因此配套有：

- heartbeat
- task reload timeout
- segment timeout
- max segments before restart
- `os.execv` soft restart

实现见：

- `openpi-comet/scripts/persistent_skill_eval_worker.py:22`
- `openpi-comet/scripts/run_skill_metric_multinode_sweep.py:2298`

## 3. 已落地的优化点

### 3.1 `renderer_texturestreaming_memory_budget`

运行时会尝试写入：

```python
/rtx-transient/resourcemanager/texturestreaming/memoryBudget
```

实现见 `BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py:68`。

作用：

- 控制 RTX texture streaming 预算；
- 降低渲染侧内存波动。

### 3.2 `skip_intermediate_obs_in_chunk`

实现见：

- `BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py:234`
- `BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py:275`

收益：

- 降低 obs 采集成本；
- 降低 websocket 往返与 wrapper preprocess 压力。

### 3.3 `partial_scene_load`

实现见 `BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py:168`。

收益：

- 只加载任务相关房间；
- 降低 scene load / reset 成本。

### 3.4 persistent worker + task affinity

这部分 orchestration 在 `openpi-comet` 仓库侧实现：

- persistent worker：`openpi-comet/scripts/persistent_skill_eval_worker.py:12`
- task-affinity job materialization：`openpi-comet/scripts/run_skill_metric_multinode_sweep.py:2057`

它们共同作用于 behavior runtime：

- 尽量同 task 连续跑；
- 同 task 复用 evaluator / env / server。

### 3.5 `segment_predicate_min_success_steps`

- 抑制过早 predicate 命中带来的假阳性；
- 实现见 `BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_segment.py:671`

### 3.6 `review_video_min_meaningful_frames`

- rollout 太短时补尾帧，保证 review 可读性；
- 实现见 `BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_segment.py:1263`

## 4. 推荐默认组合

### behavior runtime 侧

- `headless=true`
- `render_viewer_camera=false`
- `partial_scene_load=false`
- `renderer_texturestreaming_memory_budget=0.1`
- `skip_intermediate_obs_in_chunk=true`
- `segment_predicate_min_success_steps=150`

### orchestration 侧

- `EVAL_MODE=persistent`
- `RESUME=1`
- `PERSISTENT_WORKER_MAX_SEGMENTS_BEFORE_RESTART=64`
- `PERSISTENT_WORKER_HEARTBEAT_S=60`
- `PERSISTENT_WORKER_TASK_RELOAD_TIMEOUT_S=1800`
- `PERSISTENT_WORKER_SEGMENT_TIMEOUT_S=5400`

补充说明：

- `skip_intermediate_obs_in_chunk=true` 仍是默认推荐；
- `partial_scene_load` 现在不再作为正式 eval 的默认开关，而是保留给 NAS 冷加载明显偏慢时的针对性优化。

## 5. 什么时候不要过度使用这些开关

### 5.1 做逐步视觉 debug 时

不要盲目开 `skip_intermediate_obs_in_chunk=true`。

### 5.2 做 viewer / render 问题排查时

可能需要临时打开 `render_viewer_camera` 或放松 headless 路径。

### 5.3 怀疑跨段状态污染时

保留 `EVAL_MODE=process_per_segment` 作为回滚基线。
