# BEHAVIOR-1K Per-Skill Evaluation 用户指南

> Canonical 用户文档。面向需要运行、查看、解释 skill eval 的使用者。  
> 代码与 registry 永远以仓库实现为准；历史 BDDL predicate 文档已归档为 legacy 说明。

## 1. 目标

BEHAVIOR-1K 官方评测主要是 task-level BDDL goal / `q_score`。长视距任务失败时，只看最终 task success 很难定位具体失败技能。本 per-skill evaluation 体系用于：

- 按 `skill_annotation` 截取短视距 skill segment。
- 使用 skill-level metric registry 定义每个 skill 的成功条件。
- 在 OmniGibson runtime 中恢复起始状态、执行 policy、逐步评估 predicate / geometry / proxy metric。
- 输出可聚合的 metrics、视频、review images 与 predicate trace。

## 2. 当前推荐模式

正式 per-skill eval 的推荐模式是：

```text
segment_level=skill
success_mode=segment_predicates
```

当前 canonical 模型对比主线是 `pi05-b1kpt50-cs32` 的 JAX 版与 PyTorch 版 skill eval；Hamlet 不纳入当前对比，`pi05-b1kpt12` 也不是当前 A/B 对比主线。

不要把旧文档中的 `predicate_subgoal` 当作当前 per-skill eval 主路径。`predicate_subgoal` 是早期从 task-level BDDL delta 自动推 segment subgoal 的 legacy 模式，对许多中间 skill 不适用。

## 3. 核心代码入口

### BEHAVIOR-1K 侧

- 单 segment eval 入口：`/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_segment.py`
- skill metric registry：`/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/utils/segment_skill_metric_registry.py`
- registry → runtime predicate / geometry evaluator：`/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/utils/segment_predicate_eval.py`
- termination diagnostics：`/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/utils/eval_diagnostics.py`

### openpi-comet 侧

- 推荐 sweep launcher：`/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/scripts/run_skill_metric_multinode_sweep.py`
- policy server：`/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/scripts/serve_b1k.py`
- task prompt / fine-grained skill prompt wrapper：`/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/src/openpi/shared/eval_b1k_wrapper.py`

## 4. 单 segment eval 示例

```bash
cd /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K
conda activate behavior

python OmniGibson/omnigibson/learning/eval_segment.py \
  policy=websocket \
  task.name=turning_on_radio \
  demo_data_path=/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos \
  rawdata_path=/mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-rawdata \
  demo_id=00000010 \
  segment_level=skill \
  segment_idx=0 \
  success_mode=segment_predicates \
  log_path=/tmp/segment_eval_smoke \
  headless=true \
  write_video=true \
  segment_predicate_dump_trace=true \
  env_wrapper._target_=omnigibson.learning.wrappers.RGBWrapper \
  partial_scene_load=false \
  skip_intermediate_obs_in_chunk=true
```

常用参数：

| 参数 | 说明 |
|---|---|
| `task.name` | BEHAVIOR task 名称 |
| `demo_data_path` | challenge demos 根目录 |
| `rawdata_path` | raw HDF5 根目录，用于精确状态恢复 |
| `demo_id` | episode id，例如 `00040010` |
| `segment_level` | 当前 per-skill eval 使用 `skill` |
| `segment_idx` | skill annotation index |
| `success_mode` | 当前 per-skill eval 使用 `segment_predicates` |
| `segment_max_steps` | 可覆盖默认 horizon |
| `write_video` | 是否写 rollout 视频 |
| `segment_predicate_dump_trace` | 是否写逐步 predicate trace |

## 5. 批量 sweep 示例

推荐使用 openpi-comet 的 multinode sweep launcher；它会自动：

- 从 registry 与 demo annotations 中采样 skill jobs。
- 按 task 分组启动 policy server。
- 为每个 server 写入 identity，避免连接 stale server。
- 为 eval worker 设置 `CUDA_VISIBLE_DEVICES=<gpu>` 并 `unset OMNIGIBSON_GPU_ID`，避免 OmniGibson GPU ordinal 错误。
- 调用 `eval_segment.py` 的 `segment_predicates` 模式。

示例：

```bash
cd /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet
conda activate openpi-comet-nas

python -u scripts/run_skill_metric_multinode_sweep.py \
  --mode launch \
  --out-dir /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/segment_eval_runs/<run_tag> \
  --node-rank 0 \
  --num-nodes 1 \
  --gpus-per-node 8 \
  --max-steps 120 \
  --config-name pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k \
  --policy-backend torch \
  --ckpt-dir /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/checkpoints/openpi_comet/pi05-b1kpt50-cs32 \
  --behavior-dir /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K \
  --demo-data-path /mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-demos \
  --rawdata-path /mnt/bn/navigation-hl/mlx/users/chenjunting/data/2025-challenge-rawdata \
  --max-samples-per-skill 1 \
  --write-video \
  --segment-predicate-dump-trace
```

完成后 merge：

```bash
python -u scripts/run_skill_metric_multinode_sweep.py \
  --mode merge \
  --out-dir /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/segment_eval_runs/<run_tag>
```

## 6. 输出结构

每个 segment 的主要产物在：

```text
<run_dir>/raw/<task_name>/demo_<demo_id>/skill_<idx>/
  ├── segment_eval.log
  ├── metrics/segment_eval_<task>_<demo>_skill<idx>.json
  ├── videos/*.mp4                 # write_video=true 时
  └── review/*.png                  # start/end/final review frames
```

merge 后的主要产物：

```text
<run_dir>/multinode_skill_results.csv
<run_dir>/multinode_skill_results.json
<run_dir>/multinode_skill_summary.csv
<run_dir>/multinode_skill_summary.json
<run_dir>/multinode_skill_summary.md
<run_dir>/multinode_skill_task_summary.csv
```

## 7. 如何解释 result_type

| result_type | 解释 | 是否直接代表 policy 成功 |
|---|---|---|
| `predicate_satisfied` | 当前 skill registry metric 在 rollout 窗口内满足 | 是 |
| `timeout` | 跑满 skill horizon 仍未满足 metric | 否，需看 trace / video 判断是否 metric 过严 |
| `env_terminated` | 环境提前终止但 skill predicate 未满足 | 否，需重点排查 env hidden state / metric false negative |
| `pre_satisfied_start` | 起始状态已满足且 registry 要求 start 不满足，因此未 rollout | 不应简单算 policy failure；建议单独统计 |
| `restore_failed` | 起始或末帧状态恢复失败 | runtime / data 问题 |
| `no_predicates_generated` | registry/auto-mining 未生成 predicate spec | metric/annotation 问题 |

推荐汇总口径：

```text
runtime_pass = 能成功写 metrics 的样本
attemptable_segments = runtime_pass - pre_satisfied_start - restore_failed - no_predicates_generated
policy_success_rate = predicate_satisfied / attemptable_segments
```

## 8. 34 个 skill 的 metric registry

完整 registry 表见：`/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/.trae/documents/SKILL_METRIC_REGISTRY_TABLE.md`。

但最终以代码为准：`/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/utils/segment_skill_metric_registry.py`。

主要 metric family：

- `grasp_relation` / `grasp_hold` / `grasp_release`
- `relation_place_inside` / `relation_place_ontop` / `relation_attach`
- `articulation_open` / `articulation_close`
- `toggle_on` / `toggle_off`
- `geometry_base_target` / `geometry_base_facing` / `geometry_object_target`
- `contact_effect_proxy`
- `orientation_proxy`

## 9. 当前已知 caveats

1. `pre_satisfied_start` 是单独语义，不要混入普通 policy failure。
2. `attach` / `release` 当前最容易出现视觉成功但 metric 未过的 false negative，需要结合 video / review images 审计。
3. `contact_effect_proxy` 类 skill 只看接触可能过严或过窄，例如 `sweep surface` / `spray` / `wipe hard`。
4. `dynamic_max_steps = segment_duration * 2`，不加 cap 时长 segment 会很慢。
5. 当前模型对比只按 `pi05-b1kpt50-cs32` 的 JAX vs PyTorch 同口径结果解释；不要把 Hamlet 或 `pi05-b1kpt12` 作为当前 A/B 主线结论。

## 10. 当前状态快照（2026-05-10）

### 已完成能力

- 34 个 skill 已有 registry 定义。
- 早期 2026-05-03 formal run 曾达到 `runtime_pass=34/34`。
- stale server identity、localhost proxy、OmniGibson GPU ordinal、early-stop 诊断已完成主要修复。
- review artifacts / sanity manifest / multimodal audit 工具链已完成。

### 当前 `pi05-b1kpt50-cs32` PyTorch run（历史 ModelA 命名）

`/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/segment_eval_runs/full_skill_eval_modelA_pi05pt50_torch_20260509`

- planned jobs: 136
- completed jobs: 136/136
- missing jobs: 0
- runtime_ok: 136/136
- `predicate_satisfied`: 29
- `pre_satisfied_start`: 31
- `attemptable_segments = runtime_ok - pre_satisfied_start`: 105
- `policy_success_attemptable_rate`: 0.2762
- result types: `timeout=62`, `pre_satisfied_start=31`, `predicate_satisfied=29`, `env_terminated=14`
- completed unique skills: 34/34

### 当前 `pi05-b1kpt50-cs32` JAX run（历史 ModelB/round3 命名）

`/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/segment_eval_runs/full_skill_eval_modelB_pi05pt50_jax_20260509_round3`

- planned jobs: 136
- completed jobs: 136/136
- missing jobs: 0
- runtime_ok: 136/136
- `predicate_satisfied`: 33
- `pre_satisfied_start`: 31
- `attemptable_segments = runtime_ok - pre_satisfied_start`: 105
- `policy_success_attemptable_rate`: 0.3143
- result types: `timeout=55`, `pre_satisfied_start=31`, `predicate_satisfied=33`, `env_terminated=17`
- completed unique skills: 34/34

### 当前 A/B 结论（schema v2 attemptable 口径）

- 对比 artifact: `/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/segment_eval_runs/full_skill_eval_modelB_pi05pt50_jax_20260509_round3/ab_compare_torch_vs_jax.md`
- CSV artifact: `/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/openpi-comet/segment_eval_runs/full_skill_eval_modelB_pi05pt50_jax_20260509_round3/ab_compare_torch_vs_jax.csv`
- PyTorch: `29/105 = 0.2762`
- JAX: `33/105 = 0.3143`
- delta（JAX - PyTorch）: `+4` attemptable successes，`+3.81 pp`
