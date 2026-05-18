# 执行计划：pi05-b1kpt50-cs32 模型在 make_pizza 任务上的失败原因分析

## 目标

测试 `pi05-b1kpt50-cs32` checkpoint 在 `make_pizza` 任务上的表现，并进行失败原因分析：
- 区分导航失败 vs 操作失败
- 统计每个步骤（segment/primitive）的失败次数
- 汇总分析结果

## 已有资源

- **Checkpoint**: `/mnt/bn/mllm-data-yg/chenjunting/repo/openpi-comet/checkpoints/openpi_comet/pi05-b1kpt50-cs32`
- **评测脚本**: `openpi-comet/scripts/run_b1k_eval_parallel_single_task.sh`
- **Segment 评测**: `BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_segment.py`
- **使用文档**: `BEHAVIOR-1K/.trae/documents/bddl_predicate_segment_eval_usage.md`

## 任务索引

根据 `eval_utils.py`:
- `make_pizza` → task_index = 49

---

## 执行步骤

### Step 1: 环境准备

```bash
# 确认 behavior conda 环境可用
conda activate behavior

# 确认 checkpoint 存在
ls -la /mnt/bn/mllm-data-yg/chenjunting/repo/openpi-comet/checkpoints/openpi_comet/pi05-b1kpt50-cs32/
```

### Step 2: Dry-run 验证评测命令

使用 `--dry-run` 先预览命令，确保参数正确：

```bash
cd /mnt/bn/mllm-data-yg/chenjunting/repo/openpi-comet

TASK_NAME=make_pizza \
GPU_IDS=0 \
EVAL_INSTANCE_IDS=0,1,2,3,4 \
MAX_STEPS=50 \
HEADLESS=true \
bash scripts/run_b1k_eval_parallel_single_task.sh --dry-run \
  /mnt/bn/mllm-data-yg/chenjunting/repo/openpi-comet/checkpoints/openpi_comet/pi05-b1kpt50-cs32
```

### Step 3: 运行少量 Episode 烟雾测试

先跑 5 个 episode 验证流程通：

```bash
cd /mnt/bn/mllm-data-yg/chenjunting/repo/openpi-comet

TASK_NAME=make_pizza \
GPU_IDS=0,1 \
NUM_GPUS=2 \
EVAL_INSTANCE_IDS=0,1,2,3,4 \
MAX_STEPS=100 \
HEADLESS=true \
WRITE_VIDEO=false \
bash scripts/run_b1k_eval_parallel_single_task.sh \
  /mnt/bn/mllm-data-yg/chenjunting/repo/openpi-comet/checkpoints/openpi_comet/pi05-b1kpt50-cs32
```

### Step 4: 运行完整评测（10+ episodes）

烟雾测试通过后，扩大评测规模：

```bash
cd /mnt/bn/mllm-data-yg/chenjunting/repo/openpi-comet

TASK_NAME=make_pizza \
GPU_IDS=0,1,2,3,4,5,6,7 \
NUM_GPUS=8 \
EVAL_INSTANCE_IDS=0,1,2,3,4,5,6,7,8,9 \
HEADLESS=true \
WRITE_VIDEO=false \
bash scripts/run_b1k_eval_parallel_single_task.sh \
  /mnt/bn/mllm-data-yg/chenjunting/repo/openpi-comet/checkpoints/openpi_comet/pi05-b1kpt50-cs32
```

### Step 5: Segment 级别细粒度分析（可选）

使用 `eval_segment.py` 对特定 demo_id 进行 segment 级别分析：

```bash
cd /mnt/bn/mllm-data-yg/chenjunting/repo/BEHAVIOR-1K

python OmniGibson/omnigibson/learning/eval_segment.py \
  policy=websocket \
  task.name=make_pizza \
  demo_data_path=/path/to/2025-challenge-demos \
  demo_id=00000490 \
  segment_level=primitive \
  segment_idx=0 \
  success_mode=predicate_progress \
  log_path=./eval_logs/make_pizza_segment_analysis
```

### Step 6: 汇总分析

1. 收集所有 episode 的 metrics JSON
2. 统计整体成功率
3. 分析失败原因分布：
   - 统计 `result_type` 分布（timeout / predicate_satisfied / restore_failed 等）
   - 统计 `rollout.best_progress` 分布
   - 识别高频失败步骤
4. 生成分析报告

---

## 关键配置

| 配置项 | 值 |
|--------|-----|
| `TASK_NAME` | `make_pizza` |
| `CKPT_DIR` | `/mnt/bn/mllm-data-yg/chenjunting/repo/openpi-comet/checkpoints/openpi_comet/pi05-b1kpt50-cs32` |
| `EVAL_ENTRYPOINT` | `eval_custom.py` |
| `HEADLESS` | `true` |
| `BEHAVIOR_DIR` | `/mnt/bn/mllm-data-yg/chenjunting/repo/BEHAVIOR-1K` |

---

## 预期输出

评测完成后，结果将保存在：
- `openpi-comet/eval_logs/parallel_make_pizza_pi05-b1kpt50-cs32_*/`
  - `metrics/*.json` — 每个 episode 的评测结果
  - `eval_gpu*_p*.log` — 评测日志

---

## 注意事项

1. **首次初始化**: 首次 `import omnigibson` 可能耗时数分钟
2. **GPU 设置**: 如遇渲染问题，设置 `OMNIGIBSON_GPU_ID`
3. **不触发下载**: 确认数据已准备好再运行评测
