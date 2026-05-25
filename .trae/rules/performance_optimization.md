# BEHAVIOR Eval 性能优化操作手册

本文面向日常使用 `BEHAVIOR-1K` / `openpi-comet` launcher 跑 skill / segment eval 的同学，重点回答三个操作问题：

1. **数据应该放哪里才能更快？**
2. **`slim_only` / `slim_skipobs` 应该怎么开？**
3. **正式 eval 默认应该怎么配？**

---

## 1. 一页版默认操作建议

### 1.1 默认推荐配置

正式 eval / 日常批量跑分，默认建议：

```text
partial_scene_load=false
skip_intermediate_obs_in_chunk=true
```

也就是：

- **默认关闭** `slim_only`（`partial_scene_load`）
- **默认开启** `slim_skipobs`（`skip_intermediate_obs_in_chunk`）

### 1.2 默认操作顺序

优先级从高到低：

1. **先把全集资产复制到本地 SSD**
2. **让仿真从 SSD 数据根目录加载**
3. **默认开启 `skip_intermediate_obs_in_chunk=true`**
4. **默认关闭 `partial_scene_load=false`**
5. 只有在确认瓶颈主要是 **NAS 冷加载** 时，再临时打开 `partial_scene_load=true`

### 1.3 这样做的原因

- `skip_intermediate_obs_in_chunk` 能稳定降低 segment runtime，是当前最值得保留的默认优化。
- `partial_scene_load` 在当前 formal eval 上没有带来稳定收益，且 success 有回退。
- 真正的大头瓶颈通常不是 `slim_only` 本身，而是 **NAS 上的大量资产 IO**。

---

## 2. 快速开始

如果你只想快速照做，可以直接按下面三步。

### 步骤 1：把 datasets 全集复制到 SSD

假设原始数据目录是：

```text
/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/datasets
```

执行：

```bash
mkdir -p /ssd/behavior1k/datasets
rsync -a --info=progress2 /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/datasets/behavior-1k-assets /ssd/behavior1k/datasets/
rsync -a --info=progress2 /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/datasets/omnigibson-robot-assets /ssd/behavior1k/datasets/
rsync -a --info=progress2 /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/datasets/2025-challenge-task-instances /ssd/behavior1k/datasets/
rsync -a --info=progress2 /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/datasets/omnigibson.key /ssd/behavior1k/datasets/
```

### 步骤 2：让 launcher 指向 SSD 数据

推荐方式：把 `BEHAVIOR-1K` 副本整体放到 SSD，并让 launcher 的 `behavior_dir` 指过去。

例如：

```bash
python openpi-comet/scripts/run_skill_metric_multinode_sweep.py \
  --behavior-dir /ssd/behavior1k/BEHAVIOR-1K \
  ...
```

要求目录结构满足：

```text
/ssd/behavior1k/BEHAVIOR-1K/datasets
```

### 步骤 3：使用默认优化开关

```bash
python openpi-comet/scripts/run_skill_metric_multinode_sweep.py \
  --behavior-dir /ssd/behavior1k/BEHAVIOR-1K \
  --no-partial-scene-load \
  --skip-intermediate-obs-in-chunk \
  ...
```

或者在 shell launcher 中保持：

```bash
PARTIAL_SCENE_LOAD=false
SKIP_INTERMEDIATE_OBS_IN_CHUNK=true
```

---

## 3. 需要复制到 SSD 的资产范围

如果目标是系统性降低仿真启动 / scene import / task load 时间，建议把以下内容作为 **全集** 一起复制，而不是按任务裁切。

### 3.1 推荐保留的目录结构

```text
<NEW_DATA_ROOT>/
├── behavior-1k-assets/
├── omnigibson-robot-assets/
├── 2025-challenge-task-instances/
└── omnigibson.key
```

### 3.2 每部分作用

- `behavior-1k-assets/`
  - 对象全集、场景全集、系统资源
  - 这是最大的 IO 热点
- `omnigibson-robot-assets/`
  - 机器人 USD / URDF / 材质等资源
- `2025-challenge-task-instances/`
  - episode metadata、task instance json、tro_state 等
- `omnigibson.key`
  - 资产解密所需

### 3.3 为什么建议复制全集

优先复制全集，而不是任务子集，原因是：

1. **更稳**：切任务时不容易重新回退到 NAS
2. **更省心**：不用维护不同任务的资产裁剪目录
3. **更适合批量 eval**：多任务 sweep 时不会因为缺资产反复排障

---

## 4. 如何设置资产加载路径

OmniGibson 运行时会从 `OMNIGIBSON_DATA_PATH` 查找：

- `behavior-1k-assets`
- `omnigibson-robot-assets`
- `2025-challenge-task-instances`
- `omnigibson.key`

### 4.1 通用方式

直接在启动前设置：

```bash
export OMNIGIBSON_DATA_PATH=/ssd/behavior1k/datasets
```

然后再启动 BEHAVIOR eval / warmup / rollout。

### 4.2 对 `run_skill_metric_multinode_sweep.py` 的推荐方式

这个 launcher 会把：

```text
OMNIGIBSON_DATA_PATH = <behavior_dir>/datasets
```

因此最推荐的做法是：

```bash
python openpi-comet/scripts/run_skill_metric_multinode_sweep.py \
  --behavior-dir /ssd/behavior1k/BEHAVIOR-1K \
  ...
```

这样不需要额外记住环境变量覆盖关系。

### 4.3 如果只迁 datasets、不迁整个 BEHAVIOR-1K 仓库

可以先设置：

```bash
export OMNIGIBSON_DATA_PATH=/ssd/behavior1k/datasets
```

但要注意：

- 对于会**主动重写** `OMNIGIBSON_DATA_PATH` 的 launcher，仍然更推荐直接让 `behavior_dir` 指向 SSD 副本。
- 否则容易出现“你以为走 SSD，实际又回到了默认 NAS 路径”的情况。

---

## 5. 两个优化开关怎么用

本节只讲操作口径，不展开代码实现细节。

### 5.1 `slim_only` = `partial_scene_load=true`

含义：

- 只加载任务相关房间 / 房间类型
- 目的是减少场景导入量

#### 适合什么时候开

- 资产主要还在 NAS
- 你观察到瓶颈主要是 task 冷启动 / scene import
- 你当前是做针对性的加载耗时实验，而不是默认正式跑分

#### 什么时候不要默认开

- 正式 eval 默认不建议
- 资产已经迁到 SSD
- 你更关心 success 稳定性

#### 操作示例

Python launcher：

```bash
python openpi-comet/scripts/run_skill_metric_multinode_sweep.py \
  --partial-scene-load \
  --skip-intermediate-obs-in-chunk \
  ...
```

Shell launcher：

```bash
PARTIAL_SCENE_LOAD=true
SKIP_INTERMEDIATE_OBS_IN_CHUNK=true
```

### 5.2 `slim_skipobs` = `skip_intermediate_obs_in_chunk=true`

含义：

- chunk 内 cached-action step 跳过中间 observation 采集
- 只保留 chunk 边界 observation
- 主要减少渲染 / observation 开销

#### 适合什么时候开

- **默认就应该开**
- persistent eval 下尤其值得保留
- 你想稳定降低 segment runtime 时

#### 操作示例

Python launcher：

```bash
python openpi-comet/scripts/run_skill_metric_multinode_sweep.py \
  --skip-intermediate-obs-in-chunk \
  ...
```

Shell launcher：

```bash
SKIP_INTERMEDIATE_OBS_IN_CHUNK=true
```

### 5.3 推荐组合

#### 默认正式 eval 组合

```text
partial_scene_load=false
skip_intermediate_obs_in_chunk=true
```

#### NAS 冷加载排障 / 临时优化组合

```text
partial_scene_load=true
skip_intermediate_obs_in_chunk=true
```

说明：第二种组合只适合在你已经明确确认瓶颈是 NAS 冷加载时使用，不应直接取代默认配置。

---

## 6. 常用启动模板

### 6.1 默认推荐模板（正式 eval）

```bash
python openpi-comet/scripts/run_skill_metric_multinode_sweep.py \
  --mode launch-persistent \
  --behavior-dir /ssd/behavior1k/BEHAVIOR-1K \
  --no-partial-scene-load \
  --skip-intermediate-obs-in-chunk \
  ...
```

### 6.2 NAS 冷加载明显偏慢时的实验模板

```bash
python openpi-comet/scripts/run_skill_metric_multinode_sweep.py \
  --mode launch-persistent \
  --behavior-dir /mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K \
  --partial-scene-load \
  --skip-intermediate-obs-in-chunk \
  ...
```

### 6.3 对照实验：关闭 `slim_skipobs`

```bash
python openpi-comet/scripts/run_skill_metric_multinode_sweep.py \
  --no-skip-intermediate-obs-in-chunk \
  ...
```

这个模式主要用于 debug / A-B 对照，不建议作为默认设置。

---

## 7. 操作检查清单

启动前建议快速确认：

- [ ] 数据是否已经复制到本地 SSD
- [ ] `behavior_dir` 是否真的指向 SSD 副本
- [ ] `datasets` 下是否包含 4 个关键项
- [ ] 当前是否使用默认开关：`partial_scene_load=false`、`skip_intermediate_obs_in_chunk=true`
- [ ] 只有在做 NAS 冷加载实验时，才临时开启 `partial_scene_load=true`

如果你看到速度没有提升，先排查：

1. 运行时是否仍然在用 NAS 数据根目录
2. 是否只复制了部分目录，导致仍有依赖回源到 NAS
3. 是否错误地把 `partial_scene_load=true` 当成默认方案长期启用

---

## 8. 测试结论附录（供选型参考）

本节保留关键测试结果，方便在需要时回看结论来源。

### 8.1 三个 setting 的正式测速结果

- 运行目录：`openpi-comet/segment_eval_runs/renderer_tweak_fix_20260523_formal2/`
- 统计口径：按 `job_key` 去重，共 `136` 个 segment
- 对比设置：
  - `baseline`
  - `slim_only`：只开 `partial_scene_load`
  - `slim_skipobs`：只开 `skip_intermediate_obs_in_chunk`

| setting | partial_scene_load | skip_intermediate_obs_in_chunk | success | success rate | avg segment runtime (s) | median runtime (s) | p90 runtime (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | false | false | 30 / 136 | 22.1% | 257.23 | 178.16 | 538.82 |
| slim_only | true | false | 21 / 136 | 15.4% | 255.22 | 158.99 | 620.13 |
| slim_skipobs | false | true | 27 / 136 | 19.9% | 177.79 | 99.61 | 432.70 |

#### 结论

- `baseline -> slim_only`
  - 平均 runtime：`257.23s -> 255.22s`，仅快 `2.01s`
  - success：`30 -> 21`，回退明显
- `baseline -> slim_skipobs`
  - 平均 runtime：`257.23s -> 177.79s`，平均快 `79.44s`
  - success：`30 -> 27`，略低于 baseline，但明显好于 slim_only

因此：

- **`slim_only` 不适合作为默认优化**
- **`slim_skipobs` 是当前最值得默认开启的速度优化**

### 8.2 资产路径实验结果

- 任务：`sorting_household_items`
- 模式：单任务冷启动加载
- local SSD 测试数据路径：`/tmp/behavior1k_sorting_local/datasets`
- 该目录来自本任务所需子集，大小约 `4.31 GB`，文件数约 `15396`

| asset path | partial_scene_load | elapsed (s) |
| --- | --- | ---: |
| NAS `/mnt/.../BEHAVIOR-1K/datasets` | false | 1333.04 |
| NAS `/mnt/.../BEHAVIOR-1K/datasets` | true | 850.85 |
| local `/tmp/behavior1k_sorting_local/datasets` | false | 340.64 |
| local `/tmp/behavior1k_sorting_local/datasets` | true | 382.88 |

#### 结论

1. **NAS -> 本地 SSD 的收益远大于 `slim_only` 本身**
2. 在 NAS 上，`partial_scene_load=true` 能明显减少冷加载时间
3. 在本地 SSD 上，`partial_scene_load=true` 反而可能因为额外过滤开销而略慢

所以最终建议仍然是：

- **优先做 SSD 本地化**
- SSD 场景下默认保持 `partial_scene_load=false`

