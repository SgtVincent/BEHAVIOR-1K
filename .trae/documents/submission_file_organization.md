# BEHAVIOR-1K Challenge Submission：文件与组织方式

本文档记录 BEHAVIOR-1K Challenge 中与 submission 相关的文件类型、目录位置、命名规范，以及它们在评测与 leaderboard 展示中的对应关系。

## 目录与角色

submission 相关文件主要分成三类：

1. 参赛者评测产物（per-rollout 原始输出）：评测脚本运行后生成并提交
2. 评测环境 wrapper（评测时注入到环境 / 观测 / 动作接口）：官方评测时会加载
3. Leaderboard 汇总条目（per-submission 统计汇总）：用于文档与官网展示

在本仓库里，这三类内容分别对应以下位置：

- 规则说明（submission 指南）：[submission.md](file:///mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/docs/challenge/submission.md)
- 评测 wrapper 实现：[challenge_submissions/](file:///mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/wrappers/challenge_submissions)
- leaderboard 汇总 JSON：[docs/challenge_submissions/](file:///mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/docs/challenge_submissions)
- 官网 leaderboard（Submission 列表与排行）：https://behavior.stanford.edu/challenge/leaderboard.html

## 1) 参赛者评测产物（per-rollout 原始输出）

按照 [submission.md](file:///mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/docs/challenge/submission.md#L8-L75) 的描述，参赛者在本地运行评测脚本后会得到两类输出：

- `*.json`：每个 rollout 一个结果文件，包含指标（例如 q_score、time、agent_distance 等）
- `*.mp4`：每个 rollout 一个视频录制

关键约束（submission.md 明确强调）：

- 不允许以任何方式修改输出 JSON 与视频
- 允许 partial submission：如果少交了某些 rollout，对应实例会按 0 计入最终成绩

这些 per-rollout 原始输出通常数量为：

- 50 tasks × 10 instances × 1 rollout = 500 个 json（与相同数量的视频）

## 2) 评测 wrapper（`wrappers/challenge_submissions/submission_<team_slug>.py`）

### 2.1 wrapper 的用途

评测 wrapper 的作用是把评测环境的观测 / 传感器 / 动作接口改成“某队伍训练或服务时所假设的格式”，典型场景包括：

- 相机分辨率、视场角、是否包含 depth/seg 等 modality 的对齐
- 策略接口不是单步 action，而是一次输出一段 action 序列（horizon）
- 在观测中注入额外字段（历史观测、速度、映射表、aux target 等）以适配该队策略

### 2.2 wrapper 的加载方式（Hydra 注入）

在 jobqueue 的评测流程中，会通过 Hydra override 指定 wrapper 类（见 [eval_with_jobqueue.py](file:///mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_with_jobqueue.py#L153-L161)）：

- `env_wrapper._target_=omnigibson.learning.wrappers.challenge_submissions.submission_{team_slug}.WRAPPER_CLASS`

因此每个 wrapper 文件末尾都提供：

- `WRAPPER_CLASS = <YourWrapperClass>`

且文件名中的 `{team_slug}` 必须与评测系统的 team slug 一致（例如 `comet`、`the_north_star`）。

### 2.3 当前仓库内已有的 wrapper 文件与含义

- [submission_comet.py](file:///mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/wrappers/challenge_submissions/submission_comet.py)
  - 对齐 R1Pro 相机：head/wrist 分辨率与 aperture；最后 reload observation space
  - 分辨率常量来自 [eval_utils.py](file:///mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/utils/eval_utils.py#L6-L20)

- [submission_robot_learning_collective.py](file:///mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/wrappers/challenge_submissions/submission_robot_learning_collective.py)
  - 将相机分辨率统一调整为 224×224（常见视觉模型输入尺寸），并对 head 相机设置 aperture

- [submission_simpleai_robot.py](file:///mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/wrappers/challenge_submissions/submission_simpleai_robot.py)
  - 增加底盘质量（仿真稳定性相关）
  - 对齐相机分辨率与 aperture
  - 显式增加 `depth_linear / seg_semantic / seg_instance_id` modality

- [submission_the_north_star.py](file:///mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/wrappers/challenge_submissions/submission_the_north_star.py)
  - 支持“策略一次输出一段 action 序列”的接口：wrapper 内部逐步执行
  - 在需要新 action 时返回 `need_new_action=True`，并可附带历史观测与速度、seg id registry 等
  - 对齐相机设置并增加 depth/seg modality

- [submission_rapper.py](file:///mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/wrappers/challenge_submissions/submission_rapper.py)
  - 在观测中注入 `aux::...` 字段以适配规划/回放式策略
  - 依赖 `planner` 等额外资源文件；在当前仓库快照下这些资源未必齐全，因此该 wrapper 更偏“提交时的 wrapper 样例/摘录”

## 3) Leaderboard 汇总 JSON（`docs/challenge_submissions/*.json`）

### 3.1 这类 JSON 的语义

`docs/challenge_submissions/*.json` 是 per-submission 的“汇总条目”，不是 per-rollout 原始输出文件。它通常包含：

- submission 元信息：team / affiliation / date / track / testset
- overall_scores：整体统计（q_score、task_sr 等）
- per_task_scores：每个 task 汇总的统计
- per_rollout_scores：按 task → instance 的统计明细（缺失会按 0 处理）

示例见：

- [standard.public.Comet.NVIDIA_Research.20251117.json](file:///mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/docs/challenge_submissions/standard.public.Comet.NVIDIA_Research.20251117.json)

### 3.2 文件名命名规范

文件名格式：

- `<track>.<testset>.<team>.<affiliation>.<date>.json`

字段含义：

- `track`：`standard` 或 `privileged`
- `testset`：`public`（Public Validation）或 `hidden`（Held-out Test）
- `team` / `affiliation`：官网 leaderboard 的 Team 与 Affiliation（空格常用 `_` 替换）
- `date`：YYYYMMDD

该命名与官网 leaderboard 中的 Public Validation / Held-out Test 列一一对应：

- https://behavior.stanford.edu/challenge/leaderboard.html

## 4) 三类文件之间的对应关系（如何“组织起来”）

一次完整的 submission（面向 leaderboard 的一个 entry）可以理解为：

- 原始 per-rollout 输出：最多 500 个 json + 500 个 mp4（参赛者生成并提交）
- 评测 wrapper / 机器人配置 / README：用于官方在统一评测框架下复现实验与对齐接口
- 汇总条目（本仓库的 `docs/challenge_submissions/*.json`）：由评测系统对 per-rollout 结果进行汇总统计后生成，用于展示和对比

在当前仓库中，wrapper 文件与 leaderboard 汇总条目的映射主要通过队伍 slug / 名称进行匹配，例如：

- wrapper：[submission_comet.py](file:///mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/wrappers/challenge_submissions/submission_comet.py)
  - leaderboard 条目：`standard.(public|hidden).Comet.NVIDIA_Research.20251117.json`

评测时 wrapper 的选择由 `team_slug` 决定（见 [eval_with_jobqueue.py](file:///mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval_with_jobqueue.py#L146-L161)）。

