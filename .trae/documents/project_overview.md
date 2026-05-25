# BEHAVIOR-1K：项目总览与开发导航

本仓库是 BEHAVIOR-1K 的单体仓库（monorepo），用于在真实感家居仿真中训练与评测具身智能体，覆盖：
- 仿真平台：OmniGibson（基于 NVIDIA Omniverse / Isaac Sim）
- 任务与语言：BDDL（Behavior Domain Definition Language）
- 遥操作与数据采集：JoyLo
- 资产与数据：BEHAVIOR-1K 场景 / 物体 / 任务实例与资产生产管线
- 文档站点与挑战赛评测工具

主要入口参考：
- 仓库根目录 README：安装入口与引用信息
- OmniGibson/README：仿真平台介绍与文档入口
- docs/getting_started/installation.md：统一安装脚本用法与常见排障

## 本地规则（先读）

- 先遵守项目使用规则，再运行安装、下载或评测流程：
  - 见：.trae/rules/project_rules.md

## 仓库结构速览（Repo Map）

- setup.sh / setup.ps1：统一安装脚本（创建/使用 Conda 环境、安装 OmniGibson/BDDL/JoyLo、可选下载数据与评测依赖）
- OmniGibson/：核心仿真平台
  - omnigibson/：Python 包源码（环境、场景、机器人、任务、传感器、物理/渲染系统等）
  - tests/：pytest 测试集
  - docker/：容器相关脚本与配置（在 monorepo 下可能会随着版本调整）
- bddl3/：BDDL 语言与工具链（任务定义、解析、验证、文档与测试）
- joylo/：遥操作与数据采集工具（手柄/控制器、相机、机器人适配、示例脚本）
- datasets/：数据相关说明与占位（真正的大体积资产通常通过 OmniGibson 的下载接口获取）
- asset_pipeline/：场景/物体资产生产与质检管线（含 DVC 配置与工具脚本）
- docs/：文档站点内容（安装、教程、OmniGibson 与 BEHAVIOR 组件说明、挑战赛规则等）
- eval-jobqueue/：挑战赛评测分布式作业队列（FastAPI server + SLURM worker + job/resource 生成器）

## 关键工作流（常用路径）

### 1）安装 / 环境

- 推荐用统一脚本安装所需组件（可按需选择模块）：
  - 参考：docs/getting_started/installation.md
  - 常见：`./setup.sh --new-env --omnigibson --bddl --dataset`

### 2）快速运行 OmniGibson 示例

- 机器人遥操作示例：
  - `python -m omnigibson.examples.robots.robot_control_example --quickstart`
- 场景交互示例：
  - `python -m omnigibson.examples.scenes.scene_selector`

### 3）任务与 BDDL

- 新增/修改任务定义与解析逻辑，优先在 bddl3/ 内完成，并使用其自带测试/验证流程确保不破坏解析与约束。

### 4）评测与挑战赛工具

- 本地评测通常通过 OmniGibson 的任务与环境接口进行；挑战赛分布式评测可参考：
  - eval-jobqueue/README.md（jobqueue 服务、SLURM worker、资源池与输出路径约定）

### 5）资产与数据生产

- 资产生产与 QA 位于 asset_pipeline/，通常与 DVC 绑定并包含外部工具依赖；修改前先确认当前工作是否属于“生产管线”而非“仿真运行时”。

## 常见改动点（Where To Make Changes）

- 仿真行为 / 物理与渲染 / 任务执行逻辑：OmniGibson/omnigibson/
- 新增机器人、控制器、传感器或环境封装：OmniGibson/omnigibson/robots、controllers、sensors、envs
- BDDL 语法、解析、验证与任务生成：bddl3/bddl/
- 遥操作设备接入、数据采集、相机与机器人适配：joylo/
- 大规模评测调度与资源管理：eval-jobqueue/
- 文档站点改动：docs/

## 开发自检（Fast Sanity）

- 确认环境：`conda activate behavior`，Python 3.10
- 首次启动慢属于预期：首次 import OmniGibson 可能需要较长初始化
- 渲染/GPU 异常优先切换 GPU：设置 `OMNIGIBSON_GPU_ID`
- 变更后优先跑最小示例或单测子集，再扩大范围
