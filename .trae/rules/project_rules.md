# 项目规则（BEHAVIOR-1K）
1. 推荐使用仓库提供的一键安装脚本进行依赖与组件安装：Linux 用 `./setup.sh`，Windows 用 `.\setup.ps1`；默认 Conda 环境名为 `behavior`，Python 版本要求为 3.10。
2. 运行仓库代码前，优先确认已激活正确环境：`conda activate behavior`；不要混用系统 Python 与 Conda 之外的依赖环境。
3. OmniGibson / Isaac Sim 安装与运行依赖 NVIDIA EULA；数据下载依赖 BEHAVIOR Dataset License。禁止尝试解密、提取、逆向或重分发任何数据、密钥或第三方内容。
4. 数据与资产体积很大，避免在不需要时触发下载流程：不要随意运行 `./setup.sh --dataset` 或直接调用 `download_behavior_1k_assets` 等下载函数；确需下载时，确保磁盘空间与网络稳定，并明确接受条款。
5. 首次 `import omnigibson` 可能触发 Omniverse 的一次性初始化并耗时数分钟，这是预期行为；排障前先区分“首次初始化”与真正的卡死。
6. GPU/渲染问题优先通过环境变量选择设备：遇到 `HydraEngine rtx failed creating scene renderer.` 等问题时，先设置 `export OMNIGIBSON_GPU_ID=<可用GPU编号>` 再复现。
7. 避免污染 Isaac Sim 相关环境变量：不要在同一终端里混用历史安装残留；如果存在 `EXP_PATH`、`CARB_APP_PATH`、`ISAAC_PATH` 等变量，先清理后再运行安装/启动流程。
8. 修改代码遵循最小化改动与模块边界：仿真与任务逻辑在 `OmniGibson/omnigibson/`，任务语言与生成在 `bddl3/`，遥操作与数据采集在 `joylo/`，资产管线在 `asset_pipeline/`，不要跨模块硬耦合。
9. 运行测试与示例时，优先选择最小复现：先跑 `python -m omnigibson.examples...` 或单测子集，再扩大到全量 `OmniGibson/tests/`，避免一次性引入长启动与大规模资产下载带来的噪音。
