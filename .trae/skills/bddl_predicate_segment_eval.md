# BDDL Predicate Segment Eval

## 用途

基于 BDDL 谓词的 primitive/skill segment 评测能力，用于：
- 策略诊断与训练迭代
- 分析 segment 级别 subgoal 成功/失败原因
- 评估策略在子目标上的表现

## 关键参数

| 参数 | 说明 |
|------|------|
| `segment_level` | `primitive \| skill`，评测粒度 |
| `segment_idx` | segment 索引（按 start_frame 排序） |
| `success_mode` | `predicate_subgoal \| predicate_progress \| state_match` |
| `grounding_topk` | 输出的 top grounding 候选数 |
| `dry_run` | 是否仅提取 subgoal 不运行 policy |

## 命令行示例

### Primitive 级别 dry run

```bash
python OmniGibson/omnigibson/learning/eval_segment.py \
  policy=websocket task.name=turning_on_radio \
  demo_data_path=/path/to/2025-challenge-demos \
  demo_id=00000010 segment_level=primitive segment_idx=0 \
  success_mode=predicate_progress dry_run=true \
  log_path=./eval_logs/segment_eval
```

### Primitive 级别 predicate subgoal 评测

```bash
python OmniGibson/omnigibson/learning/eval_segment.py \
  policy=websocket task.name=turning_on_radio \
  env_wrapper._target_=omnigibson.learning.wrappers.wbvima_wrapper.WBVIMAWrapper \
  demo_data_path=/path/to/2025-challenge-demos \
  rawdata_path=/path/to/2025-challenge-rawdata \
  demo_id=00000010 segment_level=primitive segment_idx=0 \
  success_mode=predicate_subgoal \
  log_path=./eval_logs/segment_eval
```

### Skill 级别评测

```bash
python OmniGibson/omnigibson/learning/eval_segment.py \
  policy=websocket task.name=turning_on_radio \
  demo_data_path=/path/to/2025-challenge-demos \
  demo_id=00000010 segment_level=skill segment_idx=0 \
  success_mode=predicate_subgoal \
  log_path=./eval_logs/segment_eval
```

## 注意事项

1. 确保 `conda activate behavior` 已执行
2. 不要运行 `setup.sh --dataset` 以免触发不必要的数据下载
3. 如遇 GPU 问题，设置 `export OMNIGIBSON_GPU_ID=<可用GPU编号>`
4. 首次运行 `import omnigibson` 可能需要数分钟初始化

## 安全启动建议

1. 做正式对比实验时，优先使用 `openpi-comet/scripts/run_skill_metric_multinode_sweep.py` 启动，而不是手工分别起 `serve_b1k.py` 与 `eval_segment.py`
2. 如必须手工启动 websocket eval，务必保证 **一条 task 一个 server**，不要让多个不同 `task.name` 共享同一个 policy server
3. 正式运行时建议显式校验以下字段：`model.expected_task_name`、`model.expected_task_prompt_sha256`、`model.expected_server_run_id`、`model.expected_server_token`
4. 如果只看到 `healthz` 成功，不代表连对了 server；必须结合 metadata 做身份校验，否则旧 run 残留进程可能污染结果
5. 若同一条样例结果前后“抖动”，优先排查：旧 server 未退出、端口复用、`--resume` 复用了脏输出目录、JAX/Torch backend 与 server 不匹配、代理变量干扰 localhost websocket
6. 对比实验时固定 job 清单、`segment_max_steps` / `max_dynamic_steps_cap`、checkpoint 路径、`policy_backend`，不要一边重跑一边改配置
7. 若是查看单条 segment 失败原因，可直接读 `segment_eval.log`、`metrics/*.json` 和 `review/*.png`；若要下正式结论，必须看整次 run 的 `multinode_skill_summary.json`

## 相关文件

- 脚本：`OmniGibson/omnigibson/learning/eval_segment.py`
- 工具函数：`OmniGibson/omnigibson/learning/utils/predicate_utils.py`
- 配置：`OmniGibson/omnigibson/learning/configs/eval_segment_config.yaml`
- 接口文档：`.trae/documents/bddl_predicate_segment_eval_interface.md`
- 使用指南：`.trae/documents/bddl_predicate_segment_eval_usage.md`
