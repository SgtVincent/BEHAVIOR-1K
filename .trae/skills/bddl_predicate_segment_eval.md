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

## 相关文件

- 脚本：`OmniGibson/omnigibson/learning/eval_segment.py`
- 工具函数：`OmniGibson/omnigibson/learning/utils/predicate_utils.py`
- 配置：`OmniGibson/omnigibson/learning/configs/eval_segment_config.yaml`
- 接口文档：`.trae/documents/bddl_predicate_segment_eval_interface.md`
- 使用指南：`.trae/documents/bddl_predicate_segment_eval_usage.md`
