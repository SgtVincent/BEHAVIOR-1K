# BDDL Predicate Segment Eval 接口文档

## Annotation 结构

### primitive_annotation / skill_annotation 字段结构

```json
{
  "primitive_annotation": [
    {
      "frame_duration": [start_frame, end_frame],
      "primitive_description": ["描述字符串"],
      "skill_id": "可选的 skill 标识"
    }
  ],
  "skill_annotation": [
    {
      "frame_duration": [start_frame, end_frame],
      "skill_description": ["描述字符串"],
      "primitive_indices": [0, 1, 2]
    }
  ]
}
```

### frame_duration 语义

- `start_frame`: segment 起始帧（包含）
- `end_frame`: segment 结束帧（包含）
- segment 总帧数 = end_frame - start_frame

---

## BDDL 谓词接口

### HEAD 对象结构

`env.task.ground_goal_state_options` 返回 `List[List[HEAD]]`：

- 外层 List：多个 grounding options
- 内层 List：一个 grounding option 内的多个谓词 HEAD

### HEAD 关键属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `body` | list | BDDL 原始表达式，如 `["ontop", "?obj1", "?table"]` |
| `children[0]` | AtomicFormula | 可调用的谓词对象 |
| `terms` | list | 对象 term 列表（去前缀），如 `["obj1", "table"]` |
| `scope` | dict | 对象名→ simulator 对象的映射 |
| `object_map` | dict | category → 对象名列表 |

### HEAD.evaluate()

返回当前状态下的布尔值。

### 谓词可读化

`body[0]` 是谓词名称（如 `"ontop"`），`body[1:]` 是参数列表。

---

## 关键函数

### eval_ground_option(option: List[HEAD]) -> List[bool]

对某个 grounding option 的所有谓词求值。

### diff_subgoal(start_truth, end_truth) -> List[int]

计算 start→end 新增满足的谓词索引。

### rank_groundings(ground_options, s_start, s_end, topk) -> List[Dict]

对 grounding options 排序，输出 top-K 候选。

---

## TaskMetric 参考（q_score 计算）

```python
# TaskMetric.end_callback 中的 q_score 计算
q_score = max(
    sum(
        int(not initially_true and pred.evaluate())
        for pred, initially_true in zip(option, option_previous_state)
    )
    / len(option)
    for option, option_previous_state in zip(
        env.task.ground_goal_state_options, self.initial_predicate_states
    )
)
```
