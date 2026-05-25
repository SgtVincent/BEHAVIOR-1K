# Legacy: BDDL Predicate Segment Eval Interface

> 该文档已归档。当前 per-skill eval 的 canonical 用户文档是：
>
> `/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/.trae/documents/skill_eval_user_guide.md`
>
> 当前内部 bugfix / update 文档是：
>
> `/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/.trae/documents/skill_eval_internal_update_log.md`

## 归档原因

本文件记录的是早期从 task-level BDDL `ground_goal_state_options` 中抽取 segment predicate delta 的接口说明。该方案对理解 BDDL / `q_score` 有帮助，但不是当前 per-skill eval 的主路径。

当前正式 per-skill eval 使用：

```text
segment_level=skill
success_mode=segment_predicates
```

如需查看旧的 BDDL HEAD、`HEAD.evaluate()`、`q_score` 等背景，请从 git history 或旧版本文档中查阅。本文件不再维护具体接口细节，避免误导使用者。
