# Legacy: BDDL Predicate Segment Eval Usage

> 该文档已归档。当前 per-skill eval 的 canonical 用户文档是：
>
> `/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/.trae/documents/skill_eval_user_guide.md`

## 归档原因

本文件中的 `predicate_subgoal` / `predicate_progress` 示例属于早期 BDDL predicate segment eval 流程。当前 per-skill eval 不再把 task-level BDDL delta 作为唯一成功标准，而是使用 skill annotation + metric registry + runtime object states / geometry proxy。

当前推荐命令请看：

```text
skill_eval_user_guide.md#4-单-segment-eval-示例
skill_eval_user_guide.md#5-批量-sweep-示例
```

如果你确实需要调试 legacy BDDL predicate delta，可从 git history 恢复旧内容；正式 skill eval 不应使用本文件作为参考。
