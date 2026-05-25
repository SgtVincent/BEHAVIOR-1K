# Skill Metric Registry Table

> 当前文档保留为 registry 快照表，但最终以代码为准：
>
> `/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/OmniGibson/omnigibson/learning/utils/segment_skill_metric_registry.py`
>
> 用户指南见：`/mnt/bn/navigation-hl/mlx/users/chenjunting/repo/BEHAVIOR-1K/.trae/documents/skill_eval_user_guide.md`

| Skill | Metric Family | Object Roles | Success Rule |
|---|---|---|---|
| move to | geometry_base_target | target_or_obj | robot base stays within a target-relative proximity threshold captured from demo end state |
| pick up from | grasp_relation | obj, src_or_target | object is grasped and no longer ontop/inside original source |
| place in | relation_place_inside | obj, dst_or_target | object is inside destination and released |
| place on | relation_place_ontop | obj, dst_or_target | object is ontop destination and released |
| push to | geometry_object_target | obj, target_or_dst | object reaches demo-end target pose or becomes nextto target |
| chop | contact_effect_proxy | obj, target_obj | tool contacts target object during chopping window |
| open door | articulation_open | unary_target | target articulated object is open |
| place on next to | relation_place_ontop_nextto | obj, support_target, neighbor_target | object is ontop support, nextto neighbor, and released |
| close door | articulation_close | unary_target | target articulated object is closed |
| sweep surface | contact_effect_proxy | obj, target_obj_or_surface | tool maintains contact with target surface during sweep |
| pour | relation_transfer_proxy | payload_or_obj, dst_or_target, obj | payload reaches target/support; container end-pose proxy alone is not sufficient |
| turn on switch | toggle_on | unary_target | target is toggled on |
| close lid | articulation_close | unary_target | lid/container is closed |
| turn to | geometry_base_facing | face_target | robot base yaw faces target object within threshold from demo end state |
| turn off switch | toggle_off | unary_target | target is toggled off |
| hand over | transfer_pose_proxy | obj | object reaches demo-end handover pose while remaining grasped |
| spray | contact_effect_proxy | obj, target_obj | sprayer must make contact with the target object during the segment |
| open lid | articulation_open | unary_target | lid/container is open |
| hold | grasp_hold | obj | object is grasped |
| release | grasp_release | obj | object is no longer grasped |
| tip over | orientation_proxy | obj | object orientation matches tipped end-state proxy |
| insert | relation_place_inside | obj, dst_or_target | object is inside destination and released |
| sweep off | relation_detach_surface | obj, src_or_target | object is no longer ontop the original surface |
| open drawer | articulation_open | unary_target | drawer/openable target is open |
| close drawer | articulation_close | unary_target | drawer/openable target is closed |
| place in next to | relation_place_inside_nextto | obj, support_target, neighbor_target | object is inside container, nextto neighbor, and released |
| place under | relation_under | obj, dst_or_target | object is under the target and released |
| pull tray | articulation_open_proxy | unary_target | tray-bearing object is open (pulled out) |
| press | toggle_on | unary_target | pressed target is toggled on |
| ignite | effect_on_fire | target_obj | target object is on fire |
| hang | relation_attach | obj, dst_or_target | object is attached to hanging target and released |
| attach | relation_attach | obj, dst_or_target | object is attached to target and released |
| wipe hard | contact_effect_proxy | obj, target_obj | cleaning tool contacts target object during wiping |
| push tray | articulation_close_proxy | unary_target | tray-bearing object is closed (pushed in) |

