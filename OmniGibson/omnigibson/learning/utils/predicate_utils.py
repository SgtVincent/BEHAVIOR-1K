import logging
from typing import Any, Dict, List, Optional

from bddl.logic_base import Expression
from bddl.config import READABLE_PREDICATE_NAMES


logger = logging.getLogger("predicate_utils")
logger.setLevel(logging.INFO)


def flatten_list(li):
    for elem in li:
        if isinstance(elem, list):
            yield from flatten_list(elem)
        else:
            yield elem


def format_bddl_expr(expr_body: List[Any], backend=None) -> str:
    if not expr_body or not isinstance(expr_body, list):
        return str(expr_body)

    token = expr_body[0]

    if token == "not":
        inner = format_bddl_expr(expr_body[1], backend)
        return f"not({inner})"

    if token in ("and", "or", "imply"):
        connective = {"and": "AND", "or": "OR", "imply": "IMPLY"}.get(token, token.upper())
        parts = [format_bddl_expr(arg, backend) for arg in expr_body[1:]]
        return f"({connective} {' '.join(parts)})"

    if token in ("forall", "exists", "forn", "forpairs", "fornpairs"):
        return format_bddl_expr_quantifier(expr_body)

    predicate_name = READABLE_PREDICATE_NAMES.get(token, token)
    args = [str(arg).lstrip("?") for arg in expr_body[1:]]
    return f"({predicate_name} {' '.join(args)})"


def format_bddl_expr_quantifier(expr_body: List[Any]) -> str:
    token = expr_body[0]
    if token == "forall":
        var_info = expr_body[1]
        var_name = var_info[0].lstrip("?")
        var_type = var_info[2]
        inner = format_bddl_expr(expr_body[2])
        return f"forall({var_name}: {var_type}, {inner})"
    elif token == "exists":
        var_info = expr_body[1]
        var_name = var_info[0].lstrip("?")
        var_type = var_info[2]
        inner = format_bddl_expr(expr_body[2])
        return f"exists({var_name}: {var_type}, {inner})"
    elif token == "forn":
        n = expr_body[1][0]
        var_info = expr_body[2]
        var_name = var_info[0].lstrip("?")
        var_type = var_info[2]
        inner = format_bddl_expr(expr_body[3])
        return f"forn({n}, {var_name}: {var_type}, {inner})"
    elif token == "forpairs":
        var_info1 = expr_body[1]
        var_name1 = var_info1[0].lstrip("?")
        var_type1 = var_info1[2]
        var_info2 = expr_body[2]
        var_name2 = var_info2[0].lstrip("?")
        var_type2 = var_info2[2]
        inner = format_bddl_expr(expr_body[3])
        return f"forpairs({var_name1}: {var_type1}, {var_name2}: {var_type2}, {inner})"
    elif token == "fornpairs":
        n = expr_body[1][0]
        var_info1 = expr_body[2]
        var_name1 = var_info1[0].lstrip("?")
        var_type1 = var_info1[2]
        var_info2 = expr_body[3]
        var_name2 = var_info2[0].lstrip("?")
        var_type2 = var_info2[2]
        inner = format_bddl_expr(expr_body[4])
        return f"fornpairs({n}, {var_name1}: {var_type1}, {var_name2}: {var_type2}, {inner})"
    return str(expr_body)


def eval_ground_option(option: List, evaluate_fn_name: str = "evaluate") -> List[bool]:
    results = []
    for head in option:
        try:
            eval_method = getattr(head, evaluate_fn_name, None)
            if eval_method is None:
                logger.warning(f"HEAD has no '{evaluate_fn_name}' method")
                results.append(False)
                continue
            result = eval_method()
            results.append(bool(result))
        except Exception as e:
            logger.warning(f"Failed to evaluate HEAD: {e}")
            results.append(False)
    return results


def diff_subgoal(start_truth: List[bool], end_truth: List[bool]) -> List[int]:
    subgoal_indices = []
    min_len = min(len(start_truth), len(end_truth))
    for i in range(min_len):
        if not start_truth[i] and end_truth[i]:
            subgoal_indices.append(i)
    return subgoal_indices


def rank_groundings(
    ground_options: List[List],
    s_start: List[List[bool]],
    s_end: List[List[bool]],
    topk: int = 3,
) -> List[Dict[str, Any]]:
    candidates = []
    for opt_idx, (opt, start_truth, end_truth) in enumerate(zip(ground_options, s_start, s_end)):
        n_predicates = len(opt)
        if n_predicates == 0:
            continue

        end_satisfied = sum(1 for v in end_truth if v)
        end_satisfied_ratio = end_satisfied / n_predicates

        subgoal_indices = diff_subgoal(start_truth, end_truth)
        delta_count = len(subgoal_indices)

        candidates.append({
            "option_idx": opt_idx,
            "end_satisfied_ratio": end_satisfied_ratio,
            "delta_count": delta_count,
            "subgoal_size": delta_count,
            "subgoal_indices": subgoal_indices,
            "n_predicates": n_predicates,
        })

    candidates.sort(
        key=lambda x: (x["end_satisfied_ratio"], x["delta_count"]),
        reverse=True,
    )

    return candidates[:topk]


def select_best_grounding(
    ground_options: List[List],
    s_start: List[List[bool]],
    s_end: List[List[bool]],
) -> int:
    ranked = rank_groundings(ground_options, s_start, s_end, topk=1)
    if not ranked:
        return 0
    return ranked[0]["option_idx"]


def compute_q_score_delta(
    ground_option: List,
    start_truth: List[bool],
    end_truth: List[bool],
) -> float:
    if len(start_truth) != len(end_truth) or len(ground_option) == 0:
        return 0.0
    delta = sum(
        int(not start_truth[i] and end_truth[i])
        for i in range(len(start_truth))
    )
    return delta / len(ground_option)


def compute_subgoal_progress(
    current_truth: List[bool],
    subgoal_indices: List[int],
) -> float:
    if not subgoal_indices:
        return 1.0
    satisfied = sum(1 for i in subgoal_indices if current_truth[i])
    return satisfied / len(subgoal_indices)


def format_head_predicate(head, backend=None) -> str:
    try:
        body = getattr(head, "body", None)
        if body is not None:
            return format_bddl_expr(body, backend)
    except Exception:
        pass
    return str(head)


def get_subgoal_predicates(
    ground_option: List,
    subgoal_indices: List[int],
    backend=None,
) -> List[Dict[str, Any]]:
    predicates = []
    for idx in subgoal_indices:
        if 0 <= idx < len(ground_option):
            head = ground_option[idx]
            try:
                body = getattr(head, "body", None)
                terms = getattr(head, "terms", [])
                scope = getattr(head, "scope", {})
                readable = format_bddl_expr(body, backend) if body else str(head)
                predicate_name = body[0] if body and len(body) > 0 else "unknown"
                args = [str(arg).lstrip("?") for arg in body[1:]] if body and len(body) > 1 else []
                predicates.append({
                    "index": idx,
                    "predicate_name": predicate_name,
                    "args": args,
                    "terms": terms,
                    "readable": readable,
                    "scope_summary": {k: type(v).__name__ for k, v in list(scope.items())[:5]},
                })
            except Exception as e:
                predicates.append({
                    "index": idx,
                    "error": str(e),
                    "raw": str(head),
                })
    return predicates
