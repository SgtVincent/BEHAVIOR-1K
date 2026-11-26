"""Stdlib tests for the planner-waypoint -> physics-step time grid.

Covers explicit timestamp mapping, segment interpolation, the no-endpoint-jump
invariant, exhaustion without hold/repeat, and the bounded per-step motion
gate. No planner, simulator, torch, or numpy is imported.
"""
import math

from brvla_bridge import contracts, planner_grid
from brvla_bridge.errors import ContractViolation
from brvla_bridge.tests._assert import raises

PLANNER_DT = planner_grid.PLANNER_DT_S
PHYSICS_DT = contracts.CONTROL_DT


def _line_waypoints(count: int) -> list:
    return [[float(k)] * contracts.ACTION23 for k in range(count)]


def _grid(**kwargs) -> planner_grid.PlannerTrajectoryGrid:
    kwargs.setdefault("waypoints", _line_waypoints(5))
    kwargs.setdefault("planner_dt", PLANNER_DT)
    kwargs.setdefault("physics_dt", PHYSICS_DT)
    return planner_grid.PlannerTrajectoryGrid(**kwargs)


def test_command_maps_to_explicit_segment_fraction():
    grid = _grid()
    first = grid.command_at(0)
    expected_fraction = (PHYSICS_DT - PLANNER_DT) / PLANNER_DT
    assert first["planner_segment"] == 1
    assert abs(first["fraction"] - expected_fraction) < 1e-9
    assert abs(first["native23"][0] - (1.0 + expected_fraction)) < 1e-9
    assert first["time_s"] == PHYSICS_DT

    second = grid.command_at(1)
    expected_fraction2 = (2 * PHYSICS_DT - 2 * PLANNER_DT) / PLANNER_DT
    assert second["planner_segment"] == 2
    assert abs(second["fraction"] - expected_fraction2) < 1e-9


def test_no_endpoint_jump_on_first_step():
    grid = _grid()
    command = grid.command_at(0)["native23"]
    assert command[0] != 4.0
    assert all(math.isfinite(v) for v in command)


def test_exhausted_after_plan_duration_no_repeat():
    grid = _grid()
    grid.command_at(0)
    grid.command_at(1)
    grid.command_at(2)
    with raises(planner_grid.TrajectoryExhausted):
        grid.command_at(3)


def test_single_waypoint_plan_exhausts_immediately():
    grid = planner_grid.PlannerTrajectoryGrid(_line_waypoints(1), PLANNER_DT, PHYSICS_DT)
    assert grid.duration_s == 0.0
    with raises(planner_grid.TrajectoryExhausted):
        grid.command_at(0)


def test_bounded_motion_gate_violates_on_large_step():
    grid = _grid(max_step_delta=0.1)
    grid.command_at(0)
    with raises(ContractViolation):
        grid.command_at(1)


def test_reset_clears_motion_reference():
    grid = _grid(max_step_delta=0.1)
    grid.command_at(0)
    grid.reset()
    # After reset the gate reference is cleared; the next step passes the
    # bounded check against no previous command.
    grid.command_at(1)


def test_rejects_nonfinite_and_wrong_width():
    bad = _line_waypoints(3)
    bad[1][0] = float("nan")
    with raises(ContractViolation):
        planner_grid.PlannerTrajectoryGrid(bad, PLANNER_DT, PHYSICS_DT)
    with raises(ContractViolation):
        planner_grid.PlannerTrajectoryGrid([[0.0] * 22], PLANNER_DT, PHYSICS_DT)
    with raises(ContractViolation):
        planner_grid.PlannerTrajectoryGrid(_line_waypoints(2), 0.0, PHYSICS_DT)
    with raises(ContractViolation):
        planner_grid.PlannerTrajectoryGrid(_line_waypoints(2), PLANNER_DT, -1.0)


def test_receipt_meta_records_grid():
    grid = _grid()
    receipt = grid.receipt()
    assert receipt["waypoints"] == 5
    assert receipt["planner_dt"] == PLANNER_DT
    assert receipt["physics_dt"] == PHYSICS_DT
    assert abs(receipt["duration_s"] - 4 * PLANNER_DT) < 1e-12
    assert receipt["plan_covers_steps"] == 3
