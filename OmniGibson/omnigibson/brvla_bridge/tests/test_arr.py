"""Negative regression: ``shape_of`` must terminate on empty sequences.

P2 defect on 35a946a: an empty list fed ``[]`` back into the descent loop,
appending a leading ``0`` forever. These cases are the fail-closed guards.
"""
from brvla_bridge import _arr
from brvla_bridge.errors import SchemaContractError
from brvla_bridge.tests._assert import raises


def test_shape_of_empty_sequence():
    assert _arr.shape_of([]) == (0,)


def test_shape_of_nested_empty():
    assert _arr.shape_of([[]]) == (1, 0)
    assert _arr.shape_of([[[]]]) == (1, 1, 0)


def test_shape_of_regular_nested():
    assert _arr.shape_of([1, 2, 3]) == (3,)
    assert _arr.shape_of([[1, 2], [3, 4]]) == (2, 2)
    assert _arr.shape_of(0.5) == ()


def test_shape_of_duck_shape():
    class Duck:
        shape = (1, 23)

    assert _arr.shape_of(Duck()) == (1, 23)


def test_require_shape_empty_raises():
    with raises(SchemaContractError):
        _arr.require_shape([], (0, 5), "empty")
