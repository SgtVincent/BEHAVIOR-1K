"""Canary observation schema validation (faithful to the frozen training schema).

Online bundle mirrors the training raw dict:

    dataname, instruction, robot_type: br2,
    images: 3 RGB entries,
    br_states: (4, 45) finite history,
    br_actions: zeros (50, 45) pad,
    br_action_is_pad: zeros (50,) bool pad mask.

Target actions are never supplied at inference; padded zeros are placeholders
only. This module validates shape/finite/type contracts but does not decode
PNGs, load the model, or touch stats/weights. The expected dataname is supplied
by the private manifest, not hardcoded here.
"""
from typing import Optional

from . import contracts
from ._arr import require_finite, require_shape, to_plain
from .errors import SchemaContractError

EXPECTED_ROBOT_TYPE = "br2"
EXPECTED_IMAGE_COUNT = 3


def _check_images(images):
    plain = to_plain(images)
    if not isinstance(plain, list) or len(plain) != EXPECTED_IMAGE_COUNT:
        raise SchemaContractError(
            f"images: expected {EXPECTED_IMAGE_COUNT} RGB views, got "
            f"{len(plain) if isinstance(plain, list) else type(plain).__name__}"
        )
    out = []
    for idx, im in enumerate(plain):
        if hasattr(im, "shape"):
            shape = tuple(im.shape)
            if len(shape) != 3 or shape[2] != 3:
                raise SchemaContractError(f"image[{idx}]: expected HxWx3, got {shape}")
        else:
            rows = to_plain(im)
            if not rows or not isinstance(rows[0], list):
                raise SchemaContractError(f"image[{idx}]: expected HxWx3 pixel rows")
            pixel = rows[0][0]
            if not isinstance(pixel, list) or len(pixel) != 3:
                raise SchemaContractError(f"image[{idx}]: expected HxWx3 pixel rows")
        out.append(im)
    return out


def _check_zero_pad_actions(actions):
    plain = require_shape(actions, (contracts.CHUNK_SIZE, contracts.STATE45), "br_actions")
    require_finite(plain, "br_actions")
    for row in plain:
        for value in row:
            if float(value) != 0.0:
                raise SchemaContractError("br_actions: inference bundle must carry zero action pad only")
    return plain


def _check_pad_mask(mask):
    plain = to_plain(mask)
    if not isinstance(plain, list) or len(plain) != contracts.CHUNK_SIZE:
        raise SchemaContractError(
            f"br_action_is_pad: expected length {contracts.CHUNK_SIZE}, got {plain}"
        )
    for idx, flag in enumerate(plain):
        if flag not in (False, 0, 0.0):
            raise SchemaContractError(f"br_action_is_pad[{idx}]: expected False pad")
    return [bool(flag) for flag in plain]


def validate_canary_obs(obs, expected_dataname: Optional[str] = None):
    """Validate an online inference bundle. Returns a normalized plain dict.

    Required keys: images, instruction, br_states, br_actions, br_action_is_pad,
    robot_type. ``expected_dataname`` comes from the private frozen manifest.
    """
    if not isinstance(obs, dict):
        raise SchemaContractError("observation bundle must be a dict")

    robot_type = obs.get("robot_type")
    if robot_type != EXPECTED_ROBOT_TYPE:
        raise SchemaContractError(f"robot_type: expected {EXPECTED_ROBOT_TYPE}, got {robot_type!r}")

    dataname = obs.get("dataname")
    if expected_dataname is not None and dataname is not None and dataname != expected_dataname:
        raise SchemaContractError(f"dataname mismatch: {dataname!r} != {expected_dataname!r}")

    instruction = obs.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise SchemaContractError("instruction must be a non-empty string")

    images = _check_images(obs.get("images"))

    states = require_shape(obs.get("br_states"), (contracts.HISTORY_LEN, contracts.STATE45), "br_states")
    states = require_finite(states, "br_states")

    actions = _check_zero_pad_actions(obs.get("br_actions"))
    pad_mask = _check_pad_mask(obs.get("br_action_is_pad"))

    return {
        "dataname": dataname or expected_dataname,
        "instruction": instruction,
        "robot_type": EXPECTED_ROBOT_TYPE,
        "images": images,
        "br_states": states,
        "br_actions": actions,
        "br_action_is_pad": pad_mask,
    }
