"""Byte-level parity with the symbolic system frozen before integration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.models import MaxEntIrl, Settings
from src.representations import observe_actions


GOLDEN = Path(__file__).with_name("golden") / "symbolic_freeze_v1.json"


def test_frozen_symbolic_maxent_policy_is_unchanged():
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    observations = observe_actions(tuple(expected["actions"]))
    demonstration = [
        *((row.state, row.action) for row in observations),
        (observations[-1].next_state, "stop"),
    ]
    model = MaxEntIrl(Settings(verbose=False, **expected["settings"]))
    model.fit([demonstration], [1.0])

    weights = expected["reward_weights"]
    assert str(model.reward_weights.dtype) == weights["dtype"]
    assert list(model.reward_weights.shape) == weights["shape"]
    assert hashlib.sha256(model.reward_weights.tobytes()).hexdigest() == (
        weights["sha256"]
    )
    assert [
        model.predict(state) for state, _action in demonstration[:-1]
    ] == expected["predictions"]
