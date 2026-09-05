"""Fixed marker-to-fingertip geometry with no robot-only dependencies."""
from __future__ import annotations

import numpy as np


ARUCO_TO_FINGERTIP = {
    "right": np.array([
        [0.3420201433256655, 0.0, -0.9396926207859095, -0.01424778690840783],
        [-0.1793018049349000, 0.9816271834476666, -0.06526051994650435, 0.005896858701036393],
        [0.9224278206486285, 0.1908089953765310, 0.3357362699751409, -0.03078670811201811],
        [0.0, 0.0, 0.0, 1.0],
    ]),
    "left": np.array([
        [-0.3420201433256616, 0.0, 0.9396926207859111, 0.01424778690840872],
        [-0.1793018049348818, -0.9816271834476712, -0.06526051994648710, 0.005896858701031571],
        [0.9224278206486335, -0.1908089953765080, 0.3357362699751403, -0.03078670811202570],
        [0.0, 0.0, 0.0, 1.0],
    ]),
}


def fingertip_transforms() -> dict[str, np.ndarray]:
    """Return independent copies of the fixed transforms."""
    return {side: value.copy() for side, value in ARUCO_TO_FINGERTIP.items()}

