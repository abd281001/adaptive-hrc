"""Explicit, single-action client for supervised calibration trials."""
from __future__ import annotations

import argparse
import json
import uuid

from .config import load_lab_config
from .hardware import HttpStretchExecutor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one checked action against a calibration-mode bridge")
    parser.add_argument("--config", required=True)
    parser.add_argument("--hardware-url", default="http://127.0.0.1:9100")
    parser.add_argument("--action", required=True)
    parser.add_argument("--slot", required=True)
    parser.add_argument("--i-understand-this-moves-the-robot", action="store_true")
    parser.add_argument("--human-clear", action="store_true")
    args = parser.parse_args(argv)
    if not args.i_understand_this_moves_the_robot or not args.human_clear:
        raise SystemExit("both motion and human-clear confirmations are required")
    config = load_lab_config(args.config)
    action = config.actions.get(args.action.strip().upper())
    if action is None:
        raise SystemExit(f"unknown action {args.action!r}")
    if args.slot not in config.placement_slots:
        raise SystemExit(f"unknown placement slot {args.slot!r}")
    executor = HttpStretchExecutor(
        args.hardware_url, timeout_s=config.motion.action_timeout_s + 10.0,
    )
    try:
        status = executor.preflight(
            config.digest, require_motion=True, allow_calibration_mode=True,
        )
        if not status.get("calibration_mode"):
            raise SystemExit("probe requires a bridge started with --calibration-mode")
        result = executor.execute(
            action, execution_id=f"calibration-{uuid.uuid4()}",
            placement_slot_id=args.slot, config_digest=config.digest,
        )
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
        return 0 if result.success else 2
    finally:
        executor.close()


if __name__ == "__main__":
    raise SystemExit(main())
