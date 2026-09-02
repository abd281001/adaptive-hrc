"""Lazy, revision-checked access to the pinned Burrito environment."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Sequence, Tuple


class SubmoduleMissingError(RuntimeError):
    """Raised when a required pinned repository has not been initialized."""


class PinMismatchError(RuntimeError):
    """Raised when a checked-out upstream repository is not reproducible."""


@dataclass(frozen=True)
class UpstreamPaths:
    """All integration paths, derived relative to this installed source tree."""

    integration_root: Path

    @classmethod
    def discover(cls) -> "UpstreamPaths":
        return cls(Path(__file__).resolve().parents[2])

    @property
    def pins_file(self) -> Path:
        return self.integration_root / "pins.json"

    @property
    def talents_root(self) -> Path:
        return self.integration_root / "third_party" / "talents-zsc"

    @property
    def burrito_src(self) -> Path:
        return self.talents_root / "overcooked" / "src"

    @property
    def overcooked_ai_root(self) -> Path:
        return self.talents_root / "overcooked" / "overcooked_ai"

    @property
    def overcooked_ai_src(self) -> Path:
        return self.overcooked_ai_root / "src"

    @property
    def python_paths(self) -> Tuple[Path, Path]:
        return self.burrito_src, self.overcooked_ai_src


def _load_pins(paths: UpstreamPaths) -> Dict[str, Any]:
    with paths.pins_file.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _head(repository: Path) -> str:
    if not repository.exists():
        raise SubmoduleMissingError(
            f"missing submodule at {repository}; run "
            "'git submodule update --init --recursive' from the project root"
        )
    result = subprocess.run(
        (
            "git", "-C", str(repository), "rev-parse",
            "--show-toplevel", "HEAD",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SubmoduleMissingError(
            f"{repository} is not an initialized Git worktree: "
            f"{result.stderr.strip()}"
        )
    lines = result.stdout.splitlines()
    if len(lines) != 2 or Path(lines[0]).resolve() != repository.resolve():
        raise SubmoduleMissingError(
            f"{repository} exists but is not an initialized submodule worktree"
        )
    return lines[1].strip()


def verify_pins(paths: UpstreamPaths | None = None) -> Dict[str, str]:
    """Verify both authoritative Git links against the revision manifest."""
    resolved = paths or UpstreamPaths.discover()
    pins = _load_pins(resolved)
    repositories = {
        "talents_zsc": resolved.talents_root,
        "overcooked_ai": resolved.overcooked_ai_root,
    }
    heads: Dict[str, str] = {}
    for name, repository in repositories.items():
        actual = _head(repository)
        expected = str(pins[name]["commit"])
        if actual != expected:
            raise PinMismatchError(
                f"{name} is at {actual}, expected pinned revision {expected}"
            )
        heads[name] = actual
    return heads


def _activate_upstream_paths(paths: UpstreamPaths) -> None:
    for source_root in reversed(paths.python_paths):
        if not source_root.exists():
            raise SubmoduleMissingError(
                f"upstream Python source is missing at {source_root}"
            )
        value = str(source_root)
        if value not in sys.path:
            sys.path.insert(0, value)


def _activate_project_root(paths: UpstreamPaths) -> None:
    """Expose the repository's shared ``src`` package to the wrapper runtime."""
    project_root = str(paths.integration_root.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


@dataclass
class BurritoRuntime:
    """Construct pinned upstream environments without importing them globally."""

    paths: UpstreamPaths

    @classmethod
    def discover(cls, *, verify: bool = True) -> "BurritoRuntime":
        paths = UpstreamPaths.discover()
        if verify:
            verify_pins(paths)
        _activate_project_root(paths)
        _activate_upstream_paths(paths)
        return cls(paths)

    def upstream_types(self) -> Tuple[Any, Any, Any]:
        """Return BurritoGridworld, BurritoEnv, and HighLevelActions lazily."""
        _activate_upstream_paths(self.paths)
        from burrito.mdp.burrito_env import BurritoEnv
        from burrito.mdp.burrito_mdp import BurritoGridworld
        from burrito.planners.burrito_planner import HighLevelActions

        return BurritoGridworld, BurritoEnv, HighLevelActions

    def create_environment(
        self,
        layout: str,
        *,
        horizon: int = 400,
        player_types: Sequence[str] = ("H", "A"),
        restrict_capability: bool = True,
        info_level: int = 0,
    ) -> Any:
        """Build the unmodified upstream environment with its macro planner."""
        BurritoGridworld, BurritoEnv, _actions = self.upstream_types()
        mdp = BurritoGridworld.from_layout_name(str(layout))
        if len(player_types) != int(mdp.num_players):
            raise ValueError(
                f"layout {layout!r} has {mdp.num_players} players, but "
                f"{len(player_types)} player types were supplied"
            )
        if any(role not in {"H", "A"} for role in player_types):
            raise ValueError("player_types entries must be 'H' or 'A'")
        env = BurritoEnv.from_mdp(
            mdp, horizon=int(horizon), info_level=int(info_level),
        )
        env.setup_planner(
            str(layout), list(player_types),
            restrict_capability=bool(restrict_capability),
        )
        return env
