"""Strict, hardware-independent configuration for the Stretch lab task."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple


class ConfigurationError(ValueError):
    """Raised when a lab configuration is unsafe or internally inconsistent."""


@dataclass(frozen=True)
class StationSpec:
    station_id: str
    label: str
    heading_deg: float


@dataclass(frozen=True)
class ReferenceMarkerSpec:
    marker_id: int
    marker_length_mm: float
    expected_position_by_station_m: Mapping[str, Tuple[float, float, float]]
    tolerance_m: float


@dataclass(frozen=True)
class PlacementSlotSpec:
    slot_id: str
    label: str
    heading_offset_deg: float
    arm_extension_m: float
    lift_height_m: float
    verify_marker_xyz_m: Tuple[float, float, float]
    verify_tolerance_m: float


@dataclass(frozen=True)
class ObjectSpec:
    object_id: str
    label: str
    marker_id: int
    marker_length_mm: float
    source_station: str


@dataclass(frozen=True)
class ActionSpec:
    token: str
    label: str
    role: str
    object_id: str
    source_station: str
    destination_station: str
    requires: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RecipeSpec:
    recipe_id: str
    label: str
    actions: Tuple[str, ...]


@dataclass(frozen=True)
class MotionSpec:
    calibrated: bool = False
    calibration_id: str = ""
    calibration_record: str = ""
    calibration_record_sha256: str = ""
    action_timeout_s: float = 120.0
    command_timeout_s: float = 15.0
    grasp_timeout_s: float = 45.0
    rotation_tolerance_deg: float = 8.0
    min_station_separation_deg: float = 15.0
    placement_settle_s: float = 0.75
    velocity_scale: float = 0.35


@dataclass(frozen=True)
class PerceptionSpec:
    fingertip_marker_length_mm: float = 14.0
    min_marker_pixels: float = 18.0
    max_reprojection_error_px: float = 3.0
    min_marker_depth_m: float = 0.05
    max_marker_depth_m: float = 0.80
    stable_frames: int = 3
    max_position_jump_m: float = 0.035
    max_frame_age_s: float = 0.30
    camera_startup_timeout_s: float = 12.0
    camera_frame_timeout_ms: int = 1000
    postcondition_timeout_s: float = 8.0
    require_both_fingertips: bool = True


@dataclass(frozen=True)
class LabConfig:
    schema_version: int
    tag_dictionary: str
    home_station: str
    fingertip_marker_ids: Tuple[int, int]
    stations: Mapping[str, StationSpec]
    reference_marker: ReferenceMarkerSpec
    placement_slots: Mapping[str, PlacementSlotSpec]
    objects: Mapping[str, ObjectSpec]
    actions: Mapping[str, ActionSpec]
    recipes: Mapping[str, RecipeSpec]
    motion: MotionSpec = MotionSpec()
    perception: PerceptionSpec = PerceptionSpec()
    agent_settings: Mapping[str, Any] = field(default_factory=dict)

    @property
    def action_tokens(self) -> Tuple[str, ...]:
        return tuple(self.actions)

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tag_dictionary": self.tag_dictionary,
            "home_station": self.home_station,
            "fingertip_marker_ids": list(self.fingertip_marker_ids),
            "stations": {
                key: {"label": value.label, "heading_deg": value.heading_deg}
                for key, value in self.stations.items()
            },
            "reference_marker": {
                "marker_id": self.reference_marker.marker_id,
                "marker_length_mm": self.reference_marker.marker_length_mm,
                "expected_position_by_station_m": {
                    key: list(value)
                    for key, value in self.reference_marker.expected_position_by_station_m.items()
                },
                "tolerance_m": self.reference_marker.tolerance_m,
            },
            "placement_slots": {
                key: {
                    "label": value.label,
                    "heading_offset_deg": value.heading_offset_deg,
                    "arm_extension_m": value.arm_extension_m,
                    "lift_height_m": value.lift_height_m,
                    "verify_marker_xyz_m": list(value.verify_marker_xyz_m),
                    "verify_tolerance_m": value.verify_tolerance_m,
                }
                for key, value in self.placement_slots.items()
            },
            "objects": {
                key: {
                    "label": value.label, "marker_id": value.marker_id,
                    "marker_length_mm": value.marker_length_mm,
                    "source_station": value.source_station,
                }
                for key, value in self.objects.items()
            },
            "actions": {
                key: {
                    "label": value.label, "role": value.role,
                    "object_id": value.object_id,
                    "source_station": value.source_station,
                    "destination_station": value.destination_station,
                    "requires": list(value.requires),
                }
                for key, value in self.actions.items()
            },
            "recipes": {
                key: {"label": value.label, "actions": list(value.actions)}
                for key, value in self.recipes.items()
            },
            "motion": asdict(self.motion),
            "perception": asdict(self.perception),
            "agent_settings": dict(self.agent_settings),
        }

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "LabConfig":
        _reject_unknown(raw, {
            "schema_version", "tag_dictionary", "home_station", "fingertip_marker_ids",
            "stations", "reference_marker", "placement_slots", "objects", "actions", "recipes", "motion",
            "perception", "agent_settings",
        }, "configuration")
        if int(raw.get("schema_version", 0)) != 2:
            raise ConfigurationError("schema_version must be 2")
        dictionary = str(raw.get("tag_dictionary", ""))
        if dictionary != "DICT_6X6_250":
            raise ConfigurationError("tag_dictionary must be DICT_6X6_250")

        motion_row = dict(_mapping(raw.get("motion", {}), "motion"))
        _reject_unknown(motion_row, {
            "calibrated", "calibration_id", "action_timeout_s", "command_timeout_s",
            "calibration_record", "calibration_record_sha256",
            "grasp_timeout_s", "rotation_tolerance_deg", "min_station_separation_deg",
            "placement_settle_s", "velocity_scale",
        }, "motion")
        motion = MotionSpec(
            calibrated=_boolean(motion_row.get("calibrated", False), "motion.calibrated"),
            calibration_id=str(motion_row.get("calibration_id", "")).strip(),
            calibration_record=str(motion_row.get("calibration_record", "")).strip(),
            calibration_record_sha256=str(motion_row.get("calibration_record_sha256", "")).strip().lower(),
            action_timeout_s=_bounded(motion_row.get("action_timeout_s", 120.0), 5.0, 600.0, "motion.action_timeout_s"),
            command_timeout_s=_bounded(motion_row.get("command_timeout_s", 15.0), 0.5, 60.0, "motion.command_timeout_s"),
            grasp_timeout_s=_bounded(motion_row.get("grasp_timeout_s", 45.0), 2.0, 180.0, "motion.grasp_timeout_s"),
            rotation_tolerance_deg=_bounded(motion_row.get("rotation_tolerance_deg", 8.0), 0.25, 20.0, "motion.rotation_tolerance_deg"),
            min_station_separation_deg=_bounded(motion_row.get("min_station_separation_deg", 15.0), 1.0, 90.0, "motion.min_station_separation_deg"),
            placement_settle_s=_bounded(motion_row.get("placement_settle_s", 0.75), 0.0, 5.0, "motion.placement_settle_s", inclusive_low=True),
            velocity_scale=_bounded(motion_row.get("velocity_scale", 0.35), 0.05, 1.0, "motion.velocity_scale", inclusive_low=True, inclusive_high=True),
        )
        if motion.action_timeout_s <= motion.grasp_timeout_s:
            raise ConfigurationError("motion.action_timeout_s must exceed motion.grasp_timeout_s")
        if motion.calibrated and not motion.calibration_id:
            raise ConfigurationError("motion.calibration_id is required when calibrated is true")
        if motion.calibrated and not motion.calibration_record:
            raise ConfigurationError("motion.calibration_record is required when calibrated is true")
        if motion.calibrated and (
            len(motion.calibration_record_sha256) != 64
            or any(character not in "0123456789abcdef" for character in motion.calibration_record_sha256)
        ):
            raise ConfigurationError("motion.calibration_record_sha256 must be a SHA-256 hex digest when calibrated is true")
        if motion.calibrated:
            required_motion = {
                "calibrated", "calibration_id", "calibration_record",
                "calibration_record_sha256", "action_timeout_s",
                "command_timeout_s", "grasp_timeout_s",
                "rotation_tolerance_deg", "min_station_separation_deg",
                "placement_settle_s",
                "velocity_scale",
            }
            missing = required_motion - set(motion_row)
            if missing:
                raise ConfigurationError(f"calibrated motion may not use defaults; missing keys: {sorted(missing)}")

        perception_row = dict(_mapping(raw.get("perception", {}), "perception"))
        _reject_unknown(perception_row, {
            "fingertip_marker_length_mm", "min_marker_pixels", "max_reprojection_error_px",
            "min_marker_depth_m", "max_marker_depth_m", "stable_frames",
            "max_position_jump_m", "max_frame_age_s", "camera_startup_timeout_s",
            "camera_frame_timeout_ms", "postcondition_timeout_s", "require_both_fingertips",
        }, "perception")
        perception = PerceptionSpec(
            fingertip_marker_length_mm=_bounded(perception_row.get("fingertip_marker_length_mm", 14.0), 2.0, 100.0, "perception.fingertip_marker_length_mm"),
            min_marker_pixels=_bounded(perception_row.get("min_marker_pixels", 18.0), 4.0, 500.0, "perception.min_marker_pixels"),
            max_reprojection_error_px=_bounded(perception_row.get("max_reprojection_error_px", 3.0), 0.1, 20.0, "perception.max_reprojection_error_px"),
            min_marker_depth_m=_bounded(perception_row.get("min_marker_depth_m", 0.05), 0.01, 2.0, "perception.min_marker_depth_m"),
            max_marker_depth_m=_bounded(perception_row.get("max_marker_depth_m", 0.80), 0.05, 3.0, "perception.max_marker_depth_m"),
            stable_frames=_integer(perception_row.get("stable_frames", 3), 2, 30, "perception.stable_frames"),
            max_position_jump_m=_bounded(perception_row.get("max_position_jump_m", 0.035), 0.001, 0.25, "perception.max_position_jump_m"),
            max_frame_age_s=_bounded(perception_row.get("max_frame_age_s", 0.30), 0.05, 2.0, "perception.max_frame_age_s"),
            camera_startup_timeout_s=_bounded(perception_row.get("camera_startup_timeout_s", 12.0), 1.0, 60.0, "perception.camera_startup_timeout_s"),
            camera_frame_timeout_ms=_integer(perception_row.get("camera_frame_timeout_ms", 1000), 100, 10000, "perception.camera_frame_timeout_ms"),
            postcondition_timeout_s=_bounded(perception_row.get("postcondition_timeout_s", 8.0), 1.0, 60.0, "perception.postcondition_timeout_s"),
            require_both_fingertips=_boolean(perception_row.get("require_both_fingertips", True), "perception.require_both_fingertips"),
        )
        if perception.max_marker_depth_m <= perception.min_marker_depth_m:
            raise ConfigurationError("maximum marker depth must exceed minimum marker depth")
        if motion.calibrated and not perception.require_both_fingertips:
            raise ConfigurationError("live calibrated runs require both fingertip markers")
        if motion.calibrated:
            required_perception = {
                "fingertip_marker_length_mm", "min_marker_pixels",
                "max_reprojection_error_px", "min_marker_depth_m",
                "max_marker_depth_m", "stable_frames", "max_position_jump_m",
                "max_frame_age_s", "camera_startup_timeout_s",
                "camera_frame_timeout_ms", "postcondition_timeout_s",
                "require_both_fingertips",
            }
            missing = required_perception - set(perception_row)
            if missing:
                raise ConfigurationError(f"calibrated perception may not use defaults; missing keys: {sorted(missing)}")

        station_rows = _mapping(raw.get("stations"), "stations")
        stations: dict[str, StationSpec] = {}
        for key, value in station_rows.items():
            row = _mapping(value, f"stations.{key}")
            _reject_unknown(row, {"label", "heading_deg"}, f"stations.{key}")
            stations[str(key)] = StationSpec(
                str(key), str(row.get("label", key)),
                _bounded(row.get("heading_deg"), -180.0, 180.0, f"stations.{key}.heading_deg", inclusive_low=True, inclusive_high=True),
            )
        if not 2 <= len(stations) <= 8:
            raise ConfigurationError("between 2 and 8 stations are required")
        home_station = str(raw.get("home_station", ""))
        if home_station not in stations:
            raise ConfigurationError("home_station must name a configured station")
        _validate_station_separation(stations, motion.min_station_separation_deg)

        fingertip_ids = tuple(_marker_id(value, "fingertip_marker_ids") for value in raw.get("fingertip_marker_ids", ()))
        if len(fingertip_ids) != 2 or len(set(fingertip_ids)) != 2:
            raise ConfigurationError("exactly two distinct fingertip_marker_ids are required")

        reference_row = _mapping(raw.get("reference_marker"), "reference_marker")
        _reject_unknown(reference_row, {
            "marker_id", "marker_length_mm", "expected_position_by_station_m", "tolerance_m",
        }, "reference_marker")
        reference_id = _marker_id(reference_row.get("marker_id"), "reference_marker.marker_id")
        if reference_id in fingertip_ids:
            raise ConfigurationError("reference marker id is reserved for a fingertip marker")
        expected_rows = _mapping(
            reference_row.get("expected_position_by_station_m"),
            "reference_marker.expected_position_by_station_m",
        )
        if set(expected_rows) != set(stations):
            missing = sorted(set(stations) - set(expected_rows))
            extra = sorted(set(expected_rows) - set(stations))
            raise ConfigurationError(
                "reference marker must have one expected position for every station; "
                f"missing={missing}, extra={extra}"
            )
        reference_marker = ReferenceMarkerSpec(
            marker_id=reference_id,
            marker_length_mm=_bounded(
                reference_row.get("marker_length_mm"), 10.0, 300.0,
                "reference_marker.marker_length_mm",
            ),
            expected_position_by_station_m={
                str(key): _vector3(value, f"reference_marker.expected_position_by_station_m.{key}")
                for key, value in expected_rows.items()
            },
            tolerance_m=_bounded(
                reference_row.get("tolerance_m"), 0.005, 0.20,
                "reference_marker.tolerance_m",
            ),
        )

        slot_rows = _mapping(raw.get("placement_slots"), "placement_slots")
        placement_slots: dict[str, PlacementSlotSpec] = {}
        for key, value in slot_rows.items():
            row = _mapping(value, f"placement_slots.{key}")
            _reject_unknown(row, {"label", "heading_offset_deg", "arm_extension_m", "lift_height_m", "verify_marker_xyz_m", "verify_tolerance_m"}, f"placement_slots.{key}")
            placement_slots[str(key)] = PlacementSlotSpec(
                str(key), str(row.get("label", key)),
                _bounded(row.get("heading_offset_deg"), -45.0, 45.0, f"placement_slots.{key}.heading_offset_deg", inclusive_low=True, inclusive_high=True),
                _bounded(row.get("arm_extension_m"), 0.01, 0.45, f"placement_slots.{key}.arm_extension_m"),
                _bounded(row.get("lift_height_m"), 0.30, 1.10, f"placement_slots.{key}.lift_height_m", inclusive_high=True),
                _vector3(row.get("verify_marker_xyz_m"), f"placement_slots.{key}.verify_marker_xyz_m"),
                _bounded(row.get("verify_tolerance_m"), 0.005, 0.15, f"placement_slots.{key}.verify_tolerance_m"),
            )
        if not placement_slots:
            raise ConfigurationError("at least one placement slot is required")

        object_rows = _mapping(raw.get("objects"), "objects")
        objects: dict[str, ObjectSpec] = {}
        marker_ids = {*fingertip_ids, reference_marker.marker_id}
        for key, value in object_rows.items():
            row = _mapping(value, f"objects.{key}")
            _reject_unknown(row, {"label", "marker_id", "marker_length_mm", "source_station"}, f"objects.{key}")
            marker_id = _marker_id(row.get("marker_id"), f"objects.{key}.marker_id")
            if marker_id in marker_ids:
                raise ConfigurationError(f"duplicate or reserved marker id {marker_id}")
            marker_ids.add(marker_id)
            source = str(row.get("source_station", ""))
            if source not in stations:
                raise ConfigurationError(f"objects.{key}.source_station is unknown")
            objects[str(key)] = ObjectSpec(str(key), str(row.get("label", key)), marker_id,
                _bounded(row.get("marker_length_mm"), 2.0, 200.0, f"objects.{key}.marker_length_mm"), source)
        if not objects:
            raise ConfigurationError("at least one tagged object is required")

        action_rows = _mapping(raw.get("actions"), "actions")
        actions: dict[str, ActionSpec] = {}
        action_by_object: dict[str, str] = {}
        for key, value in action_rows.items():
            token = str(key).strip().upper()
            if token != key or not token.replace("_", "").isalnum():
                raise ConfigurationError(f"action token {key!r} must be uppercase alphanumeric/underscore")
            row = _mapping(value, f"actions.{token}")
            _reject_unknown(row, {"label", "role", "object_id", "source_station", "destination_station", "requires"}, f"actions.{token}")
            object_id = str(row.get("object_id", ""))
            if object_id not in objects:
                raise ConfigurationError(f"actions.{token}.object_id is unknown")
            source = str(row.get("source_station", objects[object_id].source_station))
            destination = str(row.get("destination_station", ""))
            if source not in stations or destination not in stations:
                raise ConfigurationError(f"actions.{token} references an unknown station")
            if source != objects[object_id].source_station:
                raise ConfigurationError(f"actions.{token} source disagrees with its object source")
            if source == destination:
                raise ConfigurationError(f"actions.{token} must move between distinct stations")
            if object_id in action_by_object:
                raise ConfigurationError(f"objects.{object_id} is used by both {action_by_object[object_id]} and {token}")
            action_by_object[object_id] = token
            requires = tuple(str(item).strip().upper() for item in row.get("requires", ()))
            actions[token] = ActionSpec(token, str(row.get("label", token)), str(row.get("role", source)), object_id, source, destination, requires)
        if not actions:
            raise ConfigurationError("at least one physical action is required")
        for action in actions.values():
            if set(action.requires) - set(actions) or action.token in action.requires:
                raise ConfigurationError(f"actions.{action.token} has invalid requirements")
        _validate_dependency_graph(actions)

        recipe_rows = _mapping(raw.get("recipes"), "recipes")
        recipes: dict[str, RecipeSpec] = {}
        goal_sets: dict[frozenset[str], str] = {}
        for key, value in recipe_rows.items():
            row = _mapping(value, f"recipes.{key}")
            _reject_unknown(row, {"label", "actions"}, f"recipes.{key}")
            tokens = tuple(str(item).strip().upper() for item in row.get("actions", ()))
            if len(tokens) < 3 or len(tokens) != len(set(tokens)):
                raise ConfigurationError(f"recipes.{key} must contain at least three distinct actions")
            if len(tokens) > len(placement_slots):
                raise ConfigurationError(f"recipes.{key} needs {len(tokens)} placement slots but only {len(placement_slots)} are configured")
            missing = set(tokens) - set(actions)
            if missing:
                raise ConfigurationError(f"recipes.{key} has unknown actions: {sorted(missing)}")
            external = {req for token in tokens for req in actions[token].requires if req not in tokens}
            if external:
                raise ConfigurationError(f"recipes.{key} omits prerequisites: {sorted(external)}")
            goal = frozenset(tokens)
            if goal in goal_sets:
                raise ConfigurationError(f"recipes.{key} duplicates the action set of {goal_sets[goal]}")
            goal_sets[goal] = str(key)
            recipes[str(key)] = RecipeSpec(str(key), str(row.get("label", key)), tokens)
        if not recipes:
            raise ConfigurationError("at least one recipe is required")

        settings = dict(_mapping(raw.get("agent_settings", {}), "agent_settings"))
        return cls(2, dictionary, home_station, fingertip_ids, stations, reference_marker, placement_slots,
                   objects, actions, recipes, motion, perception, settings)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{label} must be a JSON object")
    return value


def _reject_unknown(row: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(row) - allowed
    if unknown:
        raise ConfigurationError(f"{label} has unknown keys: {sorted(unknown)}")


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ConfigurationError(f"{label} must be finite")
    return number


def _bounded(value: Any, low: float, high: float, label: str, *, inclusive_low: bool = False, inclusive_high: bool = False) -> float:
    number = _finite(value, label)
    if not ((number >= low if inclusive_low else number > low) and (number <= high if inclusive_high else number < high)):
        raise ConfigurationError(f"{label} must be in {'[' if inclusive_low else '('}{low}, {high}{']' if inclusive_high else ')'}")
    return number


def _integer(value: Any, low: int, high: int, label: str) -> int:
    if isinstance(value, bool):
        raise ConfigurationError(f"{label} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{label} must be an integer") from exc
    if number != value or not low <= number <= high:
        raise ConfigurationError(f"{label} must be an integer in [{low}, {high}]")
    return number


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{label} must be a JSON boolean")
    return value


def _marker_id(value: Any, label: str) -> int:
    return _integer(value, 0, 249, label)


def _vector3(value: Any, label: str) -> Tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise ConfigurationError(f"{label} must be a three-element array")
    return tuple(_finite(item, f"{label}[{index}]") for index, item in enumerate(value))  # type: ignore[return-value]


def _validate_station_separation(stations: Mapping[str, StationSpec], minimum_deg: float) -> None:
    rows = list(stations.values())
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            separation = abs((left.heading_deg - right.heading_deg + 180.0) % 360.0 - 180.0)
            if separation < minimum_deg:
                raise ConfigurationError(f"stations {left.station_id!r} and {right.station_id!r} are only {separation:.2f} deg apart; minimum is {minimum_deg:.2f} deg")


def _validate_dependency_graph(actions: Mapping[str, ActionSpec]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(token: str) -> None:
        if token in visiting:
            raise ConfigurationError(f"action requirements contain a cycle at {token}")
        if token in visited:
            return
        visiting.add(token)
        for required in actions[token].requires:
            visit(required)
        visiting.remove(token)
        visited.add(token)

    for token in actions:
        visit(token)


def load_lab_config(path: str | Path) -> LabConfig:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, Mapping):
        raise ConfigurationError("configuration root must be a JSON object")
    config = LabConfig.from_mapping(raw)
    if config.motion.calibrated:
        record = Path(config.motion.calibration_record)
        if not record.is_absolute():
            record = source.resolve().parent / record
        if not record.is_file():
            raise ConfigurationError(f"calibration record does not exist: {record}")
        actual = hashlib.sha256(record.read_bytes()).hexdigest()
        if actual != config.motion.calibration_record_sha256:
            raise ConfigurationError("calibration record SHA-256 does not match motion.calibration_record_sha256")
    return config
