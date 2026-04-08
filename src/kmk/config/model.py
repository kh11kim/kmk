from __future__ import annotations

"""YAML-backed gripper config model."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping at top level in {path}")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def normalize_optional_path(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_palm_pose(palm_pose: dict[str, Any] | None) -> dict[str, list[float]]:
    if not isinstance(palm_pose, dict):
        return {"trans": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]}
    trans = [float(v) for v in palm_pose.get("trans", [0.0, 0.0, 0.0])]
    rpy = [float(v) for v in palm_pose.get("rpy", [0.0, 0.0, 0.0])]
    if len(trans) != 3 or len(rpy) != 3:
        raise ValueError("palm_pose must contain trans/rpy with 3 values each")
    return {"trans": trans, "rpy": rpy}


def normalize_palm_points_delta(value: Any | None) -> float:
    if value is None:
        return 0.05
    try:
        delta = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("palm_points_delta must be a numeric value") from exc
    if delta < 0.0:
        raise ValueError("palm_points_delta must be non-negative")
    return delta


def normalize_collision_ignore_pairs(
    pairs: Iterable[Sequence[str]] | None,
) -> list[list[str]]:
    if pairs is None:
        return []
    normalized: list[list[str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("collision ignore pair must contain exactly 2 link names")
        first = str(pair[0]).strip()
        second = str(pair[1]).strip()
        if not first or not second:
            raise ValueError("collision ignore pair link names must not be empty")
        if first == second:
            raise ValueError("collision ignore pair link names must be distinct")
        key = tuple(sorted((first, second)))
        if key in seen:
            continue
        seen.add(key)
        normalized.append([key[0], key[1]])
    return normalized


def _normalize_vector3(value: Any, *, field_name: str) -> list[float]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of 3 numeric values")
    try:
        values = [float(v) for v in value]
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a sequence of 3 numeric values") from exc
    if len(values) != 3:
        raise ValueError(f"{field_name} must contain exactly 3 values")
    return values


def _normalize_tags(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of strings")
    try:
        items = list(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a sequence of strings") from exc
    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        tag = str(item).strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        normalized.append(tag)
    return normalized


def _normalize_string_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of strings")
    try:
        items = list(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a sequence of strings") from exc
    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def _normalize_contact_radius(value: Any, *, field_name: str) -> float:
    try:
        radius = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be a numeric value") from exc
    if radius <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return radius


def normalize_contact_anchors(
    anchors: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if anchors is None:
        return {}
    if not isinstance(anchors, Mapping):
        raise TypeError("contact_anchors must be a mapping from link name to anchor data")
    normalized: dict[str, dict[str, Any]] = {}
    for raw_link_name, raw_anchor in anchors.items():
        link_name = str(raw_link_name).strip()
        if not link_name:
            raise ValueError("contact_anchors keys must be non-empty link names")
        if link_name in normalized:
            raise ValueError(f"duplicate contact anchor for link {link_name}")
        if not isinstance(raw_anchor, Mapping):
            raise TypeError(f"contact_anchors[{link_name!r}] must be a mapping")
        anchor = dict(raw_anchor)
        anchor["point"] = _normalize_vector3(anchor.get("point"), field_name=f"contact_anchors[{link_name!r}].point")
        anchor["contact_radius"] = _normalize_contact_radius(
            anchor.get("contact_radius", 0.007),
            field_name=f"contact_anchors[{link_name!r}].contact_radius",
        )
        anchor["tags"] = _normalize_tags(anchor.get("tags"), field_name=f"contact_anchors[{link_name!r}].tags")
        normalized[link_name] = anchor
    return normalized


def normalize_grasp_template(entry: Mapping[str, Any] | None) -> dict[str, Any]:
    if entry is None:
        return {}
    if not isinstance(entry, Mapping):
        raise TypeError("grasp template entry must be a mapping")

    normalized = dict(entry)
    if "q_close" in normalized:
        normalized["q_close"] = [float(v) for v in normalized["q_close"]]
    if "q_open" in normalized:
        normalized["q_open"] = [float(v) for v in normalized["q_open"]]
    if "q_close" not in normalized and "q_open" in normalized:
        normalized["q_close"] = list(normalized["q_open"])
    if "q_open" not in normalized and "q_close" in normalized:
        normalized["q_open"] = list(normalized["q_close"])
    if "q_open" not in normalized and "q_close" not in normalized:
        normalized["q_open"] = []
        normalized["q_close"] = []
    if "grasp_target_point" in normalized:
        normalized["grasp_target_point"] = _normalize_vector3(
            normalized["grasp_target_point"],
            field_name="grasp_target_point",
        )
    elif "grasp_target" in normalized:
        normalized["grasp_target_point"] = _normalize_vector3(
            normalized["grasp_target"],
            field_name="grasp_target_point",
        )
    else:
        normalized["grasp_target_point"] = [0.0, 0.0, 0.0]
    if "active_contact_anchors" in normalized:
        normalized["active_contact_anchors"] = _normalize_string_list(
            normalized["active_contact_anchors"],
            field_name="active_contact_anchors",
        )
    elif "active_contact_links" in normalized:
        normalized["active_contact_anchors"] = _normalize_string_list(
            normalized["active_contact_links"],
            field_name="active_contact_anchors",
        )
    else:
        normalized["active_contact_anchors"] = []
    return normalized


def normalize_grasp_templates(
    templates: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if templates is None:
        return {}
    if not isinstance(templates, Mapping):
        raise TypeError("grasp_templates must be a mapping from template name to template data")
    normalized: dict[str, dict[str, Any]] = {}
    for raw_name, raw_template in templates.items():
        template_name = str(raw_name).strip()
        if not template_name:
            raise ValueError("grasp_templates keys must be non-empty template names")
        if template_name in normalized:
            raise ValueError(f"duplicate grasp template for name {template_name}")
        try:
            normalized[template_name] = normalize_grasp_template(raw_template)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"grasp_templates[{template_name}]: {exc}") from exc
    return normalized


@dataclass
class GripperConfig:
    name: str
    urdf_path: str
    xml_path: str | None = None
    joint_order: list[str] = field(default_factory=list)
    q_open: list[float] = field(default_factory=list)
    xml_joint_actuator_alias: dict[str, str] = field(default_factory=dict)
    palm_pose: dict[str, Any] | None = None
    palm_points_delta: float = 0.05
    additional_collision_ignore_pairs: list[list[str]] = field(default_factory=list)
    contact_anchors: dict[str, dict[str, Any]] = field(default_factory=dict)
    grasp_templates: dict[str, dict[str, Any]] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def validate_basics(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be set")
        if not self.urdf_path.strip():
            raise ValueError("urdf_path must be set")
        self.xml_path = normalize_optional_path(self.xml_path)
        self.palm_pose = normalize_palm_pose(self.palm_pose)
        self.palm_points_delta = normalize_palm_points_delta(self.palm_points_delta)
        self.additional_collision_ignore_pairs = normalize_collision_ignore_pairs(
            self.additional_collision_ignore_pairs
        )
        self.contact_anchors = normalize_contact_anchors(self.contact_anchors)
        self.grasp_templates = normalize_grasp_templates(self.grasp_templates)

    def validate(
        self,
        *,
        urdf_actuated_joint_names: Sequence[str] | None = None,
        urdf_link_names: Sequence[str] | None = None,
    ) -> None:
        self.validate_basics()

        joint_count = len(self.joint_order)
        if urdf_actuated_joint_names is not None:
            expected = list(str(name) for name in urdf_actuated_joint_names)
            actual = list(str(name) for name in self.joint_order)
            if sorted(actual) != sorted(expected) or len(actual) != len(expected):
                raise ValueError("joint_order must be a permutation of all actuated URDF joints")
            joint_count = len(expected)
        if joint_count and self.q_open and len(self.q_open) != joint_count:
            raise ValueError("q_open must have the same length as joint_order")

        if self.xml_joint_actuator_alias:
            allowed = set(self.joint_order)
            if not set(self.xml_joint_actuator_alias).issubset(allowed):
                raise ValueError("xml_joint_actuator_alias keys must be joint names from joint_order")
            values = list(self.xml_joint_actuator_alias.values())
            if len(values) != len(set(values)):
                raise ValueError("xml_joint_actuator_alias values must be unique")

        if urdf_link_names is not None:
            link_set = set(str(name) for name in urdf_link_names)
            for pair in self.additional_collision_ignore_pairs:
                first, second = pair
                if first not in link_set or second not in link_set:
                    raise ValueError(
                        "additional_collision_ignore_pairs entries must reference valid URDF link names"
                    )
            for link_name in self.contact_anchors:
                if link_name not in link_set:
                    raise ValueError("contact_anchors keys must be valid URDF link names")
        contact_anchor_names = set(self.contact_anchors)
        if self.grasp_templates:
            for template_name, template in self.grasp_templates.items():
                if not template_name.strip():
                    raise ValueError("grasp template names must be non-empty")
                q_open = template.get("q_open", [])
                q_close = template.get("q_close", [])
                if joint_count and len(q_open) != joint_count:
                    raise ValueError("grasp_templates q_open must have the same length as joint_order")
                if joint_count and len(q_close) != joint_count:
                    raise ValueError("grasp_templates q_close must have the same length as joint_order")
                target_point = template.get("grasp_target_point")
                if not isinstance(target_point, Sequence) or len(target_point) != 3:
                    raise ValueError("grasp_templates grasp_target_point must contain exactly 3 numeric values")
                active_anchors = template.get("active_contact_anchors", [])
                if len(active_anchors) != len(set(active_anchors)):
                    raise ValueError("grasp_templates active_contact_anchors must not contain duplicates")
                if active_anchors and not set(active_anchors).issubset(contact_anchor_names):
                    raise ValueError(
                        "grasp_templates active_contact_anchors entries must reference saved contact_anchors keys"
                    )

    @classmethod
    def load(cls, path: str | Path) -> "GripperConfig":
        data = _read_yaml(Path(path))
        if "contact_anchors" not in data and "contact_points" in data:
            data = dict(data)
            data["contact_anchors"] = data["contact_points"]
        if "grasp_templates" not in data and "grasp_types" in data:
            data = dict(data)
            data["grasp_templates"] = data["grasp_types"]
        known_keys = {
            "name",
            "urdf_path",
            "xml_path",
            "joint_order",
            "q_open",
            "xml_joint_actuator_alias",
            "palm_pose",
            "palm_points_delta",
            "additional_collision_ignore_pairs",
            "contact_anchors",
            "grasp_templates",
        }
        kwargs = {key: data[key] for key in known_keys if key in data}
        extras = {key: value for key, value in data.items() if key not in known_keys}
        cfg = cls(extras=extras, **kwargs)
        cfg.validate_basics()
        return cfg

    def save(self, path: str | Path) -> None:
        self.validate_basics()
        payload: dict[str, Any] = {
            "name": self.name,
            "urdf_path": self.urdf_path,
            "joint_order": list(self.joint_order),
            "palm_pose": self.palm_pose,
            "palm_points_delta": float(self.palm_points_delta),
        }
        if self.q_open:
            payload["q_open"] = [float(v) for v in self.q_open]
        if self.xml_path is not None:
            payload["xml_path"] = self.xml_path
        if self.xml_joint_actuator_alias:
            payload["xml_joint_actuator_alias"] = dict(self.xml_joint_actuator_alias)
        if self.additional_collision_ignore_pairs:
            payload["additional_collision_ignore_pairs"] = [
                list(pair) for pair in self.additional_collision_ignore_pairs
            ]
        if self.contact_anchors:
            payload["contact_anchors"] = {
                link_name: {
                    **anchor,
                    "point": [float(v) for v in anchor["point"]],
                    "contact_radius": float(anchor["contact_radius"]),
                    "tags": list(anchor.get("tags", [])),
                }
                for link_name, anchor in self.contact_anchors.items()
            }
        if self.grasp_templates:
            payload["grasp_templates"] = {
                template_name: {
                    "q_close": [float(v) for v in template.get("q_close", [])],
                    "q_open": [float(v) for v in template.get("q_open", [])],
                    "grasp_target_point": [float(v) for v in template.get("grasp_target_point", [0.0, 0.0, 0.0])],
                    "active_contact_anchors": list(template.get("active_contact_anchors", [])),
                }
                for template_name, template in self.grasp_templates.items()
            }
        payload.update(self.extras)
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_yaml(output_path, payload)
