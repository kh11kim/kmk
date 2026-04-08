"""Stage-specific GUI apps for gripper config authoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import time
from typing import Any, Callable

import numpy as np
from scipy.spatial.transform import Rotation as R
from viser.extras import ViserUrdf

from kmk.config.model import (
    GripperConfig,
    normalize_collision_ignore_pairs,
    normalize_contact_anchors,
    normalize_grasp_template,
    normalize_grasp_templates,
    normalize_palm_pose,
)
from kmk.hand_info import PointedHandInfo
from kmk.wizard.session import WizardSession


BUTTON_COLOR_ACTIVE_EDIT = (245, 158, 11)
BUTTON_COLOR_ACTIVE_SET = (34, 197, 94)


def _resolve_path(path: str | Path, *, root: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (root / candidate).resolve()


def _display_relative(path: str | Path, *, root: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _get_widget_text(widget: Any, fallback: str) -> str:
    for attr in ("value", "content"):
        value = getattr(widget, attr, None)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return fallback


def _get_widget_string(widget: Any) -> str:
    return _get_widget_text(widget, "")


def _set_widget_value(widget: Any, value: Any) -> None:
    if hasattr(widget, "value"):
        widget.value = value
        return
    if hasattr(widget, "content"):
        widget.content = value
        return
    raise TypeError(f"Unsupported widget type: {type(widget)!r}")


def _set_widget_content(widget: Any, value: Any) -> None:
    if hasattr(widget, "content"):
        widget.content = value
        return
    if hasattr(widget, "value"):
        widget.value = value
        return
    raise TypeError(f"Unsupported widget type: {type(widget)!r}")


def _set_widget_disabled(widget: Any, disabled: bool) -> None:
    if hasattr(widget, "disabled"):
        widget.disabled = disabled


def _rpy_to_wxyz(rpy_deg: list[float]) -> tuple[float, float, float, float]:
    quat = R.from_euler("ZYX", list(reversed(rpy_deg)), degrees=True).as_quat()
    return (float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2]))


def _wxyz_to_rpy_deg(wxyz: Any) -> list[float]:
    w, x, y, z = [float(v) for v in wxyz]
    ypr = R.from_quat([x, y, z, w]).as_euler("ZYX", degrees=True)
    return [float(ypr[2]), float(ypr[1]), float(ypr[0])]


class _FallbackFrame:
    def __init__(self, position: tuple[float, float, float], wxyz: tuple[float, float, float, float]) -> None:
        self.position = position
        self.wxyz = wxyz
        self.visible = True

    def remove(self) -> None:
        return None


class _FallbackControls(_FallbackFrame):
    def __init__(self, position: tuple[float, float, float], wxyz: tuple[float, float, float, float]) -> None:
        super().__init__(position, wxyz)
        self._callbacks: list[Callable[[Any], None]] = []

    def on_update(self, fn: Callable[[Any], None]) -> Callable[[Any], None]:
        self._callbacks.append(fn)
        return fn

    def trigger_update(
        self,
        *,
        position: tuple[float, float, float] | None = None,
        wxyz: tuple[float, float, float, float] | None = None,
    ) -> None:
        if position is not None:
            self.position = position
        if wxyz is not None:
            self.wxyz = wxyz
        for callback in list(self._callbacks):
            callback(None)


class _FallbackWidget:
    def __init__(self, value: Any) -> None:
        self._value = value
        self.disabled = False
        self._callbacks: list[Callable[[Any], None]] = []

    @property
    def value(self) -> Any:
        return self._value

    @value.setter
    def value(self, value: Any) -> None:
        self._value = value
        for callback in list(self._callbacks):
            callback(None)

    def on_update(self, fn: Callable[[Any], None]) -> Callable[[Any], None]:
        self._callbacks.append(fn)
        return fn

    def trigger_update(self, value: Any) -> None:
        self.value = value
        for callback in list(self._callbacks):
            callback(None)


class _FallbackFolder:
    def __enter__(self) -> "_FallbackFolder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FallbackSphere:
    def __init__(
        self,
        position: tuple[float, float, float],
        radius: float,
        color: tuple[int, int, int] | tuple[float, float, float],
        opacity: float,
    ) -> None:
        self.position = position
        self.radius = radius
        self.color = color
        self.opacity = opacity
        self.visible = True
        self._callbacks: list[Callable[[Any], None]] = []

    def on_click(self, fn: Callable[[Any], None]) -> Callable[[Any], None]:
        self._callbacks.append(fn)
        return fn

    def trigger_click(self) -> None:
        for callback in list(self._callbacks):
            callback(None)

    def remove(self) -> None:
        return None


class _GripperScene:
    def __init__(self, server: Any, *, name: str, urdf_path: str) -> None:
        self.name = name if name.startswith("/") else f"/{name}"
        self.server = server
        self.handle = ViserUrdf(
            target=server,
            urdf_or_path=Path(urdf_path),
            root_node_name=self.name,
        )
        self._order: list[str] = []
        self.lb: dict[str, float] = {}
        self.ub: dict[str, float] = {}
        self.q_dict: dict[str, float] = {}
        self._initialize_joint_pose()

    def _initialize_joint_pose(self) -> None:
        get_limits = getattr(self.handle, "get_actuated_joint_limits", None)
        update_cfg = getattr(self.handle, "update_cfg", None)
        if not callable(get_limits) or not callable(update_cfg):
            return

        ordered_q: list[float] = []
        for joint_name, (lower, upper) in get_limits().items():
            self._order.append(joint_name)
            lower = float(lower)
            upper = float(upper)
            self.lb[joint_name] = lower
            self.ub[joint_name] = upper
            if np.isfinite(lower) and np.isfinite(upper):
                value = (lower + upper) / 2.0
            elif np.isfinite(lower):
                value = lower
            elif np.isfinite(upper):
                value = upper
            else:
                value = 0.0
            self.q_dict[joint_name] = value
            ordered_q.append(value)

        if ordered_q:
            update_cfg(np.asarray(ordered_q, dtype=float))

    @property
    def joint_order(self) -> list[str]:
        return list(self._order)

    def set_joint_angles(self, q: list[float] | np.ndarray | tuple[float, ...]) -> None:
        update_cfg = getattr(self.handle, "update_cfg", None)
        if not callable(update_cfg):
            return
        q_array = np.asarray(q, dtype=float)
        if q_array.shape[0] != len(self._order):
            raise ValueError("joint angle vector length must match actuated joint count")
        for joint_name, value in zip(self._order, q_array.tolist()):
            self.q_dict[joint_name] = float(value)
        update_cfg(q_array)


def _normalize_pair(first: str, second: str) -> tuple[str, str]:
    first = first.strip()
    second = second.strip()
    if not first or not second:
        raise ValueError("collision ignore pair links must not be empty")
    if first == second:
        raise ValueError("collision ignore pair links must be distinct")
    return tuple(sorted((first, second)))


def _current_joint_values(gripper: Any, joint_order: list[str]) -> list[float]:
    q_dict = getattr(gripper, "q_dict", {})
    return [float(q_dict.get(joint_name, 0.0)) for joint_name in joint_order]


def _canonical_joint_order(config: GripperConfig, gripper: Any) -> list[str]:
    if config.joint_order:
        return list(config.joint_order)
    return list(getattr(gripper, "joint_order", []))


def _gripper_joint_order(config: GripperConfig, gripper: Any) -> list[str]:
    order = list(getattr(gripper, "joint_order", []))
    return order if order else list(config.joint_order)


def _joint_value_map(joint_order: list[str], values: list[float]) -> dict[str, float]:
    return {joint_name: float(value) for joint_name, value in zip(joint_order, values)}


def _vector_for_gripper_order(
    config: GripperConfig,
    gripper: Any,
    canonical_values: list[float],
) -> list[float]:
    canonical_order = _canonical_joint_order(config, gripper)
    value_by_name = _joint_value_map(canonical_order, canonical_values)
    q_dict = getattr(gripper, "q_dict", {}) or {}
    return [
        float(value_by_name.get(joint_name, q_dict.get(joint_name, 0.0)))
        for joint_name in _gripper_joint_order(config, gripper)
    ]


def _clamp_slider_value(value: float, *, lower: float, upper: float) -> float:
    if lower > upper:
        lower, upper = upper, lower
    return min(max(float(value), float(lower)), float(upper))


@dataclass
class _AppContext:
    session: WizardSession | None
    config: GripperConfig
    save_path: Path
    server: Any
    gripper: Any
    root_path: Path


def _prepare_context(
    session_or_config: WizardSession | GripperConfig,
    *,
    save_path: str | Path | None,
    server: Any | None,
    gripper_factory: Callable[..., Any] | None,
    host: str,
    port: int,
) -> _AppContext:
    if server is None:
        import viser

        server = viser.ViserServer(host=host, port=port)

    if isinstance(session_or_config, WizardSession):
        session = session_or_config
        config = session.config
        root_path = session.gripper_root_abs
        resolved_save_path = Path(save_path).expanduser().resolve() if save_path is not None else session.save_path
    else:
        session = None
        config = session_or_config
        root_path = Path.cwd()
        resolved_save_path = (
            Path(save_path).expanduser().resolve()
            if save_path is not None
            else Path.cwd() / f"{config.name}.yaml"
        )

    config.palm_pose = normalize_palm_pose(config.palm_pose)
    config.additional_collision_ignore_pairs = normalize_collision_ignore_pairs(
        config.additional_collision_ignore_pairs
    )
    config.contact_anchors = normalize_contact_anchors(config.contact_anchors)

    urdf_path = _resolve_path(config.urdf_path, root=root_path)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file does not exist: {urdf_path}")

    if gripper_factory is None:
        gripper = _GripperScene(server, name=config.name, urdf_path=str(urdf_path))
    else:
        gripper = gripper_factory(urdf_path=str(urdf_path), name=f"/{config.name}")

    if hasattr(gripper, "joint_order") and hasattr(gripper, "q_dict"):
        canonical_order = _canonical_joint_order(config, gripper)
        if config.q_open and len(config.q_open) == len(canonical_order):
            set_joint_angles = getattr(gripper, "set_joint_angles", None)
            if callable(set_joint_angles):
                set_joint_angles(_vector_for_gripper_order(config, gripper, list(config.q_open)))
                for joint_name, value in zip(canonical_order, config.q_open):
                    if isinstance(getattr(gripper, "q_dict", None), dict):
                        gripper.q_dict[joint_name] = float(value)
        elif not config.q_open:
            config.q_open = _current_joint_values(gripper, canonical_order)

    _configure_initial_camera(server, config)

    return _AppContext(
        session=session,
        config=config,
        save_path=resolved_save_path,
        server=server,
        gripper=gripper,
        root_path=root_path,
    )


def _known_link_names(session: WizardSession | None, gripper: Any) -> set[str]:
    names: set[str] = set()
    if session is not None:
        names.update(str(name) for name in session.urdf.link_names)
    urdf = getattr(getattr(gripper, "handle", None), "_urdf", None)
    link_map = getattr(urdf, "link_map", None)
    if isinstance(link_map, dict):
        names.update(str(name) for name in link_map)
    return names


def _mesh_link_name(session: WizardSession | None, gripper: Any, mesh_handle: Any) -> str:
    name = getattr(mesh_handle, "name", "")
    parts = [part for part in str(name).rstrip("/").split("/") if part]
    if not parts:
        return ""
    known_names = _known_link_names(session, gripper)
    leaf = parts[-1]
    looks_like_mesh_file = bool(re.search(r"\.(?:stl|obj|ply)(?:_\d+)?$", leaf))
    search_parts = parts[:-1] if looks_like_mesh_file and len(parts) > 1 else parts
    if known_names:
        for raw_part in reversed(search_parts):
            if raw_part in known_names:
                return raw_part
            no_ext = re.sub(r"\.(?:stl|obj|ply)$", "", raw_part)
            if no_ext in known_names:
                return no_ext
        for raw_part in reversed(search_parts):
            part = re.sub(r"_(\d+)$", "", raw_part)
            part = re.sub(r"\.(?:stl|obj|ply)$", "", part)
            if part in known_names:
                return part
        leaf_no_ext = re.sub(r"\.(?:stl|obj|ply)$", "", leaf)
        if leaf_no_ext in known_names:
            return leaf_no_ext
    candidate = parts[-2] if len(parts) > 1 and looks_like_mesh_file else leaf
    candidate = re.sub(r"_(\d+)$", "", candidate)
    candidate = re.sub(r"\.(?:stl|obj|ply)$", "", candidate)
    return candidate


def _link_frame_prefix(server: Any, gripper: Any, config: GripperConfig, link_name: str) -> str:
    if hasattr(gripper, "get_viser_frame_name"):
        frame_name = gripper.get_viser_frame_name(link_name)
        if frame_name is not None:
            return frame_name
    scene_handles = getattr(server.scene, "_handle_from_node_name", None)
    if isinstance(scene_handles, dict):
        for node_name in scene_handles:
            if node_name.rstrip("/").split("/")[-1] == link_name:
                return node_name
    return f"/{config.name}/{link_name}"


def _remove_handle(handle: Any | None) -> None:
    if handle is None:
        return
    remove = getattr(handle, "remove", None)
    if callable(remove):
        remove()
        return
    if hasattr(handle, "visible"):
        handle.visible = False


def _set_handle_state(
    handle: Any | None,
    *,
    position: tuple[float, float, float] | None = None,
    radius: float | None = None,
    color: tuple[int, int, int] | tuple[float, float, float] | None = None,
    opacity: float | None = None,
    visible: bool | None = None,
) -> None:
    if handle is None:
        return
    if position is not None and hasattr(handle, "position"):
        handle.position = position
    if radius is not None and hasattr(handle, "radius"):
        handle.radius = radius
    if color is not None and hasattr(handle, "color"):
        handle.color = color
    if opacity is not None and hasattr(handle, "opacity"):
        handle.opacity = opacity
    if visible is not None and hasattr(handle, "visible"):
        handle.visible = visible


def _configure_initial_camera(server: Any, config: GripperConfig) -> None:
    initial_camera = getattr(server, "initial_camera", None)
    if initial_camera is None:
        return
    look_at = np.asarray(config.palm_pose["trans"], dtype=float)
    position = look_at + np.array([0.22, -0.18, 0.16], dtype=float)
    try:
        initial_camera.look_at = tuple(float(v) for v in look_at.tolist())
        initial_camera.position = tuple(float(v) for v in position.tolist())
    except Exception:
        return


@dataclass
class GlobalWizardGui:
    session: WizardSession | None
    config: GripperConfig
    save_path: Path
    server: Any
    gripper: Any
    root_path: Path
    status_widget: Any
    notice_widget: Any
    save_path_widget: Any
    palm_frame: Any
    palm_controls: Any
    palm_trans_widget: Any
    palm_rpy_widget: Any
    edit_palm_pose_button: Any
    set_palm_pose_button: Any
    q_open_summary_widget: Any
    q_open_joint_widgets: dict[str, Any]
    set_q_open_button: Any
    collision_summary_widget: Any
    collision_notice_widget: Any
    collision_preview_widget: Any
    collision_selected_pair_widget: Any
    add_collision_pair_button: Any
    set_collision_pair_button: Any
    delete_collision_pair_button: Any
    save_and_continue_button: Any
    validation_text: str = ""
    last_notice: str | None = None
    syncing_pose: bool = False
    waiting_for_collision_click: bool = False
    collision_pair_first_link: str | None = None
    collision_pair_preview: tuple[str, str] | None = None
    finished: bool = False

    def _pair_label(self, pair: tuple[str, str] | list[str]) -> str:
        first, second = pair
        return f"{first} / {second}"

    def _parse_pair_label(self, label: str) -> tuple[str, str] | None:
        text = label.strip()
        if not text or " / " not in text:
            return None
        first, second = text.split(" / ", 1)
        return _normalize_pair(first, second)

    def refresh_status(self) -> None:
        pair_labels = [self._pair_label(pair) for pair in self.config.additional_collision_ignore_pairs]
        preview_text = "none"
        if self.collision_pair_preview is not None:
            preview_text = (
                "**Ready to add**\n\n"
                f"`{self.collision_pair_preview[0]}` / `{self.collision_pair_preview[1]}`"
            )
        elif self.waiting_for_collision_click and self.collision_pair_first_link is None:
            preview_text = "**Waiting**\n\nclick the first link"
        elif self.waiting_for_collision_click and self.collision_pair_first_link is not None:
            preview_text = (
                "**Waiting**\n\n"
                f"first link: `{self.collision_pair_first_link}`\n"
                "second: click another link"
            )

        lines = [
            "## Global",
            f"- mode: `{self.session.mode if self.session is not None else 'config'}`",
            f"- name: `{self.config.name}`",
            f"- urdf: `{self.config.urdf_path}`",
            f"- xml: `{self.config.xml_path or 'none'}`",
            f"- joint_count: `{len(self.config.joint_order)}`",
            f"- q_open_dim: `{len(self.config.q_open)}`",
            f"- save_path: `{_display_relative(self.save_path, root=self.root_path)}`",
            f"- collision_ignore_pairs: `{len(pair_labels)}`",
        ]
        self.validation_text = "\n".join(lines)
        _set_widget_content(self.status_widget, self.validation_text)
        _set_widget_content(self.notice_widget, self.last_notice or "")
        q_open_lines = [
            f"- `{joint_name}`: `{float(value):.4f}`"
            for joint_name, value in zip(self.q_open_joint_widgets, self.config.q_open)
        ]
        _set_widget_content(self.q_open_summary_widget, "\n".join(q_open_lines) if q_open_lines else "none")
        _set_widget_content(self.collision_preview_widget, preview_text)
        _set_widget_content(
            self.collision_summary_widget,
            "\n".join(f"- `{label}`" for label in pair_labels) if pair_labels else "none",
        )
        if hasattr(self.collision_selected_pair_widget, "options"):
            self.collision_selected_pair_widget.options = tuple(pair_labels) if pair_labels else ("",)
        current_value = _get_widget_string(self.collision_selected_pair_widget)
        if current_value not in pair_labels:
            _set_widget_value(self.collision_selected_pair_widget, pair_labels[0] if pair_labels else "")
        _set_widget_value(self.save_path_widget, str(self.save_path))

    def start_palm_pose_edit(self) -> None:
        if hasattr(self.palm_controls, "visible"):
            self.palm_controls.visible = True

    def finish_palm_pose_edit(self) -> None:
        if hasattr(self.palm_controls, "visible"):
            self.palm_controls.visible = False

    def start_collision_pair_selection(self) -> None:
        self.waiting_for_collision_click = True
        self.collision_pair_first_link = None
        self.collision_pair_preview = None
        self.last_notice = "collision ignore pair: click the first link"
        self.refresh_status()

    def register_collision_link_click(self, link_name: str) -> None:
        if not self.waiting_for_collision_click:
            return
        if self.collision_pair_first_link is None:
            self.collision_pair_first_link = link_name
            self.last_notice = f"collision ignore pair: first link = {link_name}; click the second link"
            self.refresh_status()
            return
        try:
            pair = _normalize_pair(self.collision_pair_first_link, link_name)
        except ValueError:
            self.last_notice = "collision ignore pair: pick a different second link"
            self.refresh_status()
            return
        self.waiting_for_collision_click = False
        self.collision_pair_first_link = None
        self.collision_pair_preview = pair
        self.last_notice = f"collision ignore pair ready: {pair[0]} / {pair[1]}"
        self.refresh_status()

    def commit_collision_pair(self) -> tuple[str, str]:
        if self.collision_pair_preview is None:
            raise ValueError("collision ignore pair preview must be set")
        pairs = [list(pair) for pair in self.config.additional_collision_ignore_pairs]
        pairs.append(list(self.collision_pair_preview))
        self.config.additional_collision_ignore_pairs = normalize_collision_ignore_pairs(pairs)
        committed = self.collision_pair_preview
        self.collision_pair_preview = None
        self.last_notice = None
        self.refresh_status()
        _set_widget_value(self.collision_selected_pair_widget, self._pair_label(committed))
        return committed

    def delete_collision_pair(self) -> tuple[str, str] | None:
        label = _get_widget_string(self.collision_selected_pair_widget)
        pair = self._parse_pair_label(label)
        if pair is None:
            return None
        self.config.additional_collision_ignore_pairs = [
            list(existing)
            for existing in self.config.additional_collision_ignore_pairs
            if tuple(existing) != pair
        ]
        self.last_notice = "deleted!"
        self.refresh_status()
        return pair

    def save(self) -> Path:
        self.config.palm_pose = {
            "trans": [float(v) for v in self.palm_trans_widget.value],
            "rpy": [float(v) for v in self.palm_rpy_widget.value],
        }
        self.config.q_open = [float(widget.value) for widget in self.q_open_joint_widgets.values()]
        self.config.additional_collision_ignore_pairs = normalize_collision_ignore_pairs(
            self.config.additional_collision_ignore_pairs
        )
        self.config.save(self.save_path)
        self.last_notice = "saved!"
        self.refresh_status()
        return self.save_path

    def commit_q_open(self) -> list[float]:
        self.config.q_open = [float(widget.value) for widget in self.q_open_joint_widgets.values()]
        self.last_notice = "q_open set!"
        self.refresh_status()
        return list(self.config.q_open)

    def save_and_continue(self) -> Path:
        path = self.save()
        self.finished = True
        shutdown = getattr(self.server, "stop", None) or getattr(self.server, "shutdown", None)
        if callable(shutdown):
            shutdown()
        return path

    def run_until_complete(self, sleep_seconds: float = 0.1) -> Path:
        while not self.finished:
            time.sleep(sleep_seconds)
        return self.save_path


@dataclass
class KeypointWizardGui:
    session: WizardSession | None
    config: GripperConfig
    save_path: Path
    server: Any
    gripper: Any
    root_path: Path
    status_widget: Any
    notice_widget: Any
    save_path_widget: Any
    contact_summary_widget: Any
    contact_notice_widget: Any
    contact_preview_widget: Any
    contact_selected_link_widget: Any
    contact_delete_selected_widget: Any
    contact_point_widget: Any
    contact_radius_widget: Any
    contact_tags_widget: Any
    add_contact_anchor_button: Any
    set_contact_anchor_button: Any
    delete_contact_anchor_button: Any
    save_and_continue_button: Any
    validation_text: str = ""
    last_notice: str | None = None
    waiting_for_contact_anchor_click: bool = False
    contact_anchor_active_link_name: str | None = None
    contact_anchor_draft_gizmo: Any | None = None
    contact_anchor_draft_sphere: Any | None = None
    contact_anchor_saved_spheres: dict[str, Any] = field(default_factory=dict)
    syncing_contact_anchor: bool = False
    finished: bool = False

    def _known_link_names(self) -> set[str]:
        return _known_link_names(self.session, self.gripper)

    def _anchor_label(self, link_name: str, tags: list[str]) -> str:
        suffix = f" [{', '.join(tags)}]" if tags else ""
        return f"{link_name}{suffix}"

    def _saved_anchor_options(self) -> list[str]:
        return [link_name for link_name in sorted(self.config.contact_anchors)]

    def _parse_anchor_tags(self, raw: str) -> list[str]:
        tags: list[str] = []
        seen: set[str] = set()
        for token in raw.split(","):
            tag = token.strip()
            if not tag or tag in seen:
                continue
            seen.add(tag)
            tags.append(tag)
        return tags

    def _normalize_anchor_point(self, value: Any) -> list[float]:
        point = [float(v) for v in value]
        if len(point) != 3:
            raise ValueError("contact anchor point must contain exactly 3 values")
        return point

    def _contact_anchor_entry(self, link_name: str) -> dict[str, Any]:
        entry = self.config.contact_anchors.get(link_name, {})
        if not isinstance(entry, dict):
            return {"point": [0.0, 0.0, 0.0], "contact_radius": 0.007, "tags": []}
        point = entry.get("point", [0.0, 0.0, 0.0])
        tags = entry.get("tags", [])
        return {
            "point": self._normalize_anchor_point(point),
            "contact_radius": float(entry.get("contact_radius", 0.007)),
            "tags": self._parse_anchor_tags(",".join(str(tag) for tag in tags)),
        }

    def _load_contact_anchor_into_widgets(self, link_name: str) -> None:
        entry = self._contact_anchor_entry(link_name)
        self.syncing_contact_anchor = True
        try:
            _set_widget_value(self.contact_selected_link_widget, link_name)
            _set_widget_value(self.contact_delete_selected_widget, link_name)
            _set_widget_value(self.contact_point_widget, tuple(entry.get("point", [0.0, 0.0, 0.0])))
            _set_widget_value(self.contact_radius_widget, float(entry.get("contact_radius", 0.007)))
            _set_widget_value(self.contact_tags_widget, ", ".join(entry.get("tags", [])))
        finally:
            self.syncing_contact_anchor = False

    def _clear_contact_anchor_draft(self) -> None:
        if self.contact_anchor_active_link_name in self.contact_anchor_saved_spheres:
            _set_handle_state(
                self.contact_anchor_saved_spheres[self.contact_anchor_active_link_name],
                visible=True,
            )
        _remove_handle(self.contact_anchor_draft_gizmo)
        _remove_handle(self.contact_anchor_draft_sphere)
        self.contact_anchor_draft_gizmo = None
        self.contact_anchor_draft_sphere = None
        self.contact_anchor_active_link_name = None

    def _ensure_saved_contact_anchor_sphere(self, link_name: str, entry: dict[str, Any]) -> None:
        prefix = _link_frame_prefix(self.server, self.gripper, self.config, link_name)
        position = tuple(self._normalize_anchor_point(entry.get("point", [0.0, 0.0, 0.0])))
        radius = float(entry.get("contact_radius", 0.007))
        sphere = self.contact_anchor_saved_spheres.get(link_name)
        if sphere is None:
            add_icosphere = getattr(self.server.scene, "add_icosphere", None)
            if callable(add_icosphere):
                sphere = add_icosphere(
                    f"{prefix}/contact_anchor",
                    radius=radius,
                    color=(0.2, 0.7, 1.0),
                    opacity=0.75,
                    position=position,
                )
            else:
                sphere = _FallbackSphere(position, radius, (51, 178, 255), 0.75)
            self.contact_anchor_saved_spheres[link_name] = sphere
            on_click = getattr(sphere, "on_click", None)
            if callable(on_click):

                @sphere.on_click
                def _(_: Any, clicked_link_name: str = link_name) -> None:
                    self._load_contact_anchor_into_widgets(clicked_link_name)
                    self.start_contact_anchor_draft(clicked_link_name)
        else:
            _set_handle_state(
                sphere,
                position=position,
                radius=radius,
                color=(0.2, 0.7, 1.0),
                opacity=0.75,
                visible=True,
            )

    def _sync_contact_anchor_draft_visuals(
        self,
        *,
        point: tuple[float, float, float] | None = None,
        radius: float | None = None,
    ) -> None:
        if self.contact_anchor_active_link_name is None:
            return
        if point is None:
            point = tuple(self._normalize_anchor_point(self.contact_point_widget.value))
        if radius is None:
            radius = float(self.contact_radius_widget.value)
        _set_handle_state(self.contact_anchor_draft_gizmo, position=point)
        _set_handle_state(
            self.contact_anchor_draft_sphere,
            position=point,
            radius=radius,
            color=(255, 153, 0),
            opacity=0.9,
            visible=True,
        )
        if self.contact_anchor_active_link_name in self.contact_anchor_saved_spheres:
            _set_handle_state(
                self.contact_anchor_saved_spheres[self.contact_anchor_active_link_name],
                visible=False,
            )

    def _bind_contact_anchor_gizmo(self) -> None:
        gizmo = self.contact_anchor_draft_gizmo
        if gizmo is None or not callable(getattr(gizmo, "on_update", None)):
            return

        @gizmo.on_update
        def _(_: Any) -> None:
            if self.syncing_contact_anchor or self.contact_anchor_draft_gizmo is None:
                return
            point = tuple(float(v) for v in self.contact_anchor_draft_gizmo.position)
            self.syncing_contact_anchor = True
            try:
                _set_widget_value(self.contact_point_widget, point)
                self._sync_contact_anchor_draft_visuals(point=point)
            finally:
                self.syncing_contact_anchor = False

    def _begin_contact_anchor_click_selection(self) -> bool:
        meshes = getattr(getattr(self.gripper, "handle", None), "_meshes", None)
        clickable_meshes = [mesh for mesh in meshes or [] if callable(getattr(mesh, "on_click", None))]
        if not clickable_meshes:
            self.waiting_for_contact_anchor_click = False
            return False
        self._clear_contact_anchor_draft()
        self.syncing_contact_anchor = True
        try:
            _set_widget_value(self.contact_selected_link_widget, "")
            _set_widget_value(self.contact_point_widget, (0.0, 0.0, 0.0))
            _set_widget_value(self.contact_radius_widget, 0.007)
            _set_widget_value(self.contact_tags_widget, "")
        finally:
            self.syncing_contact_anchor = False
        self.waiting_for_contact_anchor_click = True
        self.last_notice = "contact anchor: click a mesh to choose a link"
        self.refresh_status()
        return True

    def register_contact_anchor_link_click(self, link_name: str) -> None:
        if not self.waiting_for_contact_anchor_click:
            return
        self.waiting_for_contact_anchor_click = False
        self._load_contact_anchor_into_widgets(link_name)
        self.start_contact_anchor_draft(link_name)

    def start_contact_anchor_draft(self, link_name: str | None = None) -> str:
        link_name = (link_name or _get_widget_string(self.contact_selected_link_widget)).strip()
        if not link_name:
            raise ValueError("selected_link must not be empty")
        known_links = self._known_link_names()
        if known_links and link_name not in known_links:
            raise ValueError("selected_link must be a valid URDF link name")
        existing = self._contact_anchor_entry(link_name)
        self._clear_contact_anchor_draft()
        self.contact_anchor_active_link_name = link_name
        self._load_contact_anchor_into_widgets(link_name)
        prefix = _link_frame_prefix(self.server, self.gripper, self.config, link_name)
        point = tuple(existing.get("point", [0.0, 0.0, 0.0]))
        radius = float(existing.get("contact_radius", 0.007))
        add_controls = getattr(self.server.scene, "add_transform_controls", None)
        if callable(add_controls):
            self.contact_anchor_draft_gizmo = add_controls(
                f"{prefix}/contact_anchor_draft_gizmo",
                position=point,
                wxyz=(1.0, 0.0, 0.0, 0.0),
                scale=0.05,
                disable_rotations=True,
            )
        else:
            self.contact_anchor_draft_gizmo = _FallbackControls(point, (1.0, 0.0, 0.0, 0.0))
        if self.contact_anchor_draft_gizmo is None:
            self.contact_anchor_draft_gizmo = _FallbackControls(point, (1.0, 0.0, 0.0, 0.0))

        add_icosphere = getattr(self.server.scene, "add_icosphere", None)
        if callable(add_icosphere):
            self.contact_anchor_draft_sphere = add_icosphere(
                f"{prefix}/contact_anchor_draft",
                radius=radius,
                color=(255, 153, 0),
                opacity=0.9,
                position=point,
            )
        else:
            self.contact_anchor_draft_sphere = _FallbackSphere(point, radius, (255, 153, 0), 0.9)
        self._bind_contact_anchor_gizmo()
        self._sync_contact_anchor_draft_visuals(point=point)
        self.last_notice = f"contact anchor editing: {link_name}"
        self.refresh_status()
        return link_name

    def commit_contact_anchor(self) -> str:
        link_name = self.contact_anchor_active_link_name or _get_widget_string(self.contact_selected_link_widget)
        if not link_name:
            raise ValueError("selected_link must not be empty")
        point = self._normalize_anchor_point(self.contact_point_widget.value)
        radius = float(self.contact_radius_widget.value)
        tags = self._parse_anchor_tags(_get_widget_string(self.contact_tags_widget))
        self.config.contact_anchors[link_name] = {
            "point": point,
            "contact_radius": radius,
            "tags": tags,
        }
        self.config.contact_anchors = normalize_contact_anchors(self.config.contact_anchors)
        self._ensure_saved_contact_anchor_sphere(link_name, self.config.contact_anchors[link_name])
        self._clear_contact_anchor_draft()
        self.last_notice = f"contact anchor saved: {link_name}"
        self.refresh_status()
        return link_name

    def delete_contact_anchor(self) -> str | None:
        link_name = _get_widget_string(self.contact_delete_selected_widget)
        if not link_name:
            return None
        self.config.contact_anchors.pop(link_name, None)
        if link_name in self.contact_anchor_saved_spheres:
            _remove_handle(self.contact_anchor_saved_spheres[link_name])
            del self.contact_anchor_saved_spheres[link_name]
        if self.contact_anchor_active_link_name == link_name:
            self._clear_contact_anchor_draft()
        self.syncing_contact_anchor = True
        try:
            _set_widget_value(self.contact_selected_link_widget, "")
            _set_widget_value(self.contact_point_widget, (0.0, 0.0, 0.0))
            _set_widget_value(self.contact_radius_widget, 0.007)
            _set_widget_value(self.contact_tags_widget, "")
        finally:
            self.syncing_contact_anchor = False
        self.last_notice = f"contact anchor deleted: {link_name}"
        self.refresh_status()
        return link_name

    def refresh_status(self) -> None:
        anchor_labels = [
            self._anchor_label(link_name, list(entry.get("tags", [])))
            for link_name, entry in sorted(self.config.contact_anchors.items())
        ]
        preview_text = "none"
        if self.waiting_for_contact_anchor_click:
            preview_text = "**Waiting**\n\nclick a mesh to choose a link"

        lines = [
            "## Keypoints",
            f"- mode: `{self.session.mode if self.session is not None else 'config'}`",
            f"- name: `{self.config.name}`",
            f"- q_open_dim: `{len(self.config.q_open)}`",
            f"- save_path: `{_display_relative(self.save_path, root=self.root_path)}`",
            f"- contact_anchors: `{len(anchor_labels)}`",
        ]
        self.validation_text = "\n".join(lines)
        _set_widget_content(self.status_widget, self.validation_text)
        _set_widget_content(self.notice_widget, self.last_notice or "")
        _set_widget_content(
            self.contact_summary_widget,
            "\n".join(f"- `{label}`" for label in anchor_labels) if anchor_labels else "none",
        )
        _set_widget_content(self.contact_preview_widget, preview_text)
        _set_widget_content(self.contact_notice_widget, self.last_notice or "")
        self.syncing_contact_anchor = True
        try:
            if hasattr(self.contact_delete_selected_widget, "options"):
                options = tuple(self._saved_anchor_options()) if self.config.contact_anchors else ("",)
                self.contact_delete_selected_widget.options = options
            current_delete = _get_widget_string(self.contact_delete_selected_widget)
            saved_options = self._saved_anchor_options()
            if current_delete not in saved_options:
                _set_widget_value(self.contact_delete_selected_widget, saved_options[0] if saved_options else "")
        finally:
            self.syncing_contact_anchor = False
        _set_widget_value(self.save_path_widget, str(self.save_path))

    def save(self) -> Path:
        self.config.contact_anchors = normalize_contact_anchors(self.config.contact_anchors)
        self.config.save(self.save_path)
        self.last_notice = "saved!"
        self.refresh_status()
        return self.save_path

    def save_and_continue(self) -> Path:
        path = self.save()
        self.finished = True
        shutdown = getattr(self.server, "stop", None) or getattr(self.server, "shutdown", None)
        if callable(shutdown):
            shutdown()
        return path

    def run_until_complete(self, sleep_seconds: float = 0.1) -> Path:
        while not self.finished:
            time.sleep(sleep_seconds)
        return self.save_path


@dataclass
class TemplateWizardGui:
    session: WizardSession | None
    config: GripperConfig
    save_path: Path
    server: Any
    gripper: Any
    root_path: Path
    status_widget: Any
    notice_widget: Any
    save_path_widget: Any
    template_summary_widget: Any
    template_notice_widget: Any
    template_mode_widget: Any
    template_active_anchor_summary_widget: Any
    template_target_summary_widget: Any
    template_palm_frame: Any
    template_name_widget: Any
    template_delete_selected_widget: Any
    template_target_widget: Any
    template_joint_widgets: dict[str, Any]
    add_edit_template_button: Any
    edit_q_open_from_q_close_button: Any
    q_open_toggle_button: Any
    q_close_toggle_button: Any
    save_template_button: Any
    delete_template_button: Any
    save_and_continue_button: Any
    validation_text: str = ""
    last_notice: str | None = None
    active_template_name: str | None = None
    template_edit_active: bool = False
    joint_edit_mode: str = "idle"
    syncing_template: bool = False
    syncing_target: bool = False
    template_target_gizmo: Any | None = None
    template_target_sphere: Any | None = None
    template_anchor_spheres: dict[str, Any] = field(default_factory=dict)
    active_contact_anchors: set[str] = field(default_factory=set)
    finished: bool = False

    @property
    def joint_edit_mode_widget(self) -> Any:
        return self.template_mode_widget

    @property
    def edit_q_open_button(self) -> Any:
        return self.q_open_toggle_button

    @property
    def set_q_open_button(self) -> Any:
        return self.q_open_toggle_button

    @property
    def edit_q_close_button(self) -> Any:
        return self.q_close_toggle_button

    @property
    def set_q_close_button(self) -> Any:
        return self.q_close_toggle_button

    @property
    def q_open_joint_widgets(self) -> dict[str, Any]:
        return self.template_joint_widgets

    @property
    def q_close_joint_widgets(self) -> dict[str, Any]:
        return self.template_joint_widgets

    @property
    def saved_contact_anchor_spheres(self) -> dict[str, Any]:
        return self.template_anchor_spheres

    @property
    def active_contact_anchor_names(self) -> list[str]:
        return sorted(self.active_contact_anchors)

    def _joint_order(self) -> list[str]:
        return list(self.template_joint_widgets)

    def _template_names(self) -> list[str]:
        return sorted(self.config.grasp_templates)

    def _refresh_joint_edit_button_styles(self) -> None:
        q_open_color = None
        edit_q_open_from_q_close_color = None
        q_close_color = None
        q_open_label = "Edit q_open"
        q_close_label = "Edit q_close"
        if self.joint_edit_mode == "editing q_open":
            q_open_color = BUTTON_COLOR_ACTIVE_SET
            edit_q_open_from_q_close_color = BUTTON_COLOR_ACTIVE_EDIT
            q_open_label = "Set q_open"
        elif self.joint_edit_mode == "editing q_close":
            q_close_color = BUTTON_COLOR_ACTIVE_SET
            q_close_label = "Set q_close"

        for handle, color, label in (
            (self.q_open_toggle_button, q_open_color, q_open_label),
            (self.edit_q_open_from_q_close_button, edit_q_open_from_q_close_color, "Edit q_open from q_close"),
            (self.q_close_toggle_button, q_close_color, q_close_label),
        ):
            if hasattr(handle, "color"):
                handle.color = color
            if hasattr(handle, "label"):
                handle.label = label

    def _refresh_template_toggle_button_style(self) -> None:
        label = "Save Template" if self.template_edit_active else "Add/Edit Template"
        color = BUTTON_COLOR_ACTIVE_SET if self.template_edit_active else None
        disabled = self.template_edit_active and self.joint_edit_mode != "idle"
        if hasattr(self.add_edit_template_button, "label"):
            self.add_edit_template_button.label = label
        if hasattr(self.add_edit_template_button, "color"):
            self.add_edit_template_button.color = color
        _set_widget_disabled(self.add_edit_template_button, disabled)

    def _template_entry(self, template_name: str) -> dict[str, Any]:
        entry = self.config.grasp_templates.get(template_name, {})
        if not isinstance(entry, dict):
            return normalize_grasp_template(None)
        return normalize_grasp_template(entry)

    def _default_template_entry(self) -> dict[str, Any]:
        joint_order = self._joint_order()
        if len(self.config.q_open) == len(joint_order):
            default_q = [float(value) for value in self.config.q_open]
        else:
            default_q = _current_joint_values(self.gripper, joint_order)
        return {
            "q_open": list(default_q),
            "q_close": list(default_q),
            "grasp_target_point": [0.0, 0.0, 0.0],
            "active_contact_anchors": [],
        }

    def _load_joint_widgets_from_values(
        self,
        values: list[float] | tuple[float, ...],
        *,
        fallback_values: list[float] | tuple[float, ...] | None = None,
    ) -> None:
        if fallback_values is None:
            fallback_values = self._default_template_entry()["q_open"]
        self.syncing_template = True
        try:
            for index, joint_name in enumerate(self._joint_order()):
                if index < len(values):
                    value = float(values[index])
                elif index < len(fallback_values):
                    value = float(fallback_values[index])
                else:
                    value = 0.0
                _set_widget_value(self.template_joint_widgets[joint_name], value)
        finally:
            self.syncing_template = False

    def _palm_frame_prefix(self) -> str:
        return f"/{self.config.name}/template_palm_pose"

    def _template_target_point(self) -> tuple[float, float, float]:
        point = self.template_target_widget.value
        return tuple(float(v) for v in point)

    def _template_target_world_point(
        self,
        local_point: tuple[float, float, float] | None = None,
    ) -> tuple[float, float, float]:
        if local_point is None:
            local_point = self._template_target_point()
        palm_trans = np.asarray(self.config.palm_pose["trans"], dtype=float)
        palm_rot = R.from_euler("ZYX", list(reversed(self.config.palm_pose["rpy"])), degrees=True)
        world = palm_trans + palm_rot.apply(np.asarray(local_point, dtype=float))
        return tuple(float(v) for v in world.tolist())

    def _template_target_local_from_world(
        self,
        world_point: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        palm_trans = np.asarray(self.config.palm_pose["trans"], dtype=float)
        palm_rot = R.from_euler("ZYX", list(reversed(self.config.palm_pose["rpy"])), degrees=True)
        local = palm_rot.inv().apply(np.asarray(world_point, dtype=float) - palm_trans)
        return tuple(float(v) for v in local.tolist())

    def _set_active_contact_anchors(self, anchors: Iterable[str]) -> None:
        self.active_contact_anchors = set(str(anchor) for anchor in anchors if str(anchor).strip())
        if self.active_template_name is None:
            return
        entry = self.config.grasp_templates.setdefault(self.active_template_name, self._default_template_entry())
        entry["active_contact_anchors"] = sorted(self.active_contact_anchors)
        self.config.grasp_templates[self.active_template_name] = normalize_grasp_template(entry)

    def _clear_template_target_draft(self) -> None:
        _remove_handle(self.template_target_gizmo)
        _remove_handle(self.template_target_sphere)
        self.template_target_gizmo = None
        self.template_target_sphere = None

    def _refresh_saved_anchor_styles(self) -> None:
        active = set(self.active_contact_anchors)
        for link_name, sphere in self.template_anchor_spheres.items():
            is_active = link_name in active
            _set_handle_state(
                sphere,
                color=(255, 51, 26) if is_active else (255, 214, 10),
                opacity=1.0 if is_active else 0.35,
                visible=True,
            )

    def _ensure_saved_anchor_sphere(self, link_name: str, entry: dict[str, Any]) -> None:
        prefix = _link_frame_prefix(self.server, self.gripper, self.config, link_name)
        position = tuple(float(v) for v in entry.get("point", [0.0, 0.0, 0.0]))
        radius = float(entry.get("contact_radius", 0.007))
        sphere = self.template_anchor_spheres.get(link_name)
        if sphere is None:
            add_icosphere = getattr(self.server.scene, "add_icosphere", None)
            if callable(add_icosphere):
                sphere = add_icosphere(
                    f"{prefix}/template_anchor",
                    radius=radius,
                    color=(255, 214, 10),
                    opacity=0.35,
                    position=position,
                )
            else:
                sphere = _FallbackSphere(position, radius, (255, 214, 10), 0.35)
            self.template_anchor_spheres[link_name] = sphere

            @sphere.on_click
            def _(_: Any, clicked_link_name: str = link_name) -> None:
                self.toggle_active_contact_anchor(clicked_link_name)
        else:
            _set_handle_state(
                sphere,
                position=position,
                radius=radius,
                color=(255, 214, 10),
                opacity=0.35,
                visible=True,
            )
        self._refresh_saved_anchor_styles()

    def _load_template_into_widgets(self, template_name: str) -> None:
        entry = self._template_entry(template_name)
        self.active_template_name = template_name
        self.active_contact_anchors = set(entry.get("active_contact_anchors", []))
        self.syncing_template = True
        try:
            _set_widget_value(self.template_name_widget, template_name)
            _set_widget_value(self.template_delete_selected_widget, template_name)
            _set_widget_value(self.template_target_widget, tuple(entry.get("grasp_target_point", [0.0, 0.0, 0.0])))
        finally:
            self.syncing_template = False
        self._clear_template_target_draft()
        local_point = self._template_target_point()
        world_point = self._template_target_world_point(local_point)
        add_controls = getattr(self.server.scene, "add_transform_controls", None)
        if callable(add_controls):
            self.template_target_gizmo = add_controls(
                f"/{self.config.name}/grasp_target/{template_name}",
                position=world_point,
                wxyz=_rpy_to_wxyz(self.config.palm_pose["rpy"]),
                scale=0.08,
                disable_rotations=True,
            )
        else:
            self.template_target_gizmo = _FallbackControls(
                world_point,
                _rpy_to_wxyz(self.config.palm_pose["rpy"]),
            )
        add_icosphere = getattr(self.server.scene, "add_icosphere", None)
        if callable(add_icosphere):
            self.template_target_sphere = add_icosphere(
                f"/{self.config.name}/grasp_target/{template_name}/point",
                radius=0.01,
                color=(255, 153, 0),
                opacity=0.9,
                position=(0.0, 0.0, 0.0),
            )
        else:
            self.template_target_sphere = _FallbackSphere(world_point, 0.01, (255, 153, 0), 0.9)
        self._bind_template_target_gizmo()
        self._refresh_saved_anchor_styles()
        self.last_notice = f"template loaded: {template_name}"
        self.refresh_status()

    def _bind_template_target_gizmo(self) -> None:
        gizmo = self.template_target_gizmo
        if gizmo is None or not callable(getattr(gizmo, "on_update", None)):
            return

        @gizmo.on_update
        def _(_: Any) -> None:
            if self.syncing_target or self.template_target_gizmo is None:
                return
            world_point = tuple(float(v) for v in self.template_target_gizmo.position)
            local_point = self._template_target_local_from_world(world_point)
            self.syncing_target = True
            try:
                _set_widget_value(self.template_target_widget, local_point)
                self._sync_template_target_visuals(point=local_point)
                if self.active_template_name is not None:
                    entry = self.config.grasp_templates.get(
                        self.active_template_name,
                        self._default_template_entry(),
                    )
                    entry = normalize_grasp_template(entry)
                    entry["grasp_target_point"] = [float(v) for v in local_point]
                    self.config.grasp_templates[self.active_template_name] = entry
                self.refresh_status()
            finally:
                self.syncing_target = False

    def _sync_template_target_visuals(self, point: tuple[float, float, float] | None = None) -> None:
        if self.template_target_gizmo is None:
            return
        if point is None:
            point = self._template_target_point()
        world_point = self._template_target_world_point(point)
        _set_handle_state(self.template_target_gizmo, position=world_point)
        if isinstance(self.template_target_sphere, _FallbackSphere):
            _set_handle_state(self.template_target_sphere, position=world_point)
        else:
            _set_handle_state(self.template_target_sphere, position=(0.0, 0.0, 0.0))

    def _set_joint_preview(self) -> None:
        joint_order = self._joint_order()
        q = [float(self.template_joint_widgets[name].value) for name in joint_order]
        set_joint_angles = getattr(self.gripper, "set_joint_angles", None)
        if callable(set_joint_angles):
            set_joint_angles(_vector_for_gripper_order(self.config, self.gripper, q))

    def refresh_status(self) -> None:
        template_names = self._template_names()
        active_anchors = sorted(self.active_contact_anchors)
        target_point = ", ".join(f"{value:.4f}" for value in self._template_target_point())
        lines = [
            "## Grasp Templates",
            f"- mode: `{self.session.mode if self.session is not None else 'config'}`",
            f"- template_name: `{self.active_template_name or _get_widget_string(self.template_name_widget) or 'none'}`",
            f"- joint_edit_mode: `{self.joint_edit_mode}`",
            f"- saved_templates: `{len(template_names)}`",
            f"- active_contact_anchors: `{len(active_anchors)}`",
            f"- target_point: `[{target_point}]`",
            f"- save_path: `{_display_relative(self.save_path, root=self.root_path)}`",
        ]
        self.validation_text = "\n".join(lines)
        _set_widget_content(self.status_widget, self.validation_text)
        _set_widget_content(self.notice_widget, self.last_notice or "")
        _set_widget_value(self.template_mode_widget, self.joint_edit_mode)
        self._refresh_joint_edit_button_styles()
        self._refresh_template_toggle_button_style()
        _set_widget_content(
            self.template_summary_widget,
            "\n".join(f"- `{name}`" for name in template_names) if template_names else "none",
        )
        _set_widget_content(
            self.template_active_anchor_summary_widget,
            "\n".join(f"- `{name}`" for name in active_anchors) if active_anchors else "none",
        )
        _set_widget_content(self.template_target_summary_widget, f"`[{target_point}]`")
        previous_sync = self.syncing_template
        self.syncing_template = True
        try:
            if hasattr(self.template_delete_selected_widget, "options"):
                self.template_delete_selected_widget.options = tuple(template_names) if template_names else ("",)
            current_delete = _get_widget_string(self.template_delete_selected_widget)
            if current_delete not in template_names:
                _set_widget_value(self.template_delete_selected_widget, template_names[0] if template_names else "")
        finally:
            self.syncing_template = previous_sync
        _set_widget_value(self.save_path_widget, str(self.save_path))

    def start_template_edit(self) -> str:
        if self.joint_edit_mode != "idle":
            return self.active_template_name or _get_widget_string(self.template_name_widget)
        template_name = _get_widget_string(self.template_name_widget)
        if not template_name:
            raise ValueError("template_name must not be empty")
        is_existing = template_name in self.config.grasp_templates
        entry = self._template_entry(template_name)
        if not is_existing:
            entry = self._default_template_entry()
        self.config.grasp_templates[template_name] = normalize_grasp_template(entry)
        self._load_template_into_widgets(template_name)
        loaded_entry = self._template_entry(template_name)
        q_open_values = loaded_entry.get("q_open", [])
        self._load_joint_widgets_from_values(q_open_values, fallback_values=self._default_template_entry()["q_open"])
        self.template_edit_active = True
        self.joint_edit_mode = "idle"
        self.last_notice = f"template ready: {template_name}"
        self._set_joint_preview()
        self.refresh_status()
        return template_name

    def _start_joint_edit(self, slot: str) -> bool:
        if self.joint_edit_mode != "idle" or self.active_template_name is None:
            return False
        entry = self._template_entry(self.active_template_name)
        values = entry.get(slot, [])
        if not values:
            values = entry.get("q_open", self._default_template_entry()["q_open"])
        default_values = self._default_template_entry()["q_open"]
        self._load_joint_widgets_from_values(values, fallback_values=default_values)
        self.joint_edit_mode = f"editing {slot}"
        self.last_notice = f"editing {slot}"
        self._set_joint_preview()
        self.refresh_status()
        return True

    def _start_q_open_edit_from_q_close(self) -> bool:
        if self.joint_edit_mode != "idle" or self.active_template_name is None:
            return False
        entry = self._template_entry(self.active_template_name)
        values = entry.get("q_close", [])
        if not values:
            values = self._default_template_entry()["q_close"]
        default_values = self._default_template_entry()["q_close"]
        self._load_joint_widgets_from_values(values, fallback_values=default_values)
        self.joint_edit_mode = "editing q_open"
        self.last_notice = "editing q_open from q_close"
        self._set_joint_preview()
        self.refresh_status()
        return True

    def _commit_joint_edit(self, slot: str) -> bool:
        if self.joint_edit_mode != f"editing {slot}" or self.active_template_name is None:
            return False
        joint_order = self._joint_order()
        values = [float(self.template_joint_widgets[name].value) for name in joint_order]
        entry = self.config.grasp_templates.get(self.active_template_name, self._default_template_entry())
        entry = normalize_grasp_template(entry)
        entry[slot] = values
        self.config.grasp_templates[self.active_template_name] = entry
        self.joint_edit_mode = "idle"
        self.last_notice = f"{slot} set"
        self.refresh_status()
        return True

    def toggle_active_contact_anchor(self, link_name: str) -> None:
        if link_name in self.active_contact_anchors:
            self.active_contact_anchors.remove(link_name)
        else:
            self.active_contact_anchors.add(link_name)
        if self.active_template_name is not None:
            entry = self.config.grasp_templates.get(self.active_template_name, self._default_template_entry())
            entry = normalize_grasp_template(entry)
            entry["active_contact_anchors"] = sorted(self.active_contact_anchors)
            self.config.grasp_templates[self.active_template_name] = entry
        self._refresh_saved_anchor_styles()
        self.last_notice = f"toggled anchor: {link_name}"
        self.refresh_status()

    def _delete_template_by_name(self, template_name: str) -> str | None:
        if not template_name:
            return None
        self.config.grasp_templates.pop(template_name, None)
        if self.active_template_name == template_name:
            self.active_template_name = None
            self.template_edit_active = False
            self.active_contact_anchors = set()
            self.joint_edit_mode = "idle"
            self._clear_template_target_draft()
            self.syncing_target = True
            try:
                _set_widget_value(self.template_target_widget, (0.0, 0.0, 0.0))
            finally:
                self.syncing_target = False
            remaining = self._template_names()
            _set_widget_value(self.template_name_widget, remaining[0] if remaining else "")
        self.last_notice = f"template deleted: {template_name}"
        self.refresh_status()
        return template_name

    def save(self) -> Path | None:
        if self.joint_edit_mode in {"editing q_open", "editing q_close"}:
            slot = self.joint_edit_mode.split(" ", 1)[1]
            self.last_notice = f"Finish {slot} editing with Set {slot} first."
            self.refresh_status()
            return None
        target_template_name = _get_widget_string(self.template_name_widget) or self.active_template_name
        if target_template_name:
            source_template_name = self.active_template_name or target_template_name
            entry = self.config.grasp_templates.get(source_template_name, self._default_template_entry())
            entry = normalize_grasp_template(entry)
            entry["grasp_target_point"] = [float(v) for v in self._template_target_point()]
            entry["active_contact_anchors"] = sorted(self.active_contact_anchors)
            self.config.grasp_templates[target_template_name] = entry
            self.active_template_name = target_template_name
        self.config.grasp_templates = normalize_grasp_templates(self.config.grasp_templates)
        self.config.save(self.save_path)
        self.template_edit_active = False
        self.last_notice = "saved!"
        self.refresh_status()
        return self.save_path

    def save_and_continue(self) -> Path:
        path = self.save()
        if path is None:
            return self.save_path
        self.finished = True
        shutdown = getattr(self.server, "stop", None) or getattr(self.server, "shutdown", None)
        if callable(shutdown):
            shutdown()
        return path

    def run_until_complete(self, sleep_seconds: float = 0.1) -> Path:
        while not self.finished:
            time.sleep(sleep_seconds)
        return self.save_path


@dataclass
class PreviewWizardGui:
    session: WizardSession | None
    config: GripperConfig
    save_path: Path
    server: Any
    gripper: Any
    root_path: Path
    hand_info: PointedHandInfo
    status_widget: Any
    notice_widget: Any
    save_path_widget: Any
    template_widget: Any
    q_mode_widget: Any
    palm_aligned_widget: Any
    palm_points_delta_widget: Any
    show_surface_widget: Any
    show_contact_widget: Any
    show_keypoints_widget: Any
    show_palm_widget: Any
    confirmed_button: Any
    palm_frame: Any | None = None
    grasp_target_handle: Any | None = None
    surface_handles: dict[str, Any] = field(default_factory=dict)
    contact_handles: dict[str, Any] = field(default_factory=dict)
    keypoint_handles: dict[str, Any] = field(default_factory=dict)
    last_notice: str | None = None
    validation_text: str = ""
    finished: bool = False

    def _template_name(self) -> str:
        return str(self.template_widget.value)

    def _q_mode(self) -> str:
        return str(self.q_mode_widget.value)

    def _clear_group(self, group: dict[str, Any]) -> None:
        for handle in list(group.values()):
            _remove_handle(handle)
        group.clear()

    def _draw_link_local_clouds(
        self,
        group: dict[str, Any],
        points_by_link: dict[str, np.ndarray],
        *,
        suffix: str,
        color: tuple[int, int, int],
        point_size: float = 0.004,
    ) -> None:
        self._clear_group(group)
        for link_name, points in points_by_link.items():
            prefix = _link_frame_prefix(self.server, self.gripper, self.config, link_name)
            group[link_name] = self.server.scene.add_point_cloud(
                f"{prefix}/{suffix}",
                points=np.asarray(points, dtype=float),
                colors=np.tile(np.asarray(color, dtype=np.uint8)[None, :], (len(points), 1)),
                point_size=point_size,
                point_shape="rounded",
            )

    def _set_q_preview(self) -> None:
        template_name = self._template_name()
        q_mode_name = self._q_mode()
        if template_name == "global" or q_mode_name == "q_open":
            q = self.hand_info.get_q_open(template_name) if template_name != "global" else self.hand_info.get_q_open()
        else:
            q = self.hand_info.get_q_close(template_name)
        set_joint_angles = getattr(self.gripper, "set_joint_angles", None)
        if callable(set_joint_angles):
            set_joint_angles(_vector_for_gripper_order(self.config, self.gripper, q.tolist()))

    def _draw_target(self) -> None:
        _remove_handle(self.grasp_target_handle)
        self.grasp_target_handle = None
        template_name = self._template_name()
        if template_name == "global":
            return
        point = self.hand_info.get_grasp_target_point(template_name)
        self.grasp_target_handle = self.server.scene.add_icosphere(
            f"/{self.config.name}/preview_palm_pose/grasp_target",
            radius=0.008,
            color=(255, 153, 0),
            position=tuple(float(v) for v in point),
        )

    def refresh_status(self) -> None:
        lines = [
            "## Preview",
            f"- template: `{self._template_name()}`",
            f"- q_mode: `{self._q_mode()}`",
            f"- save_path: `{_display_relative(self.save_path, root=self.root_path)}`",
        ]
        self.validation_text = "\n".join(lines)
        _set_widget_content(self.status_widget, self.validation_text)
        _set_widget_content(self.notice_widget, self.last_notice or "")
        _set_widget_value(self.save_path_widget, str(self.save_path))

    def render(self) -> None:
        self._set_q_preview()
        if bool(self.show_surface_widget.value):
            self._draw_link_local_clouds(
                self.surface_handles,
                self.hand_info.surface_points,
                suffix="preview_surface_points",
                color=(120, 170, 255),
            )
        else:
            self._clear_group(self.surface_handles)

        if bool(self.show_contact_widget.value):
            self._draw_link_local_clouds(
                self.contact_handles,
                self.hand_info.contact_points,
                suffix="preview_contact_points",
                color=(255, 80, 80),
            )
        else:
            self._clear_group(self.contact_handles)

        if bool(self.show_keypoints_widget.value):
            keypoints = self.hand_info.get_keypoints(
                template_name=None if self._template_name() == "global" else self._template_name(),
                palm_aligned_points=bool(self.palm_aligned_widget.value),
                palm_points_delta=float(self.palm_points_delta_widget.value),
            )
            self._draw_link_local_clouds(
                self.keypoint_handles,
                keypoints,
                suffix="preview_keypoints",
                color=(255, 214, 10),
            )
        else:
            self._clear_group(self.keypoint_handles)

        if self.palm_frame is not None and hasattr(self.palm_frame, "visible"):
            self.palm_frame.visible = bool(self.show_palm_widget.value)
        self._draw_target()
        self.refresh_status()

    def confirm(self) -> Path:
        self.finished = True
        shutdown = getattr(self.server, "stop", None) or getattr(self.server, "shutdown", None)
        if callable(shutdown):
            shutdown()
        return self.save_path

    def run_until_complete(self, sleep_seconds: float = 0.1) -> Path:
        while not self.finished:
            time.sleep(sleep_seconds)
        return self.save_path


def create_global_app(
    session_or_config: WizardSession | GripperConfig,
    *,
    save_path: str | Path | None = None,
    server: Any | None = None,
    gripper_factory: Callable[..., Any] | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> GlobalWizardGui:
    ctx = _prepare_context(
        session_or_config,
        save_path=save_path,
        server=server,
        gripper_factory=gripper_factory,
        host=host,
        port=port,
    )
    server = ctx.server
    config = ctx.config
    gripper = ctx.gripper

    palm_trans = tuple(config.palm_pose["trans"])
    palm_wxyz = _rpy_to_wxyz(config.palm_pose["rpy"])
    frame = server.scene.add_frame(
        f"/{config.name}/palm_pose",
        position=palm_trans,
        wxyz=palm_wxyz,
        axes_length=0.025,
        axes_radius=0.002,
        origin_radius=0.003,
    )
    if frame is None:
        frame = _FallbackFrame(palm_trans, palm_wxyz)

    add_controls = getattr(server.scene, "add_transform_controls", None)
    if callable(add_controls):
        controls = add_controls(
            f"/{config.name}/palm_pose_controls",
            position=palm_trans,
            wxyz=palm_wxyz,
            scale=0.12,
        )
    else:
        controls = _FallbackControls(palm_trans, palm_wxyz)
    if controls is None:
        controls = _FallbackControls(palm_trans, palm_wxyz)

    save_path_widget = server.gui.add_text("save_path", initial_value=str(ctx.save_path), disabled=True)
    notice_widget = server.gui.add_text("status", initial_value="", disabled=True)
    add_folder = getattr(server.gui, "add_folder", None)
    info_folder = add_folder("Info", expand_by_default=False) if callable(add_folder) else _FallbackFolder()
    palm_folder = add_folder("Palm Pose", expand_by_default=True) if callable(add_folder) else _FallbackFolder()
    q_open_folder = add_folder("Global q_open", expand_by_default=True) if callable(add_folder) else _FallbackFolder()
    collision_folder = add_folder("Collision Ignore Pairs", expand_by_default=True) if callable(add_folder) else _FallbackFolder()
    add_markdown = getattr(server.gui, "add_markdown", None)

    with info_folder:
        status_widget = add_markdown("") if callable(add_markdown) else _FallbackWidget("")
    with palm_folder:
        if callable(add_markdown):
            add_markdown("Convention: `+z = approach direction`, `+y = finger direction`.")
        else:
            server.gui.add_text(
                "palm_pose_note",
                initial_value="Convention: +z = approach direction, +y = finger direction.",
                disabled=True,
            )
        add_vector3 = getattr(server.gui, "add_vector3", None)
        if callable(add_vector3):
            palm_trans_widget = add_vector3("palm_trans", palm_trans, step=0.001)
            palm_rpy_widget = add_vector3("palm_rpy_deg", tuple(config.palm_pose["rpy"]), step=1.0)
        else:
            palm_trans_widget = _FallbackWidget(palm_trans)
            palm_rpy_widget = _FallbackWidget(tuple(config.palm_pose["rpy"]))
        edit_palm_pose_button = server.gui.add_button("Edit Palm Pose")
        set_palm_pose_button = server.gui.add_button("Set Palm Pose")
    with q_open_folder:
        q_open_summary_widget = add_markdown("none") if callable(add_markdown) else _FallbackWidget("none")
        set_q_open_button = server.gui.add_button("Set q_open")
        add_slider = getattr(server.gui, "add_slider", None)
        q_open_joint_widgets: dict[str, Any] = {}
        joint_order = _canonical_joint_order(config, gripper)
        q_dict = getattr(gripper, "q_dict", {})
        lb = getattr(gripper, "lb", {})
        ub = getattr(gripper, "ub", {})
        for index, joint_name in enumerate(joint_order):
            raw_initial_value = (
                float(config.q_open[index]) if index < len(config.q_open) else float(q_dict.get(joint_name, 0.0))
            )
            lower = float(lb.get(joint_name, -1.0))
            upper = float(ub.get(joint_name, 1.0))
            initial_value = _clamp_slider_value(raw_initial_value, lower=lower, upper=upper)
            if callable(add_slider):
                widget = add_slider(
                    f"q_open/{joint_name}",
                    min=lower,
                    max=upper,
                    step=1e-3,
                    initial_value=initial_value,
                )
            else:
                widget = _FallbackWidget(initial_value)
            q_open_joint_widgets[joint_name] = widget
    with collision_folder:
        collision_notice_widget = add_markdown("") if callable(add_markdown) else _FallbackWidget("")
        collision_preview_widget = add_markdown("none") if callable(add_markdown) else _FallbackWidget("none")
        collision_summary_widget = add_markdown("none") if callable(add_markdown) else _FallbackWidget("none")
        add_collision_pair_button = server.gui.add_button("Add Ignore Pair")
        set_collision_pair_button = server.gui.add_button("Set Ignore Pair")
        add_dropdown = getattr(server.gui, "add_dropdown", None)
        if callable(add_dropdown):
            collision_selected_pair_widget = add_dropdown("selected_ignore_pair", ("",))
        else:
            collision_selected_pair_widget = _FallbackWidget("")
            collision_selected_pair_widget.options = ("",)
        delete_collision_pair_button = server.gui.add_button("Delete Ignore Pair")

    save_and_continue_button = server.gui.add_button("Save and Continue")

    app = GlobalWizardGui(
        session=ctx.session,
        config=config,
        save_path=ctx.save_path,
        server=server,
        gripper=gripper,
        root_path=ctx.root_path,
        status_widget=status_widget,
        notice_widget=notice_widget,
        save_path_widget=save_path_widget,
        palm_frame=frame,
        palm_controls=controls,
        palm_trans_widget=palm_trans_widget,
        palm_rpy_widget=palm_rpy_widget,
        edit_palm_pose_button=edit_palm_pose_button,
        set_palm_pose_button=set_palm_pose_button,
        q_open_summary_widget=q_open_summary_widget,
        q_open_joint_widgets=q_open_joint_widgets,
        set_q_open_button=set_q_open_button,
        collision_summary_widget=collision_summary_widget,
        collision_notice_widget=collision_notice_widget,
        collision_preview_widget=collision_preview_widget,
        collision_selected_pair_widget=collision_selected_pair_widget,
        add_collision_pair_button=add_collision_pair_button,
        set_collision_pair_button=set_collision_pair_button,
        delete_collision_pair_button=delete_collision_pair_button,
        save_and_continue_button=save_and_continue_button,
    )
    app.refresh_status()
    app.finish_palm_pose_edit()

    for mesh in getattr(getattr(gripper, "handle", None), "_meshes", []) or []:
        if not callable(getattr(mesh, "on_click", None)):
            continue

        @mesh.on_click
        def _event_click(event: Any, mesh_handle: Any = mesh) -> None:
            target = mesh_handle if event is None else getattr(event, "target", mesh_handle)
            link_name = _mesh_link_name(app.session, app.gripper, target)
            app.register_collision_link_click(link_name)

    def _sync_pose(
        *,
        position: tuple[float, float, float] | None = None,
        rpy_deg: tuple[float, float, float] | None = None,
        wxyz: tuple[float, float, float, float] | None = None,
    ) -> None:
        if app.syncing_pose:
            return
        app.syncing_pose = True
        try:
            if position is None:
                position = tuple(float(v) for v in controls.position)
            if wxyz is None:
                wxyz = tuple(float(v) for v in controls.wxyz)
            if rpy_deg is None:
                rpy_deg = tuple(_wxyz_to_rpy_deg(wxyz))
            controls.position = position
            controls.wxyz = wxyz
            if hasattr(frame, "position"):
                frame.position = position
            if hasattr(frame, "wxyz"):
                frame.wxyz = wxyz
            _set_widget_value(palm_trans_widget, position)
            _set_widget_value(palm_rpy_widget, rpy_deg)
            app.config.palm_pose = {
                "trans": [float(v) for v in position],
                "rpy": [float(v) for v in rpy_deg],
            }
            app.refresh_status()
        finally:
            app.syncing_pose = False

    @controls.on_update
    def _(_: Any) -> None:
        _sync_pose()

    @edit_palm_pose_button.on_click
    def _(_: Any) -> None:
        app.start_palm_pose_edit()

    @set_palm_pose_button.on_click
    def _(_: Any) -> None:
        app.finish_palm_pose_edit()

    @palm_trans_widget.on_update
    def _(_: Any) -> None:
        if app.syncing_pose:
            return
        _sync_pose(position=tuple(float(v) for v in palm_trans_widget.value))

    @palm_rpy_widget.on_update
    def _(_: Any) -> None:
        if app.syncing_pose:
            return
        rpy_deg = tuple(float(v) for v in palm_rpy_widget.value)
        _sync_pose(rpy_deg=rpy_deg, wxyz=_rpy_to_wxyz(list(rpy_deg)))

    def _sync_gripper_q_open_from_widgets() -> None:
        set_joint_angles = getattr(gripper, "set_joint_angles", None)
        if not callable(set_joint_angles):
            return
        joint_names = list(app.q_open_joint_widgets)
        q_open = [float(app.q_open_joint_widgets[name].value) for name in joint_names]
        set_joint_angles(_vector_for_gripper_order(config, gripper, q_open))
        app.last_notice = "q_open preview updated"

    for joint_name, widget in q_open_joint_widgets.items():
        @widget.on_update
        def _(_: Any, current_joint_name: str = joint_name) -> None:
            _ = current_joint_name
            _sync_gripper_q_open_from_widgets()

    @set_q_open_button.on_click
    def _(_: Any) -> None:
        app.commit_q_open()

    @add_collision_pair_button.on_click
    def _(_: Any) -> None:
        app.start_collision_pair_selection()

    @set_collision_pair_button.on_click
    def _(_: Any) -> None:
        app.commit_collision_pair()

    @delete_collision_pair_button.on_click
    def _(_: Any) -> None:
        app.delete_collision_pair()

    @save_and_continue_button.on_click
    def _(_: Any) -> None:
        app.save_and_continue()

    return app


def create_keypoint_app(
    session_or_config: WizardSession | GripperConfig,
    *,
    save_path: str | Path | None = None,
    server: Any | None = None,
    gripper_factory: Callable[..., Any] | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> KeypointWizardGui:
    ctx = _prepare_context(
        session_or_config,
        save_path=save_path,
        server=server,
        gripper_factory=gripper_factory,
        host=host,
        port=port,
    )
    server = ctx.server
    config = ctx.config
    gripper = ctx.gripper

    save_path_widget = server.gui.add_text("save_path", initial_value=str(ctx.save_path), disabled=True)
    notice_widget = server.gui.add_text("status", initial_value="", disabled=True)
    add_folder = getattr(server.gui, "add_folder", None)
    info_folder = add_folder("Info", expand_by_default=False) if callable(add_folder) else _FallbackFolder()
    keypoint_folder = add_folder("Keypoints", expand_by_default=True) if callable(add_folder) else _FallbackFolder()
    add_markdown = getattr(server.gui, "add_markdown", None)

    with info_folder:
        status_widget = add_markdown("") if callable(add_markdown) else _FallbackWidget("")
    with keypoint_folder:
        contact_notice_widget = add_markdown("") if callable(add_markdown) else _FallbackWidget("")
        contact_preview_widget = add_markdown("none") if callable(add_markdown) else _FallbackWidget("none")
        contact_summary_widget = add_markdown("none") if callable(add_markdown) else _FallbackWidget("none")
        if callable(add_markdown):
            add_markdown("Workflow: `Add/Edit Point` -> click link -> adjust point -> set tags -> save.")
            add_markdown("Coordinate frame: `link-local`.")
        else:
            server.gui.add_text(
                "keypoint_note",
                initial_value="Workflow: Add/Edit Point -> click link -> adjust point -> set tags -> save.",
                disabled=True,
            )
            server.gui.add_text(
                "keypoint_frame",
                initial_value="Coordinate frame: link-local.",
                disabled=True,
            )
        add_keypoint_button = server.gui.add_button("Add/Edit Point")
        add_text = getattr(server.gui, "add_text", None)
        add_vector3 = getattr(server.gui, "add_vector3", None)
        add_dropdown = getattr(server.gui, "add_dropdown", None)
        default_anchor_link = next(iter(sorted(config.contact_anchors)), "") if config.contact_anchors else ""
        default_anchor_entry = config.contact_anchors.get(default_anchor_link, {}) if default_anchor_link else {}
        default_anchor_point = tuple(default_anchor_entry.get("point", [0.0, 0.0, 0.0]))
        default_anchor_tags = ", ".join(default_anchor_entry.get("tags", []))
        if callable(add_text):
            contact_selected_link_widget = add_text("selected_link", initial_value=default_anchor_link)
            contact_tags_widget = add_text("tags", initial_value=default_anchor_tags)
        else:
            contact_selected_link_widget = _FallbackWidget(default_anchor_link)
            contact_tags_widget = _FallbackWidget(default_anchor_tags)
        if callable(add_vector3):
            contact_point_widget = add_vector3("contact_anchor_point", default_anchor_point, step=0.001)
        else:
            contact_point_widget = _FallbackWidget(default_anchor_point)
        default_anchor_radius = float(default_anchor_entry.get("contact_radius", 0.007)) if default_anchor_entry else 0.007
        add_slider = getattr(server.gui, "add_slider", None)
        if callable(add_slider):
            contact_radius_widget = add_slider(
                "contact_anchor_radius",
                min=0.001,
                max=0.1,
                step=0.001,
                initial_value=default_anchor_radius,
            )
        else:
            contact_radius_widget = _FallbackWidget(default_anchor_radius)
        set_keypoint_button = server.gui.add_button("Save Point")
        if callable(add_dropdown):
            contact_delete_selected_widget = add_dropdown(
                "saved_points",
                tuple(sorted(config.contact_anchors)) if config.contact_anchors else ("",),
            )
        else:
            contact_delete_selected_widget = _FallbackWidget(default_anchor_link)
            contact_delete_selected_widget.options = (
                tuple(sorted(config.contact_anchors)) if config.contact_anchors else ("",)
            )
        delete_keypoint_button = server.gui.add_button("Delete Point")

    save_and_continue_button = server.gui.add_button("Save and Continue")

    app = KeypointWizardGui(
        session=ctx.session,
        config=config,
        save_path=ctx.save_path,
        server=server,
        gripper=gripper,
        root_path=ctx.root_path,
        status_widget=status_widget,
        notice_widget=notice_widget,
        save_path_widget=save_path_widget,
        contact_summary_widget=contact_summary_widget,
        contact_notice_widget=contact_notice_widget,
        contact_preview_widget=contact_preview_widget,
        contact_selected_link_widget=contact_selected_link_widget,
        contact_delete_selected_widget=contact_delete_selected_widget,
        contact_point_widget=contact_point_widget,
        contact_radius_widget=contact_radius_widget,
        contact_tags_widget=contact_tags_widget,
        add_contact_anchor_button=add_keypoint_button,
        set_contact_anchor_button=set_keypoint_button,
        delete_contact_anchor_button=delete_keypoint_button,
        save_and_continue_button=save_and_continue_button,
    )
    app.refresh_status()
    for link_name, entry in app.config.contact_anchors.items():
        app._ensure_saved_contact_anchor_sphere(link_name, entry)

    for mesh in getattr(getattr(gripper, "handle", None), "_meshes", []) or []:
        if not callable(getattr(mesh, "on_click", None)):
            continue

        @mesh.on_click
        def _event_click(event: Any, mesh_handle: Any = mesh) -> None:
            target = mesh_handle if event is None else getattr(event, "target", mesh_handle)
            link_name = _mesh_link_name(app.session, app.gripper, target)
            app.register_contact_anchor_link_click(link_name)

    @contact_point_widget.on_update
    def _(_: Any) -> None:
        if app.syncing_contact_anchor or app.contact_anchor_active_link_name is None:
            return
        point = tuple(float(v) for v in contact_point_widget.value)
        app.syncing_contact_anchor = True
        try:
            app._sync_contact_anchor_draft_visuals(point=point, radius=float(contact_radius_widget.value))
            _set_widget_value(contact_point_widget, point)
        finally:
            app.syncing_contact_anchor = False

    @contact_radius_widget.on_update
    def _(_: Any) -> None:
        if app.syncing_contact_anchor or app.contact_anchor_active_link_name is None:
            return
        radius = float(contact_radius_widget.value)
        app.syncing_contact_anchor = True
        try:
            app._sync_contact_anchor_draft_visuals(radius=radius)
            _set_widget_value(contact_radius_widget, radius)
        finally:
            app.syncing_contact_anchor = False

    @contact_tags_widget.on_update
    def _(_: Any) -> None:
        if app.syncing_contact_anchor:
            return
        app.refresh_status()

    @contact_selected_link_widget.on_update
    def _(_: Any) -> None:
        if not app.syncing_contact_anchor:
            app.refresh_status()

    @contact_delete_selected_widget.on_update
    def _(_: Any) -> None:
        if not app.syncing_contact_anchor:
            app.refresh_status()

    @add_keypoint_button.on_click
    def _(_: Any) -> None:
        if app._begin_contact_anchor_click_selection():
            return
        app.last_notice = "contact anchor: select a link first"
        app.refresh_status()

    @set_keypoint_button.on_click
    def _(_: Any) -> None:
        app.commit_contact_anchor()

    @delete_keypoint_button.on_click
    def _(_: Any) -> None:
        app.delete_contact_anchor()

    @save_and_continue_button.on_click
    def _(_: Any) -> None:
        app.save_and_continue()

    return app


def create_template_app(
    session_or_config: WizardSession | GripperConfig,
    *,
    save_path: str | Path | None = None,
    server: Any | None = None,
    gripper_factory: Callable[..., Any] | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> TemplateWizardGui:
    ctx = _prepare_context(
        session_or_config,
        save_path=save_path,
        server=server,
        gripper_factory=gripper_factory,
        host=host,
        port=port,
    )
    server = ctx.server
    config = ctx.config
    gripper = ctx.gripper
    palm_trans = tuple(config.palm_pose["trans"])
    palm_wxyz = _rpy_to_wxyz(config.palm_pose["rpy"])

    save_path_widget = server.gui.add_text("save_path", initial_value=str(ctx.save_path), disabled=True)
    notice_widget = server.gui.add_text("status", initial_value="", disabled=True)
    add_folder = getattr(server.gui, "add_folder", None)
    info_folder = add_folder("Info", expand_by_default=False) if callable(add_folder) else _FallbackFolder()
    template_folder = add_folder("Templates", expand_by_default=True) if callable(add_folder) else _FallbackFolder()
    add_markdown = getattr(server.gui, "add_markdown", None)

    template_palm_frame = server.scene.add_frame(
        f"/{config.name}/template_palm_pose",
        position=palm_trans,
        wxyz=palm_wxyz,
        axes_length=0.02,
        axes_radius=0.0015,
        origin_radius=0.003,
    )
    if template_palm_frame is None:
        template_palm_frame = _FallbackFrame(palm_trans, palm_wxyz)

    with info_folder:
        status_widget = add_markdown("") if callable(add_markdown) else _FallbackWidget("")
    with template_folder:
        if callable(add_markdown):
            add_markdown(
                "Workflow: `Add/Edit Template` -> edit `q_open` / `q_close` from idle -> set `grasp_target_point` -> toggle active anchors -> save."
            )
            add_markdown("Joint edit mode: `idle` / `editing q_open` / `editing q_close`.")
            add_markdown("`grasp_target_point` is authored in the palm-pose frame.")
        else:
            server.gui.add_text(
                "template_note",
                initial_value="Workflow: Add/Edit Template -> edit q_open / q_close from idle -> set grasp_target_point -> toggle active anchors -> save.",
                disabled=True,
            )
            server.gui.add_text(
                "template_mode_note",
                initial_value="Joint edit mode: idle / editing q_open / editing q_close.",
                disabled=True,
            )
            server.gui.add_text(
                "template_target_note",
                initial_value="grasp_target_point is authored in the palm-pose frame.",
                disabled=True,
            )
        template_mode_widget = add_markdown("idle") if callable(add_markdown) else _FallbackWidget("idle")
        template_summary_widget = add_markdown("none") if callable(add_markdown) else _FallbackWidget("none")
        template_active_anchor_summary_widget = add_markdown("none") if callable(add_markdown) else _FallbackWidget("none")
        template_target_summary_widget = add_markdown("`[0.0000, 0.0000, 0.0000]`") if callable(add_markdown) else _FallbackWidget("`[0.0000, 0.0000, 0.0000]`")
        add_text = getattr(server.gui, "add_text", None)
        add_vector3 = getattr(server.gui, "add_vector3", None)
        add_dropdown = getattr(server.gui, "add_dropdown", None)
        default_template_name = next(iter(sorted(config.grasp_templates)), "") if config.grasp_templates else ""
        if callable(add_text):
            template_name_widget = add_text("template_name", initial_value=default_template_name)
        else:
            template_name_widget = _FallbackWidget(default_template_name)
        add_edit_template_button = server.gui.add_button("Add/Edit Template")
        if callable(add_vector3):
            template_target_widget = add_vector3("grasp_target_point", (0.0, 0.0, 0.0), step=0.001)
        else:
            template_target_widget = _FallbackWidget((0.0, 0.0, 0.0))
        add_slider = getattr(server.gui, "add_slider", None)
        joint_order = _canonical_joint_order(config, gripper)
        q_dict = getattr(gripper, "q_dict", {})
        lb = getattr(gripper, "lb", {})
        ub = getattr(gripper, "ub", {})
        template_joint_widgets: dict[str, Any] = {}
        for joint_name in joint_order:
            raw_initial_value = float(config.q_open[joint_order.index(joint_name)]) if len(config.q_open) > joint_order.index(joint_name) else float(q_dict.get(joint_name, 0.0))
            lower = float(lb.get(joint_name, -1.0))
            upper = float(ub.get(joint_name, 1.0))
            initial_value = _clamp_slider_value(raw_initial_value, lower=lower, upper=upper)
            if callable(add_slider):
                widget = add_slider(
                    f"template_q/{joint_name}",
                    min=lower,
                    max=upper,
                    step=1e-3,
                    initial_value=initial_value,
                )
            else:
                widget = _FallbackWidget(initial_value)
            template_joint_widgets[joint_name] = widget
        q_close_toggle_button = server.gui.add_button("Edit q_close")
        edit_q_open_from_q_close_button = server.gui.add_button("Edit q_open from q_close")
        q_open_toggle_button = server.gui.add_button("Edit q_open")
        if callable(add_dropdown):
            template_delete_selected_widget = add_dropdown(
                "template_to_delete",
                tuple(sorted(config.grasp_templates)) if config.grasp_templates else ("",),
            )
        else:
            template_delete_selected_widget = _FallbackWidget(default_template_name)
            template_delete_selected_widget.options = tuple(sorted(config.grasp_templates)) if config.grasp_templates else ("",)
        delete_template_button = server.gui.add_button("Delete Template")
    save_and_continue_button = server.gui.add_button("Save and Continue")

    app = TemplateWizardGui(
        session=ctx.session,
        config=config,
        save_path=ctx.save_path,
        server=server,
        gripper=gripper,
        root_path=ctx.root_path,
        status_widget=status_widget,
        notice_widget=notice_widget,
        save_path_widget=save_path_widget,
        template_summary_widget=template_summary_widget,
        template_notice_widget=notice_widget,
        template_mode_widget=template_mode_widget,
        template_active_anchor_summary_widget=template_active_anchor_summary_widget,
        template_target_summary_widget=template_target_summary_widget,
        template_palm_frame=template_palm_frame,
        template_name_widget=template_name_widget,
        template_delete_selected_widget=template_delete_selected_widget,
        template_target_widget=template_target_widget,
        template_joint_widgets=template_joint_widgets,
        add_edit_template_button=add_edit_template_button,
        edit_q_open_from_q_close_button=edit_q_open_from_q_close_button,
        q_open_toggle_button=q_open_toggle_button,
        q_close_toggle_button=q_close_toggle_button,
        save_template_button=add_edit_template_button,
        delete_template_button=delete_template_button,
        save_and_continue_button=save_and_continue_button,
    )

    for link_name, entry in app.config.contact_anchors.items():
        app._ensure_saved_anchor_sphere(link_name, entry)
    app.refresh_status()
    if default_template_name:
        app._load_template_into_widgets(default_template_name)
        app.refresh_status()

    def _sync_target(
        *,
        position: tuple[float, float, float] | None = None,
    ) -> None:
        if app.syncing_target:
            return
        app.syncing_target = True
        try:
            if position is None:
                position = tuple(float(v) for v in template_target_widget.value)
            _set_widget_value(template_target_widget, position)
            if app.active_template_name is not None:
                entry = app.config.grasp_templates.get(app.active_template_name, app._default_template_entry())
                entry = normalize_grasp_template(entry)
                entry["grasp_target_point"] = [float(v) for v in position]
                app.config.grasp_templates[app.active_template_name] = entry
            app._sync_template_target_visuals(point=position)
            app.refresh_status()
        finally:
            app.syncing_target = False

    @template_target_widget.on_update
    def _(_: Any) -> None:
        if app.syncing_target:
            return
        _sync_target(position=tuple(float(v) for v in template_target_widget.value))

    def _sync_joint_preview() -> None:
        if app.syncing_template or app.joint_edit_mode == "idle":
            return
        app._set_joint_preview()

    for joint_name, widget in template_joint_widgets.items():
        @widget.on_update
        def _(_: Any, current_joint_name: str = joint_name) -> None:
            _ = current_joint_name
            _sync_joint_preview()

    @template_name_widget.on_update
    def _(_: Any) -> None:
        if app.syncing_template:
            return
        app.refresh_status()

    @template_delete_selected_widget.on_update
    def _(_: Any) -> None:
        if not app.syncing_template:
            app.refresh_status()

    @add_edit_template_button.on_click
    def _(_: Any) -> None:
        if app.template_edit_active:
            app.save()
        else:
            app.start_template_edit()

    @q_open_toggle_button.on_click
    def _(_: Any) -> None:
        if app.joint_edit_mode == "editing q_open":
            app._commit_joint_edit("q_open")
        else:
            app._start_joint_edit("q_open")

    @edit_q_open_from_q_close_button.on_click
    def _(_: Any) -> None:
        app._start_q_open_edit_from_q_close()

    @q_close_toggle_button.on_click
    def _(_: Any) -> None:
        if app.joint_edit_mode == "editing q_close":
            app._commit_joint_edit("q_close")
        else:
            app._start_joint_edit("q_close")

    @delete_template_button.on_click
    def _(_: Any) -> None:
        app._delete_template_by_name(_get_widget_string(template_delete_selected_widget))

    @save_and_continue_button.on_click
    def _(_: Any) -> None:
        app.save_and_continue()

    return app


def create_preview_app(
    session_or_config: WizardSession | GripperConfig,
    *,
    save_path: str | Path | None = None,
    server: Any | None = None,
    gripper_factory: Callable[..., Any] | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    seed: int = 0,
) -> PreviewWizardGui:
    ctx = _prepare_context(
        session_or_config,
        save_path=save_path,
        server=server,
        gripper_factory=gripper_factory,
        host=host,
        port=port,
    )
    server = ctx.server
    config = ctx.config
    gripper = ctx.gripper
    hand_info = PointedHandInfo.from_config(ctx.save_path, seed=seed)

    palm_pose = hand_info.get_palm_pose()
    palm_frame = server.scene.add_frame(
        f"/{config.name}/preview_palm_pose",
        position=tuple(float(v) for v in palm_pose[:3, 3]),
        wxyz=_rpy_to_wxyz(config.palm_pose["rpy"]),
        axes_length=0.025,
        axes_radius=0.002,
        origin_radius=0.003,
    )
    if palm_frame is None:
        palm_frame = _FallbackFrame(tuple(float(v) for v in palm_pose[:3, 3]), _rpy_to_wxyz(config.palm_pose["rpy"]))

    save_path_widget = server.gui.add_text("save_path", initial_value=str(ctx.save_path), disabled=True)
    notice_widget = server.gui.add_text("status", initial_value="", disabled=True)
    add_folder = getattr(server.gui, "add_folder", None)
    info_folder = add_folder("Info", expand_by_default=False) if callable(add_folder) else _FallbackFolder()
    preview_folder = add_folder("Preview", expand_by_default=True) if callable(add_folder) else _FallbackFolder()
    add_markdown = getattr(server.gui, "add_markdown", None)
    add_dropdown = getattr(server.gui, "add_dropdown", None)
    add_checkbox = getattr(server.gui, "add_checkbox", None)
    add_slider = getattr(server.gui, "add_slider", None)

    with info_folder:
        status_widget = add_markdown("") if callable(add_markdown) else _FallbackWidget("")
    with preview_folder:
        template_options = ("global", *hand_info.template_names)
        if callable(add_dropdown):
            template_widget = add_dropdown("template", template_options)
            q_mode_widget = add_dropdown("q_mode", ("q_open", "q_close"))
        else:
            template_widget = _FallbackWidget("global")
            template_widget.options = template_options
            q_mode_widget = _FallbackWidget("q_open")
            q_mode_widget.options = ("q_open", "q_close")
        _set_widget_value(template_widget, "global")
        _set_widget_value(q_mode_widget, "q_open")
        if callable(add_checkbox):
            show_surface_widget = add_checkbox("show_surface_points", initial_value=True)
            show_contact_widget = add_checkbox("show_contact_points", initial_value=True)
            show_keypoints_widget = add_checkbox("show_keypoints", initial_value=True)
            show_palm_widget = add_checkbox("show_palm_frame", initial_value=True)
            palm_aligned_widget = add_checkbox("palm_aligned_points", initial_value=True)
        else:
            show_surface_widget = _FallbackWidget(True)
            show_contact_widget = _FallbackWidget(True)
            show_keypoints_widget = _FallbackWidget(True)
            show_palm_widget = _FallbackWidget(True)
            palm_aligned_widget = _FallbackWidget(True)
        if callable(add_slider):
            palm_points_delta_widget = add_slider("palm_points_delta", min=0.0, max=0.2, step=1e-3, initial_value=0.05)
        else:
            palm_points_delta_widget = _FallbackWidget(0.05)
        confirmed_button = server.gui.add_button("Confirmed")

    app = PreviewWizardGui(
        session=ctx.session,
        config=config,
        save_path=ctx.save_path,
        server=server,
        gripper=gripper,
        root_path=ctx.root_path,
        hand_info=hand_info,
        status_widget=status_widget,
        notice_widget=notice_widget,
        save_path_widget=save_path_widget,
        template_widget=template_widget,
        q_mode_widget=q_mode_widget,
        palm_aligned_widget=palm_aligned_widget,
        palm_points_delta_widget=palm_points_delta_widget,
        show_surface_widget=show_surface_widget,
        show_contact_widget=show_contact_widget,
        show_keypoints_widget=show_keypoints_widget,
        show_palm_widget=show_palm_widget,
        confirmed_button=confirmed_button,
        palm_frame=palm_frame,
    )
    app.render()

    @template_widget.on_update
    def _(_: Any) -> None:
        app.render()

    @q_mode_widget.on_update
    def _(_: Any) -> None:
        app.render()

    @show_surface_widget.on_update
    def _(_: Any) -> None:
        app.render()

    @show_contact_widget.on_update
    def _(_: Any) -> None:
        app.render()

    @show_keypoints_widget.on_update
    def _(_: Any) -> None:
        app.render()

    @show_palm_widget.on_update
    def _(_: Any) -> None:
        app.render()

    @palm_aligned_widget.on_update
    def _(_: Any) -> None:
        app.render()

    @palm_points_delta_widget.on_update
    def _(_: Any) -> None:
        app.render()

    @confirmed_button.on_click
    def _(_: Any) -> None:
        app.confirm()

    return app


def create_app(
    session_or_config: WizardSession | GripperConfig,
    *,
    save_path: str | Path | None = None,
    server: Any | None = None,
    gripper_factory: Callable[..., Any] | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> GlobalWizardGui:
    return create_global_app(
        session_or_config,
        save_path=save_path,
        server=server,
        gripper_factory=gripper_factory,
        host=host,
        port=port,
    )
