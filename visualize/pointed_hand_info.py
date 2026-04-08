from __future__ import annotations

"""Visualize a YAML config with PointedHandInfo."""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import tyro
import viser
from scipy.spatial.transform import Rotation as R
from viser.extras import ViserUrdf

from kmk import PointedHandInfo


def _wxyz_from_matrix(rot: np.ndarray) -> tuple[float, float, float, float]:
    quat = R.from_matrix(np.asarray(rot, dtype=float)).as_quat()
    return (float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2]))


def _link_frame_prefix(server: Any, root_name: str, link_name: str) -> str:
    scene_handles = getattr(server.scene, "_handle_from_node_name", None)
    if isinstance(scene_handles, dict):
        for node_name in scene_handles:
            if node_name.rstrip("/").split("/")[-1] == link_name:
                return node_name
    return f"{root_name}/{link_name}"


def _remove_handle(handle: Any | None) -> None:
    if handle is None:
        return
    remove = getattr(handle, "remove", None)
    if callable(remove):
        remove()
        return
    if hasattr(handle, "visible"):
        handle.visible = False


@dataclass
class _Handles:
    surface: dict[str, Any]
    contact: dict[str, Any]
    keypoints: dict[str, Any]
    palm_frame: Any | None
    target: Any | None


def visualize(
    config_path: str,
    seed: int = 0,
    host: str = "127.0.0.1",
    port: int = 8080,
    point_size: float = 0.004,
) -> None:
    info = PointedHandInfo.from_config(config_path, seed=seed)

    server = viser.ViserServer(host=host, port=port)
    root_name = f"/{info.name}"
    urdf_handle = ViserUrdf(
        target=server,
        urdf_or_path=info.urdf_path,
        root_node_name=root_name,
    )

    palm_pose = info.get_palm_pose()
    palm_frame = server.scene.add_frame(
        f"{root_name}/palm_pose",
        position=tuple(float(v) for v in palm_pose[:3, 3]),
        wxyz=_wxyz_from_matrix(palm_pose[:3, :3]),
        axes_length=0.03,
        axes_radius=0.002,
        origin_radius=0.004,
    )

    q_dict = {}
    joint_limits = urdf_handle.get_actuated_joint_limits()
    for joint_name, (lower, upper) in joint_limits.items():
        q_dict[str(joint_name)] = float((float(lower) + float(upper)) / 2.0)

    gripper_order = [str(name) for name in joint_limits]

    def _apply_q(q: np.ndarray) -> None:
        q_map = {joint_name: float(value) for joint_name, value in zip(info.joint_order, q.tolist())}
        ordered = np.asarray([q_map.get(name, q_dict.get(name, 0.0)) for name in gripper_order], dtype=float)
        urdf_handle.update_cfg(ordered)
        for joint_name, value in zip(gripper_order, ordered.tolist()):
            q_dict[joint_name] = float(value)

    template_options = ["global", *info.template_names]
    selected_template = server.gui.add_dropdown("template", tuple(template_options), initial_value=template_options[0])
    q_mode = server.gui.add_dropdown("q_mode", ("q_open", "q_close"), initial_value="q_open")
    show_surface = server.gui.add_checkbox("show_surface_points", initial_value=True)
    show_contact = server.gui.add_checkbox("show_contact_points", initial_value=True)
    show_keypoints = server.gui.add_checkbox("show_keypoints", initial_value=True)
    show_palm = server.gui.add_checkbox("show_palm_frame", initial_value=True)
    palm_aligned = server.gui.add_checkbox("palm_aligned_points", initial_value=True)
    palm_points_delta = server.gui.add_slider("palm_points_delta", min=0.0, max=0.2, step=1e-3, initial_value=0.05)

    handles = _Handles(surface={}, contact={}, keypoints={}, palm_frame=palm_frame, target=None)

    def _clear_group(group: dict[str, Any]) -> None:
        for handle in list(group.values()):
            _remove_handle(handle)
        group.clear()

    def _draw_link_local_clouds(
        group: dict[str, Any],
        points_by_link: dict[str, np.ndarray],
        *,
        suffix: str,
        color: tuple[int, int, int],
    ) -> None:
        _clear_group(group)
        for link_name, points in points_by_link.items():
            prefix = _link_frame_prefix(server, root_name, link_name)
            group[link_name] = server.scene.add_point_cloud(
                f"{prefix}/{suffix}",
                points=np.asarray(points, dtype=float),
                colors=np.tile(np.asarray(color, dtype=np.uint8)[None, :], (len(points), 1)),
                point_size=point_size,
                point_shape="rounded",
            )

    def _draw_target(template_name: str) -> None:
        _remove_handle(handles.target)
        handles.target = None
        if template_name == "global":
            return
        point = info.get_grasp_target_point(template_name)
        handles.target = server.scene.add_icosphere(
            f"{root_name}/palm_pose/grasp_target",
            radius=0.008,
            color=(255, 153, 0),
            position=tuple(float(v) for v in point),
        )

    def _render() -> None:
        template_name = str(selected_template.value)
        q_mode_name = str(q_mode.value)
        if template_name == "global" or q_mode_name == "q_open":
            q = info.get_q_open(template_name) if template_name != "global" else info.get_q_open()
        else:
            q = info.get_q_close(template_name)
        _apply_q(q)

        if show_surface.value:
            _draw_link_local_clouds(handles.surface, info.surface_points, suffix="surface_points", color=(120, 170, 255))
        else:
            _clear_group(handles.surface)

        if show_contact.value:
            _draw_link_local_clouds(handles.contact, info.contact_points, suffix="contact_points", color=(255, 80, 80))
        else:
            _clear_group(handles.contact)

        if show_keypoints.value:
            keypoints = info.get_keypoints(
                template_name=None if template_name == "global" else template_name,
                palm_aligned_points=bool(palm_aligned.value),
                palm_points_delta=float(palm_points_delta.value),
            )
            _draw_link_local_clouds(handles.keypoints, keypoints, suffix="keypoints", color=(255, 214, 10))
        else:
            _clear_group(handles.keypoints)

        if handles.palm_frame is not None:
            handles.palm_frame.visible = bool(show_palm.value)
        _draw_target(template_name)

    @selected_template.on_update
    def _(_: Any) -> None:
        _render()

    @q_mode.on_update
    def _(_: Any) -> None:
        _render()

    @show_surface.on_update
    def _(_: Any) -> None:
        _render()

    @show_contact.on_update
    def _(_: Any) -> None:
        _render()

    @show_keypoints.on_update
    def _(_: Any) -> None:
        _render()

    @show_palm.on_update
    def _(_: Any) -> None:
        _render()

    @palm_aligned.on_update
    def _(_: Any) -> None:
        _render()

    @palm_points_delta.on_update
    def _(_: Any) -> None:
        _render()

    _render()
    print(f"[pointed_hand_info.vis] Serving {config_path} on http://{host}:{port}")
    while True:
        time.sleep(1.0)


def main(argv: list[str] | None = None) -> int:
    tyro.cli(visualize, args=argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
