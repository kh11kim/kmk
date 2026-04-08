from __future__ import annotations

"""Batch visualization for HandKinematics alignment checks."""

from pathlib import Path
import time

import numpy as np
import torch
import tyro
import viser
from scipy.spatial.transform import Rotation as R
from viser.extras import ViserUrdf

from kmk import HandKinematics, PointedHandInfo


def _wxyz_from_matrix(rot: np.ndarray) -> tuple[float, float, float, float]:
    quat = R.from_matrix(np.asarray(rot, dtype=float)).as_quat()
    return (float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2]))


def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _grid_palm_pose(row: int, col: int, spacing: float) -> np.ndarray:
    translation = np.array([col * spacing, -row * spacing, 0.0], dtype=float)
    return _make_transform(np.eye(3, dtype=float), translation)


def _set_root_pose(server: viser.ViserServer, root_name: str, transform: np.ndarray) -> None:
    root = server.scene._handle_from_node_name[root_name]
    root.position = tuple(float(v) for v in transform[:3, 3])
    root.wxyz = _wxyz_from_matrix(transform[:3, :3])


def _set_joint_configuration(handle: ViserUrdf, joint_order: list[str], q: np.ndarray) -> None:
    q_map = {joint_name: float(value) for joint_name, value in zip(joint_order, q.tolist())}
    ordered_joint_names = [str(name) for name in handle.get_actuated_joint_limits()]
    ordered_q = np.asarray([q_map.get(name, 0.0) for name in ordered_joint_names], dtype=float)
    handle.update_cfg(ordered_q)


def visualize(
    config_path: str,
    seed: int = 0,
    grid_rows: int = 4,
    grid_cols: int = 4,
    spacing: float = 0.44,
    host: str = "127.0.0.1",
    port: int = 8080,
    surface_point_size: float = 0.0025,
    contact_point_size: float = 0.004,
    keypoint_size: float = 0.006,
) -> None:
    hand_info = PointedHandInfo.from_config(config_path, seed=seed)
    kin = HandKinematics(hand_info)

    num_hands = grid_rows * grid_cols
    generator = torch.Generator(device=kin.palm_pose.device)
    generator.manual_seed(int(seed))
    lower = kin.chain.lb
    upper = kin.chain.ub

    server = viser.ViserServer(host=host, port=port)
    show_keypoints = server.gui.add_checkbox("show_keypoints", initial_value=True)
    sample_button = server.gui.add_button("Sample Batched Fingers")

    palm_pose = hand_info.get_palm_pose()
    palm_pose_inv = np.linalg.inv(palm_pose)
    urdf_handles: list[ViserUrdf] = []
    keypoint_handles = []
    q_samples = lower + (upper - lower) * torch.rand((num_hands, kin.dof), generator=generator, device=lower.device, dtype=lower.dtype)

    def _sample_q_batch() -> torch.Tensor:
        return lower + (upper - lower) * torch.rand(
            (num_hands, kin.dof),
            generator=generator,
            device=lower.device,
            dtype=lower.dtype,
        )

    def _clear_keypoints() -> None:
        for handle in list(keypoint_handles):
            remove = getattr(handle, "remove", None)
            if callable(remove):
                remove()
            elif hasattr(handle, "visible"):
                handle.visible = False
        keypoint_handles.clear()

    for index in range(num_hands):
        row = index // grid_cols
        col = index % grid_cols
        root_name = f"/hand_{index:02d}"
        handle = ViserUrdf(
            target=server,
            urdf_or_path=Path(hand_info.urdf_path),
            root_node_name=root_name,
        )
        urdf_handles.append(handle)
        q_np = q_samples[index].detach().cpu().numpy()
        _set_joint_configuration(handle, hand_info.joint_order, q_np)

        world_palm = _grid_palm_pose(row, col, spacing)
        world_base = world_palm @ palm_pose_inv
        _set_root_pose(server, root_name, world_base)

    def _render_keypoints(current_q: torch.Tensor) -> None:
        _clear_keypoints()
        keypoints_local = hand_info.get_keypoints(template_name=None, palm_aligned_points=True)
        keypoints_world = kin.transform_link_points(current_q, keypoints_local)
        for index in range(num_hands):
            q_np = current_q[index].detach().cpu().numpy()
            _set_joint_configuration(urdf_handles[index], hand_info.joint_order, q_np)
            keypoint_points = [
                points[index].detach().cpu().numpy()
                for points in keypoints_world.values()
            ]
            if keypoint_points:
                keypoint_handles.append(
                    server.scene.add_point_cloud(
                        f"/hand_{index:02d}/keypoints",
                        points=np.concatenate(keypoint_points, axis=0),
                        colors=(255, 214, 10),
                        point_size=keypoint_size,
                        point_shape="rounded",
                        visible=bool(show_keypoints.value),
                    )
                )

    _render_keypoints(q_samples)

    @show_keypoints.on_update
    def _(_: object) -> None:
        for handle in keypoint_handles:
            handle.visible = bool(show_keypoints.value)

    @sample_button.on_click
    def _(_: object) -> None:
        nonlocal q_samples
        q_samples = _sample_q_batch()
        _render_keypoints(q_samples)

    print(f"[hand_kinematics.vis] Serving {config_path} on http://{host}:{port}")
    print(f"[hand_kinematics.vis] Showing {num_hands} random samples in a {grid_rows}x{grid_cols} grid")
    while True:
        time.sleep(1.0)


def main(argv: list[str] | None = None) -> int:
    tyro.cli(visualize, args=argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
