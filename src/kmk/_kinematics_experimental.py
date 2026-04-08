from __future__ import annotations

"""Experimental hand kinematics with precomputed topological frame order."""

from pathlib import Path

import torch

from kmk.hand_info import HandInfo
from kmk.kinematics import DiffKin, HandKinematics
from kmk.pose import exp_map, transform_by_pose


def _compute_topo_frame_indices(parent_frame_indices: list[int]) -> tuple[int, ...]:
    children: list[list[int]] = [[] for _ in range(len(parent_frame_indices))]
    roots: list[int] = []
    for idx, parent_idx in enumerate(parent_frame_indices):
        if parent_idx == -1:
            roots.append(idx)
        else:
            children[parent_idx].append(idx)

    topo_order: list[int] = []
    stack = list(reversed(roots))
    while stack:
        idx = stack.pop()
        topo_order.append(idx)
        stack.extend(reversed(children[idx]))
    if len(topo_order) != len(parent_frame_indices):
        raise RuntimeError("Failed to precompute a topological frame order")
    return tuple(topo_order)


class _TopoDiffKin(DiffKin):
    def __init__(self, urdf_path: str | Path, joint_order: list[str] | None = None) -> None:
        super().__init__(urdf_path, joint_order=joint_order)
        self.topo_frame_indices = _compute_topo_frame_indices(self.parent_frame_indices)

    def forward(self, joint_angles: torch.Tensor) -> torch.Tensor:
        batch = joint_angles.shape[0]
        all_angles = self._expand_angles(joint_angles)
        local_tfs = torch.stack(
            [
                self._local_transform(
                    self.all_origins[idx],
                    self.all_axes[idx],
                    all_angles[:, idx],
                    int(self.joint_types[idx].item()),
                )
                for idx in range(self.num_frames)
            ],
            dim=1,
        )

        identity = torch.eye(4, dtype=joint_angles.dtype, device=joint_angles.device).expand(batch, 4, 4)
        link_pose_list: list[torch.Tensor | None] = [None] * self.num_frames
        for idx in self.topo_frame_indices:
            parent_idx = self.parent_frame_indices[idx]
            parent_pose = identity if parent_idx == -1 else link_pose_list[parent_idx]
            if parent_pose is None:
                raise RuntimeError("Failed to resolve parent pose in topological FK")
            link_pose_list[idx] = parent_pose @ local_tfs[:, idx, ...]
        return torch.stack([pose if pose is not None else identity for pose in link_pose_list], dim=1)


class _VectorizedTopoDiffKin(_TopoDiffKin):
    def __init__(self, urdf_path: str | Path, joint_order: list[str] | None = None) -> None:
        super().__init__(urdf_path, joint_order=joint_order)
        self.register_buffer("revolute_frame_indices", torch.where(self.joint_types == 1)[0])
        self.register_buffer("prismatic_frame_indices", torch.where(self.joint_types == 2)[0])

    def forward(self, joint_angles: torch.Tensor) -> torch.Tensor:
        batch = joint_angles.shape[0]
        all_angles = self._expand_angles(joint_angles)

        local_tfs = self.all_origins.unsqueeze(0).expand(batch, -1, -1, -1).clone()
        motion = torch.eye(4, dtype=joint_angles.dtype, device=joint_angles.device).reshape(1, 1, 4, 4).expand(
            batch, self.num_frames, 4, 4
        ).clone()

        if self.revolute_frame_indices.numel() > 0:
            revolute_angles = all_angles[:, self.revolute_frame_indices]
            revolute_axes = self.all_axes[self.revolute_frame_indices]
            revolute_rot = exp_map(revolute_axes.unsqueeze(0) * revolute_angles.unsqueeze(-1))
            motion[:, self.revolute_frame_indices, :3, :3] = revolute_rot

        if self.prismatic_frame_indices.numel() > 0:
            prismatic_angles = all_angles[:, self.prismatic_frame_indices]
            prismatic_axes = self.all_axes[self.prismatic_frame_indices]
            motion[:, self.prismatic_frame_indices, :3, 3] = prismatic_axes.unsqueeze(0) * prismatic_angles.unsqueeze(-1)

        local_tfs = local_tfs @ motion
        identity = torch.eye(4, dtype=joint_angles.dtype, device=joint_angles.device).expand(batch, 4, 4)
        link_pose_list: list[torch.Tensor | None] = [None] * self.num_frames
        for idx in self.topo_frame_indices:
            parent_idx = self.parent_frame_indices[idx]
            parent_pose = identity if parent_idx == -1 else link_pose_list[parent_idx]
            if parent_pose is None:
                raise RuntimeError("Failed to resolve parent pose in topological FK")
            link_pose_list[idx] = parent_pose @ local_tfs[:, idx, ...]
        return torch.stack([pose if pose is not None else identity for pose in link_pose_list], dim=1)


class _BaseExperimentalHandKinematics(HandKinematics):
    chain_type = _TopoDiffKin

    def __init__(
        self,
        hand_info: HandInfo | str | Path,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(hand_info, device=device, dtype=dtype)
        self.chain = self.chain_type(self.hand_info.urdf_path, joint_order=self.joint_order)
        if device is not None:
            self.chain = self.chain.to(device=device, dtype=dtype)
        else:
            self.chain = self.chain.to(dtype=dtype)

        self._link_fk_index = {self.hand_info.urdf.base_link: -1}
        for link_name, joint_idx in self.chain.lname_to_idx.items():
            self._link_fk_index[str(link_name)] = int(joint_idx)


class ExperimentalHandKinematics(_BaseExperimentalHandKinematics):
    chain_type = _TopoDiffKin


class VectorizedHandKinematics(_BaseExperimentalHandKinematics):
    chain_type = _VectorizedTopoDiffKin


class PackedTransformHandKinematics(VectorizedHandKinematics):
    def transform_link_points(
        self,
        q: torch.Tensor,
        points_by_link: dict[str, torch.Tensor | object],
    ) -> dict[str, torch.Tensor]:
        q_flat, batch_shape = self._coerce_q(q)
        fk_joint = self.chain(q_flat)
        batch = q_flat.shape[0]
        identity = torch.eye(4, dtype=fk_joint.dtype, device=fk_joint.device).expand(batch, 4, 4)

        link_names: list[str] = []
        split_sizes: list[int] = []
        packed_pose_chunks: list[torch.Tensor] = []
        packed_point_chunks: list[torch.Tensor] = []

        for link_name, raw_points in points_by_link.items():
            points = torch.as_tensor(raw_points, dtype=self.dtype, device=self.palm_pose.device)
            if points.ndim != 2 or points.shape[-1] != 3:
                raise ValueError("points_by_link values must have shape (P, 3)")
            joint_idx = self._link_fk_index.get(link_name)
            if joint_idx is None:
                raise KeyError(link_name)
            pose = identity if joint_idx < 0 else fk_joint[:, joint_idx, ...]
            point_count = int(points.shape[0])
            link_names.append(link_name)
            split_sizes.append(point_count)
            packed_pose_chunks.append(pose.unsqueeze(1).expand(batch, point_count, 4, 4))
            packed_point_chunks.append(points.unsqueeze(0).expand(batch, point_count, 3))

        packed_pose = torch.cat(packed_pose_chunks, dim=1)
        packed_points = torch.cat(packed_point_chunks, dim=1)
        packed_transformed = transform_by_pose(packed_pose, packed_points)

        transformed: dict[str, torch.Tensor] = {}
        start = 0
        for link_name, size in zip(link_names, split_sizes):
            stop = start + size
            transformed[link_name] = packed_transformed[:, start:stop, :].reshape(batch_shape + (size, 3))
            start = stop
        return transformed
