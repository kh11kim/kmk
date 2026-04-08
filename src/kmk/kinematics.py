from __future__ import annotations

"""Torch-based hand forward kinematics."""

from functools import partial
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import yourdfpy

from kmk.hand_info import HandInfo
from kmk.pose import exp_map, rR_to_T, transform_by_pose


def _load_urdf(urdf_path: str | Path) -> yourdfpy.URDF:
    path = Path(urdf_path)
    return yourdfpy.URDF.load(
        path,
        filename_handler=partial(yourdfpy.filename_handler_magic, dir=path.parent),
    )

class DiffKin(nn.Module):
    def __init__(self, urdf_path: str | Path, joint_order: list[str] | None = None) -> None:
        super().__init__()
        joint_type_map = {"fixed": 0, "revolute": 1, "prismatic": 2}
        self.urdf = _load_urdf(urdf_path)

        all_joints = list(self.urdf.joint_map.keys())
        if joint_order is None:
            self.fnames = all_joints
        else:
            self.fnames = list(joint_order) + [j for j in all_joints if j not in joint_order]

        self.link_names = list(self.urdf.link_map.keys())
        self.num_frames = len(self.fnames)
        self.fname_to_idx = {self.fnames[i]: i for i in range(self.num_frames)}
        self.idx_to_fname = {i: self.fnames[i] for i in range(self.num_frames)}

        child_link_to_fname = {jinfo.child: jinfo.name for jinfo in self.urdf.joint_map.values()}
        self.lname_to_idx = {jinfo.child: self.fname_to_idx[jinfo.name] for jinfo in self.urdf.joint_map.values()}
        self.root_name = [lname for lname in self.urdf.link_map if lname not in self.lname_to_idx][0]
        self.lname_to_idx[self.root_name] = -1

        joint_types: list[int] = []
        ctrlable_indices: list[int] = []
        mimic_dst_indices: list[int] = []
        mimic_src_indices: list[int] = []
        mimic_multipliers: list[float] = []
        mimic_offsets: list[float] = []
        lb: list[float] = []
        ub: list[float] = []
        all_axes = torch.zeros((self.num_frames, 3), dtype=torch.float32)
        all_origins = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4).repeat(self.num_frames, 1, 1)
        parent_frame_indices = [-1] * self.num_frames

        for jname in self.fnames:
            jinfo = self.urdf.joint_map[jname]
            idx = self.fname_to_idx[jname]
            joint_types.append(joint_type_map[jinfo.type])
            if jinfo.type != "fixed" and jinfo.mimic is None:
                ctrlable_indices.append(idx)
                lower = 0.0 if jinfo.limit is None or jinfo.limit.lower is None else float(jinfo.limit.lower)
                upper = 0.0 if jinfo.limit is None or jinfo.limit.upper is None else float(jinfo.limit.upper)
                lb.append(lower)
                ub.append(upper)
            if jinfo.parent in child_link_to_fname:
                parent_frame_name = child_link_to_fname[jinfo.parent]
                parent_frame_indices[idx] = self.fname_to_idx[parent_frame_name]
            if jinfo.origin is not None:
                all_origins[idx] = torch.as_tensor(jinfo.origin, dtype=torch.float32)
            if jinfo.axis is not None:
                all_axes[idx] = torch.as_tensor(jinfo.axis, dtype=torch.float32)

            if jinfo.mimic is not None:
                mimic_dst_indices.append(idx)
                mimic_src_indices.append(self.fname_to_idx[jinfo.mimic.joint])
                mimic_multipliers.append(float(jinfo.mimic.multiplier))
                mimic_offsets.append(float(jinfo.mimic.offset))

        self.parent_frame_indices = parent_frame_indices
        self.dof = len(ctrlable_indices)
        self.is_mimic_joint = len(mimic_dst_indices) > 0

        self.register_buffer("joint_types", torch.tensor(joint_types, dtype=torch.long))
        self.register_buffer("ctrlable_indices", torch.tensor(ctrlable_indices, dtype=torch.long))
        self.register_buffer("mimic_dst_indices", torch.tensor(mimic_dst_indices, dtype=torch.long))
        self.register_buffer("mimic_src_indices", torch.tensor(mimic_src_indices, dtype=torch.long))
        self.register_buffer("mimic_multipliers", torch.tensor(mimic_multipliers, dtype=torch.float32))
        self.register_buffer("mimic_offsets", torch.tensor(mimic_offsets, dtype=torch.float32))
        self.register_buffer("all_axes", all_axes)
        self.register_buffer("all_origins", all_origins)
        self.register_buffer("lb", torch.tensor(lb, dtype=torch.float32))
        self.register_buffer("ub", torch.tensor(ub, dtype=torch.float32))

        self.revolute_indices = torch.where(self.joint_types == 1)[0]
        self.prismatic_indices = torch.where(self.joint_types == 2)[0]

    def _expand_angles(self, joint_angles: torch.Tensor) -> torch.Tensor:
        batch = joint_angles.shape[0]
        all_angles = joint_angles.new_zeros((batch, self.num_frames))
        if self.ctrlable_indices.numel() > 0:
            all_angles = all_angles.scatter(
                1,
                self.ctrlable_indices.unsqueeze(0).expand(batch, -1),
                joint_angles,
            )
        if self.is_mimic_joint:
            mimic_values = all_angles[:, self.mimic_src_indices] * self.mimic_multipliers + self.mimic_offsets
            all_angles = all_angles.scatter(
                1,
                self.mimic_dst_indices.unsqueeze(0).expand(batch, -1),
                mimic_values,
            )
        return all_angles

    def _local_transform(self, origin: torch.Tensor, axis: torch.Tensor, angle: torch.Tensor, joint_type: int) -> torch.Tensor:
        batch = angle.shape[0]
        eye_rot = torch.eye(3, dtype=origin.dtype, device=origin.device).expand(batch, 3, 3)
        if joint_type == 0:
            motion = rR_to_T(
                torch.zeros(batch, 3, dtype=origin.dtype, device=origin.device),
                eye_rot,
            )
        elif joint_type == 1:
            rot = exp_map(axis.unsqueeze(0) * angle.unsqueeze(-1))
            motion = rR_to_T(
                torch.zeros(batch, 3, dtype=origin.dtype, device=origin.device),
                rot,
            )
        elif joint_type == 2:
            trans = axis.unsqueeze(0) * angle.unsqueeze(-1)
            motion = rR_to_T(trans, eye_rot)
        else:
            raise ValueError(f"Unsupported joint type: {joint_type}")
        return origin.unsqueeze(0).expand(batch, 4, 4) @ motion

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
        pending = set(range(self.num_frames))
        while pending:
            progressed = False
            for idx in list(pending):
                parent_idx = self.parent_frame_indices[idx]
                if parent_idx == -1:
                    parent_pose = identity
                else:
                    parent_pose = link_pose_list[parent_idx]
                    if parent_pose is None:
                        continue
                link_pose_list[idx] = parent_pose @ local_tfs[:, idx, ...]
                pending.remove(idx)
                progressed = True
            if not progressed:
                raise RuntimeError("Failed to topologically resolve hand kinematics")
        return torch.stack([pose if pose is not None else identity for pose in link_pose_list], dim=1)


class HandKinematics(nn.Module):
    def __init__(
        self,
        hand_info: HandInfo | str | Path,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        runtime_hand = hand_info if isinstance(hand_info, HandInfo) else HandInfo.from_config(hand_info)
        self.hand_info = runtime_hand
        self.joint_order = list(runtime_hand.joint_order)
        self.link_names = list(runtime_hand.urdf.link_map.keys())
        self.dof = len(self.joint_order)
        self.device_hint = device
        self.dtype = dtype

        self.chain = DiffKin(runtime_hand.urdf_path, joint_order=self.joint_order)
        if device is not None:
            self.chain = self.chain.to(device=device, dtype=dtype)
        else:
            self.chain = self.chain.to(dtype=dtype)

        self._link_fk_index = {runtime_hand.urdf.base_link: -1}
        for link_name, joint_idx in self.chain.lname_to_idx.items():
            self._link_fk_index[str(link_name)] = int(joint_idx)

        palm_pose = torch.as_tensor(runtime_hand.get_palm_pose(), dtype=dtype, device=device)
        self.register_buffer("palm_pose", palm_pose)

    def _coerce_q(self, q: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        q = torch.as_tensor(q, dtype=self.dtype, device=self.palm_pose.device)
        if q.shape[-1] != self.dof:
            raise ValueError(f"q last dimension must equal dof={self.dof}")
        if q.ndim == 1:
            batch_shape = (1,)
            q_flat = q.reshape(1, self.dof)
        else:
            batch_shape = tuple(int(v) for v in q.shape[:-1])
            q_flat = q.reshape(-1, self.dof)
        return q_flat, batch_shape

    def forward_kinematics(self, q: torch.Tensor) -> dict[str, torch.Tensor]:
        q_flat, batch_shape = self._coerce_q(q)
        fk_joint = self.chain(q_flat)
        batch = q_flat.shape[0]
        identity = torch.eye(4, dtype=fk_joint.dtype, device=fk_joint.device).expand(batch, 4, 4)

        result: dict[str, torch.Tensor] = {}
        for link_name in self.link_names:
            joint_idx = self._link_fk_index.get(link_name, -1)
            pose = identity if joint_idx < 0 else fk_joint[:, joint_idx, ...]
            result[link_name] = pose.reshape(batch_shape + (4, 4))
        return result

    def transform_link_points(
        self,
        q: torch.Tensor,
        points_by_link: dict[str, torch.Tensor | object],
    ) -> dict[str, torch.Tensor]:
        fk = self.forward_kinematics(q)
        transformed: dict[str, torch.Tensor] = {}
        for link_name, raw_points in points_by_link.items():
            if link_name not in fk:
                raise KeyError(link_name)
            points = torch.as_tensor(raw_points, dtype=self.dtype, device=self.palm_pose.device)
            if points.ndim != 2 or points.shape[-1] != 3:
                raise ValueError("points_by_link values must have shape (P, 3)")
            pose = fk[link_name]
            points_expanded = points.reshape((1,) * (pose.ndim - 2) + points.shape).expand(pose.shape[:-2] + points.shape)
            transformed[link_name] = transform_by_pose(pose.unsqueeze(-3), points_expanded)
        return transformed

    def get_palm_pose(self, batch_shape: Sequence[int] | None = None) -> torch.Tensor:
        if batch_shape is None:
            return self.palm_pose.clone()
        shape = tuple(int(v) for v in batch_shape)
        return self.palm_pose.reshape((1,) * len(shape) + (4, 4)).expand(shape + (4, 4)).clone()
