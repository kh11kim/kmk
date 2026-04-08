from __future__ import annotations

"""Runtime wrapper for authored hand representation configs."""

from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R
from yourdfpy import URDF

from kmk.config.model import GripperConfig, normalize_palm_pose


def _resolve_config_relative_path(path_like: str | Path | None, *, config_dir: Path) -> Path | None:
    if path_like is None:
        return None
    candidate = Path(str(path_like).strip().strip("`").strip('"').strip("'")).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (config_dir / candidate).resolve()


def _normalize_tag_query(values: Sequence[str] = ()) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        tag = str(value).strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        normalized.append(tag)
    return tuple(normalized)


def _resolve_link_name(link_name: str, urdf: URDF) -> str:
    if link_name in urdf.link_map:
        return link_name
    candidate = re.sub(r"_(\d+)$", "", link_name)
    candidate = re.sub(r"\.(?:stl|obj|ply)$", "", candidate)
    if candidate in urdf.link_map:
        return candidate
    return link_name


def _get_col_mesh_from_urdf(urdf: URDF, link_name: str):
    collisions = urdf.link_map[link_name].collisions
    if len(collisions) == 0:
        raise ValueError(f"collision geometry is required for link: {link_name}")
    meshes = []
    for collision in collisions:
        scene = urdf._geometry2trimeshscene(collision.geometry, True, True, True)
        col_mesh = scene.to_mesh()
        origin = collision.origin if collision.origin is not None else np.eye(4)
        col_mesh.apply_transform(origin)
        meshes.append(col_mesh)
    merged = meshes[0].copy()
    for mesh in meshes[1:]:
        merged = merged + mesh
    return merged


def _sample_collision_surface_points(urdf: URDF, link_name: str, density: int) -> tuple[np.ndarray, np.ndarray]:
    col_mesh = _get_col_mesh_from_urdf(urdf, link_name)
    area = float(col_mesh.area)
    num_points = max(1, int(area * density))
    points, indices = col_mesh.convex_hull.sample(num_points, return_index=True)
    normals = col_mesh.convex_hull.face_normals[indices]
    return np.asarray(points, dtype=float), np.asarray(normals, dtype=float)


def _farthest_point_down_sample(points: np.ndarray, num: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] <= num:
        return points.copy()
    selected = [0]
    min_dist = np.linalg.norm(points - points[0], axis=1)
    while len(selected) < num:
        next_index = int(np.argmax(min_dist))
        selected.append(next_index)
        dist = np.linalg.norm(points - points[next_index], axis=1)
        min_dist = np.minimum(min_dist, dist)
    return points[selected]


def _sample_surface_points(urdf: URDF, density: int = 2000) -> dict[str, np.ndarray]:
    surface_points: dict[str, np.ndarray] = {}
    for link_name, link in urdf.link_map.items():
        if len(link.collisions) == 0:
            continue
        points, _ = _sample_collision_surface_points(urdf, link_name, density)
        surface_points[link_name] = points
    return surface_points


def _sample_contact_points(
    urdf: URDF,
    contact_anchors: dict[str, Any],
    num_point_per_link: int = 10,
    default_radius: float = 0.005,
) -> dict[str, np.ndarray]:
    contact_points: dict[str, np.ndarray] = {}
    for contact_name, contact_info in contact_anchors.items():
        link_name = _resolve_link_name(contact_name, urdf)
        if link_name not in urdf.link_map:
            continue
        radius = float(contact_info.get("contact_radius", default_radius))
        ref_point = np.asarray(contact_info.get("point", [0.0, 0.0, 0.0]), dtype=float)

        points, _normals = _sample_collision_surface_points(urdf, link_name, density=100000)
        dists = np.linalg.norm(points - ref_point, axis=-1)
        mask = dists < radius
        if mask.sum() == 0:
            nearest = np.argsort(dists)[: max(num_point_per_link * 10, num_point_per_link)]
            candidate_points = points[nearest]
        else:
            candidate_points = points[mask]
        contact_points[contact_name] = _farthest_point_down_sample(candidate_points, num_point_per_link)
    return contact_points


class HandInfo:
    """Read-only runtime wrapper around an authored gripper config."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        self.config = GripperConfig.load(self.config_path)

        self.urdf_path = _resolve_config_relative_path(self.config.urdf_path, config_dir=self.config_path.parent)
        if self.urdf_path is None:
            raise ValueError("urdf_path must be set")
        self.xml_path = _resolve_config_relative_path(self.config.xml_path, config_dir=self.config_path.parent)
        self.urdf = URDF.load(str(self.urdf_path))

        actuated_joint_names = tuple(str(name) for name in getattr(self.urdf, "actuated_joint_names", ()) or ())
        link_names = tuple(str(name) for name in getattr(self.urdf, "link_map", {}).keys())
        self.config.validate(
            urdf_actuated_joint_names=actuated_joint_names or None,
            urdf_link_names=link_names or None,
        )

        self.name = self.config.name
        self.joint_order = list(self.config.joint_order)
        self.template_names = list(self.config.grasp_templates)
        self.contact_anchor_links = list(self.config.contact_anchors)

    @classmethod
    def from_config(cls, config_path: str | Path) -> "HandInfo":
        return cls(config_path=config_path)

    def _require_template(self, template_name: str) -> dict[str, object]:
        if template_name not in self.config.grasp_templates:
            raise KeyError(template_name)
        return self.config.grasp_templates[template_name]

    def _require_contact_anchor(self, link_name: str) -> dict[str, object]:
        if link_name not in self.config.contact_anchors:
            raise KeyError(link_name)
        return self.config.contact_anchors[link_name]

    def get_q_open(self, template: str = "global") -> np.ndarray:
        if template == "global":
            return np.asarray(self.config.q_open, dtype=float).copy()
        entry = self._require_template(template)
        return np.asarray(entry["q_open"], dtype=float).copy()

    def get_q_close(self, template_name: str) -> np.ndarray:
        entry = self._require_template(template_name)
        return np.asarray(entry["q_close"], dtype=float).copy()

    def get_palm_pose(self) -> np.ndarray:
        palm_pose = normalize_palm_pose(self.config.palm_pose)
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = R.from_euler("ZYX", list(reversed(palm_pose["rpy"])), degrees=True).as_matrix()
        transform[:3, 3] = np.asarray(palm_pose["trans"], dtype=float)
        return transform

    def get_contact_anchor_by_link(self, link_name: str) -> np.ndarray:
        entry = self._require_contact_anchor(link_name)
        return np.asarray(entry["point"], dtype=float).copy()

    def get_contact_anchor_by_tag(
        self,
        includes: Sequence[str] = (),
        excludes: Sequence[str] = (),
    ) -> dict[str, np.ndarray]:
        include_tags = set(_normalize_tag_query(includes))
        exclude_tags = set(_normalize_tag_query(excludes))
        matches: dict[str, np.ndarray] = {}
        for link_name, entry in self.config.contact_anchors.items():
            tags = set(str(tag) for tag in entry.get("tags", []))
            if include_tags and not include_tags.issubset(tags):
                continue
            if exclude_tags and tags.intersection(exclude_tags):
                continue
            matches[link_name] = np.asarray(entry["point"], dtype=float).copy()
        return matches

    def get_contact_anchor_by_template(self, template_name: str) -> dict[str, np.ndarray]:
        entry = self._require_template(template_name)
        matches: dict[str, np.ndarray] = {}
        for link_name in entry.get("active_contact_anchors", []):
            matches[str(link_name)] = self.get_contact_anchor_by_link(str(link_name))
        return matches

    def get_grasp_target_point(self, template_name: str) -> np.ndarray:
        entry = self._require_template(template_name)
        return np.asarray(entry["grasp_target_point"], dtype=float).copy()


class PointedHandInfo(HandInfo):
    """HandInfo with precomputed local surface/contact point samples."""

    def __init__(self, config_path: str | Path, *, seed: int) -> None:
        super().__init__(config_path=config_path)
        self.seed = int(seed)

        state = np.random.get_state()
        np.random.seed(self.seed)
        try:
            self.surface_points = {
                link_name: np.asarray(points, dtype=float).copy()
                for link_name, points in _sample_surface_points(self.urdf).items()
            }
            self.contact_points = {
                link_name: np.asarray(points, dtype=float).copy()
                for link_name, points in _sample_contact_points(self.urdf, self.config.contact_anchors).items()
            }
        finally:
            np.random.set_state(state)

    @classmethod
    def from_config(cls, config_path: str | Path, *, seed: int) -> "PointedHandInfo":
        return cls(config_path=config_path, seed=seed)

    def get_contact_points(self, template_name: str | None = None) -> dict[str, np.ndarray]:
        if template_name is None:
            selected_names = self.contact_anchor_links
        else:
            entry = self._require_template(template_name)
            selected_names = [str(link_name) for link_name in entry.get("active_contact_anchors", [])]
        return {
            link_name: np.asarray(self.contact_points[link_name], dtype=float).copy()
            for link_name in selected_names
            if link_name in self.contact_points
        }

    def get_keypoints(
        self,
        template_name: str | None = None,
        palm_aligned_points: bool = True,
        palm_points_delta: float = 0.05,
    ) -> dict[str, np.ndarray]:
        if template_name is None:
            selected = {
                link_name: self.get_contact_anchor_by_link(link_name)
                for link_name in self.contact_anchor_links
            }
        else:
            selected = self.get_contact_anchor_by_template(template_name)

        keypoints = {
            link_name: np.asarray(point, dtype=float).reshape(1, 3).copy()
            for link_name, point in selected.items()
        }

        if palm_aligned_points:
            palm_pose = self.get_palm_pose()
            center = palm_pose[:3, 3]
            basis = palm_pose[:3, :3]
            offsets = np.vstack(
                [
                    np.zeros((1, 3), dtype=float),
                    np.eye(3, dtype=float),
                    -np.eye(3, dtype=float),
                ]
            ) * float(palm_points_delta)
            palm_points = center[None, :] + offsets @ basis.T
            base_link = str(self.urdf.base_link)
            if base_link in keypoints:
                keypoints[base_link] = np.vstack([keypoints[base_link], palm_points])
            else:
                keypoints[base_link] = palm_points

        return keypoints
