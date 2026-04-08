from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import kmk.hand_info as hand_info_module
from kmk import HandInfo, PointedHandInfo
from kmk.config.model import GripperConfig


def _make_config(tmp_path: Path) -> Path:
    root = tmp_path / "hand"
    root.mkdir(parents=True, exist_ok=True)
    urdf_path = root / "demo.urdf"
    urdf_path.write_text(
        "\n".join(
            [
                "<?xml version='1.0'?>",
                "<robot name='demo'>",
                "  <link name='base' />",
                "  <link name='thumb_link' />",
                "  <link name='index_link' />",
                "  <joint name='thumb_joint' type='revolute'><parent link='base'/><child link='thumb_link'/></joint>",
                "  <joint name='index_joint' type='revolute'><parent link='base'/><child link='index_link'/></joint>",
                "</robot>",
                "",
            ]
        ),
        encoding="utf-8",
    )
    config_path = root / "demo.yaml"
    GripperConfig(
        name="demo",
        urdf_path="demo.urdf",
        joint_order=["thumb_joint", "index_joint"],
        q_open=[0.1, 0.2],
        palm_pose={"trans": [0.3, 0.4, 0.5], "rpy": [10.0, 20.0, 30.0]},
        contact_anchors={
            "thumb_link": {"point": [0.01, 0.02, 0.03], "contact_radius": 0.007, "tags": ["thumb", "tip"]},
            "index_link": {"point": [0.04, 0.05, 0.06], "contact_radius": 0.007, "tags": ["index", "tip", "outer"]},
        },
        grasp_templates={
            "pinch": {
                "q_open": [0.11, 0.22],
                "q_close": [0.44, 0.55],
                "grasp_target_point": [0.7, 0.8, 0.9],
                "active_contact_anchors": ["thumb_link"],
            },
            "power": {
                "q_open": [0.12, 0.23],
                "q_close": [0.45, 0.56],
                "grasp_target_point": [1.0, 1.1, 1.2],
                "active_contact_anchors": ["thumb_link", "index_link"],
            },
        },
    ).save(config_path)
    return config_path


def _make_collision_config(tmp_path: Path) -> Path:
    root = tmp_path / "pointed_hand"
    root.mkdir(parents=True, exist_ok=True)
    urdf_path = root / "demo_collision.urdf"
    urdf_path.write_text(
        "\n".join(
            [
                "<?xml version='1.0'?>",
                "<robot name='demo_collision'>",
                "  <link name='base'>",
                "    <collision><geometry><box size='0.04 0.04 0.04'/></geometry></collision>",
                "  </link>",
                "  <link name='thumb_link'>",
                "    <collision><geometry><box size='0.02 0.02 0.02'/></geometry></collision>",
                "  </link>",
                "  <link name='index_link'>",
                "    <collision><geometry><box size='0.02 0.02 0.02'/></geometry></collision>",
                "  </link>",
                "  <joint name='thumb_joint' type='revolute'><parent link='base'/><child link='thumb_link'/></joint>",
                "  <joint name='index_joint' type='revolute'><parent link='base'/><child link='index_link'/></joint>",
                "</robot>",
                "",
            ]
        ),
        encoding="utf-8",
    )
    config_path = root / "demo_collision.yaml"
    GripperConfig(
        name="demo_collision",
        urdf_path="demo_collision.urdf",
        joint_order=["thumb_joint", "index_joint"],
        q_open=[0.1, 0.2],
        palm_pose={"trans": [0.01, 0.02, 0.03], "rpy": [10.0, 20.0, 30.0]},
        contact_anchors={
            "thumb_link": {"point": [0.001, 0.002, 0.003], "contact_radius": 0.007, "tags": ["thumb", "tip"]},
            "index_link": {"point": [0.004, 0.005, 0.006], "contact_radius": 0.007, "tags": ["index", "tip"]},
        },
        grasp_templates={
            "pinch": {
                "q_open": [0.11, 0.22],
                "q_close": [0.44, 0.55],
                "grasp_target_point": [0.7, 0.8, 0.9],
                "active_contact_anchors": ["thumb_link"],
            },
        },
    ).save(config_path)
    return config_path


def _make_multi_collision_config(tmp_path: Path) -> Path:
    root = tmp_path / "multi_collision_hand"
    root.mkdir(parents=True, exist_ok=True)
    urdf_path = root / "multi_collision.urdf"
    urdf_path.write_text(
        "\n".join(
            [
                "<?xml version='1.0'?>",
                "<robot name='multi_collision'>",
                "  <link name='base'>",
                "    <collision><geometry><box size='0.04 0.04 0.04'/></geometry></collision>",
                "  </link>",
                "  <link name='thumb_link'>",
                "    <collision><geometry><box size='0.02 0.02 0.02'/></geometry></collision>",
                "    <collision><origin xyz='0.01 0 0'/><geometry><box size='0.01 0.01 0.01'/></geometry></collision>",
                "  </link>",
                "  <joint name='thumb_joint' type='revolute'><parent link='base'/><child link='thumb_link'/></joint>",
                "</robot>",
                "",
            ]
        ),
        encoding="utf-8",
    )
    config_path = root / "multi_collision.yaml"
    GripperConfig(
        name="multi_collision",
        urdf_path="multi_collision.urdf",
        joint_order=["thumb_joint"],
        q_open=[0.1],
        palm_pose={"trans": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
        contact_anchors={
            "thumb_link": {"point": [0.001, 0.0, 0.0], "contact_radius": 0.007, "tags": ["thumb", "tip"]},
        },
        grasp_templates={
            "pinch": {
                "q_open": [0.11],
                "q_close": [0.44],
                "grasp_target_point": [0.1, 0.2, 0.3],
                "active_contact_anchors": ["thumb_link"],
            },
        },
    ).save(config_path)
    return config_path


def test_hand_info_from_config_loads_runtime_metadata(tmp_path: Path) -> None:
    config_path = _make_config(tmp_path)

    info = HandInfo.from_config(config_path)

    assert info.name == "demo"
    assert info.config_path == config_path.resolve()
    assert info.urdf_path == (config_path.parent / "demo.urdf").resolve()
    assert info.xml_path is None
    assert info.joint_order == ["thumb_joint", "index_joint"]
    assert info.template_names == ["pinch", "power"]
    assert info.contact_anchor_links == ["thumb_link", "index_link"]


def test_hand_info_q_accessors_return_numpy_arrays(tmp_path: Path) -> None:
    info = HandInfo.from_config(_make_config(tmp_path))

    assert np.allclose(info.get_q_open(), np.array([0.1, 0.2], dtype=float))
    assert np.allclose(info.get_q_open("pinch"), np.array([0.11, 0.22], dtype=float))
    assert np.allclose(info.get_q_close("power"), np.array([0.45, 0.56], dtype=float))
    assert isinstance(info.get_q_open(), np.ndarray)


def test_hand_info_get_palm_pose_returns_world_from_palm_matrix(tmp_path: Path) -> None:
    info = HandInfo.from_config(_make_config(tmp_path))

    transform = info.get_palm_pose()
    expected = np.eye(4, dtype=float)
    expected[:3, :3] = R.from_euler("ZYX", [30.0, 20.0, 10.0], degrees=True).as_matrix()
    expected[:3, 3] = np.array([0.3, 0.4, 0.5], dtype=float)

    assert transform.shape == (4, 4)
    assert np.allclose(transform, expected)


def test_hand_info_contact_anchor_accessors(tmp_path: Path) -> None:
    info = HandInfo.from_config(_make_config(tmp_path))

    assert np.allclose(info.get_contact_anchor_by_link("thumb_link"), np.array([0.01, 0.02, 0.03], dtype=float))

    tip_matches = info.get_contact_anchor_by_tag(includes=["tip"])
    assert set(tip_matches) == {"thumb_link", "index_link"}
    assert np.allclose(tip_matches["index_link"], np.array([0.04, 0.05, 0.06], dtype=float))

    thumb_tip_matches = info.get_contact_anchor_by_tag(includes=["thumb", "tip"])
    assert set(thumb_tip_matches) == {"thumb_link"}

    excluded_matches = info.get_contact_anchor_by_tag(includes=["tip"], excludes=["outer"])
    assert set(excluded_matches) == {"thumb_link"}

    assert info.get_contact_anchor_by_tag(includes=["missing"]) == {}


def test_hand_info_template_anchor_and_target_accessors(tmp_path: Path) -> None:
    info = HandInfo.from_config(_make_config(tmp_path))

    pinch_anchors = info.get_contact_anchor_by_template("pinch")
    assert set(pinch_anchors) == {"thumb_link"}
    assert np.allclose(pinch_anchors["thumb_link"], np.array([0.01, 0.02, 0.03], dtype=float))
    assert np.allclose(info.get_grasp_target_point("power"), np.array([1.0, 1.1, 1.2], dtype=float))


def test_hand_info_raises_key_error_for_missing_entries(tmp_path: Path) -> None:
    info = HandInfo.from_config(_make_config(tmp_path))

    with pytest.raises(KeyError):
        info.get_q_close("missing")
    with pytest.raises(KeyError):
        info.get_contact_anchor_by_link("missing_link")
    with pytest.raises(KeyError):
        info.get_contact_anchor_by_template("missing")
    with pytest.raises(KeyError):
        info.get_grasp_target_point("missing")


def test_pointed_hand_info_precomputes_seeded_surface_and_contact_points(tmp_path: Path) -> None:
    config_path = _make_collision_config(tmp_path)

    info_a = PointedHandInfo.from_config(config_path, seed=7)
    info_b = PointedHandInfo.from_config(config_path, seed=7)

    assert set(info_a.surface_points) == {"base", "thumb_link", "index_link"}
    assert set(info_a.contact_points) == {"thumb_link", "index_link"}
    assert info_a.surface_points["base"].shape[1] == 3
    assert info_a.contact_points["thumb_link"].shape == (10, 3)
    assert np.allclose(info_a.surface_points["base"], info_b.surface_points["base"])
    assert np.allclose(info_a.contact_points["thumb_link"], info_b.contact_points["thumb_link"])


def test_pointed_hand_info_get_keypoints_returns_link_local_dict(tmp_path: Path) -> None:
    info = PointedHandInfo.from_config(_make_collision_config(tmp_path), seed=11)

    all_points = info.get_keypoints(template_name=None, palm_aligned_points=False)
    assert set(all_points) == {"thumb_link", "index_link"}
    assert all_points["thumb_link"].shape == (1, 3)
    assert np.allclose(all_points["thumb_link"][0], np.array([0.001, 0.002, 0.003], dtype=float))

    pinch_points = info.get_keypoints(template_name="pinch", palm_aligned_points=False)
    assert set(pinch_points) == {"thumb_link"}


def test_pointed_hand_info_get_contact_points_filters_by_template(tmp_path: Path) -> None:
    info = PointedHandInfo.from_config(_make_collision_config(tmp_path), seed=19)

    all_contact_points = info.get_contact_points()
    pinch_contact_points = info.get_contact_points("pinch")

    assert set(all_contact_points) == {"thumb_link", "index_link"}
    assert set(pinch_contact_points) == {"thumb_link"}
    assert np.allclose(pinch_contact_points["thumb_link"], info.contact_points["thumb_link"])


def test_pointed_hand_info_adds_palm_aligned_points_to_base_link(tmp_path: Path) -> None:
    info = PointedHandInfo.from_config(_make_collision_config(tmp_path), seed=13)

    keypoints = info.get_keypoints(template_name="pinch", palm_aligned_points=True, palm_points_delta=0.05)

    assert set(keypoints) == {"thumb_link", "base"}
    assert keypoints["thumb_link"].shape == (1, 3)
    assert keypoints["base"].shape == (7, 3)

    palm_pose = info.get_palm_pose()
    center = palm_pose[:3, 3]
    basis = palm_pose[:3, :3]
    expected = center[None, :] + np.vstack(
        [
            np.zeros((1, 3), dtype=float),
            np.eye(3, dtype=float),
            -np.eye(3, dtype=float),
        ]
    ) * 0.05 @ basis.T
    assert np.allclose(keypoints["base"], expected)


def test_sample_contact_points_uses_distance_only_without_contact_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    urdf = SimpleNamespace(link_map={"thumb_link": SimpleNamespace(collisions=[object()])})
    points = np.asarray(
        [
            [0.000, 0.0, 0.0],
            [0.001, 0.0, 0.0],
            [0.002, 0.0, 0.0],
            [0.003, 0.0, 0.0],
        ],
        dtype=float,
    )
    normals = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=float,
    )
    monkeypatch.setattr(
        hand_info_module,
        "_sample_collision_surface_points",
        lambda *_args, **_kwargs: (points.copy(), normals.copy()),
    )

    sampled = hand_info_module._sample_contact_points(
        urdf,
        {"thumb_link": {"point": [0.0, 0.0, 0.0], "contact_radius": 0.01}},
        num_point_per_link=4,
    )

    assert sampled["thumb_link"].shape == (4, 3)
    sampled_sorted = sampled["thumb_link"][np.argsort(sampled["thumb_link"][:, 0])]
    points_sorted = points[np.argsort(points[:, 0])]
    assert np.allclose(sampled_sorted, points_sorted)


def test_sample_contact_points_ignores_contact_axis_and_uses_radius_only(monkeypatch: pytest.MonkeyPatch) -> None:
    urdf = SimpleNamespace(link_map={"thumb_link": SimpleNamespace(collisions=[object()])})
    points = np.asarray(
        [
            [0.000, 0.0, 0.0],
            [0.001, 0.0, 0.0],
            [0.002, 0.0, 0.0],
            [0.003, 0.0, 0.0],
        ],
        dtype=float,
    )
    normals = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )
    monkeypatch.setattr(
        hand_info_module,
        "_sample_collision_surface_points",
        lambda *_args, **_kwargs: (points.copy(), normals.copy()),
    )

    sampled = hand_info_module._sample_contact_points(
        urdf,
        {
            "thumb_link": {
                "point": [0.0, 0.0, 0.0],
                "contact_radius": 0.01,
                "contact_axis": [0.0, 0.0, 1.0],
            }
        },
        num_point_per_link=4,
    )

    expected = points
    sampled_sorted = sampled["thumb_link"][np.argsort(sampled["thumb_link"][:, 0])]
    expected_sorted = expected[np.argsort(expected[:, 0])]
    assert np.allclose(sampled_sorted, expected_sorted)


def test_pointed_hand_info_supports_multiple_collision_geometries_per_link(tmp_path: Path) -> None:
    info = PointedHandInfo.from_config(_make_multi_collision_config(tmp_path), seed=17)

    assert set(info.surface_points) == {"base", "thumb_link"}
    assert set(info.contact_points) == {"thumb_link"}
    assert info.surface_points["thumb_link"].shape[1] == 3
    assert info.contact_points["thumb_link"].shape == (10, 3)
