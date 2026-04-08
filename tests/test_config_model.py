from __future__ import annotations

from pathlib import Path

import pytest

from kmk.config.model import GripperConfig


def test_save_and_load_roundtrip_preserves_optional_fields(tmp_path: Path) -> None:
    cfg = GripperConfig(
        name="demo_hand",
        urdf_path="hand.urdf",
        xml_path=None,
        joint_order=["j1", "j2"],
        q_open=[0.1, 0.2],
        palm_pose={"trans": [0.0, 0.1, 0.2], "rpy": [1.0, 2.0, 3.0]},
        palm_points_delta=0.08,
        additional_collision_ignore_pairs=[["link_b", "link_a"], ["link_a", "link_b"]],
        contact_anchors={
            "finger_link": {
                "point": [0.01, 0.02, 0.03],
                "tags": ["thumb", "tip"],
                "contact_radius": 0.01,
            },
            "palm_link": {"point": [0.1, 0.2, 0.3], "tags": [], "contact_radius": 0.02},
        },
    )

    path = tmp_path / "config.yaml"
    cfg.save(path)
    loaded = GripperConfig.load(path)

    assert loaded.name == "demo_hand"
    assert loaded.xml_path is None
    assert loaded.joint_order == ["j1", "j2"]
    assert loaded.q_open == [0.1, 0.2]
    assert loaded.palm_points_delta == pytest.approx(0.08)
    assert loaded.additional_collision_ignore_pairs == [["link_a", "link_b"]]
    assert loaded.contact_anchors == {
        "finger_link": {
            "point": [0.01, 0.02, 0.03],
            "tags": ["thumb", "tip"],
            "contact_radius": 0.01,
        },
        "palm_link": {"point": [0.1, 0.2, 0.3], "tags": [], "contact_radius": 0.02},
    }


def test_validate_rejects_invalid_joint_order_and_link_names() -> None:
    cfg = GripperConfig(
        name="demo_hand",
        urdf_path="hand.urdf",
        joint_order=["j1"],
        palm_pose={"trans": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
        additional_collision_ignore_pairs=[["base", "tip"]],
        contact_anchors={
            "finger": {"point": [0.0, 0.0, 0.0], "tags": ["tip"], "contact_radius": 0.01},
        },
    )

    with pytest.raises(ValueError, match="joint_order"):
        cfg.validate(urdf_actuated_joint_names=["j1", "j2"])

    cfg.joint_order = ["j1", "j2"]
    cfg.q_open = [0.1]
    with pytest.raises(ValueError, match="q_open"):
        cfg.validate(urdf_actuated_joint_names=["j1", "j2"])

    cfg.q_open = [0.1, 0.2]
    with pytest.raises(ValueError, match="valid URDF link names"):
        cfg.validate(
            urdf_actuated_joint_names=["j1", "j2"],
            urdf_link_names=["base", "finger"],
        )


def test_validate_rejects_invalid_contact_anchor_shape() -> None:
    cfg = GripperConfig(
        name="demo_hand",
        urdf_path="hand.urdf",
        joint_order=["j1", "j2"],
        q_open=[0.1, 0.2],
        palm_pose={"trans": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
        contact_anchors={
            "finger": {"point": [0.0, 0.0], "tags": ["tip"], "contact_radius": 0.01},
        },
    )

    with pytest.raises(ValueError, match="point"):
        cfg.validate(
            urdf_actuated_joint_names=["j1", "j2"],
            urdf_link_names=["finger"],
        )


def test_validate_rejects_nonpositive_contact_radius() -> None:
    cfg = GripperConfig(
        name="demo_hand",
        urdf_path="hand.urdf",
        joint_order=["j1", "j2"],
        q_open=[0.1, 0.2],
        palm_pose={"trans": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
        contact_anchors={
            "finger": {"point": [0.0, 0.0, 0.0], "tags": ["tip"], "contact_radius": 0.0},
        },
    )

    with pytest.raises(ValueError, match="contact_radius"):
        cfg.validate(
            urdf_actuated_joint_names=["j1", "j2"],
            urdf_link_names=["finger"],
        )


def test_save_and_load_roundtrip_preserves_grasp_templates(tmp_path: Path) -> None:
    cfg = GripperConfig(
        name="demo_hand",
        urdf_path="hand.urdf",
        xml_path=None,
        joint_order=["j1", "j2"],
        q_open=[0.1, 0.2],
        palm_pose={"trans": [0.0, 0.1, 0.2], "rpy": [1.0, 2.0, 3.0]},
        palm_points_delta=0.06,
        contact_anchors={
            "finger_link": {
                "point": [0.01, 0.02, 0.03],
                "tags": ["thumb", "tip"],
                "contact_radius": 0.01,
            }
        },
    )
    cfg.grasp_templates = {
        "pinch": {
            "q_close": [0.0, 0.0],
            "q_open": [0.3, 0.4],
            "grasp_target_point": [0.1, 0.2, 0.3],
            "active_contact_anchors": ["finger_link"],
        }
    }

    path = tmp_path / "config.yaml"
    cfg.save(path)
    loaded = GripperConfig.load(path)

    assert getattr(loaded, "grasp_templates", None) == {
        "pinch": {
            "q_close": [0.0, 0.0],
            "q_open": [0.3, 0.4],
            "grasp_target_point": [0.1, 0.2, 0.3],
            "active_contact_anchors": ["finger_link"],
        }
    }


def test_validate_rejects_invalid_grasp_template_schema() -> None:
    cfg = GripperConfig(
        name="demo_hand",
        urdf_path="hand.urdf",
        joint_order=["j1", "j2"],
        q_open=[0.1, 0.2],
        palm_pose={"trans": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
        contact_anchors={
            "finger": {"point": [0.0, 0.0, 0.0], "tags": ["tip"], "contact_radius": 0.01},
        },
    )
    cfg.grasp_templates = {
        "pinch": {
            "q_close": [0.0],
            "q_open": [0.0, 0.0],
            "grasp_target_point": [0.1, 0.2],
            "active_contact_anchors": ["finger", "finger"],
        }
    }

    with pytest.raises(ValueError, match="grasp_templates"):
        cfg.validate(
            urdf_actuated_joint_names=["j1", "j2"],
            urdf_link_names=["finger"],
        )
