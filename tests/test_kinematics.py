from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from kmk import HandInfo
from kmk.config.model import GripperConfig
from kmk.kinematics import HandKinematics


def _make_kin_config(tmp_path: Path) -> Path:
    root = tmp_path / "kin_hand"
    root.mkdir(parents=True, exist_ok=True)
    urdf_path = root / "kin.urdf"
    urdf_path.write_text(
        "\n".join(
            [
                "<?xml version='1.0'?>",
                "<robot name='kin'>",
                "  <link name='base' />",
                "  <link name='link1' />",
                "  <link name='link2' />",
                "  <joint name='j1' type='revolute'>",
                "    <parent link='base'/>",
                "    <child link='link1'/>",
                "    <origin xyz='0 0 0' rpy='0 0 0'/>",
                "    <axis xyz='0 0 1'/>",
                "    <limit lower='-3.14' upper='3.14' effort='1' velocity='1'/>",
                "  </joint>",
                "  <joint name='j2' type='revolute'>",
                "    <parent link='link1'/>",
                "    <child link='link2'/>",
                "    <origin xyz='1 0 0' rpy='0 0 0'/>",
                "    <axis xyz='0 0 1'/>",
                "    <limit lower='-3.14' upper='3.14' effort='1' velocity='1'/>",
                "  </joint>",
                "</robot>",
                "",
            ]
        ),
        encoding="utf-8",
    )
    config_path = root / "kin.yaml"
    GripperConfig(
        name="kin",
        urdf_path="kin.urdf",
        joint_order=["j1", "j2"],
        q_open=[0.0, 0.0],
        palm_pose={"trans": [0.1, 0.2, 0.3], "rpy": [0.0, 0.0, 0.0]},
    ).save(config_path)
    return config_path


def test_hand_kinematics_forward_kinematics_shapes(tmp_path: Path) -> None:
    kin = HandKinematics(_make_kin_config(tmp_path))

    fk_1 = kin.forward_kinematics(torch.zeros(2))
    fk_2 = kin.forward_kinematics(torch.zeros(4, 2))
    fk_3 = kin.forward_kinematics(torch.zeros(3, 5, 2))

    assert fk_1["base"].shape == (1, 4, 4)
    assert fk_2["link1"].shape == (4, 4, 4)
    assert fk_3["link2"].shape == (3, 5, 4, 4)


def test_hand_kinematics_forward_kinematics_numeric_pose(tmp_path: Path) -> None:
    kin = HandKinematics(_make_kin_config(tmp_path))
    q = torch.tensor([np.pi / 2.0, 0.0], dtype=torch.float32)

    fk = kin.forward_kinematics(q)

    assert torch.allclose(fk["base"][0], torch.eye(4), atol=1e-5)
    assert torch.allclose(fk["link2"][0, :3, 3], torch.tensor([0.0, 1.0, 0.0]), atol=1e-4)


def test_hand_kinematics_transform_link_points_supports_variable_p(tmp_path: Path) -> None:
    kin = HandKinematics(_make_kin_config(tmp_path))
    q = torch.tensor([[np.pi / 2.0, 0.0], [0.0, 0.0]], dtype=torch.float32)

    points = {
        "link1": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        "link2": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32),
    }
    transformed = kin.transform_link_points(q, points)

    assert transformed["link1"].shape == (2, 1, 3)
    assert transformed["link2"].shape == (2, 2, 3)
    assert torch.allclose(transformed["link2"][0, 0], torch.tensor([0.0, 1.0, 0.0]), atol=1e-4)
    assert torch.allclose(transformed["link2"][0, 1], torch.tensor([0.0, 2.0, 0.0]), atol=1e-4)


def test_hand_kinematics_transform_link_points_supports_multi_batch_dims(tmp_path: Path) -> None:
    kin = HandKinematics(_make_kin_config(tmp_path))
    q = torch.zeros(2, 3, 2, dtype=torch.float32)
    q[1, 2, 0] = torch.tensor(np.pi / 2.0, dtype=torch.float32)
    points = {"link2": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)}

    transformed = kin.transform_link_points(q, points)

    assert transformed["link2"].shape == (2, 3, 1, 3)
    assert torch.allclose(transformed["link2"][0, 0, 0], torch.tensor([2.0, 0.0, 0.0]), atol=1e-4)
    assert torch.allclose(transformed["link2"][1, 2, 0], torch.tensor([0.0, 2.0, 0.0]), atol=1e-4)


def test_hand_kinematics_get_palm_pose_supports_batch_shape(tmp_path: Path) -> None:
    kin = HandKinematics(_make_kin_config(tmp_path))

    palm = kin.get_palm_pose()
    palm_batched = kin.get_palm_pose(batch_shape=(2, 3))

    assert palm.shape == (4, 4)
    assert palm_batched.shape == (2, 3, 4, 4)
    assert torch.allclose(palm_batched[0, 0], palm)


def test_hand_kinematics_raises_for_unknown_link(tmp_path: Path) -> None:
    kin = HandKinematics(_make_kin_config(tmp_path))

    with pytest.raises(KeyError):
        kin.transform_link_points(torch.zeros(2), {"missing": torch.zeros(1, 3)})


def test_hand_kinematics_supports_autograd(tmp_path: Path) -> None:
    kin = HandKinematics(_make_kin_config(tmp_path))
    q = torch.zeros(2, requires_grad=True)
    points = {"link2": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)}

    transformed = kin.transform_link_points(q, points)
    loss = transformed["link2"].sum()
    loss.backward()

    assert q.grad is not None
    assert q.grad.shape == (2,)
    assert torch.isfinite(q.grad).all()
