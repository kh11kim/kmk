from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

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


def _make_points() -> dict[str, torch.Tensor]:
    return {
        "link1": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        "link2": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32),
    }


def _load_variant(config_path: Path, class_name: str):
    from kmk import _kinematics_experimental as experimental_module

    return getattr(experimental_module, class_name)(config_path)


@pytest.mark.parametrize(
    "class_name",
    [
        "ExperimentalHandKinematics",
        "VectorizedHandKinematics",
        "PackedTransformHandKinematics",
    ],
)
def test_experimental_matches_baseline_forward_kinematics(tmp_path: Path, class_name: str) -> None:
    config_path = _make_kin_config(tmp_path)
    baseline = HandKinematics(config_path)
    experimental = _load_variant(config_path, class_name)

    q = torch.tensor([[np.pi / 2.0, 0.0], [0.0, 0.0]], dtype=torch.float32)

    fk_baseline = baseline.forward_kinematics(q)
    fk_experimental = experimental.forward_kinematics(q)

    assert set(fk_experimental) == set(fk_baseline)
    for link_name in fk_baseline:
        assert fk_experimental[link_name].shape == fk_baseline[link_name].shape
        assert torch.allclose(fk_experimental[link_name], fk_baseline[link_name], atol=1e-5)


@pytest.mark.parametrize(
    "class_name",
    [
        "ExperimentalHandKinematics",
        "VectorizedHandKinematics",
        "PackedTransformHandKinematics",
    ],
)
def test_experimental_matches_baseline_transform_link_points_multi_batch(tmp_path: Path, class_name: str) -> None:
    config_path = _make_kin_config(tmp_path)
    baseline = HandKinematics(config_path)
    experimental = _load_variant(config_path, class_name)

    q = torch.zeros(2, 3, 2, dtype=torch.float32)
    q[1, 2, 0] = torch.tensor(np.pi / 2.0, dtype=torch.float32)
    points = _make_points()

    transformed_baseline = baseline.transform_link_points(q, points)
    transformed_experimental = experimental.transform_link_points(q, points)

    assert set(transformed_experimental) == set(transformed_baseline)
    for link_name in transformed_baseline:
        assert transformed_experimental[link_name].shape == transformed_baseline[link_name].shape
        assert torch.allclose(transformed_experimental[link_name], transformed_baseline[link_name], atol=1e-5)


@pytest.mark.parametrize(
    "class_name",
    [
        "ExperimentalHandKinematics",
        "VectorizedHandKinematics",
        "PackedTransformHandKinematics",
    ],
)
def test_experimental_supports_autograd_smoke(tmp_path: Path, class_name: str) -> None:
    config_path = _make_kin_config(tmp_path)
    experimental = _load_variant(config_path, class_name)

    q = torch.zeros(2, requires_grad=True)
    transformed = experimental.transform_link_points(q, {"link2": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)})
    loss = transformed["link2"].sum()
    loss.backward()

    assert q.grad is not None
    assert q.grad.shape == (2,)
    assert torch.isfinite(q.grad).all()
