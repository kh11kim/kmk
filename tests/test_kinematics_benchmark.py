from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from kmk.config.model import GripperConfig


def _make_benchmark_config(tmp_path: Path) -> Path:
    root = tmp_path / "bench_hand"
    root.mkdir(parents=True, exist_ok=True)
    urdf_path = root / "bench.urdf"
    urdf_path.write_text(
        "\n".join(
            [
                "<?xml version='1.0'?>",
                "<robot name='bench'>",
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
    config_path = root / "bench.yaml"
    GripperConfig(
        name="bench",
        urdf_path="bench.urdf",
        joint_order=["j1", "j2"],
        q_open=[0.0, 0.0],
        palm_pose={"trans": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
    ).save(config_path)
    return config_path


def _run_benchmark(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    script = cwd / "scripts" / "benchmark_kinematics.py"
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def test_benchmark_cli_emits_json_metrics_for_cpu(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = _make_benchmark_config(tmp_path)

    proc = _run_benchmark(
        [
            "--config-path",
            str(config_path),
            "--device",
            "cpu",
            "--batch-size",
            "4",
            "--repeats",
            "3",
            "--warmup",
            "1",
        ],
        cwd=root,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["device_requested"] == "cpu"
    assert payload["device_used"] == "cpu"
    assert set(payload["benchmarks"]) >= {"forward_kinematics", "transform_link_points"}
    assert payload["benchmarks"]["forward_kinematics"]["mean_ms"] > 0.0
    assert payload["benchmarks"]["transform_link_points"]["mean_ms"] > 0.0


def test_benchmark_cli_rejects_unknown_device(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = _make_benchmark_config(tmp_path)

    proc = _run_benchmark(
        [
            "--config-path",
            str(config_path),
            "--device",
            "banana",
        ],
        cwd=root,
    )

    assert proc.returncode != 0
