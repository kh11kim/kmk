from __future__ import annotations

from pathlib import Path

import yaml

from kmk.wizard import cli as wizard_cli


def _write_gripper_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "hand.urdf").write_text(
        "\n".join(
            [
                "<?xml version='1.0'?>",
                "<robot name='demo'>",
                "  <link name='base' />",
                "  <link name='finger_a' />",
                "  <link name='finger_b' />",
                "  <joint name='j1' type='revolute'><parent link='base'/><child link='finger_a'/></joint>",
                "  <joint name='j2' type='revolute'><parent link='base'/><child link='finger_b'/></joint>",
                "</robot>",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (root / "hand.xml").write_text("<mujoco model='demo' />\n", encoding="utf-8")


def test_cli_new_mode_builds_session_and_runs_gui(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "gripper"
    _write_gripper_root(root)

    prompted: list[str] = []
    responses = iter(["demo_cfg", "", "", ""])
    monkeypatch.setattr("builtins.input", lambda prompt: prompted.append(prompt) or next(responses))

    created = {"global": None, "keypoint": None, "template": None, "preview": None}

    class _FakeApp:
        def __init__(self, session) -> None:
            self.session = session

        def run_until_complete(self) -> Path:
            self.session.config.save(self.session.save_path)
            return self.session.save_path

    def _fake_create_global_app(session, **kwargs):
        _ = kwargs
        created["global"] = session
        return _FakeApp(session)

    def _fake_create_keypoint_app(session, **kwargs):
        _ = kwargs
        created["keypoint"] = session
        return _FakeApp(session)

    def _fake_create_template_app(session, **kwargs):
        _ = kwargs
        created["template"] = session
        return _FakeApp(session)

    def _fake_create_preview_app(session, **kwargs):
        _ = kwargs
        created["preview"] = session
        return _FakeApp(session)

    monkeypatch.setattr(wizard_cli, "create_global_app", _fake_create_global_app)
    monkeypatch.setattr(wizard_cli, "create_keypoint_app", _fake_create_keypoint_app)
    monkeypatch.setattr(wizard_cli, "create_template_app", _fake_create_template_app)
    monkeypatch.setattr(wizard_cli, "create_preview_app", _fake_create_preview_app)

    rc = wizard_cli.main(["--gripper-root", str(root)])
    assert rc == 0
    assert prompted[0] == "Config name: "
    assert any("URDF path [hand.urdf]" in prompt for prompt in prompted)
    assert any("XML path [hand.xml]" in prompt for prompt in prompted)
    session = created["global"]
    assert created["keypoint"] is session
    assert created["template"] is session
    assert created["preview"] is session
    saved = yaml.safe_load(session.save_path.read_text(encoding="utf-8"))
    assert session.config.name == "demo_cfg"
    assert session.config.joint_order == ["j1", "j2"]
    assert saved["urdf_path"] == "hand.urdf"
    assert saved["xml_path"] == "hand.xml"


def test_cli_from_config_preserves_joint_order(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "gripper"
    _write_gripper_root(root)
    config_path = root / "existing.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "loaded",
                "urdf_path": "hand.urdf",
                "xml_path": "hand.xml",
                "joint_order": ["j2", "j1"],
                "palm_pose": {"trans": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("builtins.input", lambda prompt: (_ for _ in ()).throw(AssertionError(prompt)))

    captured = {}

    class _FakeApp:
        def __init__(self, session) -> None:
            self.session = session

        def run_until_complete(self) -> Path:
            captured["joint_order"] = list(self.session.config.joint_order)
            return self.session.save_path

    monkeypatch.setattr(wizard_cli, "create_global_app", lambda session, **kwargs: _FakeApp(session))
    monkeypatch.setattr(wizard_cli, "create_keypoint_app", lambda session, **kwargs: _FakeApp(session))
    monkeypatch.setattr(wizard_cli, "create_template_app", lambda session, **kwargs: _FakeApp(session))
    monkeypatch.setattr(wizard_cli, "create_preview_app", lambda session, **kwargs: _FakeApp(session))

    rc = wizard_cli.main(["--gripper-root", str(root), "--from-config", "existing.yaml"])
    assert rc == 0
    assert captured["joint_order"] == ["j2", "j1"]


def test_cli_from_config_relative_path_resolves_from_gripper_root(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "gripper"
    _write_gripper_root(root)
    nested = root / "configs"
    nested.mkdir(parents=True, exist_ok=True)
    config_path = nested / "existing.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "loaded",
                "urdf_path": "hand.urdf",
                "joint_order": ["j2", "j1"],
                "palm_pose": {"trans": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    cwd_config = tmp_path / "configs"
    cwd_config.mkdir(parents=True, exist_ok=True)
    (cwd_config / "existing.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "wrong",
                "urdf_path": "hand.urdf",
                "joint_order": ["j1", "j2"],
                "palm_pose": {"trans": [1.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("builtins.input", lambda prompt: (_ for _ in ()).throw(AssertionError(prompt)))
    monkeypatch.chdir(tmp_path)

    captured = {}

    class _FakeApp:
        def __init__(self, session) -> None:
            self.session = session

        def run_until_complete(self) -> Path:
            captured["name"] = self.session.config.name
            captured["from_config"] = self.session.from_config
            return self.session.save_path

    monkeypatch.setattr(wizard_cli, "create_global_app", lambda session, **kwargs: _FakeApp(session))
    monkeypatch.setattr(wizard_cli, "create_keypoint_app", lambda session, **kwargs: _FakeApp(session))
    monkeypatch.setattr(wizard_cli, "create_template_app", lambda session, **kwargs: _FakeApp(session))
    monkeypatch.setattr(wizard_cli, "create_preview_app", lambda session, **kwargs: _FakeApp(session))

    rc = wizard_cli.main(["--gripper-root", str(root), "--from-config", "configs/existing.yaml"])
    assert rc == 0
    assert captured["name"] == "loaded"
    assert captured["from_config"] == "configs/existing.yaml"


def test_cli_prefills_preferred_xml_candidate_when_multiple_exist(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "gripper"
    _write_gripper_root(root)
    (root / "scene_hand.xml").write_text("<mujoco model='scene' />\n", encoding="utf-8")
    (root / "other.xml").write_text("<mujoco model='other' />\n", encoding="utf-8")

    prompted: list[str] = []
    responses = iter(["demo_cfg", "", "", ""])
    monkeypatch.setattr("builtins.input", lambda prompt: prompted.append(prompt) or next(responses))

    captured = {}

    class _FakeApp:
        def __init__(self, session) -> None:
            self.session = session

        def run_until_complete(self) -> Path:
            captured["xml_path"] = self.session.config.xml_path
            return self.session.save_path

    monkeypatch.setattr(wizard_cli, "create_global_app", lambda session, **kwargs: _FakeApp(session))
    monkeypatch.setattr(wizard_cli, "create_keypoint_app", lambda session, **kwargs: _FakeApp(session))
    monkeypatch.setattr(wizard_cli, "create_template_app", lambda session, **kwargs: _FakeApp(session))
    monkeypatch.setattr(wizard_cli, "create_preview_app", lambda session, **kwargs: _FakeApp(session))

    rc = wizard_cli.main(["--gripper-root", str(root)])
    assert rc == 0
    assert any("XML path [hand.xml]" in prompt for prompt in prompted)
    assert captured["xml_path"] == "hand.xml"


def test_cli_prefills_preferred_urdf_candidate_when_multiple_exist(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "gripper"
    _write_gripper_root(root)
    (root / "scene_hand.urdf").write_text("<robot name='scene' />\n", encoding="utf-8")
    (root / "left_hand.urdf").write_text("<robot name='left' />\n", encoding="utf-8")

    prompted: list[str] = []
    responses = iter(["demo_cfg", "", "", ""])
    monkeypatch.setattr("builtins.input", lambda prompt: prompted.append(prompt) or next(responses))

    captured = {}

    class _FakeApp:
        def __init__(self, session) -> None:
            self.session = session

        def run_until_complete(self) -> Path:
            captured["urdf_path"] = self.session.config.urdf_path
            return self.session.save_path

    monkeypatch.setattr(wizard_cli, "create_global_app", lambda session, **kwargs: _FakeApp(session))
    monkeypatch.setattr(wizard_cli, "create_keypoint_app", lambda session, **kwargs: _FakeApp(session))
    monkeypatch.setattr(wizard_cli, "create_template_app", lambda session, **kwargs: _FakeApp(session))
    monkeypatch.setattr(wizard_cli, "create_preview_app", lambda session, **kwargs: _FakeApp(session))

    rc = wizard_cli.main(["--gripper-root", str(root)])
    assert rc == 0
    assert any("URDF path [hand.urdf]" in prompt for prompt in prompted)
    assert captured["urdf_path"] == "hand.urdf"


def test_cli_runs_template_stage_after_keypoint(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "gripper"
    _write_gripper_root(root)

    prompted: list[str] = []
    responses = iter(["demo_cfg", "", "", ""])
    monkeypatch.setattr("builtins.input", lambda prompt: prompted.append(prompt) or next(responses))

    calls: list[str] = []

    class _FakeApp:
        def __init__(self, session) -> None:
            self.session = session

        def run_until_complete(self) -> Path:
            calls.append(self.session.config.name)
            return self.session.save_path

    monkeypatch.setattr(wizard_cli, "create_global_app", lambda session, **kwargs: _FakeApp(session))
    monkeypatch.setattr(wizard_cli, "create_keypoint_app", lambda session, **kwargs: _FakeApp(session))
    monkeypatch.setattr(wizard_cli, "create_template_app", lambda session, **kwargs: _FakeApp(session), raising=False)
    monkeypatch.setattr(wizard_cli, "create_preview_app", lambda session, **kwargs: _FakeApp(session), raising=False)

    rc = wizard_cli.main(["--gripper-root", str(root)])
    assert rc == 0
    assert len(calls) == 4
