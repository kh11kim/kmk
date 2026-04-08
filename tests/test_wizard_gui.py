from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from kmk.config.model import GripperConfig
from kmk.config.parse import ParsedUrdfNames
import kmk.wizard.gui as wizard_gui
from kmk.wizard.gui import PreviewWizardGui, create_global_app, create_keypoint_app
from kmk.wizard.session import WizardSession


class _Widget:
    def __init__(self, value) -> None:
        self._value = value
        self.content = value
        self._callbacks = []

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        self.content = value
        for callback in list(self._callbacks):
            callback(None)

    def on_update(self, fn):
        self._callbacks.append(fn)
        return fn


class _Button:
    def __init__(self, **kwargs) -> None:
        self._callbacks = []
        self.color = kwargs.get("color")
        self.label = kwargs.get("label")
        self.disabled = kwargs.get("disabled", False)

    def on_click(self, fn):
        self._callbacks.append(fn)
        return fn

    def click(self) -> None:
        for callback in list(self._callbacks):
            callback(None)


class _Folder:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None


class _Dropdown(_Widget):
    def __init__(self, value="") -> None:
        super().__init__(value)
        self.options = ("",)


class _Frame:
    def __init__(self, name: str, kwargs: dict[str, object]) -> None:
        self.name = name
        self.position = kwargs.get("position")
        self.wxyz = kwargs.get("wxyz")
        self.visible = kwargs.get("visible", True)
        self._callbacks = []
        self._drag_start_callbacks = []
        self._drag_end_callbacks = []

    def on_update(self, fn):
        self._callbacks.append(fn)
        return fn

    def on_drag_start(self, fn):
        self._drag_start_callbacks.append(fn)
        return fn

    def on_drag_end(self, fn):
        self._drag_end_callbacks.append(fn)
        return fn

    def trigger_update(self, *, position=None, wxyz=None) -> None:
        if position is not None:
            self.position = position
        if wxyz is not None:
            self.wxyz = wxyz
        for callback in list(self._callbacks):
            callback(None)

    def trigger_drag_start(self) -> None:
        for callback in list(self._drag_start_callbacks):
            callback(None)

    def trigger_drag_end(self) -> None:
        for callback in list(self._drag_end_callbacks):
            callback(None)

    def remove(self) -> None:
        return None


class _Mesh:
    def __init__(self, name: str) -> None:
        self.name = name
        self._callbacks = []

    def on_click(self, fn):
        self._callbacks.append(fn)
        return fn

    def trigger_click(self) -> None:
        event = SimpleNamespace(target=self)
        for callback in list(self._callbacks):
            callback(event)


class _Sphere(_Mesh):
    def __init__(self, name: str, kwargs: dict[str, object]) -> None:
        super().__init__(name)
        self.position = kwargs.get("position")
        self.radius = kwargs.get("radius")
        self.color = kwargs.get("color")
        self.opacity = kwargs.get("opacity")
        self.visible = kwargs.get("visible", True)

    def remove(self) -> None:
        return None


class _Scene:
    def __init__(self) -> None:
        self.frames = []
        self.controls = []
        self.spheres = []
        self.point_clouds = []

    def add_frame(self, name, **kwargs):
        handle = _Frame(name, kwargs)
        self.frames.append(handle)
        return handle

    def add_transform_controls(self, name, **kwargs):
        handle = _Frame(name, kwargs)
        self.controls.append(handle)
        return handle

    def add_icosphere(self, name, **kwargs):
        handle = _Sphere(name, kwargs)
        self.spheres.append(handle)
        return handle

    def add_point_cloud(self, name, **kwargs):
        handle = SimpleNamespace(name=name, **kwargs)
        self.point_clouds.append(handle)
        return handle


class _Gui:
    def add_text(self, *args, initial_value="", **kwargs):
        _ = (args, kwargs)
        return _Widget(initial_value)

    def add_markdown(self, content, **kwargs):
        _ = kwargs
        return _Widget(content)

    def add_vector3(self, *args, initial_value=(0.0, 0.0, 0.0), **kwargs):
        _ = kwargs
        value = args[1] if len(args) > 1 else initial_value
        return _Widget(tuple(value))

    def add_button(self, *args, **kwargs):
        _ = args
        return _Button(**kwargs)

    def add_slider(self, *args, initial_value=0.0, **kwargs):
        _ = (args, kwargs)
        return _Widget(float(initial_value))

    def add_dropdown(self, *args, **kwargs):
        _ = (args, kwargs)
        return _Dropdown()

    def add_checkbox(self, *args, initial_value=False, **kwargs):
        _ = (args, kwargs)
        return _Widget(bool(initial_value))

    def add_folder(self, *args, **kwargs):
        _ = (args, kwargs)
        return _Folder()


class _Server:
    def __init__(self) -> None:
        self.gui = _Gui()
        self.scene = _Scene()
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class _FakePreviewHandInfo:
    def __init__(self) -> None:
        self.template_names = ["pinch"]
        self.surface_points = {"base": np.asarray([[0.0, 0.0, 0.0]], dtype=float)}
        self.contact_points = {
            "finger_a": np.asarray([[0.01, 0.0, 0.0]], dtype=float),
            "finger_b": np.asarray([[0.02, 0.0, 0.0]], dtype=float),
        }

    def get_q_open(self, template: str = "global") -> np.ndarray:
        _ = template
        return np.asarray([0.0, 0.0], dtype=float)

    def get_q_close(self, template_name: str) -> np.ndarray:
        _ = template_name
        return np.asarray([0.1, 0.2], dtype=float)

    def get_contact_points(self, template_name: str | None = None) -> dict[str, np.ndarray]:
        if template_name == "pinch":
            return {"finger_a": self.contact_points["finger_a"].copy()}
        return {name: points.copy() for name, points in self.contact_points.items()}

    def get_keypoints(
        self,
        template_name: str | None = None,
        palm_aligned_points: bool = True,
        palm_points_delta: float = 0.05,
    ) -> dict[str, np.ndarray]:
        _ = (template_name, palm_aligned_points, palm_points_delta)
        return {"finger_a": np.asarray([[0.03, 0.0, 0.0]], dtype=float)}

    def get_grasp_target_point(self, template_name: str) -> np.ndarray:
        _ = template_name
        return np.asarray([0.0, 0.0, 0.0], dtype=float)


@dataclass
class _FakeGripper:
    handle: object
    joint_order: list[str] | None = None
    q_dict: dict[str, float] | None = None
    lb: dict[str, float] | None = None
    ub: dict[str, float] | None = None

    def set_joint_angles(self, q) -> None:
        if self.joint_order is None:
            return
        self.q_dict = dict(zip(self.joint_order, [float(v) for v in q]))


def _make_session(tmp_path: Path) -> WizardSession:
    root = tmp_path / "gripper"
    root.mkdir(parents=True, exist_ok=True)
    (root / "hand.urdf").write_text("<robot />\n", encoding="utf-8")
    config = GripperConfig(
        name="demo",
        urdf_path="hand.urdf",
        xml_path=None,
        joint_order=["j1", "j2"],
        palm_pose={"trans": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
    )
    return WizardSession(
        mode="new",
        gripper_root=".",
        gripper_root_abs=root,
        save_path=root / "demo.yaml",
        config=config,
        from_config=None,
        urdf=ParsedUrdfNames(
            joint_names=("j1", "j2"),
            link_names=("base", "finger_a", "finger_b"),
            actuated_joint_names=("j1", "j2"),
        ),
        xml=None,
    )


def test_create_global_app_syncs_palm_pose_and_collision_pairs(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    server = _Server()
    meshes = [_Mesh("/demo/finger_a/finger_a.stl"), _Mesh("/demo/finger_b/finger_b.stl")]

    def _fake_gripper_factory(**kwargs):
        _ = kwargs
        return _FakeGripper(
            handle=SimpleNamespace(_meshes=meshes),
            joint_order=["j1", "j2"],
            q_dict={"j1": 0.0, "j2": 0.0},
            lb={"j1": -1.0, "j2": -1.0},
            ub={"j1": 1.0, "j2": 1.0},
        )

    app = create_global_app(session, server=server, gripper_factory=_fake_gripper_factory)

    app.palm_trans_widget.value = (0.1, 0.2, 0.3)
    app.palm_rpy_widget.value = (10.0, 20.0, 30.0)
    app.palm_points_delta_widget.value = 0.08
    app.q_open_joint_widgets["j1"].value = 0.25
    app.q_open_joint_widgets["j2"].value = -0.5
    assert app.config.palm_pose == {"trans": [0.1, 0.2, 0.3], "rpy": [10.0, 20.0, 30.0]}
    assert app.config.palm_points_delta == pytest.approx(0.08)
    assert app.config.q_open == [0.0, 0.0]

    app.set_q_open_button.click()
    assert app.config.q_open == [0.25, -0.5]

    app.add_collision_pair_button.click()
    meshes[0].trigger_click()
    meshes[1].trigger_click()
    app.set_collision_pair_button.click()
    assert app.config.additional_collision_ignore_pairs == [["finger_a", "finger_b"]]


def test_create_global_app_preserves_palm_pose_components_on_widget_updates(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    session.config.palm_pose = {"trans": [0.1, 0.2, 0.3], "rpy": [10.0, 20.0, 30.0]}
    server = _Server()

    def _fake_gripper_factory(**kwargs):
        _ = kwargs
        return _FakeGripper(
            handle=SimpleNamespace(_meshes=[]),
            joint_order=["j1", "j2"],
            q_dict={"j1": 0.0, "j2": 0.0},
            lb={"j1": -1.0, "j2": -1.0},
            ub={"j1": 1.0, "j2": 1.0},
        )

    app = create_global_app(session, server=server, gripper_factory=_fake_gripper_factory)
    initial_controls = app.palm_controls

    app.palm_rpy_widget.value = (15.0, 25.0, 35.0)
    assert app.config.palm_pose["trans"] == [0.1, 0.2, 0.3]
    assert app.palm_controls is not initial_controls
    assert tuple(float(v) for v in app.palm_controls.position) == (0.1, 0.2, 0.3)

    second_controls = app.palm_controls
    app.palm_trans_widget.value = (0.4, 0.5, 0.6)
    assert app.config.palm_pose["rpy"] == pytest.approx([15.0, 25.0, 35.0])
    assert app.palm_controls is not second_controls
    assert tuple(float(v) for v in app.palm_controls.position) == (0.4, 0.5, 0.6)

    app.save_and_continue_button.click()
    saved = yaml.safe_load(session.save_path.read_text(encoding="utf-8"))
    assert saved["palm_pose"]["trans"] == [0.4, 0.5, 0.6]
    assert saved["palm_pose"]["rpy"] == pytest.approx([15.0, 25.0, 35.0])
    assert saved["palm_points_delta"] == pytest.approx(0.05)
    assert server.stopped is True


def test_create_global_app_ignores_stale_palm_control_update_after_widget_change(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    session.config.palm_pose = {"trans": [0.1, 0.2, 0.3], "rpy": [10.0, 20.0, 30.0]}
    server = _Server()

    def _fake_gripper_factory(**kwargs):
        _ = kwargs
        return _FakeGripper(
            handle=SimpleNamespace(_meshes=[]),
            joint_order=["j1", "j2"],
            q_dict={"j1": 0.0, "j2": 0.0},
            lb={"j1": -1.0, "j2": -1.0},
            ub={"j1": 1.0, "j2": 1.0},
        )

    app = create_global_app(session, server=server, gripper_factory=_fake_gripper_factory)

    app.palm_rpy_widget.value = (15.0, 25.0, 35.0)
    old_controls = app.palm_controls
    stale_wxyz = wizard_gui._rpy_to_wxyz([10.0, 20.0, 30.0])
    old_controls.trigger_update(position=(0.0, 0.0, 0.0), wxyz=stale_wxyz)

    assert app.config.palm_pose["trans"] == [0.1, 0.2, 0.3]
    assert app.config.palm_pose["rpy"] == pytest.approx([15.0, 25.0, 35.0])
    assert tuple(float(v) for v in app.palm_trans_widget.value) == (0.1, 0.2, 0.3)
    assert tuple(float(v) for v in app.palm_rpy_widget.value) == (15.0, 25.0, 35.0)

    fresh_wxyz = wizard_gui._rpy_to_wxyz([5.0, 6.0, 7.0])
    app.palm_controls.trigger_drag_start()
    app.palm_controls.trigger_update(position=(0.7, 0.8, 0.9), wxyz=fresh_wxyz)
    app.palm_controls.trigger_drag_end()
    assert app.config.palm_pose["trans"] == [0.7, 0.8, 0.9]
    assert app.config.palm_pose["rpy"] == pytest.approx([5.0, 6.0, 7.0])


def test_create_global_app_ignores_non_drag_palm_control_updates(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    session.config.palm_pose = {"trans": [0.1, 0.2, 0.3], "rpy": [10.0, 20.0, 30.0]}
    server = _Server()

    def _fake_gripper_factory(**kwargs):
        _ = kwargs
        return _FakeGripper(
            handle=SimpleNamespace(_meshes=[]),
            joint_order=["j1", "j2"],
            q_dict={"j1": 0.0, "j2": 0.0},
            lb={"j1": -1.0, "j2": -1.0},
            ub={"j1": 1.0, "j2": 1.0},
        )

    app = create_global_app(session, server=server, gripper_factory=_fake_gripper_factory)

    app.palm_controls.trigger_update(
        position=(0.7, 0.8, 0.9),
        wxyz=wizard_gui._rpy_to_wxyz([5.0, 6.0, 7.0]),
    )

    assert app.config.palm_pose["trans"] == [0.1, 0.2, 0.3]
    assert app.config.palm_pose["rpy"] == [10.0, 20.0, 30.0]


def test_saved_contact_anchor_sphere_uses_visible_rgb_color(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    session.config.contact_anchors = {
        "finger_a": {
            "point": [0.01, 0.02, 0.03],
            "contact_radius": 0.007,
            "tags": [],
        }
    }
    server = _Server()

    app = create_keypoint_app(
        session,
        server=server,
        gripper_factory=lambda **kwargs: _FakeGripper(
            handle=SimpleNamespace(_meshes=[]),
            joint_order=["j1", "j2"],
            q_dict={"j1": 0.0, "j2": 0.0},
            lb={"j1": -1.0, "j2": -1.0},
            ub={"j1": 1.0, "j2": 1.0},
        ),
    )

    sphere = app.contact_anchor_saved_spheres["finger_a"]
    assert sphere.color == (51, 178, 255)


def test_preview_render_filters_contact_points_by_template() -> None:
    server = _Server()
    hand_info = _FakePreviewHandInfo()
    gripper = _FakeGripper(
        handle=SimpleNamespace(_meshes=[]),
        joint_order=["j1", "j2"],
        q_dict={"j1": 0.0, "j2": 0.0},
        lb={"j1": -1.0, "j2": -1.0},
        ub={"j1": 1.0, "j2": 1.0},
    )
    app = PreviewWizardGui(
        session=None,
        config=GripperConfig(
            name="demo",
            urdf_path="hand.urdf",
            joint_order=["j1", "j2"],
            palm_pose={"trans": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
        ),
        save_path=Path("/tmp/demo.yaml"),
        server=server,
        gripper=gripper,
        root_path=Path("/tmp"),
        hand_info=hand_info,
        status_widget=_Widget(""),
        notice_widget=_Widget(""),
        save_path_widget=_Widget(""),
        template_widget=_Widget("pinch"),
        q_mode_widget=_Widget("q_open"),
        palm_aligned_widget=_Widget(True),
        show_surface_widget=_Widget(False),
        show_contact_widget=_Widget(True),
        show_keypoints_widget=_Widget(False),
        show_palm_widget=_Widget(False),
        confirmed_button=_Button(),
    )

    app.render()

    assert set(app.contact_handles) == {"finger_a"}


def test_global_app_draws_palm_points_preview_and_saves_delta(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    session.config.palm_pose = {"trans": [0.1, 0.2, 0.3], "rpy": [10.0, 20.0, 30.0]}
    session.config.palm_points_delta = 0.09
    server = _Server()

    app = create_global_app(
        session,
        server=server,
        gripper_factory=lambda **kwargs: _FakeGripper(
            handle=SimpleNamespace(_meshes=[], _urdf=SimpleNamespace(base_link="base")),
            joint_order=["j1", "j2"],
            q_dict={"j1": 0.0, "j2": 0.0},
            lb={"j1": -1.0, "j2": -1.0},
            ub={"j1": 1.0, "j2": 1.0},
        ),
    )

    assert app.palm_points_handle is not None
    assert np.asarray(app.palm_points_handle.points).shape == (7, 3)
    app.palm_points_delta_widget.value = 0.11
    assert app.config.palm_points_delta == pytest.approx(0.11)
    app.save()
    saved = yaml.safe_load(session.save_path.read_text(encoding="utf-8"))
    assert saved["palm_points_delta"] == pytest.approx(0.11)


def test_global_q_open_widgets_follow_config_joint_order(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    session.config.joint_order = ["j2", "j1"]
    session.config.q_open = [0.2, 0.1]
    server = _Server()

    app = create_global_app(
        session,
        server=server,
        gripper_factory=lambda **kwargs: _FakeGripper(
            handle=SimpleNamespace(_meshes=[]),
            joint_order=["j1", "j2"],
            q_dict={"j1": 0.0, "j2": 0.0},
            lb={"j1": -1.0, "j2": -1.0},
            ub={"j1": 1.0, "j2": 1.0},
        ),
    )

    assert list(app.q_open_joint_widgets) == ["j2", "j1"]
    app.q_open_joint_widgets["j2"].value = 0.9
    app.q_open_joint_widgets["j1"].value = -0.4

    assert app.gripper.q_dict == {"j1": -0.4, "j2": 0.9}


def test_global_q_open_slider_initial_values_are_clamped_to_limits(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    session.config.joint_order = ["j1", "j2"]
    session.config.q_open = [5.0, -5.0]
    server = _Server()

    app = create_global_app(
        session,
        server=server,
        gripper_factory=lambda **kwargs: _FakeGripper(
            handle=SimpleNamespace(_meshes=[]),
            joint_order=["j1", "j2"],
            q_dict={"j1": 0.0, "j2": 0.0},
            lb={"j1": -1.0, "j2": -0.5},
            ub={"j1": 1.0, "j2": 0.5},
        ),
    )

    assert app.q_open_joint_widgets["j1"].value == 1.0
    assert app.q_open_joint_widgets["j2"].value == -0.5


def test_delete_collision_pair(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    session.config.additional_collision_ignore_pairs = [["finger_a", "finger_b"]]
    server = _Server()

    app = create_global_app(
        session,
        server=server,
        gripper_factory=lambda **kwargs: _FakeGripper(
            handle=SimpleNamespace(_meshes=[]),
            joint_order=["j1", "j2"],
            q_dict={"j1": 0.0, "j2": 0.0},
            lb={"j1": -1.0, "j2": -1.0},
            ub={"j1": 1.0, "j2": 1.0},
        ),
    )

    app.collision_selected_pair_widget.value = "finger_a / finger_b"
    app.delete_collision_pair_button.click()
    assert app.config.additional_collision_ignore_pairs == []


def test_collision_click_resolves_urdf_link_name_instead_of_visual_leaf(tmp_path: Path) -> None:
    root = tmp_path / "gripper"
    root.mkdir(parents=True, exist_ok=True)
    (root / "hand.urdf").write_text("<robot />\n", encoding="utf-8")
    session = WizardSession(
        mode="new",
        gripper_root=".",
        gripper_root_abs=root,
        save_path=root / "demo.yaml",
        config=GripperConfig(
            name="demo",
            urdf_path="hand.urdf",
            xml_path=None,
            joint_order=["j1", "j2"],
            palm_pose={"trans": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
        ),
        from_config=None,
        urdf=ParsedUrdfNames(
            joint_names=("j1", "j2"),
            link_names=("base", "link_13.0", "link_14.0"),
            actuated_joint_names=("j1", "j2"),
        ),
        xml=None,
    )
    server = _Server()
    meshes = [_Mesh("/demo/link_13.0/visual"), _Mesh("/demo/link_14.0/visual")]

    app = create_global_app(
        session,
        server=server,
        gripper_factory=lambda **kwargs: _FakeGripper(
            handle=SimpleNamespace(_meshes=meshes),
            joint_order=["j1", "j2"],
            q_dict={"j1": 0.0, "j2": 0.0},
            lb={"j1": -1.0, "j2": -1.0},
            ub={"j1": 1.0, "j2": 1.0},
        ),
    )

    app.add_collision_pair_button.click()
    meshes[0].trigger_click()
    meshes[1].trigger_click()
    app.set_collision_pair_button.click()

    assert app.config.additional_collision_ignore_pairs == [["link_13.0", "link_14.0"]]


def test_keypoint_click_prefers_parent_link_over_reused_mesh_filename(tmp_path: Path) -> None:
    root = tmp_path / "gripper"
    root.mkdir(parents=True, exist_ok=True)
    (root / "hand.urdf").write_text("<robot />\n", encoding="utf-8")
    session = WizardSession(
        mode="new",
        gripper_root=".",
        gripper_root_abs=root,
        save_path=root / "demo.yaml",
        config=GripperConfig(
            name="demo",
            urdf_path="hand.urdf",
            xml_path=None,
            joint_order=["j1", "j2"],
            palm_pose={"trans": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
        ),
        from_config=None,
        urdf=ParsedUrdfNames(
            joint_names=("j1", "j2"),
            link_names=("base", "link_0.0", "link_4.0"),
            actuated_joint_names=("j1", "j2"),
        ),
        xml=None,
    )
    server = _Server()
    meshes = [_Mesh("/demo/link_4.0/link_0.0.stl")]

    app = create_keypoint_app(
        session,
        server=server,
        gripper_factory=lambda **kwargs: _FakeGripper(
            handle=SimpleNamespace(_meshes=meshes),
            joint_order=["j1", "j2"],
            q_dict={"j1": 0.0, "j2": 0.0},
            lb={"j1": -1.0, "j2": -1.0},
            ub={"j1": 1.0, "j2": 1.0},
        ),
    )

    app.add_contact_anchor_button.click()
    meshes[0].trigger_click()

    assert app.contact_selected_link_widget.value == "link_4.0"
    assert app.contact_anchor_active_link_name == "link_4.0"


def test_keypoint_click_allows_base_link_when_mesh_path_uses_assets_folder(tmp_path: Path) -> None:
    root = tmp_path / "gripper"
    root.mkdir(parents=True, exist_ok=True)
    (root / "hand.urdf").write_text("<robot />\n", encoding="utf-8")
    session = WizardSession(
        mode="new",
        gripper_root=".",
        gripper_root_abs=root,
        save_path=root / "demo.yaml",
        config=GripperConfig(
            name="demo",
            urdf_path="hand.urdf",
            xml_path=None,
            joint_order=["j1", "j2"],
            palm_pose={"trans": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
        ),
        from_config=None,
        urdf=ParsedUrdfNames(
            joint_names=("j1", "j2"),
            link_names=("base_link", "link_0.0"),
            actuated_joint_names=("j1", "j2"),
        ),
        xml=None,
    )
    server = _Server()
    meshes = [_Mesh("/demo/assets/base_link.stl")]

    app = create_keypoint_app(
        session,
        server=server,
        gripper_factory=lambda **kwargs: _FakeGripper(
            handle=SimpleNamespace(_meshes=meshes),
            joint_order=["j1", "j2"],
            q_dict={"j1": 0.0, "j2": 0.0},
            lb={"j1": -1.0, "j2": -1.0},
            ub={"j1": 1.0, "j2": 1.0},
        ),
    )

    app.add_contact_anchor_button.click()
    meshes[0].trigger_click()

    assert app.contact_selected_link_widget.value == "base_link"
    assert app.contact_anchor_active_link_name == "base_link"


def test_keypoint_add_edit_flow_saves_contact_anchor_and_tags(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    server = _Server()
    meshes = [_Mesh("/demo/finger_a/finger_a.stl"), _Mesh("/demo/finger_b/finger_b.stl")]

    app = create_keypoint_app(
        session,
        server=server,
        gripper_factory=lambda **kwargs: _FakeGripper(
            handle=SimpleNamespace(_meshes=meshes),
            joint_order=["j1", "j2"],
            q_dict={"j1": 0.0, "j2": 0.0},
            lb={"j1": -1.0, "j2": -1.0},
            ub={"j1": 1.0, "j2": 1.0},
        ),
    )

    assert app.add_contact_anchor_button.label == "Add/Edit Point"
    app.add_contact_anchor_button.click()
    meshes[0].trigger_click()
    app.contact_point_widget.value = (0.01, 0.02, 0.03)
    assert app.contact_anchor_draft_sphere.radius == 0.007
    app.contact_radius_widget.value = 0.02
    assert app.contact_anchor_draft_sphere.radius == 0.02
    app.contact_tags_widget.value = "thumb, tip, outer"
    assert app.add_contact_anchor_button.label == "Save Point"
    app.add_contact_anchor_button.click()

    assert app.config.contact_anchors == {
        "finger_a": {
            "point": [0.01, 0.02, 0.03],
            "tags": ["thumb", "tip", "outer"],
            "contact_radius": 0.02,
        }
    }
    assert app.add_contact_anchor_button.label == "Add/Edit Point"
    assert server.scene.spheres[0].radius == 0.02


def test_keypoint_existing_link_enters_silent_edit_mode(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    session.config.contact_anchors = {
        "finger_a": {"point": [0.01, 0.02, 0.03], "tags": ["thumb", "tip"], "contact_radius": 0.03}
    }
    server = _Server()
    meshes = [_Mesh("/demo/finger_a/finger_a.stl")]
    saved_spheres = []

    def _fake_gripper_factory(**kwargs):
        _ = kwargs
        gripper = _FakeGripper(
            handle=SimpleNamespace(_meshes=meshes),
            joint_order=["j1", "j2"],
            q_dict={"j1": 0.0, "j2": 0.0},
            lb={"j1": -1.0, "j2": -1.0},
            ub={"j1": 1.0, "j2": 1.0},
        )
        gripper.handle._saved_spheres = saved_spheres
        return gripper

    app = create_keypoint_app(session, server=server, gripper_factory=_fake_gripper_factory)

    app.add_contact_anchor_button.click()
    meshes[0].trigger_click()

    assert app.contact_selected_link_widget.value == "finger_a"
    assert app.contact_point_widget.value == (0.01, 0.02, 0.03)
    assert app.contact_tags_widget.value == "thumb, tip"
    assert app.contact_radius_widget.value == 0.03


def test_keypoint_add_button_waits_for_mesh_click_even_with_existing_selection(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    session.config.contact_anchors = {
        "finger_a": {"point": [0.01, 0.02, 0.03], "tags": ["thumb"], "contact_radius": 0.02},
    }
    server = _Server()
    meshes = [_Mesh("/demo/finger_a/finger_a.stl"), _Mesh("/demo/finger_b/finger_b.stl")]

    app = create_keypoint_app(
        session,
        server=server,
        gripper_factory=lambda **kwargs: _FakeGripper(
            handle=SimpleNamespace(_meshes=meshes),
            joint_order=["j1", "j2"],
            q_dict={"j1": 0.0, "j2": 0.0},
            lb={"j1": -1.0, "j2": -1.0},
            ub={"j1": 1.0, "j2": 1.0},
        ),
    )

    app.contact_selected_link_widget.value = "finger_a"
    app.add_contact_anchor_button.click()

    assert app.waiting_for_contact_anchor_click is True
    assert app.contact_anchor_active_link_name is None

    meshes[1].trigger_click()

    assert app.contact_selected_link_widget.value == "finger_b"
    assert app.contact_anchor_active_link_name == "finger_b"


def test_keypoint_delete_anchor_removes_saved_entry(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    session.config.contact_anchors = {
        "finger_a": {"point": [0.01, 0.02, 0.03], "tags": ["thumb", "tip"], "contact_radius": 0.03},
    }
    server = _Server()

    app = create_keypoint_app(
        session,
        server=server,
        gripper_factory=lambda **kwargs: _FakeGripper(
            handle=SimpleNamespace(_meshes=[]),
            joint_order=["j1", "j2"],
            q_dict={"j1": 0.0, "j2": 0.0},
            lb={"j1": -1.0, "j2": -1.0},
            ub={"j1": 1.0, "j2": 1.0},
        ),
    )

    app.contact_delete_selected_widget.value = "finger_a"
    app.delete_contact_anchor_button.click()

    assert app.config.contact_anchors == {}


def test_keypoint_saved_sphere_click_reenters_edit_mode(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    session.config.contact_anchors = {
        "finger_a": {"point": [0.01, 0.02, 0.03], "tags": ["thumb", "tip"], "contact_radius": 0.03},
    }
    server = _Server()
    meshes = [_Mesh("/demo/finger_a/finger_a.stl")]

    app = create_keypoint_app(
        session,
        server=server,
        gripper_factory=lambda **kwargs: _FakeGripper(
            handle=SimpleNamespace(_meshes=meshes),
            joint_order=["j1", "j2"],
            q_dict={"j1": 0.0, "j2": 0.0},
            lb={"j1": -1.0, "j2": -1.0},
            ub={"j1": 1.0, "j2": 1.0},
        ),
    )

    saved_sphere = server.scene.spheres[0]
    saved_sphere.trigger_click()

    assert app.contact_selected_link_widget.value == "finger_a"
    assert app.contact_tags_widget.value == "thumb, tip"
    assert app.contact_point_widget.value == (0.01, 0.02, 0.03)
    assert app.contact_radius_widget.value == 0.03


def test_create_template_app_supports_idle_edit_q_open_q_close_semantics(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    session.config.palm_pose = {"trans": [0.3, 0.4, 0.5], "rpy": [10.0, 20.0, 30.0]}
    session.config.contact_anchors = {
        "finger_a": {"point": [0.01, 0.02, 0.03], "tags": ["tip"], "contact_radius": 0.01},
        "finger_b": {"point": [0.04, 0.05, 0.06], "tags": ["tip"], "contact_radius": 0.01},
    }
    session.config.grasp_templates = {
        "pinch": {
            "q_close": [0.0, 0.0],
            "q_open": [0.2, 0.3],
            "grasp_target_point": [0.1, 0.2, 0.3],
            "active_contact_anchors": ["finger_a"],
        }
    }
    server = _Server()
    meshes = [_Mesh("/demo/finger_a/finger_a.stl"), _Mesh("/demo/finger_b/finger_b.stl")]

    app = wizard_gui.create_template_app(
        session,
        server=server,
        gripper_factory=lambda **kwargs: _FakeGripper(
            handle=SimpleNamespace(_meshes=meshes),
            joint_order=["j1", "j2"],
            q_dict={"j1": 0.0, "j2": 0.0},
            lb={"j1": -1.0, "j2": -1.0},
            ub={"j1": 1.0, "j2": 1.0},
        ),
    )

    assert app.joint_edit_mode_widget.value == "idle"
    assert app.q_open_toggle_button.label == "Edit q_open"
    assert app.q_close_toggle_button.label == "Edit q_close"
    assert app.q_open_toggle_button.color is None
    assert app.q_close_toggle_button.color is None
    assert app.add_edit_template_button.label == "Add/Edit Template"
    assert app.save_template_button.disabled is False
    assert app.template_palm_frame.position == (0.3, 0.4, 0.5)
    expected_world = tuple(
        float(v)
        for v in (
            wizard_gui.R.from_euler("ZYX", [30.0, 20.0, 10.0], degrees=True).apply([0.1, 0.2, 0.3])
            + np.array([0.3, 0.4, 0.5], dtype=float)
        ).tolist()
    )
    assert app.template_target_gizmo.position == expected_world
    assert app.template_target_gizmo.wxyz == wizard_gui._rpy_to_wxyz([10.0, 20.0, 30.0])
    assert app.template_target_sphere.position == (0.0, 0.0, 0.0)

    app.edit_q_open_button.click()
    assert app.joint_edit_mode_widget.value == "editing q_open"
    assert app.add_edit_template_button.label == "Add/Edit Template"
    assert app.q_open_toggle_button.label == "Set q_open"
    assert app.edit_q_open_button.color == wizard_gui.BUTTON_COLOR_ACTIVE_SET
    assert app.edit_q_open_from_q_close_button.color == wizard_gui.BUTTON_COLOR_ACTIVE_EDIT
    assert app.q_close_toggle_button.label == "Edit q_close"
    assert app.edit_q_close_button.color is None
    assert app.save_template_button.disabled is False
    assert app.q_open_joint_widgets["j1"].value == 0.2
    assert app.q_open_joint_widgets["j2"].value == 0.3

    app.edit_q_close_button.click()
    assert app.joint_edit_mode_widget.value == "editing q_open"

    app.q_open_joint_widgets["j1"].value = 0.6
    app.q_open_joint_widgets["j2"].value = 0.7
    app.set_q_open_button.click()
    assert app.joint_edit_mode_widget.value == "idle"
    assert app.add_edit_template_button.label == "Add/Edit Template"
    assert app.q_open_toggle_button.label == "Edit q_open"
    assert app.edit_q_open_button.color is None
    assert app.edit_q_open_from_q_close_button.color is None
    assert app.save_template_button.disabled is False
    assert app.config.grasp_templates["pinch"]["q_open"] == [0.6, 0.7]

    app.edit_q_close_button.click()
    assert app.joint_edit_mode_widget.value == "editing q_close"
    assert app.q_close_toggle_button.label == "Set q_close"
    assert app.edit_q_close_button.color == wizard_gui.BUTTON_COLOR_ACTIVE_SET
    assert app.save_template_button.disabled is False
    assert app.q_close_joint_widgets["j1"].value == 0.0
    assert app.q_close_joint_widgets["j2"].value == 0.0
    app.set_q_close_button.click()
    assert app.joint_edit_mode_widget.value == "idle"
    assert app.add_edit_template_button.label == "Add/Edit Template"
    assert app.q_close_toggle_button.label == "Edit q_close"
    assert app.edit_q_close_button.color is None
    assert app.save_template_button.disabled is False
    assert app.config.grasp_templates["pinch"]["q_close"] == [0.0, 0.0]


def test_template_joint_widgets_follow_config_joint_order(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    session.config.joint_order = ["j2", "j1"]
    session.config.grasp_templates = {
        "pinch": {
            "q_close": [0.8, 0.9],
            "q_open": [0.2, 0.3],
            "grasp_target_point": [0.1, 0.2, 0.3],
            "active_contact_anchors": [],
        }
    }
    server = _Server()

    app = wizard_gui.create_template_app(
        session,
        server=server,
        gripper_factory=lambda **kwargs: _FakeGripper(
            handle=SimpleNamespace(_meshes=[]),
            joint_order=["j1", "j2"],
            q_dict={"j1": 0.0, "j2": 0.0},
            lb={"j1": -1.0, "j2": -1.0},
            ub={"j1": 1.0, "j2": 1.0},
        ),
    )

    assert list(app.q_open_joint_widgets) == ["j2", "j1"]
    app.edit_q_open_button.click()
    assert app.q_open_joint_widgets["j2"].value == 0.2
    assert app.q_open_joint_widgets["j1"].value == 0.3


def test_template_joint_slider_initial_values_are_clamped_to_limits(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    session.config.joint_order = ["j1", "j2"]
    session.config.q_open = [3.0, -3.0]
    server = _Server()

    app = wizard_gui.create_template_app(
        session,
        server=server,
        gripper_factory=lambda **kwargs: _FakeGripper(
            handle=SimpleNamespace(_meshes=[]),
            joint_order=["j1", "j2"],
            q_dict={"j1": 0.0, "j2": 0.0},
            lb={"j1": -1.0, "j2": -0.5},
            ub={"j1": 1.0, "j2": 0.5},
        ),
    )

    assert app.q_open_joint_widgets["j1"].value == 1.0
    assert app.q_open_joint_widgets["j2"].value == -0.5


def test_template_stage_can_start_q_open_edit_from_q_close(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    session.config.grasp_templates = {
        "pinch": {
            "q_close": [0.8, 0.9],
            "q_open": [0.2, 0.3],
            "grasp_target_point": [0.1, 0.2, 0.3],
            "active_contact_anchors": [],
        }
    }
    server = _Server()

    app = wizard_gui.create_template_app(
        session,
        server=server,
        gripper_factory=lambda **kwargs: _FakeGripper(
            handle=SimpleNamespace(_meshes=[]),
            joint_order=["j1", "j2"],
            q_dict={"j1": 0.0, "j2": 0.0},
            lb={"j1": -1.0, "j2": -1.0},
            ub={"j1": 1.0, "j2": 1.0},
        ),
    )

    app.edit_q_open_from_q_close_button.click()

    assert app.joint_edit_mode_widget.value == "editing q_open"
    assert app.q_open_joint_widgets["j1"].value == 0.8
    assert app.q_open_joint_widgets["j2"].value == 0.9


def test_template_stage_save_uses_template_name_widget_as_target_key(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    session.config.grasp_templates = {
        "finger4": {
            "q_close": [0.8, 0.9],
            "q_open": [0.2, 0.3],
            "grasp_target_point": [0.1, 0.2, 0.3],
            "active_contact_anchors": [],
        }
    }
    server = _Server()

    app = wizard_gui.create_template_app(
        session,
        server=server,
        gripper_factory=lambda **kwargs: _FakeGripper(
            handle=SimpleNamespace(_meshes=[]),
            joint_order=["j1", "j2"],
            q_dict={"j1": 0.0, "j2": 0.0},
            lb={"j1": -1.0, "j2": -1.0},
            ub={"j1": 1.0, "j2": 1.0},
        ),
    )

    app.template_name_widget.value = "finger4_alt"
    app.save()

    assert "finger4" in app.config.grasp_templates
    assert "finger4_alt" in app.config.grasp_templates
    assert app.config.grasp_templates["finger4_alt"]["q_open"] == [0.2, 0.3]
    assert app.active_template_name == "finger4_alt"
    assert app.add_edit_template_button.label == "Add/Edit Template"


def test_template_stage_add_edit_button_toggles_into_save_template(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    session.config.grasp_templates = {}
    server = _Server()

    app = wizard_gui.create_template_app(
        session,
        server=server,
        gripper_factory=lambda **kwargs: _FakeGripper(
            handle=SimpleNamespace(_meshes=[]),
            joint_order=["j1", "j2"],
            q_dict={"j1": 0.0, "j2": 0.0},
            lb={"j1": -1.0, "j2": -1.0},
            ub={"j1": 1.0, "j2": 1.0},
        ),
    )

    app.template_name_widget.value = "pinch"
    app.add_edit_template_button.click()

    assert app.active_template_name == "pinch"
    assert app.template_edit_active is True
    assert app.add_edit_template_button.label == "Save Template"

    app.add_edit_template_button.click()

    assert "pinch" in app.config.grasp_templates
    assert app.template_edit_active is False
    assert app.add_edit_template_button.label == "Add/Edit Template"


def test_template_stage_add_uses_global_q_open_for_sliders_and_preview(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    session.config.q_open = [0.4, -0.2]
    session.config.grasp_templates = {}
    server = _Server()
    gripper = _FakeGripper(
        handle=SimpleNamespace(_meshes=[]),
        joint_order=["j1", "j2"],
        q_dict={"j1": 0.0, "j2": 0.0},
        lb={"j1": -1.0, "j2": -1.0},
        ub={"j1": 1.0, "j2": 1.0},
    )

    app = wizard_gui.create_template_app(
        session,
        server=server,
        gripper_factory=lambda **kwargs: gripper,
    )

    app.template_name_widget.value = "pinch"
    app.add_edit_template_button.click()

    assert app.q_open_joint_widgets["j1"].value == 0.4
    assert app.q_open_joint_widgets["j2"].value == -0.2
    assert gripper.q_dict == {"j1": 0.4, "j2": -0.2}


def test_template_stage_edit_uses_saved_q_open_for_sliders_and_preview(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    session.config.q_open = [0.4, -0.2]
    session.config.grasp_templates = {
        "pinch": {
            "q_close": [0.8, 0.9],
            "q_open": [0.2, 0.3],
            "grasp_target_point": [0.1, 0.2, 0.3],
            "active_contact_anchors": [],
        }
    }
    server = _Server()
    gripper = _FakeGripper(
        handle=SimpleNamespace(_meshes=[]),
        joint_order=["j1", "j2"],
        q_dict={"j1": 0.0, "j2": 0.0},
        lb={"j1": -1.0, "j2": -1.0},
        ub={"j1": 1.0, "j2": 1.0},
    )

    app = wizard_gui.create_template_app(
        session,
        server=server,
        gripper_factory=lambda **kwargs: gripper,
    )

    app.template_name_widget.value = "pinch"
    app.add_edit_template_button.click()

    assert app.q_open_joint_widgets["j1"].value == 0.2
    assert app.q_open_joint_widgets["j2"].value == 0.3
    assert gripper.q_dict == {"j1": 0.2, "j2": 0.3}


def test_template_stage_save_is_blocked_while_joint_edit_is_active(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    session.config.grasp_templates = {
        "pinch": {
            "q_close": [0.8, 0.9],
            "q_open": [0.2, 0.3],
            "grasp_target_point": [0.1, 0.2, 0.3],
            "active_contact_anchors": [],
        }
    }
    server = _Server()

    app = wizard_gui.create_template_app(
        session,
        server=server,
        gripper_factory=lambda **kwargs: _FakeGripper(
            handle=SimpleNamespace(_meshes=[]),
            joint_order=["j1", "j2"],
            q_dict={"j1": 0.0, "j2": 0.0},
            lb={"j1": -1.0, "j2": -1.0},
            ub={"j1": 1.0, "j2": 1.0},
        ),
    )

    app.edit_q_open_button.click()
    app.q_open_joint_widgets["j1"].value = 0.6
    blocked_path = app.save()

    assert blocked_path is None
    assert app.joint_edit_mode_widget.value == "editing q_open"
    assert app.config.grasp_templates["pinch"]["q_open"] == [0.2, 0.3]
    assert "Finish q_open editing with Set q_open first." in app.notice_widget.content


def test_template_stage_toggles_active_contact_anchors_from_saved_spheres(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    session.config.contact_anchors = {
        "finger_a": {"point": [0.01, 0.02, 0.03], "tags": ["tip"], "contact_radius": 0.01},
        "finger_b": {"point": [0.04, 0.05, 0.06], "tags": ["tip"], "contact_radius": 0.01},
    }
    server = _Server()
    meshes = [_Mesh("/demo/finger_a/finger_a.stl"), _Mesh("/demo/finger_b/finger_b.stl")]

    app = wizard_gui.create_template_app(
        session,
        server=server,
        gripper_factory=lambda **kwargs: _FakeGripper(
            handle=SimpleNamespace(_meshes=meshes),
            joint_order=["j1", "j2"],
            q_dict={"j1": 0.0, "j2": 0.0},
            lb={"j1": -1.0, "j2": -1.0},
            ub={"j1": 1.0, "j2": 1.0},
        ),
    )

    assert app.active_contact_anchor_names == []
    assert app.saved_contact_anchor_spheres["finger_a"].color == (255, 214, 10)
    assert app.saved_contact_anchor_spheres["finger_a"].opacity == 0.35
    app.saved_contact_anchor_spheres["finger_a"].trigger_click()
    assert app.active_contact_anchor_names == ["finger_a"]
    assert app.saved_contact_anchor_spheres["finger_a"].color == (255, 51, 26)
    assert app.saved_contact_anchor_spheres["finger_a"].opacity == 1.0
    assert app.saved_contact_anchor_spheres["finger_a"].radius == pytest.approx(0.01)
    app.saved_contact_anchor_spheres["finger_a"].trigger_click()
    assert app.active_contact_anchor_names == []
    assert app.saved_contact_anchor_spheres["finger_a"].color == (255, 214, 10)
    assert app.saved_contact_anchor_spheres["finger_a"].opacity == 0.35


def test_gripper_scene_initializes_joint_configuration_from_limits(monkeypatch) -> None:
    recorded = {}

    class _FakeViserUrdf:
        def __init__(self, *, target, urdf_or_path, root_node_name):
            recorded["target"] = target
            recorded["urdf_or_path"] = urdf_or_path
            recorded["root_node_name"] = root_node_name

        def get_actuated_joint_limits(self):
            return {
                "j1": (-1.0, 1.0),
                "j2": (0.0, 2.0),
            }

        def update_cfg(self, q):
            recorded["q"] = np.asarray(q, dtype=float)

    monkeypatch.setattr(wizard_gui, "ViserUrdf", _FakeViserUrdf)

    scene = wizard_gui._GripperScene(_Server(), name="demo", urdf_path="/tmp/demo.urdf")

    assert scene.q_dict == {"j1": 0.0, "j2": 1.0}
    assert np.allclose(recorded["q"], np.array([0.0, 1.0], dtype=float))
