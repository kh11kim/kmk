# GUI Wizard Spec

## Scope

This document defines the `Global Information` section of the robot hand gripper configuration wizard.

The goal of this section is to collect and validate hand-level configuration that applies across all per-link contact candidates and all grasp templates.

Current implementation split:

- CLI: `name`, `urdf_path`, `xml_path`, `joint_order`, `--from-config`
- GUI stage 1: `palm_pose`, `palm_points_delta`, `global q_open`, `additional_collision_ignore_pairs`
- GUI stage 2: `contact_anchors`
- GUI stage 3: `grasp_templates`
- GUI stage 4: final preview + `Confirmed`

## Global Information

### Purpose

The `Global Information` section defines the base robot model references, joint indexing convention, hand-level reference pose, and hand-level collision-ignore rules used by the rest of the configuration workflow.

### Fields

#### `name`

- Type: `string`
- Required: yes
- Description: Unique configuration name for this hand/gripper.
- Constraints:
  - Must be non-empty.
  - Should be filesystem-safe because it may be used as a default output filename.

#### `urdf_path`

- Type: `string`
- Required: yes
- Description: Path to the gripper URDF file.
- Constraints:
  - Must resolve to an existing URDF file.
  - When stored in YAML, relative paths are resolved from the config file directory at runtime.
  - During wizard CLI setup, relative `--from-config` paths are resolved from `gripper_root`.

#### `xml_path`

- Type: `string | null`
- Required: no
- Description: Optional path to the MuJoCo XML file corresponding to the gripper.
- Constraints:
  - If provided, it must resolve to an existing XML file.
  - If omitted, the system operates without MuJoCo-specific joint or actuator mapping.
  - When stored in YAML, relative paths are resolved from the config file directory at runtime.

#### `palm_points_delta`

- Type: `float`
- Required: yes
- Description: Distance used for the additional palm-aligned helper points anchored under the base link.
- Constraints:
  - Must be non-negative.
- Notes:
  - This value is authored in the global stage.
  - The final preview stage uses the stored value directly and does not expose a separate delta slider.

#### `joint_order`

- Type: `list[string]`
- Required: yes
- Description: Ordered list of actuated joint names used as the canonical joint vector order for this hand.
- Constraints:
  - Every name must be unique.
  - Every name must exist in the URDF actuated joint set.
  - The list must contain all actuated joints exactly once.
- Notes:
  - The CLI initializes this field from the URDF-detected actuated joint order.
  - The user may override it from the CLI flow before the GUI stage begins.

#### `xml_joint_actuator_alias`

- Type: `mapping[string, string] | null`
- Required: no
- Description: Optional mapping from canonical joint names to MuJoCo XML joint or actuator names.
- Constraints:
  - Keys must be joint names from `joint_order`.
  - Values must be unique within the mapping.
- Default behavior:
  - If omitted, downstream code should assume the XML side follows the same joint names as `joint_order`.
  - If `xml_path` is omitted, this field should also be omitted or ignored.

#### `palm_pose`

- Type: object
- Required: yes
- Description: Hand-level reference pose used as the palm frame convention for grasp planning and visualization.
- Shape:
  - `trans: [float, float, float]`
  - `rpy: [float, float, float]`
- Constraints:
  - `trans` must contain exactly 3 values.
  - `rpy` must contain exactly 3 values.
- Notes:
  - `rpy` is stored in degrees.
  - This pose is global to the hand config and shared across all grasp templates.

#### `q_open`

- Type: `list[float]`
- Required: yes
- Description: Hand-level default open configuration for the joint vector.
- Constraints:
  - Length must match `joint_order`.
  - Values must follow the same ordering as `joint_order`.
- Notes:
  - This is used as the default visualization and authoring pose.
  - This is distinct from any grasp-template-specific `q_open`.

#### `additional_collision_ignore_pairs`

- Type: `list[[string, string]]`
- Required: no
- Description: Additional link pairs that should be treated as collision-ignore pairs beyond the default structural or adjacency-based rules.
- Constraints:
  - Each entry must contain exactly 2 link names.
  - The two link names in a pair must be distinct.
  - Link names should exist in the URDF link set.
  - Duplicate pairs are not allowed.
  - Pair ordering is not semantically meaningful. The system should normalize each pair into a canonical sorted order before storing.
- Notes:
  - This field is global because collision-ignore policy is hand-level, not grasp-template-specific.
  - This field is intended for additional exceptions only. Default collision-ignore behavior is defined elsewhere by the runtime or simulator integration.

## Wizard Behavior

### Initialization

- The wizard loads the URDF from `urdf_path` and extracts actuated joints.
- If `xml_path` is provided, the wizard may inspect the XML to suggest alias mappings.
- The CLI prepopulates `joint_order` from the detected actuated joints.
- The wizard initializes `palm_pose` to identity if no prior config exists.
- The wizard initializes `palm_points_delta` to `0.05` if no prior config exists.
- The wizard initializes `q_open` from the current gripper default/open pose if no prior config exists.
- The wizard initializes `additional_collision_ignore_pairs` to an empty list if no prior config exists.

### Editing Rules

- The user must be able to edit every global field before moving on to later sections.
- The GUI should show validation feedback immediately for invalid paths, duplicate joint names, or incomplete pose values.
- Reordering `joint_order` in the CLI stage must update the canonical order used by later sections.
- The GUI must expose a palm-pose gizmo and vector widgets for editing `palm_pose`.
- The GUI must expose a `palm_points_delta` control and visualize the resulting palm-aligned helper points during the global stage.
- The GUI must expose joint sliders for editing `q_open`.
- The user must be able to add, review, and delete collision-ignore pairs from the global section.

### Save Semantics

- The global section is part of the same final config document as later sections.
- Saving the draft must serialize the current global state even if later sections are incomplete.

## Validation Summary

- `name` must be non-empty.
- `urdf_path` must exist.
- `xml_path` is optional, but if present it must exist.
- `joint_order` must be a permutation of all actuated URDF joints.
- `q_open` must have the same length as `joint_order`.
- `xml_joint_actuator_alias` is optional.
- If `xml_joint_actuator_alias` is omitted, XML naming defaults to canonical joint names.
- `palm_pose.trans` and `palm_pose.rpy` must each have length 3.
- `palm_points_delta` must be non-negative.
- `additional_collision_ignore_pairs` is optional.
- Every `additional_collision_ignore_pairs` entry must contain exactly two distinct valid URDF link names.
- Collision-ignore pairs must be unique after pair-order normalization.

## Per-Link Keypoint Information

### Purpose

The `Per-Link Keypoint Information` section defines one contact anchor per link for later grouping and grasp logic.

### Data Model

#### `contact_anchors`

- Type: `mapping[string, object]`
- Key: URDF `link_name`
- Required: no
- Description: Per-link contact anchor table.
- Coordinate frame: link-local

Each entry has the following shape:

- `point: [float, float, float]`  # link-local coordinates
- `contact_radius: float`  # default `0.007`
- `tags: list[string]`

### Rules

- Exactly one contact anchor is allowed per link.
- The anchor itself is defined by `point`.
- `contact_radius` controls the visual sphere size and any downstream contact neighborhood interpretation.
- If omitted in older YAML, `contact_radius` defaults to `0.007`.
- A point may have multiple tags.
- Tags are used to group anchors for downstream selection and grasp logic.
- Tags are entered in the GUI as comma-separated text, but stored in config as a string list.

### GUI Flow

The intended GUI flow for one anchor is:

1. `Add/Edit Point`
2. Select the target link by clicking the mesh
3. Adjust the anchor position with a gizmo
4. Adjust `contact_radius` with a radius control
5. Enter tags in a textbox using comma-separated text
6. `Save Point`

### Edit Behavior

- `Add/Edit Point` is used for both creation and editing.
- If the selected link does not yet have an anchor, the GUI starts a new draft for that link.
- If the selected link already has an anchor, the GUI silently enters edit mode and loads the existing anchor data.
- Existing saved anchors may also be selected by clicking their saved visualization handle.
- The primary keypoint action is a toggle button:
  - idle label: `Add/Edit Point`
  - active draft label: `Save Point`
- Saving an existing link overwrites that link's prior anchor data.
- A separate `Delete Point` action removes the saved anchor for the selected link.

### Validation Summary

- `contact_anchors` keys must be valid URDF link names.
- Each `point` must contain exactly 3 numeric values.
- `contact_radius` must be a positive numeric value.
- Each `tags` list may be empty or contain one or more strings.
- Tag storage format is normalized list form, even if the GUI input is comma-separated text.

## Grasp Template

### Purpose

The `Grasp Template` section defines reusable hand postures and target points for grasp authoring and downstream planning.

### Data Model

#### `grasp_templates`

- Type: `mapping[string, object]`
- Key: `template_name`
- Required: no
- Description: Named grasp template table.

Each entry has the following shape:

- `q_close: list[float]`
- `q_open: list[float]`
- `grasp_target_point: [float, float, float]`
- `active_contact_anchors: list[string]`

### Rules

- `template_name` must be unique within `grasp_templates`.
- `q_close` and `q_open` both use the canonical `joint_order`.
- `grasp_target_point` is a 3D point, not a full pose.
- `active_contact_anchors` is the subset of saved `contact_anchors` used by this template.
- Because the current keypoint model is `1 link = 1 anchor`, each `active_contact_anchors` entry is a `link_name`.

### Validation Summary

- Each `template_name` must be non-empty.
- `q_close` length must match `joint_order`.
- `q_open` length must match `joint_order`.
- `grasp_target_point` must contain exactly 3 numeric values.
- `active_contact_anchors` entries must be valid saved `contact_anchors` keys.
- `active_contact_anchors` must not contain duplicates.

### GUI Flow

The intended GUI flow for one grasp template is:

1. Enter `template_name`
2. `Add/Edit Template`
3. `Edit q_open` or `Edit q_close` from idle
4. Adjust the shared joint sliders
5. `Set q_open` or `Set q_close`
6. Adjust `grasp_target_point`
7. Toggle active contact anchors by clicking saved contact anchor spheres
8. `Save Template`

### Edit Behavior

- `Add/Edit Template` is used for both creation and editing.
- If the selected template does not yet exist, the GUI starts a new draft for that name.
- If the selected template already exists, the GUI silently enters edit mode and loads the existing template data.
- `Edit q_open` and `Edit q_close` are only accepted while the joint edit mode is `idle`.
- While editing `q_open`, a click on `Edit q_close` is ignored.
- While editing `q_close`, a click on `Edit q_open` is ignored.
- `Set q_open` commits the current joint slider values into `q_open` and returns the joint edit mode to `idle`.
- `Set q_close` commits the current joint slider values into `q_close` and returns the joint edit mode to `idle`.
- The GUI must show whether the joint edit mode is `idle`, `editing q_open`, or `editing q_close`.
- `active_contact_anchors` are toggled by clicking saved contact anchor spheres.
- `Save Template` serializes the currently committed template state.
- `Delete Template` removes the selected saved template entry.

## Runtime HandInfo

### Purpose

`HandInfo` is the runtime wrapper that loads an authored config YAML and exposes the minimum API needed to consume joint poses, palm pose, contact anchors, and grasp templates.

### Location

- Class: `HandInfo`
- Module: `src/kmk/hand_info.py`
- Re-export: `from kmk import HandInfo`

### API

- `HandInfo.from_config(config_path: str | Path) -> HandInfo`
- `joint_order: list[str]`
- `template_names: list[str]`
- `contact_anchor_links: list[str]`

- `get_q_open(template: str = "global") -> np.ndarray`
  - shape `(N,)`
  - follows `joint_order`
  - `template="global"` returns global `q_open`
  - any other value is interpreted as a template name and returns that template's `q_open`

- `get_q_close(template_name: str) -> np.ndarray`
  - shape `(N,)`
  - follows `joint_order`
  - template-only

- `get_palm_pose() -> np.ndarray`
  - shape `(4, 4)`
  - homogeneous transform matrix
  - represents `palm_pose wrt base(world)`

- `get_contact_anchor_by_link(link_name: str) -> np.ndarray`
  - shape `(3,)`
  - returns the link-local anchor point

- `get_contact_anchor_by_tag(includes: Sequence[str] = (), excludes: Sequence[str] = ()) -> dict[str, np.ndarray]`
  - returns `{link_name: point}`
  - each point has shape `(3,)`
  - points are link-local
  - `includes` uses AND semantics
  - `includes=()` means all anchors are eligible
  - any tag in `excludes` removes the anchor

- `get_contact_anchor_by_template(template_name: str) -> dict[str, np.ndarray]`
  - returns `{link_name: point}`
  - points are link-local
  - source links come from that template's `active_contact_anchors`

- `get_grasp_target_point(template_name: str) -> np.ndarray`
  - shape `(3,)`
  - always returned in the palm frame

### Runtime Rules

- All arrays are returned as `numpy.ndarray`.
- All returned arrays use floating-point dtype.
- Missing `template_name` values raise `KeyError`.
- Missing `link_name` values raise `KeyError`.
- Empty tag-query matches return an empty dictionary.

## Runtime PointedHandInfo

### Purpose

`PointedHandInfo` extends `HandInfo` with precomputed sampled points and a keypoint accessor for downstream point-based processing.

### API

- `PointedHandInfo.from_config(config_path: str | Path, seed: int) -> PointedHandInfo`
- `surface_points: dict[str, np.ndarray]`
- `contact_points: dict[str, np.ndarray]`
- `get_contact_points(template_name: str | None = None) -> dict[str, np.ndarray]`

- `get_keypoints(
    template_name: str | None = None,
    palm_aligned_points: bool = True,
    palm_points_delta: float = 0.05,
  ) -> dict[str, np.ndarray]`

### Runtime Rules

- Sampling is performed eagerly during initialization.
- `surface_points` stores sampled collision-surface points per link in link-local coordinates.
- `contact_points` stores sampled contact-region points per saved anchor link in link-local coordinates.
- Contact-point sampling is radius-based:
  - candidate points are collision-surface samples within `contact_radius` of the saved anchor point
  - no normal-angle filtering is applied
- `get_keypoints(template_name=None)` returns all saved contact anchors as `{link_name: points}` with each value shaped `(K, 3)`.
- `get_keypoints(template_name=...)` returns only that template's active contact anchors.
- `get_contact_points(template_name=None)` returns sampled contact-region points for all anchors.
- `get_contact_points(template_name=...)` returns sampled contact-region points only for that template's `active_contact_anchors`.
- Anchor points returned by `get_keypoints()` are link-local.
- If `palm_aligned_points=True`, additional palm-frame-axis-aligned points are added under the URDF base link key.
- Those palm-aligned points are also expressed in the base link local frame.

## Torch HandKinematics

### Purpose

`HandKinematics` is the torch-based forward-kinematics module built on top of `HandInfo`.

Its role is intentionally narrow:

- accept joint vectors in canonical `joint_order`
- compute differentiable link transforms
- transform link-local point sets into the base/world frame

It does not own template-specific point-set selection. Higher-level modules may use `PointedHandInfo.get_keypoints()` or any other point-set source and pass the resulting link-local points into this module.

### Public API

- `HandKinematics(hand_info: HandInfo | str | Path)`
- `joint_order: list[str]`
- `link_names: list[str]`
- `dof: int`

- `forward_kinematics(q: torch.Tensor) -> dict[str, torch.Tensor]`
  - input shape: `(*batch_shape, dof)`
  - output: `{link_name: T}`
  - each `T` has shape `(*batch_shape, 4, 4)`
  - transform meaning: `base/world <- link`

- `transform_link_points(
    q: torch.Tensor,
    points_by_link: dict[str, torch.Tensor | np.ndarray],
  ) -> dict[str, torch.Tensor]`
  - `q` shape: `(*batch_shape, dof)`
  - each input point set shape: `(P, 3)`
  - output: `{link_name: points}`
  - each output point set shape: `(*batch_shape, P, 3)`
  - output points are in the base/world frame

- `get_palm_pose(batch_shape: Sequence[int] | None = None) -> torch.Tensor`
  - if `batch_shape is None`, returns shape `(4, 4)`
  - otherwise returns shape `(*batch_shape, 4, 4)`
  - transform meaning: `base/world <- palm`

### Shape Rules

- `q` supports arbitrary leading batch dimensions as long as the last dimension is `dof`.
- Examples:
  - `(dof,)`
  - `(B, dof)`
  - `(B, M, dof)`
- Internal implementation may flatten leading batch dims and restore them before returning.

### Runtime Rules

- `points_by_link` is always interpreted as link-local static geometry.
- Each entry in `points_by_link` must have shape `(P, 3)`.
- Different links may have different `P`.
- Missing `link_name` values in `points_by_link` raise `KeyError`.
- All returned values are `torch.Tensor`.
- Device and dtype movement are handled through the normal torch module API, e.g. `.to(device=..., dtype=...)`.
- Implementation should avoid in-place tensor updates in the differentiable forward-kinematics path.

### Internal Design Direction

- The low-level FK engine may be named `DiffKin`.
- `DiffKin` is an internal URDF-based torch module that computes differentiable forward kinematics from canonical joint vectors.
- `HandKinematics` is the public wrapper that binds `DiffKin` to `HandInfo`.
