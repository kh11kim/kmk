# kmk

`kmk` is a utility repo for authoring and consuming keypoint-based multi-fingered hand kinematics data.

It has two main jobs:

- a staged GUI wizard that builds a hand config YAML
- runtime loaders and torch kinematics utilities that read that YAML

## What This Repo Provides

- `gripper_config_wizard`
  - staged authoring flow for:
    - global hand config
    - per-link contact anchors
    - grasp templates
    - final preview/confirmation
- `HandInfo`
  - read-only runtime wrapper around the authored YAML
- `PointedHandInfo`
  - `HandInfo` plus precomputed surface/contact point samples
- `HandKinematics`
  - torch forward kinematics for batched joint vectors and link-local point sets

## User Workflow

1. Prepare a gripper root directory containing at least a URDF.
2. Run the wizard.
3. Fill the global stage.
   - palm pose
   - palm points delta
   - global `q_open`
   - extra collision ignore pairs
4. Fill the keypoint stage.
   - one contact anchor per link
   - point, radius, tags
5. Fill the grasp template stage.
   - `q_open`, `q_close`
   - `grasp_target_point`
   - active contact anchors
6. Check the final preview and press `Confirmed`.
7. Use the saved YAML through `HandInfo`, `PointedHandInfo`, or `HandKinematics`.

## Install

```bash
uv sync
```

## Wizard

Create a new config:

```bash
uv run gripper_config_wizard --gripper-root /path/to/gripper_root
```

Edit an existing config:

```bash
uv run gripper_config_wizard \
  --gripper-root /path/to/gripper_root \
  --from-config existing.yaml
```

Notes:

- `--from-config` relative paths are resolved from `gripper_root`.
- `urdf_path` and `xml_path` stored in YAML are typically relative paths.

## Visualization

Inspect a saved config and sampled points:

```bash
uv run python visualize/pointed_hand_info.py --config-path /path/to/hand.yaml
```

Inspect batched torch kinematics:

```bash
uv run python visualize/hand_kinematics_batch.py --config-path /path/to/hand.yaml
```

## Runtime API

### `HandInfo`

```python
from kmk import HandInfo

hand = HandInfo.from_config("hand.yaml")
q_open = hand.get_q_open()
q_close = hand.get_q_close("finger4_shallow")
palm_pose = hand.get_palm_pose()
anchor = hand.get_contact_anchor_by_link("right_1thumb_distal")
anchors = hand.get_contact_anchor_by_template("finger4_shallow")
target = hand.get_grasp_target_point("finger4_shallow")
```

Main API:

- `HandInfo.from_config(config_path)`
- `joint_order`
- `template_names`
- `contact_anchor_links`
- `get_q_open(template="global")`
- `get_q_close(template_name)`
- `get_palm_pose()`
- `get_contact_anchor_by_link(link_name)`
- `get_contact_anchor_by_tag(includes=(), excludes=())`
- `get_contact_anchor_by_template(template_name)`
- `get_grasp_target_point(template_name)`

### `PointedHandInfo`

```python
from kmk import PointedHandInfo

hand = PointedHandInfo.from_config("hand.yaml", seed=0)
surface_points = hand.surface_points
contact_points = hand.get_contact_points("finger4_shallow")
keypoints = hand.get_keypoints("finger4_shallow", palm_aligned_points=True)
```

Main additions:

- `surface_points`
- `contact_points`
- `get_contact_points(template_name=None)`
- `get_keypoints(template_name=None, palm_aligned_points=True, palm_points_delta=...)`

### `HandKinematics`

```python
import torch
from kmk import HandInfo, HandKinematics

hand = HandInfo.from_config("hand.yaml")
kin = HandKinematics(hand).to(device="cpu")

q = torch.zeros(16, len(hand.joint_order))
fk = kin.forward_kinematics(q)
world_points = kin.transform_link_points(
    q,
    {"right_1thumb_distal": torch.tensor([[0.0, 0.0, 0.02]])},
)
```

Main API:

- `HandKinematics(hand_info_or_config_path)`
- `joint_order`
- `link_names`
- `dof`
- `forward_kinematics(q)`
- `transform_link_points(q, points_by_link)`
- `get_palm_pose(batch_shape=None)`

## Config Schema

The detailed spec lives in [`SPEC.md`](/Users/polde/ws/kmk/SPEC.md).
