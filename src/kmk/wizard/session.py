from __future__ import annotations

"""CLI preparation helpers for the global wizard."""

from dataclasses import dataclass
from pathlib import Path
import re

from kmk.config.model import GripperConfig, normalize_palm_pose
from kmk.config.parse import ParsedUrdfNames, ParsedXmlNames, parse_urdf_names, parse_xml_names


def _resolve_root(path: str | Path) -> Path:
    root = Path(path).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    return root.resolve()


def _resolve_from_candidates(path: str | Path, *, gripper_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    root_relative = gripper_root / candidate
    if root_relative.exists():
        return root_relative.resolve()
    return (Path.cwd() / candidate).resolve()


def _display_relative(path: Path, *, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def _require_single_file(root: Path, suffix: str) -> Path:
    matches = sorted(root.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise ValueError(f"{root} must contain exactly one '{suffix}' file; found {len(matches)}")
    return matches[0].resolve()


def _list_files(root: Path, suffix: str) -> list[Path]:
    return [path.resolve() for path in sorted(root.glob(f"*{suffix}"))]


def _pick_preferred_urdf_candidate(urdf_candidates: list[Path]) -> Path | None:
    if not urdf_candidates:
        return None
    if len(urdf_candidates) == 1:
        return urdf_candidates[0]

    scored = []
    for candidate in urdf_candidates:
        name = candidate.name.lower()
        score = 0
        try:
            parsed = parse_urdf_names(candidate)
            score += len(parsed.actuated_joint_names) * 50
        except Exception:
            parsed = None
        if "scene" in name:
            score -= 10
        if "hand" in name:
            score += 5
        if name == "hand.urdf":
            score += 20
        if "right" in name:
            score += 3
        if "left" in name:
            score += 2
        scored.append((score, candidate.name, candidate))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][2]


def _resolve_required_urdf_path(root: Path) -> Path:
    urdf_candidates = _list_files(root, ".urdf")
    if not urdf_candidates:
        raise ValueError(f"{root} must contain at least one '.urdf' file; found 0")
    default_candidate = _pick_preferred_urdf_candidate(urdf_candidates)
    if default_candidate is None:
        raise ValueError(f"{root} must contain at least one '.urdf' file; found 0")
    if len(urdf_candidates) > 1:
        print("[wizard] Detected URDF candidates:")
        for index, candidate in enumerate(urdf_candidates, start=1):
            default_mark = " (default)" if candidate == default_candidate else ""
            print(f"[wizard] {index}. {candidate.name}{default_mark}")

    prompt = f"[wizard] URDF path [{default_candidate.name}]: "
    try:
        raw = input(prompt).strip()
    except OSError:
        return default_candidate
    if not raw:
        return default_candidate
    return _resolve_from_candidates(raw, gripper_root=root)


def _pick_preferred_xml_candidate(xml_candidates: list[Path], *, urdf_file: Path) -> Path | None:
    if not xml_candidates:
        return None

    urdf_stem = urdf_file.stem.lower()
    exact_name = f"{urdf_stem}.xml"
    for candidate in xml_candidates:
        if candidate.name.lower() == exact_name:
            return candidate

    scored = []
    for candidate in xml_candidates:
        name = candidate.name.lower()
        score = 0
        if "scene" in name:
            score -= 10
        if "hand" in name:
            score += 5
        if urdf_stem and urdf_stem in name:
            score += 20
        if urdf_stem.startswith("right") and "right" in name:
            score += 8
        if urdf_stem.startswith("left") and "left" in name:
            score += 8
        scored.append((score, candidate.name, candidate))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][2]


def _resolve_optional_xml_path(root: Path, *, urdf_file: Path) -> Path | None:
    xml_candidates = _list_files(root, ".xml")
    if not xml_candidates:
        return None

    default_candidate = _pick_preferred_xml_candidate(xml_candidates, urdf_file=urdf_file)
    if default_candidate is None:
        return None

    if len(xml_candidates) > 1:
        print("[wizard] Detected XML candidates:")
        for index, candidate in enumerate(xml_candidates, start=1):
            default_mark = " (default)" if candidate == default_candidate else ""
            print(f"[wizard] {index}. {candidate.name}{default_mark}")

    default_text = default_candidate.name
    prompt = f"[wizard] XML path [{default_text}] (`-` disables XML): "
    try:
        raw = input(prompt).strip()
    except OSError:
        return default_candidate
    if not raw:
        return default_candidate
    if raw == "-":
        return None
    return _resolve_from_candidates(raw, gripper_root=root)


def _prompt_for_name() -> str:
    name = input("Config name: ").strip()
    if not name:
        raise ValueError("Config name must not be empty")
    return name


def _print_actuated_joints(joint_names: list[str]) -> None:
    print("[wizard] Detected actuated joints:")
    if not joint_names:
        print("[wizard] (none)")
        return
    for index, name in enumerate(joint_names, start=1):
        print(f"[wizard] {index}. {name}")


def _parse_joint_order(raw: str, joint_names: list[str]) -> list[str]:
    tokens = [token for token in re.split(r"[,\s]+", raw.strip()) if token]
    if len(tokens) != len(joint_names):
        raise ValueError(f"Invalid joint order: expected {len(joint_names)} indices, got {len(tokens)}")

    ordered: list[str] = []
    seen: set[int] = set()
    for token in tokens:
        index = int(token)
        normalized = index - 1
        if normalized < 0 or normalized >= len(joint_names):
            raise ValueError(f"Invalid joint order: joint index out of range {index}")
        if normalized in seen:
            raise ValueError(f"Invalid joint order: duplicate joint index {index}")
        seen.add(normalized)
        ordered.append(joint_names[normalized])
    return ordered


def _resolve_joint_order(joint_names: list[str], existing_order: list[str] | None) -> list[str]:
    _print_actuated_joints(joint_names)
    if existing_order:
        print("[wizard] Preserving joint_order from config")
        return list(existing_order)
    if not joint_names:
        return []
    try:
        raw = input(
            "[wizard] Joint order indices (comma or whitespace separated, blank keeps detected order): "
        ).strip()
    except OSError:
        return list(joint_names)
    if not raw:
        return list(joint_names)
    return _parse_joint_order(raw, joint_names)


@dataclass(frozen=True)
class WizardSession:
    mode: str
    gripper_root: str
    gripper_root_abs: Path
    save_path: Path
    config: GripperConfig
    from_config: str | None
    urdf: ParsedUrdfNames
    xml: ParsedXmlNames | None


def prepare_session(
    *,
    gripper_root: str,
    from_config: str | None = None,
    name: str | None = None,
    urdf_path: str | None = None,
    xml_path: str | None = None,
    save_path: str | None = None,
) -> WizardSession:
    root = _resolve_root(gripper_root)
    if not root.exists():
        raise FileNotFoundError(f"gripper_root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"gripper_root must be a directory: {root}")

    if from_config is not None:
        config_path = _resolve_from_candidates(from_config, gripper_root=root)
        config = GripperConfig.load(config_path)
        urdf_file = _resolve_from_candidates(config.urdf_path, gripper_root=root)
        xml_file = (
            _resolve_from_candidates(config.xml_path, gripper_root=root)
            if config.xml_path is not None
            else None
        )
        config_name = config.name
        mode = "edit"
        from_config_rel = _display_relative(config_path, base=root)
        resolved_save_path = (
            _resolve_from_candidates(save_path, gripper_root=root)
            if save_path is not None
            else config_path
        )
    else:
        config_name = name.strip() if name is not None else _prompt_for_name()
        urdf_file = (
            _resolve_from_candidates(urdf_path, gripper_root=root)
            if urdf_path is not None
            else _resolve_required_urdf_path(root)
        )
        parsed_urdf = parse_urdf_names(urdf_file)
        xml_file = (
            _resolve_from_candidates(xml_path, gripper_root=root)
            if xml_path is not None
            else _resolve_optional_xml_path(root, urdf_file=urdf_file)
        )
        joint_order = _resolve_joint_order(list(parsed_urdf.actuated_joint_names), None)
        config = GripperConfig(
            name=config_name,
            urdf_path=_display_relative(urdf_file, base=root),
            xml_path=_display_relative(xml_file, base=root) if xml_file is not None else None,
            joint_order=joint_order,
            xml_joint_actuator_alias={},
            palm_pose=normalize_palm_pose(None),
            additional_collision_ignore_pairs=[],
        )
        mode = "new"
        from_config_rel = None
        resolved_save_path = Path(save_path).expanduser().resolve() if save_path is not None else root / f"{config_name}.yaml"

    parsed_urdf = parse_urdf_names(urdf_file)
    if from_config is not None:
        config.joint_order = _resolve_joint_order(
            list(parsed_urdf.actuated_joint_names),
            list(config.joint_order) if config.joint_order else None,
        )
        config.urdf_path = _display_relative(urdf_file, base=root)
        config.xml_path = _display_relative(xml_file, base=root) if xml_file is not None else None
        config.palm_pose = normalize_palm_pose(config.palm_pose)

    parsed_xml = parse_xml_names(xml_file) if xml_file is not None else None
    config.validate(
        urdf_actuated_joint_names=parsed_urdf.actuated_joint_names,
        urdf_link_names=parsed_urdf.link_names,
    )

    return WizardSession(
        mode=mode,
        gripper_root=_display_relative(root, base=root),
        gripper_root_abs=root,
        save_path=resolved_save_path,
        config=config,
        from_config=from_config_rel,
        urdf=parsed_urdf,
        xml=parsed_xml,
    )
