from __future__ import annotations

"""URDF/XML parsing helpers."""

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


def _parse_xml(path: str | Path) -> ET.Element:
    return ET.parse(Path(path)).getroot()


def _name_sequence(elements: list[ET.Element]) -> tuple[str, ...]:
    names: list[str] = []
    for element in elements:
        name = element.attrib.get("name")
        if name:
            names.append(name)
    return tuple(names)


@dataclass(frozen=True, slots=True)
class ParsedUrdfNames:
    joint_names: tuple[str, ...]
    link_names: tuple[str, ...]
    actuated_joint_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParsedXmlNames:
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]


def parse_urdf_names(path: str | Path) -> ParsedUrdfNames:
    root = _parse_xml(path)
    link_names = _name_sequence(root.findall(".//link"))
    joints = root.findall(".//joint")
    joint_names = _name_sequence(joints)
    actuated_joint_names = tuple(
        element.attrib["name"]
        for element in joints
        if element.attrib.get("name") and element.attrib.get("type") not in {"fixed", "floating"}
    )
    return ParsedUrdfNames(
        joint_names=joint_names,
        link_names=link_names,
        actuated_joint_names=actuated_joint_names,
    )


def parse_xml_names(path: str | Path) -> ParsedXmlNames:
    root = _parse_xml(path)
    joint_names = _name_sequence(root.findall(".//joint"))
    actuator_names = _name_sequence(root.findall(".//actuator/*"))
    return ParsedXmlNames(joint_names=joint_names, actuator_names=actuator_names)
