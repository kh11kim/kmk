from __future__ import annotations

from pathlib import Path

from kmk.config.parse import parse_urdf_names, parse_xml_names


def test_parse_urdf_and_xml_names(tmp_path: Path) -> None:
    urdf_path = tmp_path / "hand.urdf"
    urdf_path.write_text(
        "\n".join(
            [
                "<?xml version='1.0'?>",
                "<robot name='demo'>",
                "  <link name='base' />",
                "  <link name='finger' />",
                "  <joint name='j1' type='revolute'><parent link='base'/><child link='finger'/></joint>",
                "  <joint name='j_fixed' type='fixed'><parent link='finger'/><child link='tip'/></joint>",
                "</robot>",
                "",
            ]
        ),
        encoding="utf-8",
    )
    xml_path = tmp_path / "hand.xml"
    xml_path.write_text(
        "\n".join(
            [
                "<?xml version='1.0'?>",
                "<mujoco model='demo'>",
                "  <worldbody><body name='hand'><joint name='j1_xml' /></body></worldbody>",
                "  <actuator><motor name='a1' joint='j1_xml' /></actuator>",
                "</mujoco>",
                "",
            ]
        ),
        encoding="utf-8",
    )

    parsed_urdf = parse_urdf_names(urdf_path)
    parsed_xml = parse_xml_names(xml_path)

    assert parsed_urdf.link_names == ("base", "finger")
    assert parsed_urdf.actuated_joint_names == ("j1",)
    assert parsed_xml.joint_names == ("j1_xml",)
    assert parsed_xml.actuator_names == ("a1",)


def test_parse_urdf_names_excludes_mimic_joints(tmp_path: Path) -> None:
    urdf_path = tmp_path / "mimic_hand.urdf"
    urdf_path.write_text(
        "\n".join(
            [
                "<?xml version='1.0'?>",
                "<robot name='demo'>",
                "  <link name='base' />",
                "  <link name='link_1' />",
                "  <link name='link_2' />",
                "  <joint name='j1' type='revolute'>",
                "    <parent link='base'/>",
                "    <child link='link_1'/>",
                "  </joint>",
                "  <joint name='j2' type='revolute'>",
                "    <parent link='link_1'/>",
                "    <child link='link_2'/>",
                "    <mimic joint='j1' multiplier='1.0' offset='0.0'/>",
                "  </joint>",
                "</robot>",
                "",
            ]
        ),
        encoding="utf-8",
    )

    parsed_urdf = parse_urdf_names(urdf_path)

    assert parsed_urdf.joint_names == ("j1", "j2")
    assert parsed_urdf.actuated_joint_names == ("j1",)
