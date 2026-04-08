"""CLI entrypoint for the staged gripper configuration wizard."""

from __future__ import annotations

import tyro

from kmk.wizard.gui import create_global_app, create_keypoint_app, create_preview_app, create_template_app
from kmk.wizard.session import prepare_session


def run(
    gripper_root: str,
    from_config: str | None = None,
    name: str | None = None,
    urdf_path: str | None = None,
    xml_path: str | None = None,
    save_path: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    session = prepare_session(
        gripper_root=gripper_root,
        from_config=from_config,
        name=name,
        urdf_path=urdf_path,
        xml_path=xml_path,
        save_path=save_path,
    )
    print(f"[wizard] Prepared global stage mode={session.mode} save_path={session.save_path}")
    global_app = create_global_app(session, host=host, port=port)
    global_app.run_until_complete()
    print(f"[wizard] Global stage completed: {session.save_path}")

    keypoint_app = create_keypoint_app(session, host=host, port=port)
    keypoint_app.run_until_complete()
    print(f"[wizard] Keypoint stage completed: {session.save_path}")

    template_app = create_template_app(session, host=host, port=port)
    template_app.run_until_complete()
    print(f"[wizard] Template stage completed: {session.save_path}")

    preview_app = create_preview_app(session, host=host, port=port)
    preview_app.run_until_complete()
    print(f"[wizard] Preview stage confirmed: {session.save_path}")


def main(argv: list[str] | None = None) -> int:
    tyro.cli(run, args=argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
