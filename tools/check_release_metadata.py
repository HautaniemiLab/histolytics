"""Check version agreement without importing runtime dependencies."""

import ast
import os
from pathlib import Path

import tomllib


def check_versions(root: Path, tag: str = "") -> str:
    """Require matching project, module, and optional release tag versions."""
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    module = ast.parse((root / "src/histolytics/__init__.py").read_text())
    module_version = None
    for statement in module.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in statement.targets
        ):
            module_version = ast.literal_eval(statement.value)
    if module_version != version:
        raise ValueError(
            f"Project version {version} != module version {module_version}"
        )
    if tag and tag.removeprefix("v") != version:
        raise ValueError(f"Release tag {tag} != package version {version}")
    return version


if __name__ == "__main__":
    print(
        check_versions(
            Path(__file__).resolve().parents[1], os.environ.get("RELEASE_TAG", "")
        )
    )
