"""Verify release archives include the package, typing marker, and bundled data."""

import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

from check_release_metadata import check_versions


def check_archive(path: Path, expected: set[str], version: str) -> None:
    """Check archive contents and metadata without extracting files."""
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            metadata_names = [
                name for name in names if name.endswith(".dist-info/METADATA")
            ]
            metadata = (
                archive.read(metadata_names[0]) if len(metadata_names) == 1 else b""
            )
    else:
        with tarfile.open(path) as archive:
            members = [member for member in archive.getmembers() if member.isfile()]
            names = {
                member.name.split("/", 1)[-1].removeprefix("src/") for member in members
            }
            metadata_members = [
                member for member in members if member.name.endswith("/PKG-INFO")
            ]
            metadata = (
                archive.extractfile(metadata_members[0]).read()
                if len(metadata_members) == 1
                else b""
            )
    missing = expected - names
    if missing:
        raise ValueError(f"{path.name} is missing package files: {sorted(missing)}")
    parsed = BytesParser().parsebytes(metadata)
    if parsed["Name"] != "histolytics" or parsed["Version"] != version:
        raise ValueError(f"{path.name} has incorrect or missing package metadata")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    expected = {
        path.relative_to(root / "src").as_posix()
        for path in (root / "src/histolytics").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    wheels = list((root / "dist").glob("*.whl"))
    sdists = list((root / "dist").glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Expected exactly one wheel and one source distribution")
    for path in wheels + sdists:
        check_archive(path, expected, check_versions(root))
        print(f"Validated {path.name}: {len(expected)} package files")
