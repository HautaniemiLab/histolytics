"""Regression checks for release version and archive guards."""

import importlib
import sys
import tarfile
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

check_archive = importlib.import_module("check_distributions").check_archive
check_versions = importlib.import_module("check_release_metadata").check_versions


class ReleaseGuardsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "src/histolytics").mkdir(parents=True)
        (self.root / "pyproject.toml").write_text('[project]\nversion = "0.2.5"\n')
        (self.root / "src/histolytics/__init__.py").write_text(
            '__version__ = "0.2.5"\n'
        )

    def test_matching_versions(self):
        for tag in ("", "0.2.5", "v0.2.5"):
            self.assertEqual(check_versions(self.root, tag), "0.2.5")

    def test_module_mismatch(self):
        (self.root / "src/histolytics/__init__.py").write_text(
            '__version__ = "0.2.4"\n'
        )
        with self.assertRaisesRegex(ValueError, "module version"):
            check_versions(self.root)

    def test_missing_module_version(self):
        (self.root / "src/histolytics/__init__.py").write_text("")
        with self.assertRaises(ValueError):
            check_versions(self.root)

    def test_tag_mismatch(self):
        with self.assertRaisesRegex(ValueError, "Release tag"):
            check_versions(self.root, "v0.2.6")

    def test_archive_guards(self):
        for kind in ("wheel", "sdist"):
            for missing_data, version in (
                (False, "0.2.5"),
                (True, "0.2.5"),
                (False, "0.2.4"),
            ):
                with self.subTest(
                    kind=kind, missing_data=missing_data, version=version
                ):
                    files = {"histolytics/py.typed": b""}
                    if not missing_data:
                        files["histolytics/data/sample.parquet"] = b"sample"
                    metadata = f"Name: histolytics\nVersion: {version}\n".encode()
                    if kind == "wheel":
                        path = self.root / "example.whl"
                        with zipfile.ZipFile(path, "w") as archive:
                            for name, content in files.items():
                                archive.writestr(name, content)
                            archive.writestr("histolytics.dist-info/METADATA", metadata)
                    else:
                        path = self.root / "example.tar.gz"
                        files = {
                            f"example/src/{name}": content
                            for name, content in files.items()
                        }
                        files["example/PKG-INFO"] = metadata
                        with tarfile.open(path, "w:gz") as archive:
                            for name, content in files.items():
                                member = tarfile.TarInfo(name)
                                member.size = len(content)
                                archive.addfile(member, BytesIO(content))
                    expected = {
                        "histolytics/py.typed",
                        "histolytics/data/sample.parquet",
                    }
                    if missing_data or version != "0.2.5":
                        with self.assertRaises(ValueError):
                            check_archive(path, expected, "0.2.5")
                    else:
                        check_archive(path, expected, "0.2.5")
