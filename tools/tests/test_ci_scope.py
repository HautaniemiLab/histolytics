"""Keep routine CI small without dropping release installation coverage."""

import json
import re
import unittest
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github/workflows"


class CIScopeTests(unittest.TestCase):
    def test_installation_scopes(self):
        workflow = (WORKFLOWS / "package.yml").read_text()
        full, compatibility, routine = [
            json.loads(value)
            for value in re.findall(r"'(\[\s*\{.*?\}\s*\])'", workflow, re.S)
        ]

        def key(row):
            return (
                row["python-version"],
                row["distribution"],
                row["profile"],
                row["os"],
            )

        full_keys = {key(row) for row in full}
        expected = {
            (version, archive, profile, "ubuntu-latest")
            for version in ("3.10", "3.11", "3.12")
            for archive in ("wheel", "sdist")
            for profile in ("analysis", "full")
        }
        expected |= {
            ("3.12", archive, profile, platform)
            for archive in ("wheel", "sdist")
            for profile, platform in (
                ("analysis", "macos-latest"),
                ("segmentation", "macos-latest"),
                ("segmentation", "ubuntu-latest"),
                ("cuda-analysis", "ubuntu-latest"),
            )
        }
        self.assertEqual(full_keys, expected)
        self.assertEqual(len(full), 20)
        self.assertEqual(len(routine), 6)
        self.assertEqual(len({key(row) for row in routine}), 6)
        self.assertTrue({key(row) for row in routine} <= full_keys)
        self.assertEqual(
            {row["profile"] for row in routine},
            {"analysis", "full", "segmentation", "cuda-analysis"},
        )
        self.assertEqual({row["distribution"] for row in routine}, {"wheel", "sdist"})
        self.assertEqual(
            {key(row) for row in compatibility},
            {
                ("3.13", archive, profile, "ubuntu-latest")
                for archive in ("wheel", "sdist")
                for profile in ("analysis", "full")
            },
        )

    def test_release_requires_full_scope(self):
        publish = (WORKFLOWS / "publish.yml").read_text()
        self.assertRegex(
            publish, r"uses: ./\.github/workflows/tests.yml\s+with:\s+scope: full"
        )
        self.assertIn("needs: validate", publish)
        source = (WORKFLOWS / "tests.yml").read_text()
        versions = re.findall(r"'(\[\"3\..*?\])'", source)
        self.assertEqual(
            [json.loads(value) for value in versions],
            [["3.13"], ["3.10", "3.12"], ["3.10", "3.11", "3.12"]],
        )

    def test_checkpoint_downloads_require_explicit_scope(self):
        source = (WORKFLOWS / "tests.yml").read_text()
        self.assertRegex(
            source,
            r"name: Check pinned public checkpoint on CPU\s+if: inputs.scope == 'checkpoint'",
        )
        self.assertIn('revision=checkpoint["revision"]', source)
        self.assertIn("token=False", source)
        self.assertRegex(
            source,
            r"name: Fetch pinned public checkpoint\s+if: inputs.scope == 'checkpoint'",
        )
        self.assertIn(
            'tools/check_checkpoint_reference.py "$HISTOLYTICS_CHECKPOINT_PATH"', source
        )
        self.assertRegex(source, r"packages:\s+if: inputs.scope != 'checkpoint'")
