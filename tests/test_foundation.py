import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from reference_system.foundation import initialize_project


class FoundationTests(unittest.TestCase):
    def test_initialize_creates_manifest_quotas_directories_and_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "personal-reference"

            result = initialize_project(root)

            self.assertEqual(result["mode"], "LOCAL_ONLY")
            self.assertEqual(result["storage_quota"], {
                "total_gib": 30,
                "live_gib": 28,
                "backup_gib": 0,
                "temporary_gib": 2,
                "minimum_free_disk_gib": 20,
            })
            manifest_path = root / "manifest.yaml"
            checksum_path = root / "manifest.sha256"
            self.assertTrue(manifest_path.is_file())
            self.assertTrue(checksum_path.is_file())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertFalse(manifest["backup_enabled"])
            expected_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            self.assertEqual(checksum_path.read_text(encoding="ascii").strip(), expected_hash)

            for relative in (
                "config", "db/migrations", "store/blobs/sha256", "quarantine",
                "artifacts", "indexes/vector", "indexes/manifests", "processors",
                "viewer", "reports/ingest", "reports/deletion", "reports/integrity",
                "logs", "tests/approved-fixtures", "tests/expected",
            ):
                self.assertTrue((root / relative).is_dir(), relative)

            db_path = root / "db/reference.sqlite3"
            self.assertTrue(db_path.is_file())
            with closing(sqlite3.connect(db_path)) as conn:
                tables = {row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )}
            self.assertTrue({"material", "material_version", "file_blob", "fragment", "fragment_fts", "job"} <= tables)


if __name__ == "__main__":
    unittest.main()
