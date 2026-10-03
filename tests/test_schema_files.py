"""The committed SQL schema must match the code that generates migrations.

`reference_system.foundation` carries the canonical migration SQL as
constants and writes it into a project's db/migrations directory at init. The
repository also ships a readable copy under schema/migrations so the schema can
be reviewed without running anything. This test fails if the two ever drift.
"""

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from reference_system.foundation import MIGRATIONS, write_migration_files

SCHEMA_DIR = PROJECT_ROOT / "schema" / "migrations"


class SchemaFileTests(unittest.TestCase):
    def test_committed_schema_files_match_the_code(self):
        self.assertTrue(SCHEMA_DIR.is_dir(), SCHEMA_DIR)
        for filename, sql in MIGRATIONS:
            path = SCHEMA_DIR / filename
            self.assertTrue(path.is_file(), path)
            self.assertEqual(path.read_text(encoding="utf-8"), sql, filename)

    def test_write_migration_files_reproduces_the_same_sql(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = write_migration_files(tmp, overwrite=True)
            self.assertEqual(len(written), len(MIGRATIONS))
            for filename, sql in MIGRATIONS:
                self.assertEqual((Path(tmp) / filename).read_text(encoding="utf-8"), sql)

    def test_migration_versions_are_ordered_and_unique(self):
        versions = [int(filename.split("_", 1)[0]) for filename, _ in MIGRATIONS]
        self.assertEqual(versions, sorted(versions))
        self.assertEqual(len(versions), len(set(versions)))


if __name__ == "__main__":
    unittest.main()
