import contextlib
import io
import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from reference_system.cli import main as cli_main
from reference_system.core import ReferenceService
from reference_system.foundation import (
    FoundationError,
    apply_migrations,
    applied_migration_versions,
    ensure_migration_files,
    initialize_project,
)

EXPECTED_VERSIONS = [1, 2, 3, 4]
EXPECTED_TABLES = {
    "schema_version",
    "knowledge_base",
    "ingest_event",
    "file_blob",
    "material",
    "material_version",
    "processing_run",
    "artifact",
    "fragment",
    "fragment_fts",
    "job",
    "ocr_cache",
    "embedding",
}


def _table_names(db_path: Path) -> set[str]:
    with closing(sqlite3.connect(db_path)) as conn:
        return {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
        }


class MigrationTests(unittest.TestCase):
    def test_migrations_build_schema_on_an_empty_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "kb"
            migrations_dir = ensure_migration_files(root)
            db_path = root / "db" / "reference.sqlite3"

            applied = apply_migrations(db_path, migrations_dir)

            self.assertEqual(applied, EXPECTED_VERSIONS)
            self.assertTrue(EXPECTED_TABLES.issubset(_table_names(db_path)))
            self.assertEqual(applied_migration_versions(db_path), EXPECTED_VERSIONS)

    def test_migrations_are_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "kb"
            migrations_dir = ensure_migration_files(root)
            db_path = root / "db" / "reference.sqlite3"

            apply_migrations(db_path, migrations_dir)
            second = apply_migrations(db_path, migrations_dir)
            third = apply_migrations(db_path, migrations_dir)

            self.assertEqual(second, [])
            self.assertEqual(third, [])
            with closing(sqlite3.connect(db_path)) as conn:
                self.assertEqual(
                    conn.execute("SELECT count(*) FROM schema_version").fetchone()[0],
                    len(EXPECTED_VERSIONS),
                )

    def test_initialize_project_is_repeatable_and_keeps_manifest_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "kb"
            first = initialize_project(root)
            second = initialize_project(root)

            self.assertEqual(first["knowledge_base_id"], second["knowledge_base_id"])
            manifest = json.loads((root / "manifest.yaml").read_text(encoding="utf-8"))
            self.assertEqual(manifest["knowledge_base_id"], first["knowledge_base_id"])

    def test_reinitializing_with_reset_rebuilds_database_from_migrations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "kb"
            initialize_project(root)
            source = Path(tmp) / "note.txt"
            source.write_text("Температура в норме.\n", encoding="utf-8")
            ReferenceService(root).add_file(source)

            initialize_project(root, reset_database=True)

            self.assertEqual(
                applied_migration_versions(root / "db" / "reference.sqlite3"), EXPECTED_VERSIONS
            )
            service = ReferenceService(root)
            self.assertEqual(service.status()["materials"], 0)

    def test_unknown_migration_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            migrations_dir = Path(tmp) / "migrations"
            migrations_dir.mkdir()
            (migrations_dir / "not_versioned.sql").write_text("SELECT 1;", encoding="utf-8")
            with self.assertRaises(FoundationError):
                apply_migrations(Path(tmp) / "db.sqlite3", migrations_dir)

    def test_duplicate_migration_versions_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            migrations_dir = Path(tmp) / "migrations"
            migrations_dir.mkdir()
            (migrations_dir / "0001_a.sql").write_text("SELECT 1;", encoding="utf-8")
            (migrations_dir / "0001_b.sql").write_text("SELECT 1;", encoding="utf-8")
            with self.assertRaises(FoundationError):
                apply_migrations(Path(tmp) / "db.sqlite3", migrations_dir)


class CliTests(unittest.TestCase):
    def _run(self, argv: list[str]) -> int:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            return cli_main(argv)

    def _run_capture(self, argv: list[str]) -> tuple[int, str]:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = cli_main(argv)
        return code, buffer.getvalue()

    def test_cli_search_supports_both_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "kb"
            self._run(["--root", str(root), "init"])
            source = Path(tmp) / "note.txt"
            source.write_text("Насос P-101 вибрирует.\n", encoding="utf-8")
            self._run(["--root", str(root), "add", str(source)])

            code, output = self._run_capture(["--root", str(root), "search", "P-101", "--exact"])
            self.assertEqual(code, 0)
            exact = json.loads(output)
            self.assertEqual(exact["mode"], "exact")
            self.assertEqual(len(exact["results"]), 1)
            self.assertEqual(exact["results"][0]["match_mode"], "exact")

            code, output = self._run_capture(["--root", str(root), "search", "P-101"])
            self.assertEqual(code, 0)
            full_text = json.loads(output)
            self.assertEqual(full_text["mode"], "full-text")

    def test_full_cli_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "kb"
            workspace = Path(tmp)

            self.assertEqual(self._run(["--root", str(root), "init"]), 0)

            source = workspace / "report.txt"
            source.write_text("Насос P-101 вибрирует при пуске.\n", encoding="utf-8")
            self.assertEqual(self._run(["--root", str(root), "add", str(source)]), 0)

            self.assertEqual(self._run(["--root", str(root), "status"]), 0)
            self.assertEqual(self._run(["--root", str(root), "search", "вибрирует"]), 0)

            service = ReferenceService(root)
            fragment_id = service.search("вибрирует")[0]["fragment_id"]
            material_id = service.search("вибрирует")[0]["material_id"]

            self.assertEqual(self._run(["--root", str(root), "show-source", fragment_id]), 0)
            self.assertEqual(self._run(["--root", str(root), "reprocess", material_id]), 0)
            self.assertEqual(self._run(["--root", str(root), "integrity-check"]), 0)

            self.assertEqual(self._run(["--root", str(root), "delete", material_id]), 1)
            self.assertEqual(self._run(["--root", str(root), "delete", material_id, "--confirm"]), 0)
            self.assertEqual(service.search("вибрирует"), [])

    def test_cli_reports_unsupported_input_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "kb"
            self._run(["--root", str(root), "init"])
            source = Path(tmp) / "unknown.bin"
            source.write_bytes(b"\x00\x01\x02")
            self.assertEqual(self._run(["--root", str(root), "add", str(source)]), 1)

    def test_cli_integrity_exit_code_flags_damage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "kb"
            self._run(["--root", str(root), "init"])
            source = Path(tmp) / "note.txt"
            source.write_text("Подшипник перегрет.\n", encoding="utf-8")
            self._run(["--root", str(root), "add", str(source)])

            blob = next(
                path for path in (root / "store" / "blobs" / "sha256").rglob("*") if path.is_file()
            )
            blob.write_bytes(b"corrupted")

            self.assertEqual(self._run(["--root", str(root), "integrity-check"]), 2)


if __name__ == "__main__":
    unittest.main()
