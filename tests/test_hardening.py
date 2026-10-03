import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from reference_system import core, foundation
from reference_system.core import QuotaExceededError, ReferenceError, ReferenceService
from reference_system.foundation import initialize_project
from reference_system.processors import UnsupportedFormatError

_GIB = 1024 ** 3


class _TempProject:
    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "personal-reference"
        initialize_project(self.root)
        self.workspace = Path(self._tmp.name)
        return self

    def __exit__(self, *exc):
        self._tmp.cleanup()
        return False


def _count_files(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file())


class HardeningTests(unittest.TestCase):
    def test_scanned_pdf_without_text_layer_is_rejected_without_residue(self):
        import fitz

        with _TempProject() as project:
            source = project.workspace / "scan.pdf"
            document = fitz.open()
            document.new_page()
            document.save(source)
            document.close()

            service = ReferenceService(project.root)
            with self.assertRaises(UnsupportedFormatError) as ctx:
                service.add_file(source)
            self.assertIn("OCR", str(ctx.exception))

            self.assertEqual(service.search("scan"), [])
            self.assertEqual(_count_files(project.root / "store"), 0)
            self.assertEqual(_count_files(project.root / "quarantine"), 0)
            self.assertEqual(_count_files(project.root / "artifacts"), 0)
            with closing(sqlite3.connect(project.root / "db" / "reference.sqlite3")) as conn:
                self.assertEqual(conn.execute("SELECT count(*) FROM material").fetchone()[0], 0)
                self.assertEqual(conn.execute("SELECT count(*) FROM fragment_fts").fetchone()[0], 0)

    def test_unreadable_zip_containers_are_rejected(self):
        with _TempProject() as project:
            service = ReferenceService(project.root)
            for name in ("broken.docx", "broken.xlsx"):
                source = project.workspace / name
                source.write_bytes(b"PK\x03\x04" + b"\x00" * 64)
                with self.assertRaises(UnsupportedFormatError):
                    service.add_file(source)
            self.assertEqual(_count_files(project.root / "store"), 0)
            self.assertEqual(_count_files(project.root / "quarantine"), 0)

    def test_zip_without_expected_parts_is_rejected(self):
        import zipfile

        with _TempProject() as project:
            source = project.workspace / "partial.docx"
            with zipfile.ZipFile(source, "w") as package:
                package.writestr("readme.txt", "no content types here")
            service = ReferenceService(project.root)
            with self.assertRaises(UnsupportedFormatError):
                service.add_file(source)

    def test_identical_bytes_are_deduplicated_into_one_blob(self):
        with _TempProject() as project:
            payload = "Дефект подшипника обнаружен при осмотре.\n"
            first = project.workspace / "note-a.txt"
            second = project.workspace / "note-b.txt"
            first.write_text(payload, encoding="utf-8")
            second.write_text(payload, encoding="utf-8")

            service = ReferenceService(project.root)
            added_first = service.add_file(first)
            added_second = service.add_file(second)

            self.assertFalse(added_first["deduplicated_blob"])
            self.assertTrue(added_second["deduplicated_blob"])
            self.assertEqual(added_first["sha256"], added_second["sha256"])
            self.assertEqual(added_first["blob_relative_path"], added_second["blob_relative_path"])
            self.assertEqual(len(service.search("подшипника")), 2)
            with closing(sqlite3.connect(project.root / "db" / "reference.sqlite3")) as conn:
                self.assertEqual(conn.execute("SELECT count(*) FROM file_blob").fetchone()[0], 1)

    def test_directory_input_is_rejected(self):
        with _TempProject() as project:
            service = ReferenceService(project.root)
            with self.assertRaises(ReferenceError):
                service.add_file(project.workspace)

    def test_oversized_input_is_rejected_before_copying(self):
        with _TempProject() as project:
            source = project.workspace / "big.txt"
            source.write_text("x" * 4096, encoding="utf-8")
            original = foundation.STORAGE_QUOTA["temporary_gib"]
            foundation.STORAGE_QUOTA["temporary_gib"] = 0
            try:
                service = ReferenceService(project.root)
                with self.assertRaises(QuotaExceededError):
                    service.add_file(source)
            finally:
                foundation.STORAGE_QUOTA["temporary_gib"] = original
            self.assertEqual(_count_files(project.root / "quarantine"), 0)

    def test_unsupported_extension_is_rejected(self):
        with _TempProject() as project:
            source = project.workspace / "vector.svg"
            source.write_text("<svg/>", encoding="utf-8")
            service = ReferenceService(project.root)
            with self.assertRaises(UnsupportedFormatError):
                service.add_file(source)

    def test_delete_requires_explicit_confirmation(self):
        with _TempProject() as project:
            source = project.workspace / "note.txt"
            source.write_text("Насос требует замены сальника.\n", encoding="utf-8")
            service = ReferenceService(project.root)
            added = service.add_file(source)

            with self.assertRaises(ReferenceError):
                service.delete_material(added["material_id"])
            self.assertEqual(len(service.search("сальника")), 1)

            report = service.delete_material(added["material_id"], confirm=True)
            self.assertTrue(report["irreversible_without_backup"])
            self.assertEqual(service.search("сальника"), [])
            self.assertEqual(_count_files(project.root / "store"), 0)

    def test_integrity_check_detects_tampered_blob(self):
        with _TempProject() as project:
            source = project.workspace / "note.txt"
            source.write_text("Клапан открыт частично.\n", encoding="utf-8")
            service = ReferenceService(project.root)
            added = service.add_file(source)

            blob = project.root / added["blob_relative_path"]
            blob.write_bytes(blob.read_bytes() + b"tampered")

            report = service.integrity_check()
            self.assertFalse(report["ok"])
            self.assertIn("BLOB_HASH_MISMATCH", [issue["code"] for issue in report["issues"]])

    def test_integrity_check_detects_missing_fts_rows(self):
        with _TempProject() as project:
            source = project.workspace / "note.txt"
            source.write_text("Уплотнение изношено.\n", encoding="utf-8")
            service = ReferenceService(project.root)
            service.add_file(source)

            db_path = project.root / "db" / "reference.sqlite3"
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute("DELETE FROM fragment_fts")
                conn.commit()

            report = service.integrity_check()
            self.assertFalse(report["ok"])
            self.assertIn("FTS_COUNT_MISMATCH", [issue["code"] for issue in report["issues"]])

    def test_integrity_check_detects_manifest_tampering(self):
        with _TempProject() as project:
            manifest_path = project.root / "manifest.yaml"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["backup_enabled"] = True
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            service = ReferenceService(project.root)
            report = service.integrity_check()
            self.assertFalse(report["ok"])
            self.assertIn("MANIFEST_HASH_MISMATCH", [issue["code"] for issue in report["issues"]])

    def test_integrity_check_reports_dangling_blob_path_outside_root(self):
        with _TempProject() as project:
            source = project.workspace / "note.txt"
            source.write_text("Фильтр забит.\n", encoding="utf-8")
            service = ReferenceService(project.root)
            service.add_file(source)

            with closing(sqlite3.connect(project.root / "db" / "reference.sqlite3")) as conn:
                conn.execute("UPDATE file_blob SET relative_path='../../etc/passwd'")
                conn.commit()

            report = service.integrity_check()
            self.assertFalse(report["ok"])
            self.assertIn("BLOB_MISSING", [issue["code"] for issue in report["issues"]])

    def test_integrity_check_is_clean_for_fresh_project(self):
        with _TempProject() as project:
            source = project.workspace / "note.txt"
            source.write_text("Агрегат в норме.\n", encoding="utf-8")
            service = ReferenceService(project.root)
            service.add_file(source)
            report = service.integrity_check()
            self.assertTrue(report["ok"], report["issues"])
            self.assertFalse(report["restore_available"])

    def test_status_reports_quotas_and_local_only_mode(self):
        with _TempProject() as project:
            source = project.workspace / "note.txt"
            source.write_text("Температура подшипника 68 C.\n", encoding="utf-8")
            service = ReferenceService(project.root)
            service.add_file(source)

            status = service.status()
            self.assertEqual(status["mode"], "LOCAL_ONLY")
            self.assertEqual(status["live_quota_gib"], 28)
            self.assertEqual(status["temporary_quota_gib"], 2)
            self.assertEqual(status["backup_quota_gib"], 0)
            self.assertEqual(status["minimum_free_disk_gib"], 20)
            self.assertFalse(status["backup_enabled"])
            self.assertEqual(status["materials"], 1)
            self.assertGreater(status["used_bytes"], 0)

    def test_empty_text_file_is_indexed_without_fragments(self):
        with _TempProject() as project:
            source = project.workspace / "empty.txt"
            source.write_text("", encoding="utf-8")
            service = ReferenceService(project.root)
            added = service.add_file(source)
            self.assertEqual(added["status"], "READY")
            self.assertEqual(service.search("empty"), [])
            with closing(sqlite3.connect(project.root / "db" / "reference.sqlite3")) as conn:
                self.assertEqual(conn.execute("SELECT count(*) FROM fragment").fetchone()[0], 0)
                self.assertEqual(conn.execute("SELECT count(*) FROM fragment_fts").fetchone()[0], 0)
                self.assertEqual(conn.execute("SELECT count(*) FROM material").fetchone()[0], 1)
            report = service.integrity_check()
            self.assertTrue(report["ok"], report["issues"])

    def test_reprocess_uses_immutable_blob_and_refreshes_fragments(self):
        with _TempProject() as project:
            source = project.workspace / "note.txt"
            source.write_text("Зазор увеличен.\n", encoding="utf-8")
            service = ReferenceService(project.root)
            added = service.add_file(source)

            report = service.reprocess(added["material_id"])
            self.assertEqual(report["source"], "immutable_blob")
            self.assertEqual(report["fragment_count"], 1)
            self.assertEqual(len(service.search("Зазор")), 1)
            self.assertTrue((project.root / "reports" / "reprocess" / f"{report['processing_run_id']}.json").is_file())

    def test_reprocess_reports_missing_blob(self):
        with _TempProject() as project:
            source = project.workspace / "note.txt"
            source.write_text("Муфта смещена.\n", encoding="utf-8")
            service = ReferenceService(project.root)
            added = service.add_file(source)
            (project.root / added["blob_relative_path"]).unlink()

            with self.assertRaises(ReferenceError):
                service.reprocess(added["material_id"])

    def test_search_rejects_unknown_fragment(self):
        with _TempProject() as project:
            service = ReferenceService(project.root)
            with self.assertRaises(ReferenceError):
                service.show_source("frag_missing")

    def test_extraction_uses_the_quarantined_copy_not_the_original(self):
        with _TempProject() as project:
            source = project.workspace / "note.txt"
            source.write_text("Исходное содержимое AAAA.\n", encoding="utf-8")
            untouched = source.read_bytes()
            original_extract = core.extract_file

            def racing_extract(path, *, original_name=None, **kwargs):
                source.write_text("Подменённое содержимое BBBB.\n", encoding="utf-8")
                return original_extract(path, original_name=original_name, **kwargs)

            core.extract_file = racing_extract
            try:
                service = ReferenceService(project.root)
                added = service.add_file(source)
            finally:
                core.extract_file = original_extract

            blob_path = project.root / added["blob_relative_path"]
            self.assertEqual(blob_path.read_bytes(), untouched)
            self.assertEqual(len(service.search("AAAA")), 1)
            self.assertEqual(service.search("BBBB"), [])
            self.assertTrue(service.integrity_check()["ok"])

    def test_delete_refuses_material_id_that_escapes_the_root(self):
        with _TempProject() as project:
            service = ReferenceService(project.root)
            with self.assertRaises(ReferenceError):
                service.delete_material("../../outside", confirm=True)
            with self.assertRaises(ReferenceError):
                service.reprocess("../../outside")
            outside = project.workspace / "outside"
            outside.mkdir()
            (outside / "keep.txt").write_text("must survive\n", encoding="utf-8")
            with self.assertRaises(ReferenceError):
                service.delete_material("..", confirm=True)
            self.assertTrue((outside / "keep.txt").is_file())


if __name__ == "__main__":
    unittest.main()
