import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from reference_system.core import ReferenceService
from reference_system.foundation import initialize_project


class LifecycleTests(unittest.TestCase):
    def test_status_reports_counts_quotas_and_backup_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "personal-reference"
            initialize_project(root)
            source = Path(tmp) / "note.txt"
            source.write_text("Локальный справочник.\n", encoding="utf-8")
            service = ReferenceService(root)
            service.add_file(source)

            status = service.status()

            self.assertEqual(status["materials"], 1)
            self.assertEqual(status["file_blobs"], 1)
            self.assertEqual(status["fragments"], 1)
            self.assertEqual(status["live_quota_gib"], 28)
            self.assertEqual(status["temporary_quota_gib"], 2)
            self.assertEqual(status["backup_quota_gib"], 0)
            self.assertFalse(status["backup_enabled"])
            self.assertGreater(status["used_bytes"], 0)

    def test_delete_requires_confirmation_and_purges_unshared_material(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "personal-reference"
            initialize_project(root)
            source = Path(tmp) / "delete-me.txt"
            source.write_text("Этот материал будет удалён без резервной копии.\n", encoding="utf-8")
            service = ReferenceService(root)
            added = service.add_file(source, title="Удаляемый материал")
            blob_path = root / added["blob_relative_path"]

            with self.assertRaisesRegex(Exception, "explicit confirmation"):
                service.delete_material(added["material_id"], confirm=False)
            self.assertTrue(blob_path.exists())

            report = service.delete_material(added["material_id"], confirm=True)

            self.assertEqual(report["status"], "DELETED")
            self.assertTrue(report["irreversible_without_backup"])
            self.assertFalse(blob_path.exists())
            self.assertEqual(service.search("удалён"), [])
            status = service.status()
            self.assertEqual(status["materials"], 0)
            self.assertEqual(status["file_blobs"], 0)
            self.assertEqual(status["fragments"], 0)

    def test_integrity_check_detects_blob_corruption_without_claiming_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "personal-reference"
            initialize_project(root)
            source = Path(tmp) / "integrity.txt"
            source.write_text("Проверка целостности.\n", encoding="utf-8")
            service = ReferenceService(root)
            added = service.add_file(source)

            healthy = service.integrity_check()
            self.assertTrue(healthy["ok"])
            self.assertEqual(healthy["issues"], [])
            self.assertFalse(healthy["restore_available"])

            blob_path = root / added["blob_relative_path"]
            blob_path.write_bytes(blob_path.read_bytes() + b"corruption")
            damaged = service.integrity_check()

            self.assertFalse(damaged["ok"])
            self.assertFalse(damaged["restore_available"])
            self.assertTrue(any(issue["code"] == "BLOB_HASH_MISMATCH" for issue in damaged["issues"]))

    def test_reprocess_uses_immutable_blob_after_source_is_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "personal-reference"
            initialize_project(root)
            source = Path(tmp) / "reprocess.txt"
            source.write_text("Повторная обработка из неизменяемого оригинала.\n", encoding="utf-8")
            service = ReferenceService(root)
            added = service.add_file(source)
            source.unlink()

            report = service.reprocess(added["material_id"])

            self.assertEqual(report["status"], "READY")
            self.assertEqual(report["material_id"], added["material_id"])
            self.assertNotEqual(report["processing_run_id"], added["processing_run_id"])
            hits = service.search("неизменяемого оригинала")
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0]["material_id"], added["material_id"])


if __name__ == "__main__":
    unittest.main()
