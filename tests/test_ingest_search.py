import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from reference_system.core import ReferenceService
from reference_system.foundation import initialize_project


class IngestSearchTests(unittest.TestCase):
    def test_add_txt_creates_blob_and_searchable_fragment_with_locator(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "personal-reference"
            initialize_project(root)
            source = base / "specification.txt"
            source.write_text(
                "Насос Н-1. Рабочее давление не менее 10 бар.\nТемпература до 95 °C.\n",
                encoding="utf-8",
            )
            service = ReferenceService(root)

            added = service.add_file(source, title="Характеристики насоса")

            self.assertEqual(added["status"], "READY")
            self.assertEqual(added["detected_type"], "txt")
            self.assertEqual(len(added["sha256"]), 64)
            blob_path = root / added["blob_relative_path"]
            self.assertTrue(blob_path.is_file())
            self.assertEqual(blob_path.read_bytes(), source.read_bytes())

            results = service.search('"рабочее давление"')
            self.assertEqual(len(results), 1)
            hit = results[0]
            self.assertEqual(hit["material_id"], added["material_id"])
            self.assertEqual(hit["title"], "Характеристики насоса")
            self.assertEqual(hit["locator"]["type"], "text-lines")
            self.assertEqual(hit["locator"]["line_start"], 1)
            self.assertEqual(hit["original_sha256"], added["sha256"])

            source_view = service.show_source(hit["fragment_id"])
            self.assertEqual(source_view["original_sha256"], added["sha256"])
            self.assertEqual(source_view["locator"], hit["locator"])
            self.assertTrue((root / source_view["blob_relative_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
