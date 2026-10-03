"""OCR pipeline (stage 2, first increment).

The engine is an injected interface, so the whole pipeline — rasterising PDF
pages with PyMuPDF, OCR cache keyed by blob + engine + language, locators with
boxes, search integration, integrity checking — is built and verified here with
a test double. No engine binary, no installation, no network, no subprocess.

Stage-1 behaviour must survive untouched: with no engine injected, images stay
metadata-only and text-less PDFs are still refused.
"""

import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from reference_system.core import ReferenceService
from reference_system.foundation import applied_migration_versions, initialize_project
from reference_system.ocr import OcrResult, StubOcrEngine


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

    def image(self, name: str, size=(32, 18)) -> Path:
        from PIL import Image

        path = self.workspace / name
        Image.new("RGB", size, color=(240, 240, 235)).save(path)
        return path

    def scanned_pdf(self, name: str, pages: int = 1) -> Path:
        import fitz

        path = self.workspace / name
        document = fitz.open()
        for _ in range(pages):
            document.new_page()
        document.save(path)
        document.close()
        return path

    def service(self, engine=None, language="ru") -> ReferenceService:
        return ReferenceService(self.root, ocr_engine=engine, ocr_language=language)


def _engine(text: str, *, language: str = "ru") -> StubOcrEngine:
    return StubOcrEngine(
        engine_id="stub-test-double",
        engine_version="0.1",
        languages=("ru", "en-US"),
        respond=lambda image_bytes, lang: OcrResult(
            text=text,
            lines=[{"text": text, "bbox": [1.0, 2.0, 30.0, 12.0]}],
            engine_id="stub-test-double",
            engine_version="0.1",
            language=lang,
        ),
    )


class NoEngineRegressionTests(unittest.TestCase):
    def test_image_without_engine_stays_metadata_only(self):
        with _TempProject() as project:
            added = project.service().add_file(project.image("scan.png"))
            self.assertEqual(added["detected_type"], "png")
            hits = project.service().search("Дефект", exact=True)
            self.assertEqual(hits, [])

    def test_textless_pdf_without_engine_is_still_refused(self):
        from reference_system.processors import UnsupportedFormatError

        with _TempProject() as project:
            with self.assertRaises(UnsupportedFormatError) as ctx:
                project.service().add_file(project.scanned_pdf("blank.pdf"))
            self.assertIn("OCR", str(ctx.exception))


class MigrationTests(unittest.TestCase):
    def test_ocr_cache_migration_is_applied(self):
        with _TempProject() as project:
            versions = applied_migration_versions(project.root / "db" / "reference.sqlite3")
            self.assertEqual(versions, [1, 2, 3, 4])

            with closing(sqlite3.connect(project.root / "db" / "reference.sqlite3")) as conn:
                tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
            self.assertIn("ocr_cache", tables)

    def test_migrations_remain_idempotent_with_ocr(self):
        from reference_system.foundation import apply_migrations, ensure_migration_files

        with _TempProject() as project:
            migrations_dir = ensure_migration_files(project.root)
            self.assertEqual(
                apply_migrations(project.root / "db" / "reference.sqlite3", migrations_dir), []
            )


class ImageOcrTests(unittest.TestCase):
    def test_image_text_becomes_searchable_with_ocr_locator(self):
        with _TempProject() as project:
            engine = _engine("Дефект подшипника обнаружен при осмотре")
            service = project.service(engine)
            added = service.add_file(project.image("scan.png"), title="Скан акта")

            hits = service.search("подшипника")
            self.assertEqual(len(hits), 1)
            hit = hits[0]
            self.assertEqual(hit["locator"]["type"], "image-ocr")
            self.assertTrue(hit["locator"]["ocr_performed"])
            self.assertEqual(hit["locator"]["engine"], "stub-test-double")
            self.assertEqual(hit["locator"]["engine_version"], "0.1")
            self.assertEqual(hit["locator"]["language"], "ru")
            self.assertEqual(hit["locator"]["width"], 32)
            self.assertEqual(hit["locator"]["height"], 18)
            self.assertEqual(len(hit["locator"]["lines"]), 1)
            self.assertEqual(added["detected_type"], "png")

            source_ref = service.show_source(hit["fragment_id"])
            self.assertEqual(source_ref["original_sha256"], added["sha256"])

    def test_exact_search_finds_ocr_text(self):
        with _TempProject() as project:
            service = project.service(_engine("Насос P-101 вибрирует"))
            service.add_file(project.image("scan.png"))
            self.assertEqual(len(service.search("P-101", exact=True)), 1)

    def test_ocr_text_is_cached_per_blob_and_engine_runs_once(self):
        with _TempProject() as project:
            calls = {"n": 0}
            base = _engine("Задвижка DN100 закрыта")

            def respond(image_bytes, lang):
                calls["n"] += 1
                return base.respond(image_bytes, lang)

            engine = StubOcrEngine(
                engine_id="stub-test-double",
                engine_version="0.1",
                languages=("ru",),
                respond=respond,
            )
            service = project.service(engine)

            first = project.image("a.png")
            second = project.workspace / "b.png"
            second.write_bytes(first.read_bytes())

            added_first = service.add_file(first)
            added_second = service.add_file(second)

            self.assertEqual(calls["n"], 1)
            self.assertEqual(added_first["sha256"], added_second["sha256"])
            self.assertEqual(len(service.search("DN100", exact=True)), 2)

            with closing(sqlite3.connect(project.root / "db" / "reference.sqlite3")) as conn:
                rows = conn.execute("SELECT count(*) FROM ocr_cache").fetchone()[0]
            self.assertEqual(rows, 1)

    def test_reprocess_reuses_cached_ocr_without_rebuilding(self):
        with _TempProject() as project:
            calls = {"n": 0}
            base = _engine("Клапан открыт частично")

            def respond(image_bytes, lang):
                calls["n"] += 1
                return base.respond(image_bytes, lang)

            engine = StubOcrEngine(
                engine_id="stub-test-double",
                engine_version="0.1",
                languages=("ru",),
                respond=respond,
            )
            service = project.service(engine)
            added = service.add_file(project.image("scan.png"))
            self.assertEqual(calls["n"], 1)

            service.reprocess(added["material_id"])
            self.assertEqual(calls["n"], 1)
            self.assertEqual(len(service.search("Клапан")), 1)

    def test_engine_language_is_passed_through(self):
        with _TempProject() as project:
            seen = []

            def respond(image_bytes, lang):
                seen.append(lang)
                return OcrResult(text="text", lines=[], engine_id="stub-test-double",
                                 engine_version="0.1", language=lang)

            engine = StubOcrEngine(
                engine_id="stub-test-double",
                engine_version="0.1",
                languages=("ru", "en-US"),
                respond=respond,
            )
            project.service(engine, language="en-US").add_file(project.image("scan.png"))
            self.assertEqual(seen, ["en-US"])

    def test_unsupported_language_keeps_metadata_only(self):
        with _TempProject() as project:
            service = project.service(_engine("Дефект"), language="de-DE")
            service.add_file(project.image("scan.png"))
            self.assertEqual(service.search("Дефект", exact=True), [])

    def test_empty_ocr_result_does_not_create_an_empty_fragment(self):
        with _TempProject() as project:
            service = project.service(_engine("   "))
            service.add_file(project.image("scan.png"))
            with closing(sqlite3.connect(project.root / "db" / "reference.sqlite3")) as conn:
                self.assertEqual(conn.execute("SELECT count(*) FROM fragment").fetchone()[0], 0)


class ScannedPdfOcrTests(unittest.TestCase):
    def test_textless_pdf_pages_are_ocrd_when_an_engine_is_present(self):
        with _TempProject() as project:
            service = project.service(_engine("Скан документа: акт осмотра насоса"))
            added = service.add_file(project.scanned_pdf("scan.pdf", pages=2))

            self.assertEqual(added["detected_type"], "pdf")
            hits = service.search("осмотра")
            self.assertEqual(len(hits), 2)
            for hit in hits:
                self.assertEqual(hit["locator"]["type"], "pdf-ocr-page")
                self.assertTrue(hit["locator"]["ocr_performed"])
            self.assertEqual(sorted(hit["locator"]["page"] for hit in hits), [1, 2])

    def test_pdf_with_text_layer_is_not_ocrd(self):
        import fitz

        with _TempProject() as project:
            calls = {"n": 0}
            base = _engine("ОБРАЗЕЦ")

            def respond(image_bytes, lang):
                calls["n"] += 1
                return base.respond(image_bytes, lang)

            engine = StubOcrEngine(
                engine_id="stub-test-double",
                engine_version="0.1",
                languages=("ru",),
                respond=respond,
            )
            source = project.workspace / "manual.pdf"
            document = fitz.open()
            page = document.new_page()
            page.insert_text((72, 72), "Pump P-101 inspection record")
            document.save(source)
            document.close()

            service = project.service(engine)
            service.add_file(source)

            self.assertEqual(calls["n"], 0)
            hit = service.search("inspection")[0]
            self.assertEqual(hit["locator"]["type"], "pdf-page")
            self.assertFalse(hit["locator"].get("ocr_performed", False))


class OcrIntegrityTests(unittest.TestCase):
    def test_integrity_check_covers_ocr_cache_files(self):
        with _TempProject() as project:
            service = project.service(_engine("Дефект"))
            service.add_file(project.image("scan.png"))
            self.assertTrue(service.integrity_check()["ok"])

    def test_integrity_check_detects_tampered_ocr_cache(self):
        with _TempProject() as project:
            service = project.service(_engine("Дефект подшипника"))
            added = service.add_file(project.image("scan.png"))

            with closing(sqlite3.connect(project.root / "db" / "reference.sqlite3")) as conn:
                relative = conn.execute("SELECT relative_path FROM ocr_cache").fetchone()[0]
            (project.root / relative).write_text("подменённый текст", encoding="utf-8")

            report = service.integrity_check()
            self.assertFalse(report["ok"])
            self.assertIn("OCR_CACHE_HASH_MISMATCH", [issue["code"] for issue in report["issues"]])
            self.assertEqual(added["detected_type"], "png")

    def test_integrity_check_reports_missing_ocr_cache_file(self):
        with _TempProject() as project:
            service = project.service(_engine("Дефект"))
            service.add_file(project.image("scan.png"))

            with closing(sqlite3.connect(project.root / "db" / "reference.sqlite3")) as conn:
                relative = conn.execute("SELECT relative_path FROM ocr_cache").fetchone()[0]
            (project.root / relative).unlink()

            report = service.integrity_check()
            self.assertFalse(report["ok"])
            self.assertIn("OCR_CACHE_MISSING", [issue["code"] for issue in report["issues"]])

    def test_deleting_a_material_drops_its_ocr_cache_entries(self):
        with _TempProject() as project:
            service = project.service(_engine("Дефект"))
            added = service.add_file(project.image("scan.png"))
            service.delete_material(added["material_id"], confirm=True)

            with closing(sqlite3.connect(project.root / "db" / "reference.sqlite3")) as conn:
                self.assertEqual(conn.execute("SELECT count(*) FROM ocr_cache").fetchone()[0], 0)
            self.assertTrue(service.integrity_check()["ok"])


if __name__ == "__main__":
    unittest.main()
