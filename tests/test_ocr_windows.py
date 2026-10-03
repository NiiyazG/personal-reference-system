"""Real Windows OCR adapter (stage 2, second increment).

These tests exercise the *real* engine on synthetic fixtures — no mocks. They
skip cleanly when `winsdk` or a recogniser is absent, so the suite still passes
on a host without Windows OCR.

What is asserted here is what the engine actually does, including the parts that
are not flattering: prose is recognised exactly, but a Latin code like `P-101`
comes back with Cyrillic lookalikes. That limitation is pinned by a test so it
cannot be forgotten, and so a future engine that fixes it makes this test fail
and forces the docs and search strategy to be revisited.
"""

import io
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from reference_system.core import ReferenceService
from reference_system.foundation import initialize_project
from reference_system.ocr_windows import (
    WindowsOcrEngine,
    WindowsOcrError,
    WindowsOcrUnavailableError,
)
from reference_system.processors import UnsupportedFormatError

FONT_PATH = r"C:/Windows/Fonts/segoeui.ttf"
PROSE_LINE = "вибрирует при открытии"
CODE_LINE = "P-101"


def _engine_or_none() -> WindowsOcrEngine | None:
    try:
        return WindowsOcrEngine()
    except (WindowsOcrUnavailableError, OSError):
        return None


ENGINE = _engine_or_none()
requires_engine = unittest.skipUnless(ENGINE is not None, "Windows OCR is not available on this host")


def render_text_png(
    text: str, *, font_size: int = 34, width: int = 900, scale: int = 1
) -> bytes:
    """A synthetic, fully controlled image: known text, known layout."""
    from PIL import Image, ImageDraw, ImageFont

    lines = text.splitlines()
    font = ImageFont.truetype(FONT_PATH, font_size)
    height = 40 + len(lines) * int(font_size * 1.6)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(lines):
        draw.text((30, 20 + index * int(font_size * 1.6)), line, fill="black", font=font)
    if scale != 1:
        image = image.resize((image.width * scale, image.height * scale), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


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

    def text_image(self, name: str, text: str = PROSE_LINE, **kwargs) -> Path:
        path = self.workspace / name
        path.write_bytes(render_text_png(text, **kwargs))
        return path

    def scanned_pdf(self, name: str, text: str = PROSE_LINE, pages: int = 1) -> Path:
        """A page that is only a raster image, so it has no text layer at all."""
        import fitz

        png = render_text_png(text)
        path = self.workspace / name
        document = fitz.open()
        for _ in range(pages):
            page = document.new_page(width=900, height=200)
            page.insert_image(fitz.Rect(0, 0, 900, 200), stream=png)
        document.save(path)
        document.close()
        return path

    def service(self, engine=None, language="ru") -> ReferenceService:
        return ReferenceService(self.root, ocr_engine=engine, ocr_language=language)


@requires_engine
class TestWindowsOcrEngine(unittest.TestCase):
    def test_engine_reports_its_own_identity(self):
        self.assertEqual(ENGINE.engine_id, "windows-media-ocr")
        self.assertIn("winsdk-", ENGINE.engine_version)
        self.assertIn("windows-", ENGINE.engine_version)
        self.assertIn("ru", ENGINE.languages)
        self.assertFalse(ENGINE.is_test_double)

    def test_supports_real_languages_and_rejects_impossible_tags(self):
        self.assertTrue(ENGINE.supports("ru"))
        self.assertFalse(ENGINE.supports(""))
        self.assertFalse(ENGINE.supports("definitely not a language tag"))

    def test_recognises_russian_prose_exactly(self):
        result = ENGINE.recognize(render_text_png("Щит управления\n" + PROSE_LINE), language="ru")

        self.assertIn(PROSE_LINE, result.text)
        self.assertEqual(result.engine_id, "windows-media-ocr")
        self.assertEqual(result.language, "ru")
        self.assertTrue(result.lines)
        for line in result.lines:
            self.assertIn("text", line)
            self.assertLess(line["left"], line["right"])
            self.assertLess(line["top"], line["bottom"])

    def test_words_are_joined_with_separators(self):
        result = ENGINE.recognize(render_text_png(PROSE_LINE), language="ru")

        # Words arrive as separate items; glueing them without a space would
        # produce "вибрируетприоткрытии" and make the text unsearchable.
        self.assertNotIn("вибрируетприоткрытии", result.text)

    def test_language_is_recorded_in_canonical_form(self):
        result = ENGINE.recognize(render_text_png(PROSE_LINE), language="ru-RU")

        self.assertEqual(result.language, "ru")

    def test_latin_lookalike_codes_are_a_known_limitation(self):
        """The ru recogniser returns Cyrillic lookalikes for Latin codes.

        Documented, not hidden: the stored text is what the engine produced, so a
        search for the Latin `P-101` will not match. If this ever starts passing,
        the engine improved and the workaround note in README must be revisited.
        """
        result = ENGINE.recognize(render_text_png(f"Код {CODE_LINE}"), language="ru")

        self.assertIn("Код", result.text)
        self.assertNotIn(CODE_LINE, result.text)
        self.assertTrue(
            any(character in result.text for character in "РОЗСАВЕН"),
            f"expected a Cyrillic lookalike in {result.text!r}",
        )

    def test_empty_bytes_are_refused(self):
        with self.assertRaises(WindowsOcrError):
            ENGINE.recognize(b"", language="ru")

    def test_undecodable_bytes_are_refused_loudly(self):
        with self.assertRaises(WindowsOcrError) as caught:
            ENGINE.recognize(b"this is definitely not an image", language="ru")

        self.assertIn("cannot decode", str(caught.exception))

    def test_unavailable_language_is_refused(self):
        with self.assertRaises(WindowsOcrError):
            ENGINE.recognize(render_text_png(PROSE_LINE), language="ja-JP")

    def test_oversized_image_is_downscaled_and_says_so(self):
        png = render_text_png("Щит управления", width=ENGINE.max_image_dimension + 50)

        result = ENGINE.recognize(png, language="ru")

        self.assertTrue(result.notes, "a downscaled image must be reported, not hidden")
        self.assertIn("downscaled", result.notes[0])
        self.assertIn(str(ENGINE.max_image_dimension), result.notes[0])

    def test_image_within_the_limit_reports_no_notes(self):
        result = ENGINE.recognize(render_text_png(PROSE_LINE), language="ru")

        self.assertEqual(result.notes, [])


@requires_engine
class TestWindowsOcrThroughTheService(unittest.TestCase):
    def test_image_is_recognised_indexed_and_searchable(self):
        with _TempProject() as project:
            source = project.text_image("shield.png", "Щит управления\n" + PROSE_LINE)
            service = project.service(engine=ENGINE)

            added = service.add_file(source)
            self.assertEqual(added["status"], "READY")
            self.assertEqual(added["detected_type"], "png")

            hits = service.search("вибрирует")
            self.assertEqual(len(hits), 1)
            locator = hits[0]["locator"]
            self.assertEqual(locator["type"], "image-ocr")
            self.assertTrue(locator["ocr_performed"])
            self.assertEqual(locator["engine"], "windows-media-ocr")
            self.assertTrue(locator["lines"])
            self.assertEqual(locator["ocr_notes"], [])

            source_view = service.show_source(hits[0]["fragment_id"])
            self.assertEqual(source_view["locator"]["type"], "image-ocr")
            self.assertEqual(source_view["original_sha256"], added["sha256"])

    def test_page_without_a_text_layer_is_recognised(self):
        with _TempProject() as project:
            source = project.scanned_pdf("scan.pdf", PROSE_LINE)
            service = project.service(engine=ENGINE)

            service.add_file(source)

            hits = service.search("вибрирует")
            self.assertEqual(len(hits), 1)
            locator = hits[0]["locator"]
            self.assertEqual(locator["type"], "pdf-ocr-page")
            self.assertEqual(locator["page"], 1)
            self.assertTrue(locator["ocr_performed"])

    def test_the_same_page_without_an_engine_is_still_refused(self):
        with _TempProject() as project:
            source = project.scanned_pdf("scan.pdf", PROSE_LINE)
            service = project.service(engine=None)

            with self.assertRaises(UnsupportedFormatError) as caught:
                service.add_file(source)

            self.assertIn("OCR", str(caught.exception))
            self.assertEqual(service.search("вибрирует"), [])

    def test_recognition_is_cached_across_reprocess(self):
        with _TempProject() as project:
            source = project.text_image("shield.png", PROSE_LINE)
            service = project.service(engine=ENGINE)
            added = service.add_file(source)

            first = service.status()["ocr_cache_entries"]
            self.assertEqual(first, 1)

            service.reprocess(added["material_id"])

            self.assertEqual(service.status()["ocr_cache_entries"], 1)
            self.assertEqual(len(service.search("вибрирует")), 1)

    def test_status_reports_the_active_engine(self):
        with _TempProject() as project:
            service = project.service(engine=ENGINE)

            status = service.status()

            self.assertEqual(status["ocr_engine"], "windows-media-ocr")
            self.assertEqual(status["ocr_language"], "ru")
            self.assertEqual(status["ocr_cache_entries"], 0)

    def test_status_reports_no_engine_by_default(self):
        with _TempProject() as project:
            service = project.service(engine=None)

            status = service.status()

            self.assertIsNone(status["ocr_engine"])
            self.assertEqual(status["ocr_cache_entries"], 0)

    def test_deleting_an_ocrd_material_counts_blobs_and_cache_separately(self):
        """The deletion report must not inflate one counter with the other's files."""
        with _TempProject() as project:
            service = project.service(engine=ENGINE)
            added = service.add_file(project.text_image("shield.png", PROSE_LINE))
            self.assertEqual(service.status()["ocr_cache_entries"], 1)

            report = service.delete_material(added["material_id"], confirm=True)

            self.assertEqual(report["deleted_fragments"], 1)
            self.assertEqual(report["deleted_blobs"], 1)
            self.assertEqual(report["deleted_ocr_cache_entries"], 1)
            self.assertTrue(report["irreversible_without_backup"])
            self.assertEqual(service.status()["ocr_cache_entries"], 0)
            self.assertEqual(list((project.root / "indexes" / "ocr").rglob("*.json")), [])

    def test_integrity_check_covers_the_real_ocr_cache(self):
        with _TempProject() as project:
            service = project.service(engine=ENGINE)
            service.add_file(project.text_image("shield.png", PROSE_LINE))

            report = service.integrity_check()

            self.assertTrue(report["ok"], report["issues"])
            self.assertEqual(report["issues"], [])

            # now damage the cache file and confirm the report is not blind
            cache_files = list((project.root / "indexes" / "ocr").rglob("*.json"))
            self.assertEqual(len(cache_files), 1)
            cache_files[0].write_text("{}", encoding="utf-8")

            damaged = service.integrity_check()

            self.assertFalse(damaged["ok"])
            codes = {issue["code"] for issue in damaged["issues"]}
            self.assertIn("OCR_CACHE_HASH_MISMATCH", codes)


if __name__ == "__main__":
    unittest.main()
