"""Page images: extraction, storage, search and lifecycle (stage 2, fourth increment).

An image is stored as its own object, linked to the page and to that page's
fragment — so the page's text is the image's searchable context. Images are not
materials: 59 photographs are not 59 documents.

Fixtures are built with PyMuPDF, so the embedded bytes are real embedded bytes,
not a stand-in. Page text in the fixtures is Latin because PyMuPDF's built-in
fonts carry no Cyrillic; Cyrillic page text is covered by the text pipeline's
own tests and by the end-to-end run on the real catalogue.
"""

import hashlib
import io
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import fitz
from PIL import Image

from reference_system.core import ReferenceService, ReferenceError
from reference_system.foundation import applied_migration_versions, initialize_project
from reference_system.processors import extract_file, extract_pdf_images


def image_bytes(colour, size=(140, 100), fmt="PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format=fmt)
    return buffer.getvalue()


def build_pdf(path: Path, pages: list[dict]) -> Path:
    """`pages` is a list of {"text": str | None, "images": [bytes, ...]}."""
    document = fitz.open()
    for spec in pages:
        page = document.new_page(width=595, height=842)
        if spec.get("text"):
            page.insert_text((72, 72), spec["text"], fontsize=12)
        top = 160
        for raw in spec.get("images", []):
            page.insert_image(fitz.Rect(72, top, 300, top + 140), stream=raw)
            top += 170
    document.save(path)
    document.close()
    return path


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

    def pdf(self, name, pages) -> Path:
        return build_pdf(self.workspace / name, pages)

    def text(self, name, body) -> Path:
        path = self.workspace / name
        path.write_text(body, encoding="utf-8")
        return path

    def service(self, **kwargs) -> ReferenceService:
        return ReferenceService(self.root, **kwargs)

    def rows(self, sql, params=()):
        with closing(sqlite3.connect(self.root / "db" / "reference.sqlite3")) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute(sql, params)]


class TestPdfImageExtraction(unittest.TestCase):
    def test_every_placed_image_is_found(self):
        with _TempProject() as project:
            pdf = project.pdf(
                "catalogue.pdf",
                [
                    {"text": "Page one about pumps", "images": [image_bytes("red"), image_bytes("green")]},
                    {"text": "Page two about valves", "images": [image_bytes("blue")]},
                ],
            )

            images = extract_pdf_images(pdf)

            self.assertEqual(len(images), 3)
            self.assertEqual([entry["page"] for entry in images], [1, 1, 2])
            self.assertEqual([entry["ordinal"] for entry in images], [1, 2, 1])

    def test_geometry_and_format_are_reported(self):
        with _TempProject() as project:
            pdf = project.pdf("one.pdf", [{"text": "Only page", "images": [image_bytes("red", (200, 120))]}])

            image = extract_pdf_images(pdf)[0]

            self.assertEqual(image["width"], 200)
            self.assertEqual(image["height"], 120)
            self.assertEqual(image["image_format"], "png")
            self.assertEqual(image["byte_size"], len(image["bytes"]))
            self.assertEqual(image["sha256"], hashlib.sha256(image["bytes"]).hexdigest())
            self.assertLess(image["bbox"]["y0"], image["bbox"]["y1"])
            self.assertLess(image["bbox"]["x0"], image["bbox"]["x1"])

    def test_the_same_image_placed_twice_is_reported_twice(self):
        with _TempProject() as project:
            shared = image_bytes("red")
            pdf = project.pdf("twice.pdf", [{"text": "A", "images": [shared]}, {"text": "B", "images": [shared]}])

            images = extract_pdf_images(pdf)

            self.assertEqual(len(images), 2)
            self.assertEqual(images[0]["sha256"], images[1]["sha256"])

    def test_a_pdf_without_images_returns_nothing(self):
        with _TempProject() as project:
            pdf = project.pdf("plain.pdf", [{"text": "No pictures here"}])

            self.assertEqual(extract_pdf_images(pdf), [])

    def test_extract_file_carries_images_alongside_fragments(self):
        with _TempProject() as project:
            pdf = project.pdf("mixed.pdf", [{"text": "Pump page", "images": [image_bytes("red")]}])

            extracted = extract_file(pdf, original_name="mixed.pdf")

            self.assertEqual(len(extracted["fragments"]), 1)
            self.assertEqual(len(extracted["images"]), 1)

    def test_extract_file_can_skip_images(self):
        with _TempProject() as project:
            pdf = project.pdf("skip.pdf", [{"text": "Pump page", "images": [image_bytes("red")]}])

            extracted = extract_file(pdf, original_name="skip.pdf", images=False)

            self.assertEqual(extracted["images"], [])

    def test_a_text_file_has_no_images(self):
        with _TempProject() as project:
            note = project.text("note.txt", "There is no picture here.\n")

            self.assertEqual(extract_file(note)["images"], [])


class TestImageStorage(unittest.TestCase):
    def test_importing_a_pdf_stores_its_images(self):
        with _TempProject() as project:
            service = project.service()
            pdf = project.pdf("cat.pdf", [{"text": "Pumps", "images": [image_bytes("red"), image_bytes("blue")]}])

            report = service.add_file(pdf)

            self.assertEqual(report["page_images"], 2)
            self.assertEqual(report["image_files"], 2)
            self.assertGreater(report["image_bytes"], 0)
            self.assertEqual(len(project.rows("SELECT id FROM page_image")), 2)

    def test_images_can_be_skipped_at_import(self):
        with _TempProject() as project:
            service = project.service()
            pdf = project.pdf("cat.pdf", [{"text": "Pumps", "images": [image_bytes("red")]}])

            report = service.add_file(pdf, images=False)

            self.assertEqual(report["page_images"], 0)
            self.assertEqual(project.rows("SELECT id FROM page_image"), [])

    def test_the_same_image_on_two_pages_is_stored_once(self):
        with _TempProject() as project:
            service = project.service()
            shared = image_bytes("red")
            pdf = project.pdf("twice.pdf", [{"text": "A", "images": [shared]}, {"text": "B", "images": [shared]}])

            report = service.add_file(pdf)

            self.assertEqual(report["page_images"], 2)
            self.assertEqual(report["image_files"], 1)
            rows = project.rows("SELECT sha256, relative_path FROM page_image")
            self.assertEqual({row["relative_path"] for row in rows}, {rows[0]["relative_path"]})

    def test_an_image_is_linked_to_its_page_fragment(self):
        with _TempProject() as project:
            service = project.service()
            pdf = project.pdf("linked.pdf", [{"text": "Pump page text", "images": [image_bytes("red")]}])
            service.add_file(pdf)

            row = project.rows(
                "SELECT i.page, i.fragment_id, f.text AS page_text FROM page_image i "
                "JOIN fragment f ON f.id = i.fragment_id"
            )[0]

            self.assertEqual(row["page"], 1)
            self.assertTrue(row["fragment_id"].startswith("frag_"))
            self.assertIn("Pump page text", row["page_text"])

    def test_an_image_on_a_text_less_page_is_kept_without_a_fragment(self):
        """No text, no fragment — but the photograph is still an object in the base."""
        with _TempProject() as project:
            service = project.service()
            pdf = project.pdf(
                "quiet.pdf",
                [{"text": "Text page", "images": []}, {"text": None, "images": [image_bytes("black")]}],
            )

            report = service.add_file(pdf)

            self.assertEqual(report["page_images"], 1)
            row = project.rows("SELECT page, fragment_id FROM page_image")[0]
            self.assertEqual(row["page"], 2)
            self.assertIsNone(row["fragment_id"])

    def test_stored_bytes_are_the_embedded_bytes(self):
        with _TempProject() as project:
            service = project.service()
            raw = image_bytes("red", (180, 130))
            pdf = project.pdf("bytes.pdf", [{"text": "Page", "images": [raw]}])
            service.add_file(pdf)

            row = project.rows("SELECT sha256, relative_path FROM page_image")[0]
            stored = project.root / row["relative_path"]

            self.assertTrue(stored.is_file())
            self.assertEqual(hashlib.sha256(stored.read_bytes()).hexdigest(), row["sha256"])
            embedded = extract_pdf_images(pdf)[0]
            self.assertEqual(stored.read_bytes(), embedded["bytes"])

    def test_images_live_under_the_hash_store_as_real_files(self):
        with _TempProject() as project:
            service = project.service()
            pdf = project.pdf("path.pdf", [{"text": "Page", "images": [image_bytes("red")]}])
            service.add_file(pdf)

            row = project.rows("SELECT sha256, relative_path FROM page_image")[0]

            self.assertTrue(row["relative_path"].startswith("store/images/sha256/"))
            self.assertIn(row["sha256"][:2], row["relative_path"])
            self.assertFalse((project.root / row["relative_path"]).is_symlink())

    def test_status_counts_images(self):
        with _TempProject() as project:
            service = project.service()
            pdf = project.pdf("counted.pdf", [{"text": "A", "images": [image_bytes("red")]}, {"text": "B", "images": [image_bytes("blue")]}])
            service.add_file(pdf)

            status = service.status()

            self.assertEqual(status["page_images"], 2)
            self.assertEqual(status["image_files"], 2)
            self.assertGreater(status["image_bytes"], 0)


class TestImageSearch(unittest.TestCase):
    def _seeded(self, project, service):
        pdf = project.pdf(
            "catalogue.pdf",
            [
                {"text": "The pump vibrates when it starts", "images": [image_bytes("red")]},
                {"text": "Painting the walls in blue", "images": [image_bytes("blue")]},
                {"text": "Pump fasteners must be checked", "images": [image_bytes("green"), image_bytes("yellow")]},
            ],
        )
        service.add_file(pdf)
        return pdf

    def test_an_image_is_found_through_its_page_text(self):
        with _TempProject() as project:
            service = project.service()
            self._seeded(project, service)

            report = service.search_images("pump vibrates")

            self.assertEqual(report["mode"], "images")
            self.assertEqual(report["match_mode"], "full-text")
            self.assertEqual(report["matched_images"], 1)
            self.assertEqual(report["results"][0]["page"], 1)
            self.assertIn("vibrates", report["results"][0]["text"])

    def test_several_images_on_one_page_are_returned_together(self):
        with _TempProject() as project:
            service = project.service()
            self._seeded(project, service)

            report = service.search_images("fasteners")

            self.assertEqual(report["matched_images"], 2)
            self.assertEqual([hit["page"] for hit in report["results"]], [3, 3])
            self.assertEqual([hit["ordinal"] for hit in report["results"]], [1, 2])

    def test_search_results_carry_geometry_and_the_file(self):
        with _TempProject() as project:
            service = project.service()
            self._seeded(project, service)

            hit = service.search_images("fasteners", limit=1)["results"][0]

            self.assertEqual(hit["width"], 140)
            self.assertEqual(hit["image_format"], "png")
            self.assertTrue(hit["relative_path"].startswith("store/images/"))
            self.assertTrue(Path(hit["absolute_path"]).is_file())
            self.assertEqual(hit["locator"]["page"], 3)
            self.assertEqual(hit["title"], "catalogue.pdf")

    def test_matched_and_returned_are_reported_separately(self):
        with _TempProject() as project:
            service = project.service()
            self._seeded(project, service)

            report = service.search_images("pump", limit=1)

            self.assertEqual(report["matched_images"], 3)
            self.assertEqual(report["returned"], 1)
            self.assertEqual(len(report["results"]), 1)

    def test_exact_mode_matches_a_literal_substring(self):
        with _TempProject() as project:
            service = project.service()
            self._seeded(project, service)

            report = service.search_images("walls in blue", exact=True)

            self.assertEqual(report["match_mode"], "exact")
            self.assertEqual(report["matched_images"], 1)
            self.assertEqual(report["results"][0]["page"], 2)

    def test_exact_mode_is_case_insensitive(self):
        with _TempProject() as project:
            service = project.service()
            self._seeded(project, service)

            self.assertEqual(service.search_images("WALLS IN BLUE", exact=True)["matched_images"], 1)

    def test_no_matches_returns_an_empty_list_not_an_error(self):
        with _TempProject() as project:
            service = project.service()
            self._seeded(project, service)

            report = service.search_images("hydraulic accumulator")

            self.assertEqual(report["results"], [])
            self.assertEqual(report["matched_images"], 0)

    def test_fts_metacharacters_do_not_break_the_query(self):
        with _TempProject() as project:
            service = project.service()
            self._seeded(project, service)

            for query in ("-", '"', "(", "P-101", "AND", "*", "---"):
                with self.subTest(query=query):
                    self.assertIsInstance(service.search_images(query)["results"], list)

    def test_an_empty_query_is_refused(self):
        with _TempProject() as project:
            service = project.service()
            self._seeded(project, service)

            with self.assertRaises(ReferenceError):
                service.search_images("   ")

    def test_search_skips_images_of_deleted_materials(self):
        with _TempProject() as project:
            service = project.service()
            self._seeded(project, service)
            material_id = service.search_images("pump vibrates")["results"][0]["material_id"]

            service.delete_material(material_id, confirm=True)

            self.assertEqual(service.search_images("pump vibrates")["matched_images"], 0)

    def test_show_image_returns_a_verified_file(self):
        with _TempProject() as project:
            service = project.service()
            self._seeded(project, service)
            image_id = service.search_images("fasteners", limit=1)["results"][0]["image_id"]

            report = service.show_image(image_id)

            self.assertEqual(report["image_id"], image_id)
            self.assertEqual(report["page"], 3)
            self.assertTrue(report["file_present"])
            self.assertTrue(report["sha256_verified"])
            self.assertTrue(Path(report["absolute_path"]).is_file())

    def test_show_image_rejects_an_unknown_id(self):
        with _TempProject() as project:
            service = project.service()

            with self.assertRaises(ReferenceError):
                service.show_image("img_does_not_exist")

    def test_show_image_rejects_a_malformed_id(self):
        with _TempProject() as project:
            service = project.service()

            with self.assertRaises(ReferenceError):
                service.show_image("../../etc/passwd")


class TestImageLifecycle(unittest.TestCase):
    def test_deleting_a_material_removes_its_images_and_files(self):
        with _TempProject() as project:
            service = project.service()
            pdf = project.pdf("gone.pdf", [{"text": "Pump page", "images": [image_bytes("red"), image_bytes("blue")]}])
            service.add_file(pdf)
            paths = [project.root / row["relative_path"] for row in project.rows("SELECT relative_path FROM page_image")]
            material_id = project.rows("SELECT id FROM material")[0]["id"]

            report = service.delete_material(material_id, confirm=True)

            self.assertEqual(report["deleted_images"], 2)
            self.assertEqual(report["deleted_image_files"], 2)
            self.assertEqual(project.rows("SELECT id FROM page_image"), [])
            self.assertTrue(all(not path.exists() for path in paths))

    def test_a_shared_image_file_survives_deleting_one_material(self):
        with _TempProject() as project:
            service = project.service()
            shared = image_bytes("red")
            first = project.pdf("first.pdf", [{"text": "First", "images": [shared]}])
            second = project.pdf("second.pdf", [{"text": "Second", "images": [shared]}])
            service.add_file(first)
            service.add_file(second)
            rows = project.rows("SELECT relative_path FROM page_image")
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["relative_path"], rows[1]["relative_path"])
            path = project.root / rows[0]["relative_path"]
            material_id = service.search("First")[0]["material_id"]

            report = service.delete_material(material_id, confirm=True)

            self.assertEqual(report["deleted_images"], 1)
            self.assertEqual(report["deleted_image_files"], 0)
            self.assertTrue(path.is_file())
            self.assertEqual(len(project.rows("SELECT id FROM page_image")), 1)

    def test_reprocess_rebuilds_images_without_duplicating_them(self):
        with _TempProject() as project:
            service = project.service()
            pdf = project.pdf("redo.pdf", [{"text": "Pump page", "images": [image_bytes("red")]}])
            service.add_file(pdf)
            material_id = project.rows("SELECT id FROM material")[0]["id"]

            report = service.reprocess(material_id)

            self.assertEqual(report["page_images"], 1)
            self.assertEqual(len(project.rows("SELECT id FROM page_image")), 1)
            self.assertEqual(len(project.rows("SELECT id FROM file_blob")), 1)

    def test_add_images_fills_in_an_existing_material_without_touching_text_or_vectors(self):
        from reference_system.embeddings import StubEmbeddingModel

        model = StubEmbeddingModel(model_id="keyword-stub", model_version="kw-1", dimensions=4)
        with _TempProject() as project:
            service = project.service(embedding_model=model)
            pdf = project.pdf("later.pdf", [{"text": "Pump page", "images": [image_bytes("red"), image_bytes("blue")]}])
            service.add_file(pdf, images=False)
            service.embed_fragments()
            fragments_before = project.rows("SELECT id, text FROM fragment")

            report = service.add_images(project.rows("SELECT id FROM material")[0]["id"])

            self.assertEqual(report["page_images"], 2)
            self.assertEqual(report["new_files"], 2)
            self.assertEqual(project.rows("SELECT id, text FROM fragment"), fragments_before)
            self.assertEqual(service.status()["embeddings"], len(fragments_before))

    def test_add_images_is_idempotent(self):
        with _TempProject() as project:
            service = project.service()
            pdf = project.pdf("twice.pdf", [{"text": "Pump page", "images": [image_bytes("red")]}])
            service.add_file(pdf, images=False)
            material_id = project.rows("SELECT id FROM material")[0]["id"]

            first = service.add_images(material_id)
            second = service.add_images(material_id)

            self.assertEqual(first["page_images"], 1)
            self.assertEqual(second["page_images"], 0)
            self.assertEqual(second["existing"], 1)
            self.assertEqual(len(project.rows("SELECT id FROM page_image")), 1)

    def test_add_images_rejects_an_unknown_material(self):
        with _TempProject() as project:
            service = project.service()

            with self.assertRaises(ReferenceError):
                service.add_images("mat_missing")


class TestImageIntegrity(unittest.TestCase):
    def test_integrity_is_clean_after_importing_images(self):
        with _TempProject() as project:
            service = project.service()
            pdf = project.pdf("clean.pdf", [{"text": "Pump page", "images": [image_bytes("red")]}])
            service.add_file(pdf)

            report = service.integrity_check()

            self.assertTrue(report["ok"], report["issues"])
            self.assertEqual(report["issues"], [])

    def test_a_missing_image_file_is_reported(self):
        with _TempProject() as project:
            service = project.service()
            pdf = project.pdf("lost.pdf", [{"text": "Pump page", "images": [image_bytes("red")]}])
            service.add_file(pdf)
            path = project.root / project.rows("SELECT relative_path FROM page_image")[0]["relative_path"]
            path.unlink()

            report = service.integrity_check()

            self.assertFalse(report["ok"])
            self.assertIn("IMAGE_MISSING", {issue["code"] for issue in report["issues"]})

    def test_a_corrupted_image_file_is_reported(self):
        with _TempProject() as project:
            service = project.service()
            pdf = project.pdf("broken.pdf", [{"text": "Pump page", "images": [image_bytes("red")]}])
            service.add_file(pdf)
            path = project.root / project.rows("SELECT relative_path FROM page_image")[0]["relative_path"]
            path.write_bytes(b"not an image any more")

            report = service.integrity_check()

            self.assertFalse(report["ok"])
            self.assertIn("IMAGE_HASH_MISMATCH", {issue["code"] for issue in report["issues"]})

    def test_an_orphan_image_row_is_reported(self):
        with _TempProject() as project:
            service = project.service()
            pdf = project.pdf("orphan.pdf", [{"text": "Pump page", "images": [image_bytes("red")]}])
            service.add_file(pdf)
            with closing(sqlite3.connect(project.root / "db" / "reference.sqlite3")) as conn:
                conn.execute("UPDATE page_image SET fragment_id = 'frag_0000000000000000000000000000dead'")
                conn.commit()

            report = service.integrity_check()

            self.assertFalse(report["ok"])
            self.assertIn("IMAGE_ORPHAN", {issue["code"] for issue in report["issues"]})


class TestOutdatedSchemaForImages(unittest.TestCase):
    def test_status_reports_an_outdated_schema(self):
        with _TempProject() as project:
            with closing(sqlite3.connect(project.root / "db" / "reference.sqlite3")) as conn:
                conn.execute("DROP TABLE page_image")
                conn.commit()

            status = project.service().status()

            self.assertFalse(status["schema_current"])
            self.assertIsNone(status["page_images"])

    def test_search_images_names_the_migration(self):
        with _TempProject() as project:
            service = project.service()
            pdf = project.pdf("old.pdf", [{"text": "Pump page", "images": [image_bytes("red")]}])
            service.add_file(pdf)
            with closing(sqlite3.connect(project.root / "db" / "reference.sqlite3")) as conn:
                conn.execute("DROP TABLE page_image")
                conn.commit()

            with self.assertRaises(ReferenceError) as caught:
                service.search_images("pump")

            self.assertIn("0004", str(caught.exception))

    def test_migration_versions_include_the_images_migration(self):
        with _TempProject() as project:
            self.assertEqual(
                applied_migration_versions(project.root / "db" / "reference.sqlite3"), [1, 2, 3, 4]
            )


if __name__ == "__main__":
    unittest.main()
