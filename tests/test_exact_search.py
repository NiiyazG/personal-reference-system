"""Stage 1 requires BOTH an exact search and a full-text search.

Exact search must match a literal, contiguous, case-insensitive substring of the
fragment text, must not treat LIKE wildcards as wildcards, and must not match
non-contiguous word sets the way FTS5 does.
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
from reference_system.foundation import initialize_project


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

    def add(self, name: str, text: str, *, title: str | None = None) -> dict:
        path = self.workspace / name
        path.write_text(text, encoding="utf-8")
        return ReferenceService(self.root).add_file(path, title=title)


class ExactSearchTests(unittest.TestCase):
    def test_exact_search_finds_literal_substring_that_fts_splits(self):
        with _TempProject() as project:
            project.add("note.txt", "Насос P-101 вибрирует при пуске.\n")
            service = ReferenceService(project.root)

            exact = service.search("P-101", exact=True)
            self.assertEqual(len(exact), 1)
            self.assertEqual(exact[0]["match_mode"], "exact")
            self.assertEqual(exact[0]["match_start"], 6)

            source_ref = service.show_source(exact[0]["fragment_id"])
            self.assertEqual(source_ref["locator"]["type"], "text-lines")

    def test_exact_search_does_not_match_non_contiguous_words(self):
        with _TempProject() as project:
            project.add("note.txt", "насос сильно вибрирует при пуске\n")
            service = ReferenceService(project.root)

            # FTS5 treats these as separate tokens and matches; exact must not.
            self.assertTrue(service.search("насос вибрирует"))
            self.assertEqual(service.search("насос вибрирует", exact=True), [])

    def test_exact_search_is_case_insensitive_for_cyrillic(self):
        with _TempProject() as project:
            project.add("note.txt", "Насос вибрирует при пуске.\n")
            service = ReferenceService(project.root)
            self.assertEqual(len(service.search("ВИБРИРУЕТ", exact=True)), 1)
            self.assertEqual(len(service.search("вибрирует", exact=True)), 1)

    def test_exact_search_does_not_use_like_wildcards(self):
        with _TempProject() as project:
            project.add("note.txt", "Гарантия 100% по договору.\n")
            project.add("other.txt", "Обычная строка без знаков препинания.\n")
            service = ReferenceService(project.root)

            # A wildcard would make these match every fragment; literal search
            # must match only the fragment that really contains the character.
            self.assertEqual(len(service.search("100%", exact=True)), 1)
            percent = service.search("%", exact=True)
            self.assertEqual(len(percent), 1)
            self.assertIn("100%", percent[0]["text"])
            self.assertEqual(service.search("_", exact=True), [])

    def test_exact_search_finds_partial_word(self):
        with _TempProject() as project:
            project.add("note.txt", "Наблюдается вибрация подшипника.\n")
            service = ReferenceService(project.root)
            self.assertEqual(len(service.search("вибрац", exact=True)), 1)

    def test_exact_search_matches_within_a_word(self):
        with _TempProject() as project:
            project.add("note.txt", "Арматура задвижка DN100 установлена.\n")
            service = ReferenceService(project.root)
            exact = service.search("движ", exact=True)
            self.assertEqual(len(exact), 1)
            self.assertIn("движ", exact[0]["text"])

    def test_exact_search_ignores_deleted_materials(self):
        with _TempProject() as project:
            added = project.add("note.txt", "Насос P-101 вибрирует.\n")
            service = ReferenceService(project.root)
            self.assertEqual(len(service.search("P-101", exact=True)), 1)

            service.delete_material(added["material_id"], confirm=True)
            self.assertEqual(service.search("P-101", exact=True), [])

    def test_exact_search_respects_limit(self):
        with _TempProject() as project:
            for index in range(3):
                project.add(f"note-{index}.txt", "Общая строка ABCDEF для проверки.\n")
            service = ReferenceService(project.root)

            self.assertEqual(len(service.search("ABCDEF", exact=True)), 3)
            self.assertEqual(len(service.search("ABCDEF", exact=True, limit=2)), 2)

    def test_exact_search_returns_empty_for_blank_query(self):
        with _TempProject() as project:
            project.add("note.txt", "Насос вибрирует.\n")
            service = ReferenceService(project.root)
            self.assertEqual(service.search("", exact=True), [])
            self.assertEqual(service.search("   ", exact=True), [])

    def test_exact_search_result_shape_matches_full_text_result_shape(self):
        with _TempProject() as project:
            project.add("note.txt", "Насос P-101 вибрирует при пуске.\n", title="Акт осмотра")
            service = ReferenceService(project.root)

            fts_keys = set(service.search("вибрирует")[0])
            exact_keys = set(service.search("вибрирует", exact=True)[0])
            self.assertTrue(fts_keys - {"rank"} <= exact_keys)
            self.assertIn("material_id", exact_keys)
            self.assertIn("original_sha256", exact_keys)
            self.assertEqual(service.search("вибрирует", exact=True)[0]["title"], "Акт осмотра")

    def test_full_text_search_survives_fts_metacharacters(self):
        with _TempProject() as project:
            project.add("note.txt", "Насос P-101 вибрирует при пуске.\n")
            service = ReferenceService(project.root)

            for query in ('-', '"', '^', '(', ')', ':', '*', 'NEAR', 'AND', 'OR',
                          'NOT', 'P-101', '""', 'a"b', '---', '**', '(('):
                with self.subTest(query=query):
                    self.assertIsInstance(service.search(query), list)

            self.assertEqual(len(service.search("P-101")), 1)
            self.assertEqual(service.search("---"), [])

    def test_exact_search_does_not_create_a_second_index(self):
        with _TempProject() as project:
            project.add("note.txt", "Насос P-101 вибрирует.\n")
            ReferenceService(project.root).search("P-101", exact=True)

            with closing(sqlite3.connect(project.root / "db" / "reference.sqlite3")) as conn:
                names = [row[0] for row in conn.execute("SELECT name FROM sqlite_master")]
                tables = [
                    row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                ]
            base_tables = {
                name for name in tables
                if not name.startswith(("fragment_fts", "sqlite_"))
            }
            self.assertEqual(base_tables, {
                "schema_version", "knowledge_base", "ingest_event", "file_blob",
                "material", "material_version", "processing_run", "artifact",
                "fragment", "job", "ocr_cache", "embedding", "page_image",
            })
            self.assertIn("fragment_fts", names)


if __name__ == "__main__":
    unittest.main()
