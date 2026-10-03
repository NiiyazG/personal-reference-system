"""Embeddings: contract, storage and semantic search (stage 2, third increment).

Everything here runs on the injected test double, so the whole layer — vector
packing, the `embedding` table, ranking, lifecycle and integrity checking — is
verified without a model on disk and without any network access.

Ranking is tested with a deterministic keyword stub rather than the hash stub:
a hash produces noise, and noise cannot demonstrate that the *closest* fragment
wins. The stub is a stand-in for a language model, never presented as one.
"""

import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from reference_system.core import ReferenceService, ReferenceError
from reference_system.embeddings import (
    EmbeddingError,
    StubEmbeddingModel,
    cosine_similarity,
    l2_normalize,
    pack_vector,
    model_spec,
    unpack_vector,
)
from reference_system.foundation import applied_migration_versions, initialize_project

VOCABULARY = ("вибрирует", "задвижк", "окраск", "насос")


def keyword_vectors(texts):
    """A 4-dimension, hand-made stand-in for a semantic model."""
    vectors = []
    for text in texts:
        folded = text.casefold()
        vectors.append(
            l2_normalize([1.0 if token in folded else 0.0 for token in VOCABULARY] + [0.05])
        )
    return vectors


def keyword_model(**kwargs):
    return StubEmbeddingModel(
        model_id="keyword-stub",
        model_version="kw-1",
        dimensions=len(VOCABULARY) + 1,
        max_tokens=512,
        pooling="stub",
        respond=keyword_vectors,
        **kwargs,
    )


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

    def text(self, name: str, body: str) -> Path:
        path = self.workspace / name
        path.write_text(body, encoding="utf-8")
        return path

    def service(self, model=None) -> ReferenceService:
        return ReferenceService(self.root, embedding_model=model)

    def seed(self, service: ReferenceService) -> None:
        service.add_file(self.text("valve.txt", "Задвижка DN100 вибрирует при открытии.\n"))
        service.add_file(self.text("paint.txt", "Окраска стен в синий цвет по RAL 5010.\n"))
        service.add_file(self.text("pump.txt", "Насос вибрирует, проверить крепёж.\n"))


class TestVectorPrimitives(unittest.TestCase):
    def test_pack_and_unpack_round_trip(self):
        values = [0.5, -1.25, 3.0, 0.0]
        blob = pack_vector(values)

        self.assertEqual(len(blob), len(values) * 4)
        self.assertEqual(unpack_vector(blob, len(values)), values)

    def test_unpack_rejects_a_wrong_length(self):
        with self.assertRaises(EmbeddingError):
            unpack_vector(b"\x00" * 6, 1024)

    def test_l2_normalize_produces_unit_length(self):
        vector = l2_normalize([3.0, 4.0])

        self.assertAlmostEqual(vector[0], 0.6, places=6)
        self.assertAlmostEqual(vector[1], 0.8, places=6)

    def test_l2_normalize_leaves_a_zero_vector_alone(self):
        self.assertEqual(l2_normalize([0.0, 0.0]), [0.0, 0.0])

    def test_cosine_similarity_is_one_for_identical_vectors(self):
        vector = l2_normalize([0.3, 0.4, 0.5])

        self.assertAlmostEqual(cosine_similarity(vector, vector), 1.0, places=6)

    def test_cosine_similarity_rejects_a_dimension_mismatch(self):
        with self.assertRaises(EmbeddingError):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_vector_size_for_bge_m3_dimensions(self):
        # 1024 float32 dimensions, the shape the reference model would produce
        self.assertEqual(len(pack_vector([0.0] * 1024)), 4096)


class TestStubContract(unittest.TestCase):
    def test_deterministic_vectors_for_identical_text(self):
        model = StubEmbeddingModel(dimensions=8)

        first = model.encode(["Задвижка DN100"])
        second = model.encode(["Задвижка DN100"])

        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), 8)

    def test_wrong_vector_count_is_refused(self):
        model = StubEmbeddingModel(dimensions=4, respond=lambda texts: [[0.0] * 4])

        with self.assertRaises(EmbeddingError):
            model.encode(["a", "b"])

    def test_wrong_dimensions_are_refused(self):
        model = StubEmbeddingModel(dimensions=4, respond=lambda texts: [[0.0] * 3 for _ in texts])

        with self.assertRaises(EmbeddingError):
            model.encode(["a"])

    def test_model_spec_captures_the_comparability_fingerprint(self):
        spec = model_spec(keyword_model()).as_dict()

        self.assertEqual(spec["model_id"], "keyword-stub")
        self.assertEqual(spec["model_version"], "kw-1")
        self.assertEqual(spec["dimensions"], 5)
        self.assertEqual(spec["pooling"], "stub")


class TestEmbeddingStorage(unittest.TestCase):
    def test_migration_adds_the_embedding_table(self):
        with _TempProject() as project:
            self.assertEqual(applied_migration_versions(project.root / "db" / "reference.sqlite3"), [1, 2, 3, 4])
            with closing(sqlite3.connect(project.root / "db" / "reference.sqlite3")) as conn:
                tables = {
                    row[0]
                    for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                }
            self.assertIn("embedding", tables)

    def test_status_reports_no_model_by_default(self):
        with _TempProject() as project:
            status = project.service().status()

            self.assertIsNone(status["embedding_model"])
            self.assertEqual(status["embeddings"], 0)

    def test_embed_fragments_stores_one_vector_per_fragment(self):
        with _TempProject() as project:
            service = project.service(keyword_model())
            project.seed(service)

            report = service.embed_fragments()

            self.assertEqual(report["embedded"], 3)
            self.assertEqual(report["dimensions"], 5)
            self.assertEqual(report["model_id"], "keyword-stub")
            self.assertEqual(service.status()["embeddings"], 3)

    def test_embedding_is_idempotent(self):
        with _TempProject() as project:
            service = project.service(keyword_model())
            project.seed(service)
            service.embed_fragments()

            again = service.embed_fragments()

            self.assertEqual(again["embedded"], 0)
            self.assertEqual(again["already_embedded"], 3)
            self.assertEqual(service.status()["embeddings"], 3)

    def test_new_material_is_embedded_without_touching_the_old_ones(self):
        with _TempProject() as project:
            service = project.service(keyword_model())
            project.seed(service)
            service.embed_fragments()

            service.add_file(project.text("extra.txt", "Насос вибрирует сильнее.\n"))
            report = service.embed_fragments()

            self.assertEqual(report["embedded"], 1)
            self.assertEqual(service.status()["embeddings"], 4)

    def test_a_different_model_version_produces_its_own_vectors(self):
        with _TempProject() as project:
            service = project.service(keyword_model())
            project.seed(service)
            service.embed_fragments()

            other = StubEmbeddingModel(
                model_id="keyword-stub",
                model_version="kw-2",
                dimensions=len(VOCABULARY) + 1,
                pooling="stub",
                respond=keyword_vectors,
            )
            swapped = project.service(other)
            report = swapped.embed_fragments()

            self.assertEqual(report["embedded"], 3)
            self.assertEqual(report["model_version"], "kw-2")

    def test_embedding_without_a_model_is_refused(self):
        with _TempProject() as project:
            service = project.service(None)
            project.seed(service)

            with self.assertRaises(EmbeddingError):
                service.embed_fragments()

    def test_vectors_are_stored_as_float32_of_the_declared_dimensions(self):
        with _TempProject() as project:
            service = project.service(keyword_model())
            project.seed(service)
            service.embed_fragments()

            with closing(sqlite3.connect(project.root / "db" / "reference.sqlite3")) as conn:
                rows = conn.execute("SELECT dimensions, vector, sha256 FROM embedding").fetchall()

            self.assertEqual(len(rows), 3)
            for dimensions, blob, sha in rows:
                self.assertEqual(dimensions, 5)
                self.assertEqual(len(blob), 20)
                self.assertEqual(len(sha), 64)


class TestSemanticSearch(unittest.TestCase):
    def test_semantic_search_ranks_the_closest_fragment_first(self):
        with _TempProject() as project:
            service = project.service(keyword_model())
            project.seed(service)
            service.embed_fragments()

            report = service.semantic_search("когда насос вибрирует", limit=3)

            self.assertEqual(report["mode"], "semantic")
            titles = [hit["title"] for hit in report["results"]]
            self.assertEqual(titles[0], "pump.txt")
            self.assertIn("valve.txt", titles)
            self.assertEqual(titles[-1], "paint.txt")
            scores = [hit["score"] for hit in report["results"]]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_semantic_search_honours_the_limit(self):
        with _TempProject() as project:
            service = project.service(keyword_model())
            project.seed(service)
            service.embed_fragments()

            report = service.semantic_search("вибрирует", limit=2)

            self.assertEqual(len(report["results"]), 2)

    def test_semantic_search_can_filter_by_minimum_score(self):
        with _TempProject() as project:
            service = project.service(keyword_model())
            project.seed(service)
            service.embed_fragments()

            report = service.semantic_search("окраска", limit=10, min_score=0.5)

            self.assertEqual([hit["title"] for hit in report["results"]], ["paint.txt"])

    def test_semantic_search_reports_the_model_that_produced_the_vectors(self):
        with _TempProject() as project:
            service = project.service(keyword_model())
            project.seed(service)
            service.embed_fragments()

            report = service.semantic_search("вибрирует")

            self.assertEqual(report["model_id"], "keyword-stub")
            self.assertEqual(report["dimensions"], 5)

    def test_semantic_search_without_a_model_is_refused(self):
        with _TempProject() as project:
            service = project.service(None)

            with self.assertRaises(EmbeddingError):
                service.semantic_search("вибрирует")

    def test_semantic_search_without_vectors_is_refused_not_empty(self):
        with _TempProject() as project:
            service = project.service(keyword_model())
            project.seed(service)

            with self.assertRaises(EmbeddingError) as caught:
                service.semantic_search("вибрирует")

            self.assertIn("embed", str(caught.exception).casefold())

    def test_semantic_search_never_compares_vectors_of_a_foreign_model(self):
        """Vectors from another model live in another space: refuse, never mix."""
        with _TempProject() as project:
            service = project.service(keyword_model())
            project.seed(service)
            service.embed_fragments()

            other = StubEmbeddingModel(
                model_id="keyword-stub",
                model_version="kw-2",
                dimensions=len(VOCABULARY) + 1,
                pooling="stub",
                respond=keyword_vectors,
            )

            with self.assertRaises(EmbeddingError) as caught:
                project.service(other).semantic_search("вибрирует")

            self.assertIn("kw-2", str(caught.exception))

    def test_semantic_search_returns_the_same_locator_as_full_text(self):
        with _TempProject() as project:
            service = project.service(keyword_model())
            project.seed(service)
            service.embed_fragments()

            semantic = service.semantic_search("насос вибрирует", limit=1)["results"][0]
            full_text = service.search("вибрирует")[0]

            self.assertEqual(semantic["locator"]["type"], full_text["locator"]["type"])
            self.assertEqual(semantic["material_id"], full_text["material_id"])


class TestEmbeddingLifecycle(unittest.TestCase):
    def test_deleting_a_material_drops_its_vectors(self):
        with _TempProject() as project:
            service = project.service(keyword_model())
            project.seed(service)
            service.embed_fragments()
            material_id = service.search("окраска")[0]["material_id"]

            report = service.delete_material(material_id, confirm=True)

            self.assertEqual(report["deleted_embeddings"], 1)
            self.assertEqual(service.status()["embeddings"], 2)

    def test_integrity_check_is_clean_after_embedding(self):
        with _TempProject() as project:
            service = project.service(keyword_model())
            project.seed(service)
            service.embed_fragments()

            report = service.integrity_check()

            self.assertTrue(report["ok"], report["issues"])
            self.assertEqual(report["issues"], [])

    def test_integrity_check_detects_a_corrupted_vector(self):
        with _TempProject() as project:
            service = project.service(keyword_model())
            project.seed(service)
            service.embed_fragments()

            with closing(sqlite3.connect(project.root / "db" / "reference.sqlite3")) as conn:
                conn.execute("UPDATE embedding SET vector=? WHERE id=(SELECT min(id) FROM embedding)", (b"\x00" * 20,))
                conn.commit()

            report = service.integrity_check()

            self.assertFalse(report["ok"])
            codes = {issue["code"] for issue in report["issues"]}
            self.assertIn("EMBEDDING_HASH_MISMATCH", codes)

    def test_integrity_check_detects_a_length_mismatch(self):
        with _TempProject() as project:
            service = project.service(keyword_model())
            project.seed(service)
            service.embed_fragments()

            with closing(sqlite3.connect(project.root / "db" / "reference.sqlite3")) as conn:
                conn.execute("UPDATE embedding SET dimensions=99")
                conn.commit()

            report = service.integrity_check()

            self.assertFalse(report["ok"])
            codes = {issue["code"] for issue in report["issues"]}
            self.assertIn("EMBEDDING_LENGTH_MISMATCH", codes)

    def test_reprocess_does_not_duplicate_vectors(self):
        with _TempProject() as project:
            service = project.service(keyword_model())
            project.seed(service)
            service.embed_fragments()
            material_id = service.search("вибрирует")[0]["material_id"]

            service.reprocess(material_id)

            self.assertEqual(service.status()["embeddings"], 3)

    def test_unknown_material_still_raises(self):
        with _TempProject() as project:
            service = project.service(keyword_model())

            with self.assertRaises(ReferenceError):
                service.delete_material("mat_does_not_exist", confirm=True)


class TestOutdatedSchema(unittest.TestCase):
    """A project created before the embeddings migration must say so, not crash.

    `data/` is exactly this case until the migration is applied deliberately.
    """

    @staticmethod
    def _drop_embedding_table(project: _TempProject) -> None:
        with closing(sqlite3.connect(project.root / "db" / "reference.sqlite3")) as conn:
            conn.execute("DROP TABLE embedding")
            conn.commit()

    def test_status_reports_an_outdated_schema_instead_of_failing(self):
        with _TempProject() as project:
            self._drop_embedding_table(project)

            status = project.service(keyword_model()).status()

            self.assertFalse(status["schema_current"])
            self.assertIsNone(status["embeddings"])

    def test_embed_on_an_outdated_schema_names_the_migration(self):
        with _TempProject() as project:
            service = project.service(keyword_model())
            project.seed(service)
            self._drop_embedding_table(project)

            with self.assertRaises(ReferenceError) as caught:
                service.embed_fragments()

            self.assertIn("0003", str(caught.exception))

    def test_semantic_search_on_an_outdated_schema_names_the_migration(self):
        with _TempProject() as project:
            service = project.service(keyword_model())
            project.seed(service)
            self._drop_embedding_table(project)

            with self.assertRaises(ReferenceError) as caught:
                service.semantic_search("вибрирует")

            self.assertIn("0003", str(caught.exception))

    def test_integrity_check_survives_an_outdated_schema(self):
        with _TempProject() as project:
            service = project.service(keyword_model())
            project.seed(service)
            self._drop_embedding_table(project)

            report = service.integrity_check()

            self.assertFalse(report["schema_current"])
            self.assertTrue(report["ok"], report["issues"])


if __name__ == "__main__":
    unittest.main()
