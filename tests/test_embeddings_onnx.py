"""Embeddings against the real bge-m3 ONNX model.

Skipped cleanly when the model is not in the local Hugging Face cache, so the
suite stays honest on a machine without it — never silently green.

These tests are the ones that can fail for *modelling* reasons rather than code
reasons: does the model actually place Russian paraphrases closer together, is
the vector space cross-lingual, is inference deterministic. Claims about the
model are only made here after the model has been run.
"""

import math
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from reference_system.core import ReferenceService
from reference_system.embeddings import EmbeddingModelUnavailableError, cosine_similarity
from reference_system.embeddings_onnx import (
    DEFAULT_ONNX_RELPATH,
    DEFAULT_REPO_ID,
    OnnxEmbeddingModel,
    default_cache_dir,
    load_default_model,
    resolve_cached_model,
)
from reference_system.foundation import initialize_project

_MODEL: OnnxEmbeddingModel | None = None


def model() -> OnnxEmbeddingModel:
    """One model instance for the whole run: loading it is the slow part."""
    global _MODEL
    if _MODEL is None:
        try:
            _MODEL = load_default_model()
        except EmbeddingModelUnavailableError as exc:
            raise unittest.SkipTest(f"{DEFAULT_REPO_ID} is not available locally: {exc}")
    return _MODEL


def similarity(left: str, right: str) -> float:
    active = model()
    vectors = active.encode([left, right])
    return cosine_similarity(vectors[0], vectors[1])


class TestModelResolution(unittest.TestCase):
    def test_the_model_resolves_from_the_local_cache(self):
        directory = resolve_cached_model()

        self.assertTrue((directory / DEFAULT_ONNX_RELPATH).is_file())
        self.assertTrue((directory / "tokenizer.json").is_file())
        self.assertTrue(directory.is_relative_to(default_cache_dir()))

    def test_an_unknown_repository_is_a_clean_error(self):
        with self.assertRaises(EmbeddingModelUnavailableError):
            resolve_cached_model("Xenova/definitely-not-a-real-model-xyz")


class TestModelShape(unittest.TestCase):
    def test_declared_shape_matches_the_graph(self):
        active = model()

        self.assertEqual(active.model_id, DEFAULT_REPO_ID)
        self.assertEqual(active.dimensions, 1024)
        self.assertEqual(active.max_tokens, 8192)
        self.assertEqual(active.pooling, "cls")

    def test_version_is_a_fingerprint_of_the_files_and_the_pooling_rule(self):
        active = model()

        self.assertIn("@", active.model_version)
        self.assertIn("pool=cls", active.model_version)
        self.assertGreater(len(active.model_version), 20)

    def test_the_adapter_never_downloads(self):
        source = (PROJECT_ROOT / "reference_system" / "embeddings_onnx.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("huggingface_hub", source)
        self.assertNotIn("requests", source)
        self.assertNotIn("urllib", source)


class TestRealVectors(unittest.TestCase):
    def test_a_single_text_yields_one_unit_vector_of_1024_dimensions(self):
        vectors = model().encode(["Задвижка DN100 вибрирует при открытии."])

        self.assertEqual(len(vectors), 1)
        self.assertEqual(len(vectors[0]), 1024)
        norm = math.sqrt(sum(value * value for value in vectors[0]))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_a_batch_keeps_one_vector_per_text_in_order(self):
        texts = ["Насос вибрирует.", "Покраска стен в синий цвет.", "Проверить крепёж."]

        vectors = model().encode(texts)

        self.assertEqual(len(vectors), 3)
        single = model().encode([texts[1]])[0]
        self.assertEqual(len(single), 1024)
        # One text per inference run: no vector may depend on its batch mates,
        # otherwise stored vectors and query vectors drift apart.
        largest = max(abs(a - b) for a, b in zip(vectors[1], single))
        self.assertLess(largest, 1e-6)

    def test_an_empty_text_is_refused_rather_than_embedded(self):
        from reference_system.embeddings import EmbeddingError

        with self.assertRaises(EmbeddingError):
            model().encode(["   "])

    def test_inference_repeats_to_float32_precision(self):
        """Not bit-exact, and claiming otherwise would be a lie.

        ONNX Runtime's threaded reductions reorder floating-point additions, so
        two runs differ at ~1e-8 — five orders of magnitude below anything that
        can move a cosine similarity. Measured, not assumed.
        """
        text = "Герметичность соединения проверяется опрессовкой."

        first = model().encode([text])[0]
        second = model().encode([text])[0]
        largest = max(abs(a - b) for a, b in zip(first, second))

        self.assertLess(largest, 1e-5)

    def test_a_very_long_text_is_truncated_not_refused(self):
        long_text = "Задвижка DN100 вибрирует при открытии. " * 4000

        vectors = model().encode([long_text])

        self.assertEqual(len(vectors), 1)
        self.assertEqual(len(vectors[0]), 1024)


class TestSemanticQuality(unittest.TestCase):
    """What the vectors are actually for. These can honestly fail."""

    def test_russian_paraphrases_are_closer_than_unrelated_text(self):
        related = similarity(
            "Задвижка DN100 вибрирует при открытии.",
            "Клапан DN100 дрожит, когда его открывают.",
        )
        unrelated = similarity(
            "Задвижка DN100 вибрирует при открытии.",
            "Краска для стен поставляется в банках по 9 литров.",
        )

        self.assertGreater(related, unrelated)
        self.assertGreater(related, 0.6)

    def test_the_space_is_cross_lingual(self):
        russian = "Насос вибрирует, нужно проверить крепёж."

        cross_lingual = similarity(russian, "The pump vibrates, check the fasteners.")
        unrelated = similarity(russian, "Paint the walls in a blue colour.")

        self.assertGreater(cross_lingual, unrelated)

    def test_similarity_is_symmetric(self):
        left = "Проверить крепёж насоса."
        right = "Осмотреть крепление насосного агрегата."

        self.assertAlmostEqual(similarity(left, right), similarity(right, left), places=6)


class TestSemanticSearchOnRealText(unittest.TestCase):
    def test_a_query_without_shared_words_finds_the_right_material(self):
        """The point of embeddings: no keyword overlap between query and answer."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "personal-reference"
            initialize_project(root)
            workspace = Path(tmp)
            documents = {
                "pump.txt": "Насос вибрирует, проверить крепёж и центровку вала.\n",
                "paint.txt": "Окраска стен в синий цвет по RAL 5010, два слоя.\n",
                "weld.txt": "Сварной шов проверен визуально, дефектов нет.\n",
            }
            for name, body in documents.items():
                path = workspace / name
                path.write_text(body, encoding="utf-8")

            service = ReferenceService(root, embedding_model=model())
            for path in sorted(workspace.glob("*.txt")):
                service.add_file(path)

            report = service.embed_fragments()
            self.assertEqual(report["embedded"], 3)

            hits = service.semantic_search("оборудование трясётся при работе", limit=3)

            self.assertEqual(hits["model_id"], DEFAULT_REPO_ID)
            self.assertEqual(len(hits["results"]), 3)
            self.assertEqual(hits["results"][0]["title"], "pump.txt")
            self.assertEqual(hits["results"][-1]["title"], "paint.txt")
            scores = [hit["score"] for hit in hits["results"]]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_stored_vectors_match_what_the_model_produces_now(self):
        """Re-embedding identical text must reproduce the stored vector.

        To float32 precision, not bit-exactly: the same threaded-reduction
        noise as above. A difference bigger than ~1e-5 would mean the model,
        the pooling rule or the preprocessing changed under the stored index.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "personal-reference"
            initialize_project(root)
            source = Path(tmp) / "note.txt"
            body = "Задвижка DN100 вибрирует при открытии, нужен осмотр.\n"
            source.write_text(body, encoding="utf-8")

            active = model()
            service = ReferenceService(root, embedding_model=active)
            service.add_file(source)
            service.embed_fragments()

            import sqlite3
            from contextlib import closing

            from reference_system.embeddings import unpack_vector

            with closing(sqlite3.connect(root / "db" / "reference.sqlite3")) as conn:
                blob, dimensions = conn.execute(
                    "SELECT vector, dimensions FROM embedding"
                ).fetchone()
                fragment_text = conn.execute("SELECT text FROM fragment").fetchone()[0]

            recomputed = active.encode([fragment_text])[0]
            stored = unpack_vector(blob, dimensions)

            self.assertEqual(dimensions, 1024)
            largest = max(abs(a - b) for a, b in zip(stored, recomputed))
            self.assertLess(largest, 1e-5)


if __name__ == "__main__":
    unittest.main()
