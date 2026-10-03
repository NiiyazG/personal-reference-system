"""Embedding model interface, vector packing and a test double.

Same shape as the OCR layer, for the same reasons: the model is **injected**,
never discovered, and the core stays free of third-party imports. Vectors are
packed with `struct` as little-endian float32, so storing and scanning them
needs no `numpy` in `core.py`.

Every vector carries the model id and version that produced it. `model_version`
is derived from the model files themselves (a hash), so re-downloading a model,
switching quantization or changing the pooling rule all invalidate stored
vectors instead of silently mixing two different vector spaces — the failure
mode this project has already been bitten by twice with caches.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable


class EmbeddingError(RuntimeError):
    """Embeddings are unavailable, misconfigured, or produced nothing usable."""


class EmbeddingModelUnavailableError(EmbeddingError):
    """No model is present locally, or its runtime is missing."""


@runtime_checkable
class EmbeddingModel(Protocol):
    """What a real embedding adapter must provide.

    `encode` receives texts in order and must return one vector per text, in the
    same order, each of `dimensions` floats. Returning fewer vectors than inputs
    is an error, not a shortcut — a silently dropped text would make a fragment
    invisible to semantic search with nothing to show for it.
    """

    model_id: str
    model_version: str
    dimensions: int
    max_tokens: int
    pooling: str

    def encode(self, texts: list[str]) -> list[list[float]]: ...


def l2_normalize(values: list[float]) -> list[float]:
    """Unit-length copy. Cosine similarity is a dot product once vectors are unit."""
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return list(values)
    return [value / norm for value in values]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise EmbeddingError(f"vector length mismatch: {len(left)} != {len(right)}")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def pack_vector(values: list[float]) -> bytes:
    """float32 little-endian, the storage format in the `embedding` table."""
    return struct.pack(f"<{len(values)}f", *values)


def unpack_vector(blob: bytes, dimensions: int) -> list[float]:
    expected = dimensions * 4
    if len(blob) != expected:
        raise EmbeddingError(f"vector blob is {len(blob)} bytes, expected {expected}")
    return list(struct.unpack(f"<{dimensions}f", blob))


def vector_sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class ModelSpec:
    """Everything that decides whether a stored vector is still comparable."""

    model_id: str
    model_version: str
    dimensions: int
    pooling: str
    max_tokens: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "dimensions": self.dimensions,
            "pooling": self.pooling,
            "max_tokens": self.max_tokens,
        }


class StubEmbeddingModel:
    """A deterministic test double, never a language model.

    Vectors are derived from the text hash, so they are stable across runs and
    identical texts produce identical vectors — which is what lets the storage
    and ranking tests be meaningful without any model on disk.
    """

    is_test_double = True

    def __init__(
        self,
        *,
        model_id: str = "stub-embedder",
        model_version: str = "stub-1",
        dimensions: int = 16,
        max_tokens: int = 512,
        pooling: str = "stub",
        respond: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> None:
        if dimensions <= 0:
            raise EmbeddingError("dimensions must be positive")
        self.model_id = model_id
        self.model_version = model_version
        self.dimensions = dimensions
        self.max_tokens = max_tokens
        self.pooling = pooling
        self.respond = respond
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.respond is not None:
            vectors = self.respond(list(texts))
        else:
            vectors = [self._vector_for(text) for text in texts]
        if len(vectors) != len(texts):
            raise EmbeddingError(f"model returned {len(vectors)} vectors for {len(texts)} texts")
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise EmbeddingError(
                    f"model returned {len(vector)} dimensions, declared {self.dimensions}"
                )
        return vectors

    def _vector_for(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = struct.unpack(f"<{self.dimensions}H", (digest * ((self.dimensions * 2) // 32 + 1))[: self.dimensions * 2])
        # map 0..65535 onto -1..1, then normalise so cosine is a dot product
        return l2_normalize([(value / 32767.5) - 1.0 for value in raw])


def model_spec(model: EmbeddingModel) -> ModelSpec:
    return ModelSpec(
        model_id=str(model.model_id),
        model_version=str(model.model_version),
        dimensions=int(model.dimensions),
        pooling=str(getattr(model, "pooling", "unknown")),
        max_tokens=int(getattr(model, "max_tokens", 0)),
    )
