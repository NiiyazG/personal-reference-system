"""ONNX embedding adapter on `onnxruntime` + `tokenizers`.

Two deliberate choices:

* **No network, by construction.** The model is resolved by walking the local
  Hugging Face cache directory; this module never imports a Hub client and
  cannot download anything. A missing model is an error the caller can act on,
  not a silent fetch.
* **`model_version` is the model's own fingerprint** — the SHA-256 of the ONNX
  file, plus the pooling rule. Re-downloading the model, switching quantization
  or changing how vectors are pooled all produce a different version, so stored
  vectors from two different vector spaces can never be compared by accident.
  That failure mode is the reason this project treats caches as suspects.

Pooling is explicit and recorded: bge-m3 ships CLS-pooled dense vectors, and
`mean`/`cls` must not be mixed silently.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .embeddings import EmbeddingError, EmbeddingModelUnavailableError, l2_normalize

DEFAULT_REPO_ID = "Xenova/bge-m3"
DEFAULT_ONNX_RELPATH = "onnx/model_quantized.onnx"
SUPPORTED_POOLING = ("cls", "mean")


def default_cache_dir() -> Path:
    """The Hugging Face hub cache, honouring the usual environment overrides."""
    for variable in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(variable)
        if value:
            return Path(value)
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def repository_directory(repo_id: str, *, cache_dir: Path | None = None) -> Path:
    return (cache_dir or default_cache_dir()) / ("models--" + repo_id.replace("/", "--"))


def resolve_cached_model(
    repo_id: str = DEFAULT_REPO_ID,
    *,
    cache_dir: Path | None = None,
    onnx_relpath: str = DEFAULT_ONNX_RELPATH,
) -> Path:
    """Newest local snapshot that has both the ONNX graph and a tokenizer."""
    base = repository_directory(repo_id, cache_dir=cache_dir)
    snapshots = base / "snapshots"
    if not snapshots.is_dir():
        raise EmbeddingModelUnavailableError(
            f"{repo_id} is not in the local cache ({base}); this module never downloads"
        )
    candidates = sorted(
        (entry for entry in snapshots.iterdir() if entry.is_dir()),
        key=lambda entry: entry.stat().st_mtime,
        reverse=True,
    )
    for snapshot in candidates:
        if (snapshot / onnx_relpath).is_file() and (snapshot / "tokenizer.json").is_file():
            return snapshot
    raise EmbeddingModelUnavailableError(
        f"no snapshot of {repo_id} in {snapshots} contains both {onnx_relpath} and tokenizer.json"
    )


def _sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class OnnxEmbeddingModel:
    """Dense sentence embeddings from a local ONNX graph, on the CPU."""

    is_test_double = False

    def __init__(
        self,
        directory: Path | str,
        *,
        repo_id: str = DEFAULT_REPO_ID,
        onnx_relpath: str = DEFAULT_ONNX_RELPATH,
        pooling: str = "cls",
        normalize: bool = True,
        max_tokens: int = 8192,
        model_id: str | None = None,
        model_version: str | None = None,
        threads: int | None = None,
    ) -> None:
        if pooling not in SUPPORTED_POOLING:
            raise EmbeddingError(f"unsupported pooling {pooling!r}; use one of {SUPPORTED_POOLING}")
        self.directory = Path(directory)
        onnx_path = self.directory / onnx_relpath
        if not onnx_path.is_file():
            raise EmbeddingModelUnavailableError(f"model file not found: {onnx_path}")
        tokenizer_path = self.directory / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise EmbeddingModelUnavailableError(f"tokenizer not found: {tokenizer_path}")

        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise EmbeddingModelUnavailableError("onnxruntime is not installed") from exc
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise EmbeddingModelUnavailableError("tokenizers is not installed") from exc

        self.model_id = model_id or repo_id
        self.pooling = pooling
        self.normalize = normalize
        self.max_tokens = max_tokens
        self.onnx_relpath = onnx_relpath

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads or max(1, (os.cpu_count() or 2))
        self._session = ort.InferenceSession(
            str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_names = {item.name for item in self._session.get_inputs()}
        self._output_name = self._session.get_outputs()[0].name

        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=max_tokens)

        self.dimensions = self._read_dimensions()
        self.model_version = model_version or self._fingerprint(onnx_path)

    # -- identity ---------------------------------------------------------

    def _read_dimensions(self) -> int:
        """From `config.json`; a bad value silently mis-sizing vectors would be worse."""
        config_path = self.directory / "config.json"
        if config_path.is_file():
            try:
                payload: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise EmbeddingError(f"cannot read {config_path}: {exc}") from exc
            for key in ("hidden_size", "d_model", "dim"):
                value = payload.get(key)
                if isinstance(value, int) and value > 0:
                    return value
        shape = self._session.get_outputs()[0].shape
        if len(shape) == 3 and isinstance(shape[-1], int) and shape[-1] > 0:
            return int(shape[-1])
        raise EmbeddingError(
            "cannot determine embedding dimensions: no hidden_size in config.json and no static output shape"
        )

    def _fingerprint(self, onnx_path: Path) -> str:
        digest = _sha256_of_file(onnx_path)
        suffix = "" if self.normalize else ";raw"
        return f"{self.model_id}@{digest[:16]};pool={self.pooling};batch=1{suffix}"

    def describe(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "dimensions": self.dimensions,
            "pooling": self.pooling,
            "max_tokens": self.max_tokens,
            "directory": str(self.directory),
            "onnx": self.onnx_relpath,
        }

    # -- inference --------------------------------------------------------

    def encode(self, texts: list[str]) -> list[list[float]]:
        """One text per inference run, deliberately.

        The graph is dynamically quantized, so activation quantization ranges
        are derived from the tensor actually fed in — including padding. The
        same text therefore comes out differently in a batch of 1 and in a
        padded batch of 16 (measured: up to 2e-2 per component, far above
        float32 noise). Stored vectors and query vectors have to live in one
        consistent space, so every vector here is produced the same way: alone.
        The cost in throughput is real and is recorded in `model_version` as
        `batch=1`, so a future batched path cannot silently mix with these.
        """
        if not texts:
            return []
        return [self._encode_one(self._prepare(text)) for text in texts]

    def _encode_one(self, text: str) -> list[float]:
        import numpy as np

        encoding = self._tokenizer.encode(text, add_special_tokens=True)
        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
        feeds: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)
        missing = self._input_names - set(feeds)
        if missing:
            raise EmbeddingError(
                f"the model expects inputs this adapter does not provide: {sorted(missing)}"
            )

        hidden = self._session.run([self._output_name], feeds)[0]
        vector = self._pool(hidden, attention_mask)[0]
        return l2_normalize(vector) if self.normalize else vector

    def _prepare(self, text: str) -> str:
        if not isinstance(text, str):
            raise EmbeddingError("encode() takes strings")
        if not text.strip():
            # An empty string has no meaning to embed; an empty vector would
            # enter the index and match everything equally badly.
            raise EmbeddingError("refusing to embed an empty text")
        return text

    def _pool(self, hidden: Any, attention_mask: Any) -> list[list[float]]:
        import numpy as np

        if self.pooling == "cls":
            pooled = hidden[:, 0, :]
        elif self.pooling == "mean":
            mask = attention_mask.astype(np.float32)[:, :, None]
            pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)
        else:  # pragma: no cover - guarded in __init__
            raise EmbeddingError(f"unsupported pooling {self.pooling!r}")
        return [row.astype(np.float32).tolist() for row in pooled]


def load_default_model(
    repo_id: str = DEFAULT_REPO_ID,
    *,
    cache_dir: Path | None = None,
    onnx_relpath: str = DEFAULT_ONNX_RELPATH,
    **kwargs: Any,
) -> OnnxEmbeddingModel:
    """Resolve the model from the local cache and instantiate it."""
    directory = resolve_cached_model(repo_id, cache_dir=cache_dir, onnx_relpath=onnx_relpath)
    return OnnxEmbeddingModel(directory, repo_id=repo_id, onnx_relpath=onnx_relpath, **kwargs)
