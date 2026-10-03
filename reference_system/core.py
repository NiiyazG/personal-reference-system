from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config import DEFAULT_MODE
from .embeddings import (
    EmbeddingError,
    EmbeddingModel,
    ModelSpec,
    cosine_similarity,
    model_spec,
    pack_vector,
    unpack_vector,
    vector_sha256,
)
from .foundation import STORAGE_QUOTA, disk_free_bytes, project_size_bytes
from .ocr import OcrEngine, OcrResult, supports_language
from .processors import extract_file

_GIB = 1024 ** 3
_EMBED_BATCH = 16


class ReferenceError(RuntimeError):
    pass


class QuotaExceededError(ReferenceError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _table_present(conn: sqlite3.Connection, name: str) -> bool:
    """A project created before a migration exists must say so, not crash."""
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_size(path: Path) -> int:
    return project_size_bytes(path)


def _project_mode(root: Path) -> str | None:
    """The operating mode recorded in a project manifest, if one exists."""
    manifest_path = root / "manifest.yaml"
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = payload.get("mode")
    return value if isinstance(value, str) and value else None


_FTS_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def _fts_match_query(query: str) -> str | None:
    """Build a safe FTS5 MATCH expression from raw user input.

    Passing user input straight to MATCH lets its metacharacters act as syntax:
    "P-101" means "P NOT 101", a stray quote or "(" is a syntax error, and the
    whole search fails with sqlite3.OperationalError. Here the input is split on
    the same word boundaries the unicode61 tokenizer uses at index time, and each
    word is quoted and AND-combined, so any query is accepted literally.
    Returns None when the input holds no searchable words.
    """
    tokens = _FTS_TOKEN_PATTERN.findall(query)
    if not tokens:
        return None
    return " AND ".join(f'"{token}"' for token in tokens)


def _image_relative_path(sha256: str, image_format: str) -> Path:
    safe_ext = "".join(character for character in image_format.lower() if character.isalnum()) or "bin"
    return Path("store") / "images" / "sha256" / sha256[:2] / f"{sha256}.{safe_ext}"


def _image_path(root: Path, sha256: str, image_format: str) -> tuple[Path, Path]:
    relative = _image_relative_path(sha256, image_format)
    return relative, root / relative


def _validate_id(value: str, prefix: str) -> str:
    """Reject identifiers that could be used to escape the project root in a path."""
    if not isinstance(value, str) or not value.startswith(f"{prefix}_"):
        raise ReferenceError(f"malformed identifier: expected a {prefix}_ id")
    if not all(character.isalnum() or character == "_" for character in value):
        raise ReferenceError("malformed identifier: illegal characters")
    return value


class _OcrCache:
    """Content-addressed OCR cache.

    One entry per (image bytes, engine id, engine version, language). The raw
    rendered image is hashed, not the source document, so a multi-page PDF caches
    each page separately and an engine upgrade never reuses stale text. Cached
    text is verified by SHA-256 on read: a damaged entry is treated as a miss
    rather than trusted.
    """

    def __init__(self, root: Path, db_path: Path):
        self.root = Path(root)
        self.db_path = Path(db_path)

    def _path_for(self, engine_id: str, image_sha: str) -> Path:
        safe_engine = "".join(c for c in engine_id if c.isalnum() or c in "-_.") or "unknown"
        return self.root / "indexes" / "ocr" / safe_engine / f"{image_sha}.json"

    def get(
        self, image_sha: str, engine_id: str, engine_version: str, language: str
    ) -> OcrResult | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT relative_path, sha256 FROM ocr_cache WHERE image_sha256=? AND engine_id=? "
                "AND engine_version=? AND language=?",
                (image_sha, engine_id, engine_version, language),
            ).fetchone()
        if row is None:
            return None
        path = (self.root / row["relative_path"]).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            return None
        if _sha256(path) != row["sha256"]:
            return None
        try:
            return OcrResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, KeyError, TypeError):
            return None

    def put(
        self,
        *,
        image_sha: str,
        blob_sha: str,
        engine_id: str,
        engine_version: str,
        language: str,
        result: OcrResult,
    ) -> None:
        path = self._path_for(engine_id, image_sha)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        relative = path.relative_to(self.root).as_posix()
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO ocr_cache(image_sha256, blob_sha256, engine_id, engine_version, "
                "language, relative_path, sha256, created_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    image_sha,
                    blob_sha,
                    engine_id,
                    engine_version,
                    language,
                    relative,
                    _sha256(path),
                    _now(),
                ),
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn


class _CachingOcrEngine:
    """Wraps an engine so each distinct image is recognised at most once."""

    def __init__(self, engine: OcrEngine, cache: _OcrCache, blob_sha: str):
        self._engine = engine
        self._cache = cache
        self._blob_sha = blob_sha
        self.engine_id = engine.engine_id
        self.engine_version = engine.engine_version
        self.languages = tuple(getattr(engine, "languages", ()))

    def supports(self, language: str) -> bool:
        return supports_language(self._engine, language)

    def recognize(self, image_bytes: bytes, *, language: str) -> OcrResult:
        image_sha = hashlib.sha256(image_bytes).hexdigest()
        cached = self._cache.get(image_sha, self.engine_id, self.engine_version, language)
        if cached is not None:
            return cached
        result = self._engine.recognize(image_bytes, language=language)
        self._cache.put(
            image_sha=image_sha,
            blob_sha=self._blob_sha,
            engine_id=self.engine_id,
            engine_version=self.engine_version,
            language=language,
            result=result,
        )
        return result


class ReferenceService:
    def __init__(
        self,
        root: Path | str,
        *,
        ocr_engine: OcrEngine | None = None,
        ocr_language: str = "ru",
        embedding_model: EmbeddingModel | None = None,
        storage_quota: Mapping[str, int] | None = None,
        mode: str | None = None,
    ):
        """`ocr_engine` and `embedding_model` are injected, never discovered:
        with no engine the service behaves exactly as in stage 1 — images are
        metadata-only and text-less PDFs are refused instead of being guessed
        at; with no embedding model nothing is embedded and semantic search
        refuses instead of quietly returning an empty result.

        `storage_quota` and `mode` default to the module defaults and to the
        value recorded in the project manifest, so an existing project keeps
        reporting the settings it was created with.
        """
        self.root = Path(root).resolve()
        self.db_path = self.root / "db" / "reference.sqlite3"
        self.ocr_engine = ocr_engine
        self.ocr_language = ocr_language
        self.embedding_model = embedding_model
        self.storage_quota = (
            dict(storage_quota) if storage_quota is not None else dict(STORAGE_QUOTA)
        )
        self.mode = mode or _project_mode(self.root) or DEFAULT_MODE
        if not self.db_path.is_file():
            raise ReferenceError("reference project is not initialized")

    def _ocr_for_blob(self, blob_sha: str) -> OcrEngine | None:
        if self.ocr_engine is None:
            return None
        return _CachingOcrEngine(self.ocr_engine, _OcrCache(self.root, self.db_path), blob_sha)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _check_source(self, source: Path) -> None:
        if source.is_symlink():
            raise ReferenceError("symbolic links are not accepted")
        if not source.is_file():
            raise ReferenceError(f"input file not found: {source}")
        quota = self.storage_quota
        size = source.stat().st_size
        if size > quota["temporary_gib"] * _GIB:
            raise QuotaExceededError(
                f"input exceeds the {quota['temporary_gib']} GiB temporary quota"
            )
        free = disk_free_bytes(self.root)
        if free - size < quota["minimum_free_disk_gib"] * _GIB:
            raise QuotaExceededError("minimum free disk floor would be violated")
        if _tree_size(self.root) + size > quota["live_gib"] * _GIB:
            raise QuotaExceededError(
                f"{quota['live_gib']} GiB live quota would be exceeded"
            )

    def add_file(
        self,
        source: Path | str,
        *,
        title: str | None = None,
        images: bool = True,
    ) -> dict[str, Any]:
        source = Path(source).resolve()
        self._check_source(source)
        event_id = _id("ing")
        quarantine_dir = self.root / "quarantine" / event_id
        quarantine_dir.mkdir(parents=True, exist_ok=False)
        quarantine_path = quarantine_dir / "input.bin"
        shutil.copyfile(source, quarantine_path)
        created_blob = False
        created_image_files: list[Path] = []
        try:
            digest = _sha256(quarantine_path)
            extracted = extract_file(
                quarantine_path,
                original_name=source.name,
                ocr=self._ocr_for_blob(digest),
                ocr_language=self.ocr_language,
                images=images,
            )
            extracted_images = extracted.get("images", [])
            image_file_bytes = 0
            image_paths: dict[str, str] = {}
            for image in extracted_images:
                sha256 = image["sha256"]
                relative, image_path = _image_path(self.root, sha256, image["image_format"])
                image_paths[sha256] = relative.as_posix()
                if not image_path.is_file():
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image_path.write_bytes(image["bytes"])
                    created_image_files.append(image_path)
                    image_file_bytes += image["byte_size"]
            blob_relative = Path("store") / "blobs" / "sha256" / digest[:2] / digest
            blob_path = self.root / blob_relative
            blob_path.parent.mkdir(parents=True, exist_ok=True)
            if not blob_path.exists():
                os.replace(quarantine_path, blob_path)
                created_blob = True

            now = _now()
            material_id = _id("mat")
            version_id = _id("ver")
            run_id = _id("run")
            blob_id = _id("blob")
            display_title = (title or source.name).strip() or source.name
            artifact_dir = self.root / "artifacts" / material_id / version_id / run_id
            artifact_dir.mkdir(parents=True, exist_ok=False)
            normalized_path = artifact_dir / "normalized.txt"
            normalized_text = "\n\n".join(fragment["text"] for fragment in extracted["fragments"])
            normalized_path.write_text(normalized_text, encoding="utf-8")
            normalized_hash = _sha256(normalized_path)
            artifact_relative = normalized_path.relative_to(self.root)

            with closing(self._connect()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "INSERT INTO ingest_event(id, source_path, requested_at, status) VALUES(?, ?, ?, ?)",
                    (event_id, str(source), now, "PROCESSING"),
                )
                existing = conn.execute(
                    "SELECT id, relative_path FROM file_blob WHERE sha256=?", (digest,)
                ).fetchone()
                if existing is None:
                    conn.execute(
                        "INSERT INTO file_blob(id, sha256, size, detected_type, relative_path, created_at) "
                        "VALUES(?, ?, ?, ?, ?, ?)",
                        (blob_id, digest, source.stat().st_size, extracted["detected_type"], blob_relative.as_posix(), now),
                    )
                else:
                    blob_id = existing["id"]
                    blob_relative = Path(existing["relative_path"])
                conn.execute(
                    "INSERT INTO material(id, title, created_at) VALUES(?, ?, ?)",
                    (material_id, display_title, now),
                )
                conn.execute(
                    "INSERT INTO material_version(id, material_id, file_blob_id, version_number, status, created_at) "
                    "VALUES(?, ?, ?, 1, 'READY', ?)",
                    (version_id, material_id, blob_id, now),
                )
                config_hash = hashlib.sha256(b"{}").hexdigest()
                conn.execute(
                    "INSERT INTO processing_run(id, material_version_id, processor_id, processor_version, "
                    "config_hash, status, started_at, finished_at) VALUES(?, ?, ?, ?, ?, 'SUCCESS', ?, ?)",
                    (run_id, version_id, extracted["processor_id"], extracted["processor_version"], config_hash, now, now),
                )
                conn.execute(
                    "INSERT INTO artifact(id, processing_run_id, kind, relative_path, sha256, created_at) "
                    "VALUES(?, ?, 'normalized-text', ?, ?, ?)",
                    (_id("art"), run_id, artifact_relative.as_posix(), normalized_hash, now),
                )
                fragment_ids_by_page: dict[int, str] = {}
                for ordinal, fragment in enumerate(extracted["fragments"]):
                    fragment_id = _id("frag")
                    locator = fragment["locator"]
                    locator_json = json.dumps(locator, ensure_ascii=False, sort_keys=True)
                    conn.execute(
                        "INSERT INTO fragment(id, material_version_id, ordinal, text, locator_json, created_at) "
                        "VALUES(?, ?, ?, ?, ?, ?)",
                        (fragment_id, version_id, ordinal, fragment["text"], locator_json, now),
                    )
                    conn.execute(
                        "INSERT INTO fragment_fts(fragment_id, title, text) VALUES(?, ?, ?)",
                        (fragment_id, display_title, fragment["text"]),
                    )
                    if locator.get("page") is not None:
                        fragment_ids_by_page[int(locator["page"])] = fragment_id
                for image in extracted_images:
                    relative_path = image_paths[image["sha256"]]
                    conn.execute(
                        "INSERT INTO page_image(id, material_version_id, fragment_id, page, ordinal, "
                        "bbox_json, width, height, image_format, byte_size, sha256, relative_path, created_at) "
                        "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            _id("img"),
                            version_id,
                            fragment_ids_by_page.get(int(image["page"])),
                            int(image["page"]),
                            int(image["ordinal"]),
                            json.dumps(image["bbox"], sort_keys=True),
                            int(image["width"]),
                            int(image["height"]),
                            image["image_format"],
                            int(image["byte_size"]),
                            image["sha256"],
                            relative_path,
                            now,
                        ),
                    )
                conn.execute(
                    "UPDATE ingest_event SET status='READY' WHERE id=?", (event_id,)
                )
                conn.commit()
            return {
                "status": "READY",
                "ingest_event_id": event_id,
                "material_id": material_id,
                "version_id": version_id,
                "processing_run_id": run_id,
                "sha256": digest,
                "detected_type": extracted["detected_type"],
                "blob_relative_path": blob_relative.as_posix(),
                "deduplicated_blob": not created_blob,
                "page_images": len(extracted_images),
                "image_files": len(created_image_files),
                "image_bytes": image_file_bytes,
            }
        except Exception:
            if created_blob:
                with closing(self._connect()) as conn:
                    referenced = conn.execute(
                        "SELECT 1 FROM file_blob WHERE relative_path=?", (blob_relative.as_posix(),)
                    ).fetchone()
                if referenced is None:
                    blob_path.unlink(missing_ok=True)
            with closing(self._connect()) as conn:
                for image_path in created_image_files:
                    relative = image_path.relative_to(self.root).as_posix()
                    referenced = conn.execute(
                        "SELECT 1 FROM page_image WHERE relative_path=? LIMIT 1", (relative,)
                    ).fetchone()
                    if referenced is None:
                        image_path.unlink(missing_ok=True)
            raise
        finally:
            quarantine_root = quarantine_dir.resolve()
            if quarantine_root.is_relative_to(self.root):
                shutil.rmtree(quarantine_root, ignore_errors=True)

    def _require_page_images(self, conn: sqlite3.Connection) -> None:
        if not _table_present(conn, "page_image"):
            raise ReferenceError(
                "this project predates migration 0004_page_images.sql; run init to apply it"
            )

    def add_images(self, material_id: str) -> dict[str, Any]:
        """Extract only page images for an existing material.

        Text fragments and embeddings are deliberately untouched. This is the
        migration path for a PDF already imported before image extraction was
        enabled.
        """
        _validate_id(material_id, "mat")
        with closing(self._connect()) as conn:
            self._require_page_images(conn)
            row = conn.execute(
                """
                SELECT m.id AS material_id, m.title, mv.id AS version_id,
                       fb.sha256, fb.relative_path, fb.detected_type
                FROM material m
                JOIN material_version mv ON mv.material_id=m.id
                JOIN file_blob fb ON fb.id=mv.file_blob_id
                WHERE m.id=? AND m.deleted_at IS NULL
                ORDER BY mv.version_number DESC LIMIT 1
                """,
                (material_id,),
            ).fetchone()
        if row is None:
            raise ReferenceError("unknown material_id")
        blob_path = (self.root / row["relative_path"]).resolve()
        if not blob_path.is_relative_to(self.root) or not blob_path.is_file():
            raise ReferenceError("immutable blob is missing")
        if _sha256(blob_path) != row["sha256"]:
            raise ReferenceError("immutable blob failed SHA-256 verification")
        if row["detected_type"] != "pdf":
            return {
                "status": "READY",
                "material_id": material_id,
                "page_images": 0,
                "existing": 0,
                "new_files": 0,
                "image_bytes": 0,
            }

        from .processors import extract_pdf_images

        extracted_images = extract_pdf_images(blob_path)
        created_files: list[Path] = []
        image_bytes = 0
        image_paths: dict[str, str] = {}
        for image in extracted_images:
            relative, image_path = _image_path(self.root, image["sha256"], image["image_format"])
            image_paths[image["sha256"]] = relative.as_posix()
            if not image_path.is_file():
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(image["bytes"])
                created_files.append(image_path)
                image_bytes += image["byte_size"]

        new_images = 0
        existing_images = 0
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            fragment_ids_by_page = {
                int(json.loads(fragment["locator_json"])["page"]): fragment["id"]
                for fragment in conn.execute(
                    "SELECT id, locator_json FROM fragment WHERE material_version_id=?",
                    (row["version_id"],),
                ).fetchall()
                if json.loads(fragment["locator_json"]).get("page") is not None
            }
            for image in extracted_images:
                already = conn.execute(
                    "SELECT 1 FROM page_image WHERE material_version_id=? AND page=? AND ordinal=?",
                    (row["version_id"], image["page"], image["ordinal"]),
                ).fetchone()
                if already is not None:
                    existing_images += 1
                    continue
                conn.execute(
                    "INSERT INTO page_image(id, material_version_id, fragment_id, page, ordinal, "
                    "bbox_json, width, height, image_format, byte_size, sha256, relative_path, created_at) "
                    "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        _id("img"), row["version_id"], fragment_ids_by_page.get(image["page"]),
                        image["page"], image["ordinal"], json.dumps(image["bbox"], sort_keys=True),
                        image["width"], image["height"], image["image_format"], image["byte_size"],
                        image["sha256"], image_paths[image["sha256"]], _now(),
                    ),
                )
                new_images += 1
            conn.commit()
        return {
            "status": "READY",
            "material_id": material_id,
            "page_images": new_images,
            "existing": existing_images,
            "new_files": len(created_files),
            "image_bytes": image_bytes,
        }

    def _image_result(self, row: sqlite3.Row) -> dict[str, Any]:
        locator = json.loads(row["locator_json"]) if row["locator_json"] else {
            "type": "pdf-page-image", "page": row["page"]
        }
        relative = row["relative_path"]
        absolute = (self.root / relative).resolve()
        result = {
            "image_id": row["image_id"],
            "material_id": row["material_id"],
            "title": row["title"],
            "page": row["page"],
            "ordinal": row["ordinal"],
            "fragment_id": row["fragment_id"],
            "text": row["text"] or "",
            "locator": locator,
            "bbox": json.loads(row["bbox_json"]),
            "width": row["width"],
            "height": row["height"],
            "image_format": row["image_format"],
            "byte_size": row["byte_size"],
            "sha256": row["sha256"],
            "relative_path": relative,
            "absolute_path": str(absolute),
        }
        if "rank" in row.keys():
            result["rank"] = row["rank"]
        return result

    def search_images(
        self, query: str, *, limit: int = 20, exact: bool = False
    ) -> dict[str, Any]:
        """Find image objects through the text of their containing page."""
        if not query or not query.strip():
            raise ReferenceError("image search query must not be empty")
        match_mode = "exact" if exact else "full-text"
        bounded = max(1, min(int(limit), 100))
        with closing(self._connect()) as conn:
            self._require_page_images(conn)
            base = (
                "SELECT i.id AS image_id, i.fragment_id, i.page, i.ordinal, "
                "i.bbox_json, i.width, i.height, i.image_format, i.byte_size, i.sha256, i.relative_path, "
                "f.locator_json, f.text, m.id AS material_id, m.title "
                "FROM page_image i JOIN material_version mv ON mv.id=i.material_version_id "
                "JOIN material m ON m.id=mv.material_id "
                "LEFT JOIN fragment f ON f.id=i.fragment_id "
                "WHERE m.deleted_at IS NULL"
            )
            if exact:
                rows = conn.execute(base + " ORDER BY m.created_at, i.page, i.ordinal").fetchall()
                needle = query.casefold()
                matching = [row for row in rows if needle in (row["text"] or "").casefold()]
            else:
                match_query = _fts_match_query(query)
                if match_query is None:
                    matching = []
                else:
                    matching = conn.execute(
                        base + " AND i.fragment_id IN (SELECT fragment_id FROM fragment_fts WHERE fragment_fts MATCH ?) "
                        "ORDER BY m.created_at, i.page, i.ordinal",
                        (match_query,),
                    ).fetchall()
        results = [self._image_result(row) for row in matching[:bounded]]
        return {
            "mode": "images",
            "query": query,
            "match_mode": match_mode,
            "matched_images": len(matching),
            "returned": len(results),
            "results": results,
        }

    def show_image(self, image_id: str) -> dict[str, Any]:
        _validate_id(image_id, "img")
        with closing(self._connect()) as conn:
            self._require_page_images(conn)
            row = conn.execute(
                "SELECT i.id AS image_id, i.fragment_id, i.page, i.ordinal, i.bbox_json, i.width, i.height, "
                "i.image_format, i.byte_size, i.sha256, i.relative_path, f.locator_json, f.text, "
                "m.id AS material_id, m.title "
                "FROM page_image i JOIN material_version mv ON mv.id=i.material_version_id "
                "JOIN material m ON m.id=mv.material_id LEFT JOIN fragment f ON f.id=i.fragment_id "
                "WHERE i.id=? AND m.deleted_at IS NULL",
                (image_id,),
            ).fetchone()
        if row is None:
            raise ReferenceError("unknown image_id")
        result = self._image_result(row)
        path = Path(result["absolute_path"])
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ReferenceError("image file is missing")
        verified = _sha256(path) == row["sha256"]
        if not verified:
            raise ReferenceError("image file failed SHA-256 verification")
        result.update({"file_present": True, "sha256_verified": True})
        return result

    def search(self, query: str, *, limit: int = 20, exact: bool = False) -> list[dict[str, Any]]:
        """Search indexed fragments.

        exact=False (default) runs a SQLite FTS5 full-text query: tokens are
        matched independently, so "насос вибрирует" matches a fragment where
        those words are not adjacent.
        exact=True matches the query as one literal, contiguous, case-insensitive
        substring of the fragment text, which is what "find exact string"
        needs (and what FTS cannot express for text like "P-101").
        """
        if not query or not query.strip():
            return []
        if exact:
            return self._search_exact(query, limit)
        match_query = _fts_match_query(query)
        if match_query is None:
            return []
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT f.id AS fragment_id, f.text, f.locator_json, m.id AS material_id, m.title, "
                "fb.sha256 AS original_sha256, bm25(fragment_fts) AS rank "
                "FROM fragment_fts "
                "JOIN fragment f ON f.id=fragment_fts.fragment_id "
                "JOIN material_version mv ON mv.id=f.material_version_id "
                "JOIN material m ON m.id=mv.material_id "
                "JOIN file_blob fb ON fb.id=mv.file_blob_id "
                "WHERE fragment_fts MATCH ? AND m.deleted_at IS NULL "
                "ORDER BY rank LIMIT ?",
                (match_query, max(1, min(int(limit), 100))),
            ).fetchall()
        return [
            {
                "fragment_id": row["fragment_id"],
                "material_id": row["material_id"],
                "title": row["title"],
                "text": row["text"],
                "locator": json.loads(row["locator_json"]),
                "original_sha256": row["original_sha256"],
                "rank": row["rank"],
            }
            for row in rows
        ]

    def _search_exact(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Literal, contiguous, case-insensitive substring search over fragments.

        Implemented with a casefolded comparison in Python rather than SQL LIKE
        so that Cyrillic matches regardless of case and LIKE metacharacters
        (% and _) are treated as ordinary characters. This is a linear scan over
        indexed fragments; the FTS5 path stays the primary search for prose.
        """
        needle = query.casefold()
        bounded = max(1, min(int(limit), 100))
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT f.id AS fragment_id, f.text, f.locator_json, f.ordinal, "
                "m.id AS material_id, m.title, fb.sha256 AS original_sha256 "
                "FROM fragment f "
                "JOIN material_version mv ON mv.id=f.material_version_id "
                "JOIN material m ON m.id=mv.material_id "
                "JOIN file_blob fb ON fb.id=mv.file_blob_id "
                "WHERE m.deleted_at IS NULL "
                "ORDER BY m.created_at, mv.version_number, f.ordinal"
            ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            haystack = row["text"].casefold()
            position = haystack.find(needle)
            if position < 0:
                continue
            results.append({
                "fragment_id": row["fragment_id"],
                "material_id": row["material_id"],
                "title": row["title"],
                "text": row["text"],
                "locator": json.loads(row["locator_json"]),
                "original_sha256": row["original_sha256"],
                "match_mode": "exact",
                "match_start": position,
            })
            if len(results) >= bounded:
                break
        return results

    # -- embeddings -------------------------------------------------------

    def _embedding_spec(self) -> ModelSpec:
        if self.embedding_model is None:
            raise EmbeddingError(
                "no embedding model is configured; pass one to ReferenceService or use --embeddings"
            )
        return model_spec(self.embedding_model)

    def embed_fragments(self, limit: int | None = None) -> dict[str, Any]:
        """Embed every live fragment that has no vector for this model version.

        Safe to re-run: fragments already carrying a vector for this exact
        model version are skipped, so this can be called after every import
        without recomputing the whole corpus.
        """
        spec = self._embedding_spec()
        with closing(self._connect()) as conn:
            if not _table_present(conn, "embedding"):
                raise ReferenceError(
                    "this project predates migration 0003_embeddings.sql; run init to apply it"
                )
            pending = conn.execute(
                """
                SELECT f.id AS fragment_id, f.text AS text
                FROM fragment f
                JOIN material_version mv ON mv.id = f.material_version_id
                JOIN material m ON m.id = mv.material_id
                WHERE m.deleted_at IS NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM embedding e
                      WHERE e.fragment_id = f.id AND e.model_version = ?
                  )
                ORDER BY f.rowid
                """,
                (spec.model_version,),
            ).fetchall()
            already = conn.execute(
                "SELECT count(*) FROM embedding WHERE model_version=?", (spec.model_version,)
            ).fetchone()[0]
            fragments_total = conn.execute("SELECT count(*) FROM fragment").fetchone()[0]

        selected = pending[:limit] if limit is not None else pending
        vector_count = 0
        for start in range(0, len(selected), _EMBED_BATCH):
            batch = selected[start : start + _EMBED_BATCH]
            vectors = self.embedding_model.encode([row["text"] for row in batch])
            if len(vectors) != len(batch):
                raise EmbeddingError(
                    f"model returned {len(vectors)} vectors for {len(batch)} fragments"
                )
            with closing(self._connect()) as conn:
                conn.execute("BEGIN IMMEDIATE")
                for row, vector in zip(batch, vectors):
                    if len(vector) != spec.dimensions:
                        raise EmbeddingError(
                            f"model returned {len(vector)} dimensions, declared {spec.dimensions}"
                        )
                    blob = pack_vector(vector)
                    conn.execute(
                        """
                        INSERT INTO embedding(
                            id, fragment_id, model_id, model_version, pooling,
                            dimensions, vector, sha256, created_at
                        ) VALUES(?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(fragment_id, model_version) DO UPDATE SET
                            model_id=excluded.model_id,
                            pooling=excluded.pooling,
                            dimensions=excluded.dimensions,
                            vector=excluded.vector,
                            sha256=excluded.sha256,
                            created_at=excluded.created_at
                        """,
                        (
                            _id("emb"),
                            row["fragment_id"],
                            spec.model_id,
                            spec.model_version,
                            spec.pooling,
                            spec.dimensions,
                            blob,
                            vector_sha256(blob),
                            _now(),
                        ),
                    )
                    vector_count += 1
                conn.commit()

        return {
            **spec.as_dict(),
            "embedded": vector_count,
            "already_embedded": already,
            "pending": len(selected),
            "fragments": fragments_total,
        }

    def semantic_search(
        self, query: str, *, limit: int = 20, min_score: float | None = None
    ) -> dict[str, Any]:
        """Rank fragments by cosine similarity to the query.

        Vectors stored under another `model_version` are never compared: they
        live in a different vector space, so matching them would produce
        confident nonsense. With nothing embedded for this model the call
        refuses and says so, rather than returning an empty list that reads
        like "no such material".
        """
        spec = self._embedding_spec()
        if not isinstance(query, str) or not query.strip():
            raise EmbeddingError("semantic search needs a non-empty query")
        query_vector = self.embedding_model.encode([query])[0]
        if len(query_vector) != spec.dimensions:
            raise EmbeddingError(
                f"model returned {len(query_vector)} dimensions, declared {spec.dimensions}"
            )

        with closing(self._connect()) as conn:
            if not _table_present(conn, "embedding"):
                raise ReferenceError(
                    "this project predates migration 0003_embeddings.sql; run init to apply it"
                )
            rows = conn.execute(
                """
                SELECT e.id AS embedding_id, e.vector AS vector, e.dimensions AS dimensions,
                       f.id AS fragment_id, f.text AS text, f.locator_json AS locator_json,
                       mv.material_id AS material_id, m.title AS title,
                       fb.sha256 AS original_sha256
                FROM embedding e
                JOIN fragment f ON f.id = e.fragment_id
                JOIN material_version mv ON mv.id = f.material_version_id
                JOIN material m ON m.id = mv.material_id
                JOIN file_blob fb ON fb.id = mv.file_blob_id
                WHERE e.model_version = ? AND m.deleted_at IS NULL
                """,
                (spec.model_version,),
            ).fetchall()

        if not rows:
            raise EmbeddingError(
                f"no vectors are stored for {spec.model_version!r}; run embed first"
            )

        hits: list[dict[str, Any]] = []
        for row in rows:
            if row["dimensions"] != len(query_vector):
                raise EmbeddingError(
                    f"stored vector has {row['dimensions']} dimensions, query has {len(query_vector)}"
                )
            score = cosine_similarity(query_vector, unpack_vector(row["vector"], row["dimensions"]))
            if min_score is not None and score < min_score:
                continue
            hits.append(
                {
                    "fragment_id": row["fragment_id"],
                    "material_id": row["material_id"],
                    "title": row["title"],
                    "text": row["text"],
                    "locator": json.loads(row["locator_json"]),
                    "original_sha256": row["original_sha256"],
                    "score": score,
                    "mode": "semantic",
                }
            )

        hits.sort(key=lambda hit: (-hit["score"], hit["title"], hit["fragment_id"]))
        for position, hit in enumerate(hits, start=1):
            hit["rank"] = position

        return {
            **spec.as_dict(),
            "query": query,
            "mode": "semantic",
            "candidates": len(rows),
            "results": hits[:limit],
        }

    def show_source(self, fragment_id: str) -> dict[str, Any]:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT f.id AS fragment_id, f.locator_json, m.id AS material_id, m.title, "
                "fb.sha256 AS original_sha256, fb.relative_path AS blob_relative_path "
                "FROM fragment f "
                "JOIN material_version mv ON mv.id=f.material_version_id "
                "JOIN material m ON m.id=mv.material_id "
                "JOIN file_blob fb ON fb.id=mv.file_blob_id "
                "WHERE f.id=? AND m.deleted_at IS NULL",
                (fragment_id,),
            ).fetchone()
        if row is None:
            raise ReferenceError("unknown fragment_id")
        return {
            "fragment_id": row["fragment_id"],
            "material_id": row["material_id"],
            "title": row["title"],
            "locator": json.loads(row["locator_json"]),
            "original_sha256": row["original_sha256"],
            "blob_relative_path": row["blob_relative_path"],
        }

    def status(self) -> dict[str, Any]:
        with closing(self._connect()) as conn:
            counts = {
                "materials": conn.execute(
                    "SELECT count(*) FROM material WHERE deleted_at IS NULL"
                ).fetchone()[0],
                "file_blobs": conn.execute("SELECT count(*) FROM file_blob").fetchone()[0],
                "fragments": conn.execute("SELECT count(*) FROM fragment").fetchone()[0],
                "jobs": conn.execute("SELECT count(*) FROM job").fetchone()[0],
                "ocr_cache_entries": conn.execute("SELECT count(*) FROM ocr_cache").fetchone()[0],
                "ocr_engine": getattr(self.ocr_engine, "engine_id", None),
                "ocr_language": self.ocr_language,
                "embeddings": (
                    conn.execute("SELECT count(*) FROM embedding").fetchone()[0]
                    if _table_present(conn, "embedding")
                    else None
                ),
                "page_images": (
                    conn.execute("SELECT count(*) FROM page_image").fetchone()[0]
                    if _table_present(conn, "page_image")
                    else None
                ),
                "image_files": (
                    conn.execute("SELECT count(DISTINCT relative_path) FROM page_image").fetchone()[0]
                    if _table_present(conn, "page_image")
                    else None
                ),
                "image_bytes": (
                    conn.execute(
                        "SELECT coalesce(sum(byte_size), 0) FROM "
                        "(SELECT relative_path, max(byte_size) AS byte_size FROM page_image GROUP BY relative_path)"
                    ).fetchone()[0]
                    if _table_present(conn, "page_image")
                    else None
                ),
                "schema_current": _table_present(conn, "embedding") and _table_present(conn, "page_image"),
                "embedding_model": getattr(self.embedding_model, "model_id", None),
                "embedding_dimensions": getattr(self.embedding_model, "dimensions", None),
            }
        return {
            **counts,
            "used_bytes": _tree_size(self.root),
            "storage_root": str(self.root),
            "live_quota_gib": self.storage_quota["live_gib"],
            "temporary_quota_gib": self.storage_quota["temporary_gib"],
            "backup_quota_gib": self.storage_quota["backup_gib"],
            "minimum_free_disk_gib": self.storage_quota["minimum_free_disk_gib"],
            "backup_enabled": False,
            "mode": self.mode,
        }

    def delete_material(self, material_id: str, *, confirm: bool = False) -> dict[str, Any]:
        _validate_id(material_id, "mat")
        if not confirm:
            raise ReferenceError("explicit confirmation is required because deletion is irreversible without backup")
        blob_files: list[Path] = []
        cache_files: list[Path] = []
        image_files: list[Path] = []
        deleted_cache_entries = 0
        deleted_images = 0
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            material = conn.execute(
                "SELECT id, title FROM material WHERE id=? AND deleted_at IS NULL", (material_id,)
            ).fetchone()
            if material is None:
                conn.rollback()
                raise ReferenceError("unknown material_id")
            versions = conn.execute(
                "SELECT id, file_blob_id FROM material_version WHERE material_id=?", (material_id,)
            ).fetchall()
            version_ids = [row["id"] for row in versions]
            if version_ids:
                placeholders = ",".join("?" for _ in version_ids)
                if _table_present(conn, "page_image"):
                    image_rows = conn.execute(
                        f"SELECT relative_path FROM page_image WHERE material_version_id IN ({placeholders})",
                        version_ids,
                    ).fetchall()
                    for image_row in image_rows:
                        candidate = (self.root / image_row["relative_path"]).resolve()
                        if candidate.is_relative_to(self.root):
                            image_files.append(candidate)
                    deleted_images = conn.execute(
                        f"DELETE FROM page_image WHERE material_version_id IN ({placeholders})",
                        version_ids,
                    ).rowcount
            fragment_ids: list[str] = []
            run_ids: list[str] = []
            if version_ids:
                placeholders = ",".join("?" for _ in version_ids)
                fragment_ids = [row[0] for row in conn.execute(
                    f"SELECT id FROM fragment WHERE material_version_id IN ({placeholders})", version_ids
                )]
                run_ids = [row[0] for row in conn.execute(
                    f"SELECT id FROM processing_run WHERE material_version_id IN ({placeholders})", version_ids
                )]
            deleted_embeddings = 0
            if fragment_ids:
                placeholders = ",".join("?" for _ in fragment_ids)
                # Vectors must go before their fragments: with foreign keys on,
                # deleting a parent row that still has children is an error.
                conn.execute("SAVEPOINT delete_embeddings")
                deleted_embeddings = conn.execute(
                    f"DELETE FROM embedding WHERE fragment_id IN ({placeholders})", fragment_ids
                ).rowcount
                conn.execute("RELEASE delete_embeddings")
            for fragment_id in fragment_ids:
                conn.execute("DELETE FROM fragment_fts WHERE fragment_id=?", (fragment_id,))
            if fragment_ids:
                placeholders = ",".join("?" for _ in fragment_ids)
                conn.execute(f"DELETE FROM fragment WHERE id IN ({placeholders})", fragment_ids)
            if run_ids:
                placeholders = ",".join("?" for _ in run_ids)
                conn.execute(f"DELETE FROM artifact WHERE processing_run_id IN ({placeholders})", run_ids)
                conn.execute(f"DELETE FROM processing_run WHERE id IN ({placeholders})", run_ids)
            if version_ids:
                placeholders = ",".join("?" for _ in version_ids)
                conn.execute(f"DELETE FROM material_version WHERE id IN ({placeholders})", version_ids)
            conn.execute("DELETE FROM material WHERE id=?", (material_id,))
            removed_digests: list[str] = []
            for blob_id in {row["file_blob_id"] for row in versions}:
                remaining = conn.execute(
                    "SELECT count(*) FROM material_version WHERE file_blob_id=?", (blob_id,)
                ).fetchone()[0]
                if remaining == 0:
                    blob = conn.execute(
                        "SELECT relative_path, sha256 FROM file_blob WHERE id=?", (blob_id,)
                    ).fetchone()
                    if blob is not None:
                        candidate = (self.root / blob["relative_path"]).resolve()
                        if candidate.is_relative_to(self.root):
                            blob_files.append(candidate)
                        removed_digests.append(blob["sha256"])
                    conn.execute("DELETE FROM file_blob WHERE id=?", (blob_id,))
            if removed_digests:
                placeholders = ",".join("?" for _ in removed_digests)
                for cache_row in conn.execute(
                    f"SELECT relative_path FROM ocr_cache WHERE blob_sha256 IN ({placeholders})",
                    removed_digests,
                ):
                    candidate = (self.root / cache_row["relative_path"]).resolve()
                    if candidate.is_relative_to(self.root):
                        cache_files.append(candidate)
                deleted_cache_entries = conn.execute(
                    f"DELETE FROM ocr_cache WHERE blob_sha256 IN ({placeholders})", removed_digests
                ).rowcount
            conn.commit()

        artifact_root = (self.root / "artifacts" / material_id).resolve()
        if artifact_root.is_relative_to(self.root):
            shutil.rmtree(artifact_root, ignore_errors=True)
        for removable in (*blob_files, *cache_files):
            removable.unlink(missing_ok=True)
        deleted_image_files = 0
        for removable in set(image_files):
            relative = removable.relative_to(self.root).as_posix()
            with closing(self._connect()) as conn:
                still_used = (
                    conn.execute(
                        "SELECT 1 FROM page_image WHERE relative_path=? LIMIT 1", (relative,)
                    ).fetchone()
                    if _table_present(conn, "page_image")
                    else None
                )
            if still_used is None and removable.is_file():
                removable.unlink()
                deleted_image_files += 1
        report = {
            "status": "DELETED",
            "material_id": material_id,
            "deleted_fragments": len(fragment_ids),
            "deleted_embeddings": deleted_embeddings,
            "deleted_images": deleted_images,
            "deleted_image_files": deleted_image_files,
            "deleted_blobs": len(blob_files),
            "deleted_ocr_cache_entries": deleted_cache_entries,
            "irreversible_without_backup": True,
            "deleted_at": _now(),
        }
        report_path = self.root / "reports" / "deletion" / f"{material_id}.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report

    def reprocess(self, material_id: str) -> dict[str, Any]:
        _validate_id(material_id, "mat")
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT m.id AS material_id, m.title, mv.id AS version_id,
                       fb.sha256, fb.relative_path, fb.detected_type
                FROM material m
                JOIN material_version mv ON mv.material_id=m.id
                JOIN file_blob fb ON fb.id=mv.file_blob_id
                WHERE m.id=? AND m.deleted_at IS NULL
                ORDER BY mv.version_number DESC LIMIT 1
                """,
                (material_id,),
            ).fetchone()
        if row is None:
            raise ReferenceError("unknown material_id")
        blob_path = (self.root / row["relative_path"]).resolve()
        if not blob_path.is_relative_to(self.root) or not blob_path.is_file():
            raise ReferenceError("immutable blob is missing")
        if _sha256(blob_path) != row["sha256"]:
            raise ReferenceError("immutable blob failed SHA-256 verification")

        extracted = extract_file(
            blob_path,
            original_name=f"original.{row['detected_type']}",
            ocr=self._ocr_for_blob(row["sha256"]),
            ocr_language=self.ocr_language,
            images=True,
        )
        extracted_images = extracted.get("images", [])
        created_image_files: list[Path] = []
        image_file_bytes = 0
        image_paths: dict[str, str] = {}
        for image in extracted_images:
            relative, image_path = _image_path(self.root, image["sha256"], image["image_format"])
            image_paths[image["sha256"]] = relative.as_posix()
            if not image_path.is_file():
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(image["bytes"])
                created_image_files.append(image_path)
                image_file_bytes += image["byte_size"]
        run_id = _id("run")
        now = _now()
        artifact_dir = self.root / "artifacts" / material_id / row["version_id"] / run_id
        artifact_dir.mkdir(parents=True, exist_ok=False)
        normalized_path = artifact_dir / "normalized.txt"
        normalized_text = "\n\n".join(fragment["text"] for fragment in extracted["fragments"])
        normalized_path.write_text(normalized_text, encoding="utf-8")
        artifact_relative = normalized_path.relative_to(self.root).as_posix()

        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            old_fragment_ids = [r[0] for r in conn.execute(
                "SELECT id FROM fragment WHERE material_version_id=?", (row["version_id"],)
            )]
            old_image_paths: list[str] = []
            if _table_present(conn, "page_image"):
                old_image_paths = [
                    r[0] for r in conn.execute(
                        "SELECT relative_path FROM page_image WHERE material_version_id=?",
                        (row["version_id"],),
                    )
                ]
                conn.execute(
                    "DELETE FROM page_image WHERE material_version_id=?", (row["version_id"],)
                )
            for fragment_id in old_fragment_ids:
                conn.execute("DELETE FROM fragment_fts WHERE fragment_id=?", (fragment_id,))
                # Vectors go before their fragment: foreign keys are on.
                conn.execute("DELETE FROM embedding WHERE fragment_id=?", (fragment_id,))
            conn.execute("DELETE FROM fragment WHERE material_version_id=?", (row["version_id"],))
            conn.execute(
                "INSERT INTO processing_run(id, material_version_id, processor_id, processor_version, "
                "config_hash, status, started_at, finished_at) VALUES (?, ?, ?, ?, ?, 'SUCCESS', ?, ?)",
                (
                    run_id,
                    row["version_id"],
                    extracted["processor_id"],
                    extracted["processor_version"],
                    hashlib.sha256(b"{}").hexdigest(),
                    now,
                    now,
                ),
            )
            conn.execute(
                "INSERT INTO artifact(id, processing_run_id, kind, relative_path, sha256, created_at) "
                "VALUES (?, ?, 'normalized_text', ?, ?, ?)",
                (_id("art"), run_id, artifact_relative, _sha256(normalized_path), now),
            )
            fragment_ids_by_page: dict[int, str] = {}
            for ordinal, fragment in enumerate(extracted["fragments"]):
                fragment_id = _id("frag")
                locator = fragment["locator"]
                locator_json = json.dumps(locator, ensure_ascii=False, sort_keys=True)
                conn.execute(
                    "INSERT INTO fragment(id, material_version_id, ordinal, text, locator_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (fragment_id, row["version_id"], ordinal, fragment["text"], locator_json, now),
                )
                conn.execute(
                    "INSERT INTO fragment_fts(fragment_id, title, text) VALUES (?, ?, ?)",
                    (fragment_id, row["title"], fragment["text"]),
                )
                if locator.get("page") is not None:
                    fragment_ids_by_page[int(locator["page"])] = fragment_id
            for image in extracted_images:
                conn.execute(
                    "INSERT INTO page_image(id, material_version_id, fragment_id, page, ordinal, "
                    "bbox_json, width, height, image_format, byte_size, sha256, relative_path, created_at) "
                    "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        _id("img"),
                        row["version_id"],
                        fragment_ids_by_page.get(int(image["page"])),
                        int(image["page"]),
                        int(image["ordinal"]),
                        json.dumps(image["bbox"], sort_keys=True),
                        int(image["width"]),
                        int(image["height"]),
                        image["image_format"],
                        int(image["byte_size"]),
                        image["sha256"],
                        image_paths[image["sha256"]],
                        now,
                    ),
                )
            conn.execute("UPDATE material_version SET status='READY' WHERE id=?", (row["version_id"],))
            conn.commit()

        for relative in set(old_image_paths):
            candidate = (self.root / relative).resolve()
            if not candidate.is_relative_to(self.root) or not candidate.is_file():
                continue
            with closing(self._connect()) as conn:
                still_used = conn.execute(
                    "SELECT 1 FROM page_image WHERE relative_path=? LIMIT 1", (relative,)
                ).fetchone()
            if still_used is None:
                candidate.unlink(missing_ok=True)

        # Fragments were rebuilt from scratch, so their vectors are gone too.
        # Re-embed them here when a model is configured, otherwise the material
        # would silently drop out of semantic search until the next embed run.
        embedded = None
        if self.embedding_model is not None:
            embedded = self.embed_fragments()["embedded"]

        report = {
            "status": "READY",
            "material_id": material_id,
            "version_id": row["version_id"],
            "processing_run_id": run_id,
            "fragment_count": len(extracted["fragments"]),
            "page_images": len(extracted_images),
            "image_files": len(created_image_files),
            "image_bytes": image_file_bytes,
            "embedded": embedded,
            "source": "immutable_blob",
        }
        report_path = self.root / "reports" / "reprocess" / f"{run_id}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report

    def integrity_check(self) -> dict[str, Any]:
        issues: list[dict[str, Any]] = []
        manifest_path = self.root / "manifest.yaml"
        checksum_path = self.root / "manifest.sha256"
        if not manifest_path.is_file() or not checksum_path.is_file():
            issues.append({"code": "MANIFEST_MISSING"})
        else:
            expected = checksum_path.read_text(encoding="ascii").strip()
            actual = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            if actual != expected:
                issues.append({"code": "MANIFEST_HASH_MISMATCH", "expected": expected, "actual": actual})

        with closing(self._connect()) as conn:
            quick = conn.execute("PRAGMA quick_check").fetchone()[0]
            if quick != "ok":
                issues.append({"code": "SQLITE_QUICK_CHECK_FAILED", "detail": quick})
            for row in conn.execute("PRAGMA foreign_key_check"):
                issues.append({"code": "SQLITE_FOREIGN_KEY_ERROR", "detail": list(row)})
            blobs = conn.execute("SELECT sha256, relative_path FROM file_blob").fetchall()
            artifact_rows = conn.execute(
                "SELECT sha256, relative_path FROM artifact WHERE relative_path IS NOT NULL AND sha256 IS NOT NULL"
            ).fetchall()
            ocr_rows = conn.execute(
                "SELECT image_sha256, relative_path, sha256 FROM ocr_cache"
            ).fetchall()
            images_current = _table_present(conn, "page_image")
            image_rows = (
                conn.execute("SELECT id, sha256, relative_path FROM page_image").fetchall()
                if images_current
                else []
            )
            orphan_images = (
                conn.execute(
                    "SELECT count(*) FROM page_image i "
                    "WHERE i.fragment_id IS NOT NULL AND NOT EXISTS "
                    "(SELECT 1 FROM fragment f WHERE f.id=i.fragment_id)"
                ).fetchone()[0]
                if images_current
                else 0
            )
            if orphan_images:
                issues.append({"code": "IMAGE_ORPHAN", "count": orphan_images})
            schema_current = _table_present(conn, "embedding") and images_current
            embedding_rows = (
                conn.execute(
                    "SELECT id, fragment_id, dimensions, vector, sha256 FROM embedding"
                ).fetchall()
                if schema_current
                else []
            )
            orphan_embeddings = (
                conn.execute(
                    "SELECT count(*) FROM embedding e "
                    "WHERE NOT EXISTS (SELECT 1 FROM fragment f WHERE f.id = e.fragment_id)"
                ).fetchone()[0]
                if schema_current
                else 0
            )
            if orphan_embeddings:
                issues.append({"code": "EMBEDDING_ORPHAN", "count": orphan_embeddings})
            fragment_count = conn.execute("SELECT count(*) FROM fragment").fetchone()[0]
            fts_count = conn.execute("SELECT count(*) FROM fragment_fts").fetchone()[0]
            if fragment_count != fts_count:
                issues.append({
                    "code": "FTS_COUNT_MISMATCH",
                    "fragments": fragment_count,
                    "fts_rows": fts_count,
                })

        for row in blobs:
            path = (self.root / row["relative_path"]).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                issues.append({"code": "BLOB_MISSING", "sha256": row["sha256"]})
                continue
            actual = _sha256(path)
            if actual != row["sha256"]:
                issues.append({
                    "code": "BLOB_HASH_MISMATCH",
                    "expected": row["sha256"],
                    "actual": actual,
                })

        for row in image_rows:
            path = (self.root / row["relative_path"]).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                issues.append({"code": "IMAGE_MISSING", "image_id": row["id"], "sha256": row["sha256"]})
                continue
            actual = _sha256(path)
            if actual != row["sha256"]:
                issues.append({
                    "code": "IMAGE_HASH_MISMATCH",
                    "image_id": row["id"],
                    "expected": row["sha256"],
                    "actual": actual,
                })

        for row in artifact_rows:
            path = (self.root / row["relative_path"]).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                issues.append({"code": "ARTIFACT_MISSING", "relative_path": row["relative_path"]})
                continue
            actual = _sha256(path)
            if actual != row["sha256"]:
                issues.append({
                    "code": "ARTIFACT_HASH_MISMATCH",
                    "relative_path": row["relative_path"],
                    "expected": row["sha256"],
                    "actual": actual,
                })

        for row in ocr_rows:
            path = (self.root / row["relative_path"]).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                issues.append({"code": "OCR_CACHE_MISSING", "image_sha256": row["image_sha256"]})
                continue
            actual = _sha256(path)
            if actual != row["sha256"]:
                issues.append({
                    "code": "OCR_CACHE_HASH_MISMATCH",
                    "image_sha256": row["image_sha256"],
                    "relative_path": row["relative_path"],
                    "expected": row["sha256"],
                    "actual": actual,
                })

        for row in embedding_rows:
            blob = row["vector"]
            if len(blob) != row["dimensions"] * 4:
                issues.append(
                    {
                        "code": "EMBEDDING_LENGTH_MISMATCH",
                        "embedding_id": row["id"],
                        "dimensions": row["dimensions"],
                        "bytes": len(blob),
                    }
                )
                continue
            actual = vector_sha256(blob)
            if actual != row["sha256"]:
                issues.append(
                    {
                        "code": "EMBEDDING_HASH_MISMATCH",
                        "embedding_id": row["id"],
                        "expected": row["sha256"],
                        "actual": actual,
                    }
                )

        report = {
            "ok": not issues,
            "schema_current": schema_current,
            "issues": issues,
            "checked_at": _now(),
            "restore_available": False,
            "backup_enabled": False,
        }
        report_path = self.root / "reports" / "integrity" / "latest.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report
