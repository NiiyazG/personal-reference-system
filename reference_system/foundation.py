from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .config import DEFAULT_ALLOWED_PROFILE, DEFAULT_MODE, DEFAULT_STORAGE_QUOTA

DIRECTORIES = (
    "config",
    "db/migrations",
    "store/blobs/sha256",
    "store/images/sha256",
    "quarantine",
    "artifacts",
    "indexes/vector",
    "indexes/manifests",
    "indexes/ocr",
    "processors",
    "viewer",
    "reports/ingest",
    "reports/deletion",
    "reports/integrity",
    "reports/reprocess",
    "logs",
    "tests/approved-fixtures",
    "tests/expected",
)

# Mutable module-level default kept for callers that override it in place.
# The authoritative defaults live in `reference_system.config`; a project
# manifest records the values it was initialized with.
STORAGE_QUOTA = dict(DEFAULT_STORAGE_QUOTA)

INITIAL_MIGRATION_FILENAME = "0001_initial_schema.sql"
OCR_CACHE_MIGRATION_FILENAME = "0002_ocr_cache.sql"

OCR_CACHE_SQL = """PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS ocr_cache (
    image_sha256 TEXT NOT NULL,
    blob_sha256 TEXT NOT NULL,
    engine_id TEXT NOT NULL,
    engine_version TEXT NOT NULL,
    language TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (image_sha256, engine_id, engine_version, language)
);
CREATE INDEX IF NOT EXISTS ocr_cache_blob_idx ON ocr_cache(blob_sha256);
"""

EMBEDDINGS_MIGRATION_FILENAME = "0003_embeddings.sql"
PAGE_IMAGES_MIGRATION_FILENAME = "0004_page_images.sql"

PAGE_IMAGES_SQL = """PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS page_image (
    id TEXT PRIMARY KEY,
    material_version_id TEXT NOT NULL REFERENCES material_version(id),
    fragment_id TEXT REFERENCES fragment(id),
    page INTEGER NOT NULL CHECK(page > 0),
    ordinal INTEGER NOT NULL CHECK(ordinal > 0),
    bbox_json TEXT NOT NULL,
    width INTEGER NOT NULL CHECK(width > 0),
    height INTEGER NOT NULL CHECK(height > 0),
    image_format TEXT NOT NULL,
    byte_size INTEGER NOT NULL CHECK(byte_size >= 0),
    sha256 TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(material_version_id, page, ordinal)
);
CREATE INDEX IF NOT EXISTS page_image_fragment_idx ON page_image(fragment_id);
CREATE INDEX IF NOT EXISTS page_image_sha256_idx ON page_image(sha256);
"""

EMBEDDINGS_SQL = """PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS embedding (
    id TEXT PRIMARY KEY,
    fragment_id TEXT NOT NULL REFERENCES fragment(id),
    model_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    pooling TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK(dimensions > 0),
    vector BLOB NOT NULL,
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(fragment_id, model_version)
);
CREATE INDEX IF NOT EXISTS embedding_model_version_idx ON embedding(model_version);
CREATE INDEX IF NOT EXISTS embedding_fragment_idx ON embedding(fragment_id);
"""

SCHEMA_SQL = """PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS knowledge_base (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    mode TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ingest_event (
    id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT
);
CREATE TABLE IF NOT EXISTS file_blob (
    id TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL UNIQUE,
    size INTEGER NOT NULL CHECK(size >= 0),
    detected_type TEXT NOT NULL,
    relative_path TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS material (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    deleted_at TEXT
);
CREATE TABLE IF NOT EXISTS material_version (
    id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL REFERENCES material(id),
    file_blob_id TEXT NOT NULL REFERENCES file_blob(id),
    version_number INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(material_id, version_number)
);
CREATE TABLE IF NOT EXISTS processing_run (
    id TEXT PRIMARY KEY,
    material_version_id TEXT NOT NULL REFERENCES material_version(id),
    processor_id TEXT NOT NULL,
    processor_version TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    warning_json TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS artifact (
    id TEXT PRIMARY KEY,
    processing_run_id TEXT NOT NULL REFERENCES processing_run(id),
    kind TEXT NOT NULL,
    relative_path TEXT,
    sha256 TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fragment (
    id TEXT PRIMARY KEY,
    material_version_id TEXT NOT NULL REFERENCES material_version(id),
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    locator_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(material_version_id, ordinal)
);
CREATE VIRTUAL TABLE IF NOT EXISTS fragment_fts USING fts5(
    fragment_id UNINDEXED,
    title,
    text,
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TABLE IF NOT EXISTS job (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Canonical migration sources: (filename, SQL). `ensure_migration_files` writes
# these into a project's db/migrations directory; a committed export under
# schema/migrations is checked against them by the test suite.
MIGRATIONS = (
    (INITIAL_MIGRATION_FILENAME, SCHEMA_SQL),
    (OCR_CACHE_MIGRATION_FILENAME, OCR_CACHE_SQL),
    (EMBEDDINGS_MIGRATION_FILENAME, EMBEDDINGS_SQL),
    (PAGE_IMAGES_MIGRATION_FILENAME, PAGE_IMAGES_SQL),
)


def write_migration_files(destination: Path | str, *, overwrite: bool = False) -> list[Path]:
    """Write the canonical migration SQL into `destination`; return the paths."""
    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, sql in MIGRATIONS:
        path = directory / filename
        if overwrite or not path.is_file():
            path.write_text(sql, encoding="utf-8")
        written.append(path)
    return written


def _manifest(
    *,
    mode: str | None = None,
    storage_quota: Mapping[str, int] | None = None,
    allowed_profile: str | None = None,
) -> dict:
    return {
        "knowledge_base_id": str(uuid.uuid4()),
        "name": "Personal Reference",
        "schema_version": 1,
        "allowed_profile": allowed_profile or DEFAULT_ALLOWED_PROFILE,
        "root_layout_version": 1,
        "mode": mode or DEFAULT_MODE,
        "storage_quota": dict(storage_quota) if storage_quota is not None else dict(STORAGE_QUOTA),
        "backup_policy_id": "disabled-by-user",
        "backup_enabled": False,
        "fts_index_version": 1,
        "vector_index_version": None,
        "embedding_model": None,
        "created_at": _now(),
    }


class FoundationError(RuntimeError):
    pass


def migration_files(migrations_dir: Path | str) -> list[tuple[int, Path]]:
    directory = Path(migrations_dir)
    if not directory.is_dir():
        raise FoundationError(f"migrations directory not found: {directory}")
    found: list[tuple[int, Path]] = []
    for path in sorted(directory.glob("*.sql")):
        prefix = path.stem.split("_", 1)[0]
        if not prefix.isdigit():
            raise FoundationError(f"migration file must start with a numeric version: {path.name}")
        found.append((int(prefix), path))
    versions = [version for version, _ in found]
    if len(set(versions)) != len(versions):
        raise FoundationError("duplicate migration version detected")
    return sorted(found)


def applied_migration_versions(db_path: Path | str) -> list[int]:
    path = Path(db_path)
    if not path.is_file():
        return []
    with closing(sqlite3.connect(path)) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "schema_version" not in tables:
            return []
        return sorted(row[0] for row in conn.execute("SELECT version FROM schema_version"))


def apply_migrations(db_path: Path | str, migrations_dir: Path | str) -> list[int]:
    """Apply pending .sql migrations in version order; returns the versions applied."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_versions: list[int] = []
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        already = {row[0] for row in conn.execute("SELECT version FROM schema_version")}
        for version, migration_path in migration_files(migrations_dir):
            if version in already:
                continue
            conn.executescript(migration_path.read_text(encoding="utf-8"))
            conn.execute(
                "INSERT OR IGNORE INTO schema_version(version, applied_at) VALUES(?, ?)",
                (version, _now()),
            )
            conn.commit()
            new_versions.append(version)
    return new_versions


def ensure_migration_files(root: Path | str) -> Path:
    migrations_dir = Path(root) / "db" / "migrations"
    write_migration_files(migrations_dir)
    return migrations_dir


def initialize_project(
    root: Path | str,
    *,
    reset_database: bool = False,
    mode: str | None = None,
    storage_quota: Mapping[str, int] | None = None,
    allowed_profile: str | None = None,
) -> dict:
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for relative in DIRECTORIES:
        (root / relative).mkdir(parents=True, exist_ok=True)

    manifest_path = root / "manifest.yaml"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = _manifest(
            mode=mode, storage_quota=storage_quota, allowed_profile=allowed_profile
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (root / "manifest.sha256").write_text(digest + "\n", encoding="ascii")

    db_path = root / "db" / "reference.sqlite3"
    if reset_database and db_path.exists():
        db_path.unlink()
    migrations_dir = ensure_migration_files(root)
    apply_migrations(db_path, migrations_dir)

    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            "INSERT OR IGNORE INTO knowledge_base(id, name, mode, created_at) VALUES(?, ?, ?, ?)",
            (manifest["knowledge_base_id"], manifest["name"], manifest["mode"], manifest["created_at"]),
        )
        conn.commit()
    return manifest


def project_size_bytes(root: Path | str) -> int:
    """Total size of regular files under the project root, ignoring symlinks."""
    total = 0
    for item in Path(root).rglob("*"):
        if item.is_file() and not item.is_symlink():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def disk_free_bytes(root: Path | str) -> int:
    return shutil.disk_usage(Path(root)).free

