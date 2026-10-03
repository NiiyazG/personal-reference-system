PRAGMA foreign_keys = ON;
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
