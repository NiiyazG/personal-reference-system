PRAGMA foreign_keys = ON;
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
