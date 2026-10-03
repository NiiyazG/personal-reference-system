PRAGMA foreign_keys = ON;
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
