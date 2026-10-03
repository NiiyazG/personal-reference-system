PRAGMA foreign_keys = ON;
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
