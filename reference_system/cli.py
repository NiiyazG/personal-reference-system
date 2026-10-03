from __future__ import annotations

import argparse
import json
import sqlite3
import sys

from .config import ConfigError, load_config
from .core import ReferenceError, ReferenceService
from .embeddings import EmbeddingError
from .foundation import initialize_project
from .ocr_windows import WindowsOcrEngine, WindowsOcrError
from .processors import UnsupportedFormatError

OCR_ENGINES = ("none", "windows")
EMBEDDING_MODELS = ("none", "bge-m3")


def _ocr_engine(name: str):
    """Build the OCR engine a command asked for. Opt-in: the default is none."""
    if name == "none":
        return None
    if name == "windows":
        return WindowsOcrEngine()
    raise ReferenceError(f"unknown OCR engine: {name!r} (available: {', '.join(OCR_ENGINES)})")


def _embedding_model(name: str):
    """Build the embedding model a command asked for. Opt-in: the default is none.

    `bge-m3` is resolved from the local Hugging Face cache by walking directories;
    the adapter has no Hub client and never downloads, so a missing model is a
    plain error rather than a silent fetch.
    """
    if name == "none":
        return None
    if name == "bge-m3":
        from .embeddings_onnx import load_default_model

        return load_default_model()
    raise ReferenceError(
        f"unknown embedding model: {name!r} (available: {', '.join(EMBEDDING_MODELS)})"
    )


def _emit(payload: object) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    sys.stdout.write(text + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reference",
        description="Personal reference system (local full-text and optional semantic search)",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="project root directory (default: $REFERENCE_ROOT or ./data)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="path to a JSON config file (default: $REFERENCE_CONFIG or ./reference.config.json)",
    )
    parser.add_argument(
        "--mode",
        default=None,
        help="operating mode recorded in the manifest and reported by status (default: LOCAL_ONLY)",
    )
    parser.add_argument("--ocr-engine", choices=OCR_ENGINES, default=None,
                        help="default OCR engine (default: none)")
    parser.add_argument("--ocr-language", default=None,
                        help="default OCR language tag (default: ru)")
    parser.add_argument("--embedding-model", choices=EMBEDDING_MODELS, default=None,
                        help="default embedding model (default: none)")
    parser.add_argument("--total-gib", type=int, default=None, help="total storage quota in GiB")
    parser.add_argument("--live-gib", type=int, default=None, help="live storage quota in GiB")
    parser.add_argument("--backup-gib", type=int, default=None, help="backup storage quota in GiB")
    parser.add_argument("--temporary-gib", type=int, default=None, help="temporary quota in GiB")
    parser.add_argument("--minimum-free-disk-gib", type=int, default=None,
                        help="minimum free disk space to keep, in GiB")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create the project layout, manifest and database")

    add = sub.add_parser("add", help="import one file into the immutable store")
    add.add_argument("path")
    add.add_argument("--title", default=None)
    add.add_argument(
        "--ocr",
        choices=OCR_ENGINES,
        default=None,
        help="recognise text on pages/images without a text layer (default: the configured engine)",
    )
    add.add_argument(
        "--no-images",
        action="store_true",
        help="skip embedded PDF image extraction (default: images are stored)",
    )

    images = sub.add_parser("images", help="extract images for an already imported material")
    images.add_argument("material_id")

    status = sub.add_parser("status", help="show counters, quotas and mode")
    status.add_argument(
        "--ocr",
        choices=OCR_ENGINES,
        default=None,
        help="report this engine in the output, to verify it is available",
    )
    status.add_argument(
        "--embeddings",
        choices=EMBEDDING_MODELS,
        default=None,
        help="report this embedding model in the output, to verify it is available",
    )

    search = sub.add_parser("search", help="search indexed fragments (full-text, exact or semantic)")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=20)
    search.add_argument(
        "--images",
        action="store_true",
        help="return image objects found through page text",
    )
    search.add_argument(
        "--exact",
        action="store_true",
        help="match the query as one literal contiguous substring instead of FTS5 tokens",
    )
    search.add_argument(
        "--semantic",
        action="store_true",
        help="rank by cosine similarity of stored vectors instead of matching words",
    )
    search.add_argument(
        "--embeddings",
        choices=EMBEDDING_MODELS,
        default=None,
        help="embedding model used for --semantic (default: the configured model)",
    )
    search.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="drop semantic hits below this cosine similarity",
    )

    show = sub.add_parser("show-source", help="resolve a fragment back to its original")
    show.add_argument("fragment_id")

    reprocess = sub.add_parser("reprocess", help="re-extract a material from its immutable blob")
    reprocess.add_argument("material_id")
    reprocess.add_argument(
        "--ocr",
        choices=OCR_ENGINES,
        default=None,
        help="recognise text on pages/images without a text layer (default: the configured engine)",
    )

    delete = sub.add_parser("delete", help="irreversibly delete a material")
    delete.add_argument("material_id")
    delete.add_argument("--confirm", action="store_true", help="required: deletion has no backup")

    sub.add_parser("integrity-check", help="verify manifest, database, blobs and indexes")

    embed = sub.add_parser("embed", help="store vectors for fragments that have none yet")
    embed.add_argument(
        "--embeddings",
        choices=EMBEDDING_MODELS,
        default=None,
        help="embedding model to use (default: the configured model)",
    )
    embed.add_argument("--limit", type=int, default=None, help="embed at most this many fragments")
    return parser


def _quota_overrides(args: argparse.Namespace) -> dict[str, int]:
    """Only the quota flags the caller actually set, ready for `load_config`."""
    mapping = {
        "total_gib": args.total_gib,
        "live_gib": args.live_gib,
        "backup_gib": args.backup_gib,
        "temporary_gib": args.temporary_gib,
        "minimum_free_disk_gib": args.minimum_free_disk_gib,
    }
    return {key: value for key, value in mapping.items() if value is not None}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(
            root=args.root,
            config_file=args.config,
            mode=args.mode,
            ocr_engine=args.ocr_engine,
            ocr_language=args.ocr_language,
            embedding_model=args.embedding_model,
            storage_quota=_quota_overrides(args) or None,
        )
        root = config.root
        if args.command == "init":
            manifest = initialize_project(
                root,
                mode=config.mode,
                storage_quota=config.storage_quota,
                allowed_profile=config.allowed_profile,
            )
            _emit({
                "status": "INITIALIZED",
                "root": str(root.resolve()),
                "knowledge_base_id": manifest["knowledge_base_id"],
                "mode": manifest["mode"],
                "storage_quota": manifest["storage_quota"],
                "backup_enabled": manifest["backup_enabled"],
            })
            return 0

        service = ReferenceService(
            root,
            ocr_engine=_ocr_engine(getattr(args, "ocr", None) or config.ocr_engine),
            ocr_language=config.ocr_language,
            embedding_model=_embedding_model(
                getattr(args, "embeddings", None) or config.embedding_model
            ),
            storage_quota=config.storage_quota,
            mode=config.mode,
        )
        if args.command == "add":
            _emit(service.add_file(args.path, title=args.title, images=not args.no_images))
            return 0
        if args.command == "images":
            _emit(service.add_images(args.material_id))
            return 0
        if args.command == "status":
            _emit(service.status())
            return 0
        if args.command == "search":
            if args.images:
                _emit(service.search_images(args.query, limit=args.limit, exact=args.exact))
                return 0
            if args.semantic:
                _emit(
                    service.semantic_search(
                        args.query, limit=args.limit, min_score=args.min_score
                    )
                )
                return 0
            _emit({
                "query": args.query,
                "mode": "exact" if args.exact else "full-text",
                "results": service.search(args.query, limit=args.limit, exact=args.exact),
            })
            return 0
        if args.command == "embed":
            _emit(service.embed_fragments(limit=args.limit))
            return 0
        if args.command == "show-source":
            _emit(service.show_source(args.fragment_id))
            return 0
        if args.command == "reprocess":
            _emit(service.reprocess(args.material_id))
            return 0
        if args.command == "delete":
            _emit(service.delete_material(args.material_id, confirm=args.confirm))
            return 0
        if args.command == "integrity-check":
            report = service.integrity_check()
            _emit(report)
            return 0 if report["ok"] else 2
    except (
        ReferenceError,
        UnsupportedFormatError,
        WindowsOcrError,
        EmbeddingError,
        ConfigError,
        sqlite3.Error,
    ) as exc:
        _emit({"status": "ERROR", "error": str(exc)})
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
