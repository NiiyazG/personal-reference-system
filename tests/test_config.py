import contextlib
import io
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from reference_system.cli import main as cli_main
from reference_system.config import ConfigError, load_config


@contextlib.contextmanager
def _environment(**values):
    """Set REFERENCE_* variables for one test and restore the environment."""
    saved = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class ConfigTests(unittest.TestCase):
    def test_defaults_are_generic_and_opt_out(self):
        with _environment(REFERENCE_CONFIG=None, REFERENCE_ROOT=None):
            config = load_config(root=Path("somewhere"))
        self.assertEqual(config.root, Path("somewhere"))
        self.assertEqual(config.mode, "LOCAL_ONLY")
        self.assertEqual(config.storage_quota, {
            "total_gib": 30,
            "live_gib": 28,
            "backup_gib": 0,
            "temporary_gib": 2,
            "minimum_free_disk_gib": 20,
        })
        self.assertEqual(config.ocr_engine, "none")
        self.assertEqual(config.embedding_model, "none")

    def test_environment_overrides_quotas_and_engines(self):
        with _environment(
            REFERENCE_LIVE_GIB="10",
            REFERENCE_TEMPORARY_GIB="1",
            REFERENCE_MODE="LOCAL_ONLY",
            REFERENCE_OCR_ENGINE="none",
            REFERENCE_EMBEDDING_MODEL="none",
        ):
            config = load_config(root=Path("somewhere"))
        self.assertEqual(config.storage_quota["live_gib"], 10)
        self.assertEqual(config.storage_quota["temporary_gib"], 1)

    def test_config_file_is_read_and_overridden_by_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "reference.config.json"
            config_path.write_text(json.dumps({
                "mode": "LOCAL_ONLY",
                "storage_quota": {"live_gib": 5},
                "ocr_engine": "none",
            }), encoding="utf-8")
            with _environment(REFERENCE_ROOT=None):
                config = load_config(
                    root=Path("somewhere"),
                    config_file=config_path,
                    storage_quota={"live_gib": 7},
                )
        self.assertEqual(config.storage_quota["live_gib"], 7)
        self.assertEqual(config.storage_quota["temporary_gib"], 2)

    def test_unknown_config_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "reference.config.json"
            config_path.write_text(json.dumps({"mystery": True}), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(config_file=config_path)


class ConfigCliTests(unittest.TestCase):
    def _run(self, argv):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = cli_main(argv)
        return code, buffer.getvalue()

    def test_init_records_custom_quota_and_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "kb"
            code, output = self._run([
                "--root", str(root),
                "--live-gib", "12",
                "--temporary-gib", "3",
                "--mode", "LOCAL_ONLY",
                "init",
            ])
            self.assertEqual(code, 0)
            payload = json.loads(output)
            self.assertEqual(payload["storage_quota"]["live_gib"], 12)
            self.assertEqual(payload["storage_quota"]["temporary_gib"], 3)
            self.assertEqual(payload["mode"], "LOCAL_ONLY")

            manifest = json.loads((root / "manifest.yaml").read_text(encoding="utf-8"))
            self.assertEqual(manifest["storage_quota"]["live_gib"], 12)
            self.assertEqual(manifest["storage_quota"]["temporary_gib"], 3)
            self.assertEqual(manifest["mode"], "LOCAL_ONLY")

            # status reports the mode recorded in the project manifest
            code, output = self._run(["--root", str(root), "status"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output)["mode"], "LOCAL_ONLY")

    def test_quota_override_reaches_enforcement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "kb"
            source = Path(tmp) / "note.txt"
            source.write_text("Насос в норме.\n", encoding="utf-8")
            self.assertEqual(self._run(["--root", str(root), "init"])[0], 0)

            code, output = self._run([
                "--root", str(root), "--temporary-gib", "0", "add", str(source)
            ])
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(output)["status"], "ERROR")

            code, _ = self._run(["--root", str(root), "add", str(source)])
            self.assertEqual(code, 0)

    def test_fresh_install_creates_an_empty_database_and_search_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "kb"
            self.assertEqual(self._run(["--root", str(root), "init"])[0], 0)

            with closing(sqlite3.connect(root / "db" / "reference.sqlite3")) as conn:
                materials = conn.execute("SELECT count(*) FROM material").fetchone()[0]
                fragments = conn.execute("SELECT count(*) FROM fragment").fetchone()[0]
            self.assertEqual(materials, 0)
            self.assertEqual(fragments, 0)

            code, output = self._run(["--root", str(root), "status"])
            self.assertEqual(code, 0)
            status = json.loads(output)
            self.assertEqual(status["materials"], 0)
            self.assertEqual(status["fragments"], 0)

            code, output = self._run(["--root", str(root), "search", "тест"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output)["results"], [])

    def test_root_can_come_from_the_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "kb"
            with _environment(REFERENCE_ROOT=str(root)):
                code, output = self._run(["init"])
            self.assertEqual(code, 0)
            self.assertTrue((root / "manifest.yaml").is_file())
            self.assertEqual(Path(json.loads(output)["root"]), root.resolve())


if __name__ == "__main__":
    unittest.main()
