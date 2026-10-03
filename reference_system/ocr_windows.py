"""Windows OCR adapter, called in-process through the `winsdk` WinRT projection.

Recognition runs inside this process: nothing is spawned, nothing is downloaded
at run time. The recogniser itself ships with Windows, and `winsdk` is the only
extra package — chosen over a PowerShell call precisely so that
`tools/security_scan.py` can keep banning `subprocess` outright.

Boundaries this adapter does not cross, on purpose:

* It never decides which text is "good enough". Whatever the engine returns is
  recorded verbatim, together with the engine id, its version, the canonical
  language tag and any note about how the image had to be prepared.
* It never silently degrades. An image larger than the engine limit is
  downscaled and the fact is reported in `OcrResult.notes`; an undecodable
  image raises instead of contributing an empty fragment.

Known limitations of this engine, measured on synthetic fixtures:

* The Russian recogniser maps Latin lookalikes onto Cyrillic: `P-101` comes back
  as `Р-1О1`. A later search for the Latin code therefore misses; the text is
  stored as the engine produced it, and the mismatch is the caller's to handle.
* Line text is assembled from word items, which the engine returns without
  separators — joining with a space is required, not cosmetic.
* `OcrWord` exposes no confidence value, so none is reported.
"""

from __future__ import annotations

import asyncio
import io
import platform
from typing import Any

from .ocr import OcrResult

ENGINE_ID = "windows-media-ocr"
DEFAULT_LANGUAGES: tuple[str, ...] = ("ru", "en-US")


class WindowsOcrError(RuntimeError):
    """The Windows OCR engine could not be used, or failed while running."""


class WindowsOcrUnavailableError(WindowsOcrError):
    """The WinRT projection (`winsdk`) or a recogniser for the language is absent."""


def _winrt() -> dict[str, Any]:
    """Import the WinRT projections lazily, so this module imports without them."""
    try:
        from winsdk.windows.globalization import Language
        from winsdk.windows.graphics.imaging import BitmapDecoder
        from winsdk.windows.media.ocr import OcrEngine
        from winsdk.windows.storage.streams import DataWriter, InMemoryRandomAccessStream
    except ImportError as exc:  # pragma: no cover - depends on the host
        raise WindowsOcrUnavailableError(
            "winsdk is not installed, so Windows OCR cannot run in-process; "
            "install it, or run without an OCR engine"
        ) from exc
    return {
        "Language": Language,
        "BitmapDecoder": BitmapDecoder,
        "OcrEngine": OcrEngine,
        "DataWriter": DataWriter,
        "InMemoryRandomAccessStream": InMemoryRandomAccessStream,
    }


def _engine_version() -> str:
    """A version string that changes whenever the recogniser might have changed.

    Windows exposes no model version, so the winsdk projection plus the OS build
    is the closest honest proxy: a Windows update that replaces the OCR model
    changes the build number and therefore invalidates the cache.
    """
    from importlib import metadata

    try:
        winsdk_version = metadata.version("winsdk")
    except metadata.PackageNotFoundError:  # pragma: no cover - defensive
        winsdk_version = "unknown"
    return f"winsdk-{winsdk_version};windows-{platform.version()}"


class WindowsOcrEngine:
    """Real recognition through `Windows.Media.Ocr`."""

    engine_id = ENGINE_ID
    is_test_double = False

    def __init__(self, *, languages: tuple[str, ...] = DEFAULT_LANGUAGES) -> None:
        self._winrt = _winrt()
        self.engine_version = _engine_version()
        self.max_image_dimension = int(self._winrt["OcrEngine"].max_image_dimension)
        available = tuple(tag for tag in languages if self.supports(tag))
        if not available:
            raise WindowsOcrUnavailableError(
                "Windows OCR is installed but has no recogniser for any of: "
                + ", ".join(languages)
            )
        # What this engine advertises as usable; `supports` answers for any tag.
        self.languages = available

    def supports(self, language: str) -> bool:
        try:
            return bool(
                self._winrt["OcrEngine"].is_language_supported(self._winrt["Language"](str(language)))
            )
        except (ValueError, OSError):
            return False

    def recognize(self, image_bytes: bytes, *, language: str) -> OcrResult:
        if not image_bytes:
            raise WindowsOcrError("empty image passed to OCR")
        data, notes = self._prepare(bytes(image_bytes))
        try:
            return asyncio.run(self._recognize_async(data, language, notes))
        except RuntimeError as exc:  # a running event loop in the calling thread
            if "asyncio.run()" not in str(exc):
                raise
            raise WindowsOcrError(
                "Windows OCR cannot be called from a thread that already runs an event loop"
            ) from exc

    async def _recognize_async(
        self, data: bytes, language: str, notes: list[str]
    ) -> OcrResult:
        winrt = self._winrt
        engine = winrt["OcrEngine"].try_create_from_language(winrt["Language"](str(language)))
        if engine is None:
            raise WindowsOcrError(f"Windows OCR has no recogniser for language {language!r}")
        # The engine reports the canonical tag (`ru-RU` -> `ru`): cache on that.
        canonical = engine.recognizer_language.language_tag

        stream = winrt["InMemoryRandomAccessStream"]()
        writer = winrt["DataWriter"](stream)
        writer.write_bytes(data)
        await writer.store_async()
        await writer.flush_async()
        stream.seek(0)
        decoder = await winrt["BitmapDecoder"].create_async(stream)
        bitmap = await decoder.get_software_bitmap_async()
        result = await engine.recognize_async(bitmap)

        lines: list[dict[str, Any]] = []
        for line in result.lines:
            # WinRT projected vectors reject negative indexes: materialise first.
            words = list(line.words)
            if not words:
                continue
            first, last = words[0], words[-1]
            lines.append(
                {
                    "text": " ".join(word.text for word in words),
                    "left": int(first.bounding_rect.x),
                    "top": int(first.bounding_rect.y),
                    "right": int(last.bounding_rect.x + last.bounding_rect.width),
                    "bottom": int(last.bounding_rect.y + last.bounding_rect.height),
                }
            )

        return OcrResult(
            text="\n".join(line["text"] for line in lines),
            lines=lines,
            engine_id=self.engine_id,
            engine_version=self.engine_version,
            language=canonical,
            notes=notes,
        )

    def _prepare(self, data: bytes) -> tuple[bytes, list[str]]:
        """Return bytes the engine accepts, plus notes on any change made."""
        from PIL import Image

        try:
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
                if max(width, height) <= self.max_image_dimension:
                    return data, []
                scale = self.max_image_dimension / max(width, height)
                size = (max(1, int(width * scale)), max(1, int(height * scale)))
                resized = image.convert("RGB").resize(size, Image.LANCZOS)
        except (OSError, ValueError) as exc:
            raise WindowsOcrError(f"cannot decode image for OCR: {exc}") from exc

        buffer = io.BytesIO()
        resized.save(buffer, format="PNG")
        note = (
            f"downscaled from {width}x{height} to {size[0]}x{size[1]} "
            f"to fit the engine limit of {self.max_image_dimension}px"
        )
        return buffer.getvalue(), [note]
