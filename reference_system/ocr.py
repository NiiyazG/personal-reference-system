"""OCR engine interface and test double.

The engine is injected, never imported implicitly: `reference_system` ships no
engine adapter in this increment, so nothing here installs software, spawns a
process, or reaches the network. A real adapter (Windows OCR, Tesseract, ...)
is registered from outside by implementing `OcrEngine` and passing an instance
to `ReferenceService`.

Every `OcrResult` must carry the engine id and version that produced it, so a
later reader can tell which engine produced a given text and invalidate the
cache when the engine changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


@dataclass(frozen=True)
class OcrResult:
    """Recognised text plus the provenance needed to trust and cache it."""

    text: str
    lines: list[dict[str, Any]]
    engine_id: str
    engine_version: str
    language: str
    # Anything a reader must know about how this text was obtained, e.g. an
    # image that had to be downscaled to fit the engine's limit. Never silent.
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "lines": self.lines,
            "engine_id": self.engine_id,
            "engine_version": self.engine_version,
            "language": self.language,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OcrResult":
        return cls(
            text=payload["text"],
            lines=list(payload.get("lines", [])),
            engine_id=payload["engine_id"],
            engine_version=payload["engine_version"],
            language=payload["language"],
            notes=list(payload.get("notes", [])),
        )


@runtime_checkable
class OcrEngine(Protocol):
    """What a real engine adapter must provide.

    `recognize` receives the raw bytes of one image (a whole image file, or one
    rasterised PDF page) and the requested language tag, and returns an
    `OcrResult` whose engine fields must echo the adapter's own identity.
    """

    engine_id: str
    engine_version: str
    languages: tuple[str, ...]

    def supports(self, language: str) -> bool: ...

    def recognize(self, image_bytes: bytes, *, language: str) -> OcrResult: ...


def supports_language(engine: OcrEngine | None, language: str) -> bool:
    """True when the engine is present and declares the requested language."""
    if engine is None:
        return False
    try:
        return bool(engine.supports(language))
    except AttributeError:
        return language in tuple(getattr(engine, "languages", ()))


class StubOcrEngine:
    """A test double, never a recogniser.

    It returns exactly what the test hands it, so the pipeline can be verified
    without any engine present. `is_test_double` marks it so that nothing can
    mistake its output for real recognition.
    """

    is_test_double = True

    def __init__(
        self,
        *,
        engine_id: str,
        engine_version: str,
        languages: tuple[str, ...],
        respond: Callable[[bytes, str], OcrResult],
        fail_on_call: bool = False,
    ) -> None:
        self.engine_id = engine_id
        self.engine_version = engine_version
        self.languages = tuple(languages)
        self.respond = respond
        self._fail_on_call = fail_on_call

    def supports(self, language: str) -> bool:
        return language in self.languages

    def recognize(self, image_bytes: bytes, *, language: str) -> OcrResult:
        if self._fail_on_call:
            raise RuntimeError("stub engine was told to fail")
        return self.respond(image_bytes, language)
