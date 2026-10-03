from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from .ocr import OcrEngine, supports_language


class UnsupportedFormatError(ValueError):
    pass


_TEXTUAL_PATTERN = re.compile(r"[\w\s\-.,;:!?()\[\]/\\+=%$#@'\"<>*&^~`|{}\u00a0]", re.UNICODE)

# Legacy Russian encodings in priority order. cp1251 dominates Windows-era
# documents, cp866 covers DOS-era ones; priority decides when both decode.
_LEGACY_ENCODINGS = ("cp1251", "cp866", "koi8-r", "cp1252")


def _looks_like_text(text: str) -> bool:
    """Guard against an encoding guess that decodes technically but yields garbage."""
    if not text:
        return False
    textual = sum(1 for character in text if _TEXTUAL_PATTERN.match(character))
    return textual / len(text) >= 0.9


def _decode_text(data: bytes) -> tuple[str, str]:
    """Decode bytes to text, returning the text and the encoding actually used.

    charset-normalizer alone is not trustworthy for legacy Russian data: on a
    cp1251 sample it guessed big5hkscs and returned Chinese mojibake without any
    error, which would have indexed the document as unsearchable garbage. So a
    fixed priority chain runs first, and the heuristic is kept only as a last
    resort whose output must still look like text.
    """
    if b"\x00" in data:
        raise UnsupportedFormatError("text input contains NUL bytes")
    try:
        return data.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        pass
    for encoding in _LEGACY_ENCODINGS:
        try:
            candidate = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        if _looks_like_text(candidate):
            return candidate, encoding
    try:
        from charset_normalizer import from_bytes

        match = from_bytes(data).best()
    except ImportError as exc:
        raise UnsupportedFormatError("input is not UTF-8 and charset-normalizer is unavailable") from exc
    if match is None:
        raise UnsupportedFormatError("unable to determine text encoding")
    guessed = str(match)
    if not _looks_like_text(guessed):
        raise UnsupportedFormatError(
            f"input is not UTF-8 and no usable legacy encoding was found (guessed {match.encoding})"
        )
    return guessed, match.encoding or "unknown"


def _page_images(doc: Any, page: Any, page_number: int) -> list[dict[str, Any]]:
    """Images actually placed on one page, with bytes exactly as the PDF embeds them."""
    collected: list[dict[str, Any]] = []
    for ordinal, info in enumerate(page.get_image_info(xrefs=True), start=1):
        xref = info.get("xref")
        if not xref:
            continue
        payload = doc.extract_image(int(xref))
        raw = payload["image"]
        bbox = info.get("bbox") or (0.0, 0.0, 0.0, 0.0)
        collected.append({
            "page": page_number,
            "ordinal": ordinal,
            "bbox": {
                "x0": float(bbox[0]),
                "y0": float(bbox[1]),
                "x1": float(bbox[2]),
                "y1": float(bbox[3]),
            },
            "width": int(payload["width"]),
            "height": int(payload["height"]),
            "image_format": str(payload["ext"]),
            "byte_size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": raw,
        })
    return collected


def extract_pdf_images(path: Path, *, max_images: int | None = None) -> list[dict[str, Any]]:
    """Every image placed in a PDF — no text, no OCR, no index touched.

    Occurrences, not unique pictures: the same image on two pages yields two
    entries with identical bytes, which is what lets the store keep one file and
    two records. A soft mask (alpha channel) is *not* baked into `bytes`: the
    base image is stored the way the PDF embeds it, without re-encoding.
    """
    import fitz

    data = path.read_bytes()
    if data[:5] != b"%PDF-":
        raise UnsupportedFormatError("PDF signature mismatch")
    collected: list[dict[str, Any]] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page_index, page in enumerate(doc):
            collected.extend(_page_images(doc, page, page_index + 1))
            if max_images is not None and len(collected) >= max_images:
                break
    return collected


def _extract_pdf(
    path: Path, *, ocr: OcrEngine | None = None, language: str = "ru", images: bool = True
) -> dict[str, Any]:
    import io

    import fitz

    data = path.read_bytes()
    if data[:5] != b"%PDF-":
        raise UnsupportedFormatError("PDF signature mismatch")
    fragments: list[dict[str, Any]] = []
    page_images: list[dict[str, Any]] = []
    ocr_pages = 0
    ocr_active = supports_language(ocr, language)
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page_index, page in enumerate(doc):
            if images:
                page_images.extend(_page_images(doc, page, page_index + 1))
            blocks = page.get_text("blocks")
            text_parts = [block[4].strip() for block in blocks if block[4].strip()]
            if text_parts:
                fragments.append({
                    "text": "\n".join(text_parts),
                    "locator": {
                        "type": "pdf-page",
                        "page": page_index + 1,
                        "blocks": [
                            {"x0": block[0], "y0": block[1], "x1": block[2], "y1": block[3]}
                            for block in blocks if block[4].strip()
                        ],
                        "ocr_performed": False,
                    },
                })
                continue
            if not ocr_active:
                continue
            image_bytes = page.get_pixmap(dpi=200).tobytes("png")
            result = ocr.recognize(image_bytes, language=language)
            text = result.text.strip()
            if not text:
                continue
            ocr_pages += 1
            fragments.append({
                "text": text,
                "locator": {
                    "type": "pdf-ocr-page",
                    "page": page_index + 1,
                    "ocr_performed": True,
                    "engine": result.engine_id,
                    "engine_version": result.engine_version,
                    "language": result.language,
                    "lines": result.lines,
                    "ocr_notes": list(result.notes),
                },
            })
        page_count = doc.page_count
    if not fragments:
        if ocr_active and ocr_pages == 0:
            raise UnsupportedFormatError("PDF has no usable text layer and OCR returned no text")
        raise UnsupportedFormatError(
            "PDF has no usable text layer and no OCR engine is configured; configure an engine to index it"
        )
    return {
        "detected_type": "pdf",
        "processor_id": "pymupdf-text-layer" if ocr_pages == 0 else "pymupdf-text-layer+ocr",
        "processor_version": getattr(fitz, "VersionBind", "unknown"),
        "fragments": fragments,
        "images": page_images,
        "metadata": {
            "page_count": page_count,
            "text_layer_pages": len(fragments) - ocr_pages,
            "ocr_pages": ocr_pages,
            "image_count": len(page_images),
        },
    }


def _extract_docx(path: Path) -> dict[str, Any]:
    import io
    import zipfile

    import docx

    data = path.read_bytes()
    if data[:4] != b"PK\x03\x04":
        raise UnsupportedFormatError("DOCX container signature mismatch")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as package:
            names = set(package.namelist())
    except zipfile.BadZipFile as exc:
        raise UnsupportedFormatError("invalid DOCX package: not a readable ZIP container") from exc
    if "[Content_Types].xml" not in names or "word/document.xml" not in names:
        raise UnsupportedFormatError("file is not a DOCX package")
    try:
        document = docx.Document(io.BytesIO(data))
    except (KeyError, ValueError) as exc:
        raise UnsupportedFormatError("invalid DOCX package") from exc

    fragments: list[dict[str, Any]] = []
    for index, paragraph in enumerate(document.paragraphs, start=1):
        text = paragraph.text.strip()
        if text:
            fragments.append({
                "text": text,
                "locator": {"type": "docx-paragraph", "paragraph": index},
            })
    for table_index, table in enumerate(document.tables, start=1):
        for row_index, row in enumerate(table.rows, start=1):
            for column_index, cell in enumerate(row.cells, start=1):
                text = cell.text.strip()
                if text:
                    fragments.append({
                        "text": text,
                        "locator": {
                            "type": "docx-table-cell",
                            "table": table_index,
                            "row": row_index,
                            "column": column_index,
                        },
                    })
    return {
        "detected_type": "docx",
        "processor_id": "python-docx",
        "processor_version": getattr(docx, "__version__", "unknown"),
        "fragments": fragments,
        "metadata": {
            "paragraph_count": len(document.paragraphs),
            "table_count": len(document.tables),
        },
    }


def _extract_xlsx(path: Path) -> dict[str, Any]:
    import io
    import zipfile

    import openpyxl

    data = path.read_bytes()
    if data[:4] != b"PK\x03\x04":
        raise UnsupportedFormatError("XLSX container signature mismatch")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as package:
            names = set(package.namelist())
    except zipfile.BadZipFile as exc:
        raise UnsupportedFormatError("invalid XLSX package: not a readable ZIP container") from exc
    if "[Content_Types].xml" not in names or "xl/workbook.xml" not in names:
        raise UnsupportedFormatError("file is not an XLSX package")
    try:
        workbook = openpyxl.load_workbook(
            io.BytesIO(data), read_only=True, data_only=False, keep_links=False
        )
    except (KeyError, ValueError) as exc:
        raise UnsupportedFormatError("invalid XLSX package") from exc

    fragments: list[dict[str, Any]] = []
    sheet_names = list(workbook.sheetnames)
    try:
        for sheet in workbook.worksheets:
            for row in sheet.iter_rows():
                for cell in row:
                    if cell.value is None:
                        continue
                    value = cell.value.isoformat() if hasattr(cell.value, "isoformat") else str(cell.value)
                    if value.strip():
                        fragments.append({
                            "text": value,
                            "locator": {
                                "type": "xlsx-cell",
                                "sheet": sheet.title,
                                "cell": cell.coordinate,
                            },
                        })
    finally:
        workbook.close()
    return {
        "detected_type": "xlsx",
        "processor_id": "openpyxl",
        "processor_version": getattr(openpyxl, "__version__", "unknown"),
        "fragments": fragments,
        "metadata": {"sheets": sheet_names, "cell_count": len(fragments)},
    }


def _extract_csv(path: Path, *, delimiter: str | None = None) -> dict[str, Any]:
    import csv
    import io

    text, encoding = _decode_text(path.read_bytes())
    if delimiter is not None:
        dialect = type("FixedDialect", (csv.excel,), {"delimiter": delimiter})
    else:
        sample = text[:8192]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
    fragments: list[dict[str, Any]] = []
    reader = csv.reader(io.StringIO(text), dialect)
    row_count = 0
    for row_index, row in enumerate(reader, start=1):
        row_count = row_index
        for column_index, value in enumerate(row, start=1):
            value = value.strip()
            if value:
                fragments.append({
                    "text": value,
                    "locator": {
                        "type": "csv-cell",
                        "row": row_index,
                        "column": column_index,
                    },
                })
    return {
        "detected_type": "csv",
        "processor_id": "csv-stdlib",
        "processor_version": "1",
        "fragments": fragments,
        "metadata": {"delimiter": dialect.delimiter, "row_count": row_count, "encoding": encoding},
    }


def _image_dimensions(data: bytes, kind: str) -> tuple[int, int]:
    import struct

    if kind == "png":
        if len(data) < 24 or data[12:16] != b"IHDR":
            raise UnsupportedFormatError("PNG is missing an IHDR header")
        width, height = struct.unpack(">II", data[16:24])
        return width, height
    index = 2
    while index + 4 <= len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        length = struct.unpack(">H", data[index + 2:index + 4])[0]
        if length < 2:
            break
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if index + 9 > len(data):
                break
            height, width = struct.unpack(">HH", data[index + 5:index + 9])
            return width, height
        index += 2 + length
    raise UnsupportedFormatError("JPEG frame header not found")


def _extract_image(
    path: Path, kind: str, *, ocr: OcrEngine | None = None, language: str = "ru"
) -> dict[str, Any]:
    data = path.read_bytes()
    if kind == "png":
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise UnsupportedFormatError("PNG signature mismatch")
    elif not data.startswith(b"\xff\xd8\xff"):
        raise UnsupportedFormatError("JPEG signature mismatch")
    width, height = _image_dimensions(data, kind)
    if width <= 0 or height <= 0:
        raise UnsupportedFormatError("image dimensions are not positive")
    metadata: dict[str, Any] = {
        "format": kind,
        "width": width,
        "height": height,
        "byte_size": len(data),
        "ocr_performed": False,
        "text_layer": None,
    }
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            metadata["mode"] = image.mode
    except Exception:
        metadata["mode"] = None

    ocr_active = supports_language(ocr, language)
    metadata["ocr_engine"] = getattr(ocr, "engine_id", None) if ocr_active else None
    if ocr_active:
        result = ocr.recognize(data, language=language)
        text = result.text.strip()
        if not text:
            metadata["ocr_performed"] = True
            return {
                "detected_type": kind,
                "processor_id": f"image-ocr:{result.engine_id}",
                "processor_version": result.engine_version,
                "fragments": [],
                "metadata": metadata,
            }
        metadata["ocr_performed"] = True
        return {
            "detected_type": kind,
            "processor_id": f"image-ocr:{result.engine_id}",
            "processor_version": result.engine_version,
            "fragments": [
                {
                    "text": text,
                    "locator": {
                        "type": "image-ocr",
                        "width": width,
                        "height": height,
                        "format": kind,
                        "ocr_performed": True,
                        "engine": result.engine_id,
                        "engine_version": result.engine_version,
                        "language": result.language,
                        "lines": result.lines,
                        "ocr_notes": list(result.notes),
                    },
                }
            ],
            "metadata": metadata,
        }

    return {
        "detected_type": kind,
        "processor_id": "image-header-metadata",
        "processor_version": "1",
        "fragments": [
            {
                "text": f"Image file {path.name} ({kind}, {width}x{height} pixels)",
                "locator": {
                    "type": "image-file",
                    "width": width,
                    "height": height,
                    "format": kind,
                    "ocr_performed": False,
                },
            }
        ],
        "metadata": metadata,
    }


def _with_images(result: dict[str, Any]) -> dict[str, Any]:
    """Formats that carry no page images report an empty list, never a missing key."""
    result.setdefault("images", [])
    return result


def extract_file(
    path: Path,
    *,
    original_name: str | None = None,
    ocr: OcrEngine | None = None,
    ocr_language: str = "ru",
    images: bool = True,
) -> dict[str, Any]:
    """Extract text fragments and, for PDFs, the embedded page images.

    `images=False` skips image extraction entirely; every other format reports an
    empty list, so callers never have to test for the key's presence.
    """
    suffix = Path(original_name).suffix.lower() if original_name else path.suffix.lower()
    if suffix == ".pdf":
        return _with_images(_extract_pdf(path, ocr=ocr, language=ocr_language, images=images))
    if suffix == ".docx":
        return _with_images(_extract_docx(path))
    if suffix == ".xlsx":
        return _with_images(_extract_xlsx(path))
    if suffix == ".csv":
        return _with_images(_extract_csv(path))
    if suffix in (".tsv", ".tab"):
        return _with_images(_extract_csv(path, delimiter="\t"))
    if suffix == ".png":
        return _with_images(_extract_image(path, "png", ocr=ocr, language=ocr_language))
    if suffix in (".jpg", ".jpeg"):
        return _with_images(_extract_image(path, "jpeg", ocr=ocr, language=ocr_language))
    if suffix != ".txt":
        raise UnsupportedFormatError(f"unsupported format: {suffix or '<none>'}")
    data = path.read_bytes()
    decoded, encoding = _decode_text(data)
    text = decoded.replace("\r\n", "\n").replace("\r", "\n")
    line_count = max(1, len(text.splitlines()))
    fragments = []
    if text.strip():
        fragments.append({
            "text": text,
            "locator": {
                "type": "text-lines",
                "line_start": 1,
                "line_end": line_count,
            },
        })
    return _with_images({
        "detected_type": "txt",
        "processor_id": "txt-stdlib",
        "processor_version": "1",
        "fragments": fragments,
        "metadata": {"encoding": encoding, "newlines_normalized": True, "line_count": line_count},
    })
