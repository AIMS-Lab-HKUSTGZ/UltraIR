"""Responsibly download requested IR JCAMP-DX records from NIST WebBook.

The command does not enumerate or crawl NIST records. Users provide an ID list
and remain responsible for the WebBook terms of use and dataset citation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Protocol
from urllib.parse import urlencode

from .http_client import RateLimitedClient


NIST_ENDPOINT = "https://webbook.nist.gov/cgi/cbook.cgi"
NIST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
JCAMP_HEADER_RE = re.compile(r"^##\s*JCAMP-DX\s*=", re.IGNORECASE | re.MULTILINE)
TITLE_RE = re.compile(r"^##\s*TITLE\s*=", re.IGNORECASE | re.MULTILINE)
DATA_RE = re.compile(r"^##\s*(?:XYDATA|PEAK TABLE)\s*=", re.IGNORECASE | re.MULTILINE)


class TextClient(Protocol):
    def get_text(self, url: str, *, accept: str = "text/plain") -> str: ...


def normalize_nist_id(value: str) -> str:
    identifier = value.strip()
    if Path(identifier).suffix.lower() in {".jdx", ".dx", ".txt", ".npy"}:
        identifier = Path(identifier).stem
    if not NIST_ID_RE.fullmatch(identifier):
        raise ValueError(f"invalid NIST WebBook ID: {value!r}")
    return identifier


def load_ids(source: Path) -> list[str]:
    """Load IDs from one-ID-per-line text or from supported file stems."""
    if source.is_dir():
        values = [
            path.stem
            for path in source.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".jdx", ".dx", ".txt", ".npy"}
            and ".ipynb_checkpoints" not in path.parts
        ]
    elif source.is_file():
        values = [
            line.strip()
            for line in source.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        raise FileNotFoundError(source)
    identifiers = sorted({normalize_nist_id(value) for value in values})
    if not identifiers:
        raise ValueError(f"no NIST IDs found in {source}")
    return identifiers


def nist_jcamp_url(identifier: str) -> str:
    query = urlencode({"JCAMP": normalize_nist_id(identifier), "Type": "IR"})
    return f"{NIST_ENDPOINT}?{query}"


def validate_jcamp(text: str, identifier: str = "record") -> None:
    if not JCAMP_HEADER_RE.search(text) or not TITLE_RE.search(text):
        raise ValueError(f"{identifier}: response is not JCAMP-DX")
    if not DATA_RE.search(text):
        raise ValueError(f"{identifier}: JCAMP-DX response contains no spectral data")


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.part")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def download_jcamp(
    identifiers: list[str],
    output_dir: Path,
    *,
    client: TextClient | None = None,
    overwrite: bool = False,
    strict: bool = False,
) -> dict[str, object]:
    normalized = sorted({normalize_nist_id(value) for value in identifiers})
    if not normalized:
        raise ValueError("at least one NIST ID is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    http = client or RateLimitedClient(min_interval=1.0)
    records: list[dict[str, str]] = []
    first_error: Exception | None = None

    for identifier in normalized:
        destination = output_dir / f"{identifier}.jdx"
        if destination.is_file() and not overwrite:
            try:
                validate_jcamp(
                    destination.read_text(encoding="utf-8", errors="replace"),
                    identifier,
                )
            except (OSError, ValueError):
                pass
            else:
                records.append({"id": identifier, "status": "existing"})
                continue
        try:
            text = http.get_text(nist_jcamp_url(identifier), accept="text/plain")
            validate_jcamp(text, identifier)
            _write_text_atomic(destination, text)
            records.append({"id": identifier, "status": "downloaded"})
        except (OSError, RuntimeError, ValueError) as exc:
            records.append(
                {
                    "id": identifier,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if first_error is None:
                first_error = exc
            if strict:
                break

    summary: dict[str, object] = {
        "source": "NIST Chemistry WebBook",
        "requested": len(normalized),
        "downloaded": sum(row["status"] == "downloaded" for row in records),
        "existing": sum(row["status"] == "existing" for row in records),
        "failed": sum(row["status"] == "failed" for row in records),
        "records": records,
    }
    (output_dir / "download_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    failed_ids = [row["id"] for row in records if row["status"] == "failed"]
    (output_dir / "failed_ids.txt").write_text(
        "".join(f"{identifier}\n" for identifier in failed_ids), encoding="utf-8"
    )
    if strict and first_error is not None:
        raise RuntimeError(
            "NIST download stopped after the first failure"
        ) from first_error
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ids",
        type=Path,
        required=True,
        help="One-ID-per-line file or directory whose supported file stems are IDs",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--user-agent", default="UltraIR-data-preparation/1.0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    client = RateLimitedClient(
        user_agent=args.user_agent,
        min_interval=args.min_interval,
        timeout=args.timeout,
        retries=args.retries,
    )
    summary = download_jcamp(
        load_ids(args.ids),
        args.output_dir,
        client=client,
        overwrite=args.overwrite,
        strict=args.strict,
    )
    public_summary = {
        key: value for key, value in summary.items() if key != "records"
    }
    print(json.dumps(public_summary, indent=2))


if __name__ == "__main__":
    main()
