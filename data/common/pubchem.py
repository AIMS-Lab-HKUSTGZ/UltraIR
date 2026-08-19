"""Add PubChem ConnectivitySMILES to local NIST or SDBS metadata."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

from .http_client import HttpStatusError, RateLimitedClient


PUBCHEM_ENDPOINT = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CAS_RE = re.compile(
    r"^##CAS REGISTRY NO\s*=\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE
)


class JsonClient(Protocol):
    def get_json(self, url: str) -> Any: ...


def pubchem_url(identifier: str, namespace: str = "name") -> str:
    value = identifier.strip()
    if not value:
        raise ValueError("PubChem identifier must not be empty")
    if namespace not in {"name", "inchi", "inchikey"}:
        raise ValueError(f"unsupported PubChem namespace: {namespace!r}")
    encoded = quote(value, safe="")
    return (
        f"{PUBCHEM_ENDPOINT}/compound/{namespace}/{encoded}"
        "/property/ConnectivitySMILES/JSON"
    )


def fetch_connectivity_smiles(
    identifier: str,
    *,
    client: JsonClient,
    namespace: str = "name",
) -> str | None:
    """Return one ConnectivitySMILES, or ``None`` when PubChem reports not found."""
    url = pubchem_url(identifier, namespace)
    try:
        payload = client.get_json(url)
    except HttpStatusError as exc:
        if exc.status == 404:
            return None
        raise
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected PubChem response type for {identifier!r}")
    fault = payload.get("Fault")
    if isinstance(fault, dict):
        detail = f"{fault.get('Message', '')} {fault.get('Details', '')}"
        if "notfound" in detail.lower():
            return None
        raise ValueError(f"PubChem fault for {identifier!r}: {detail.strip()}")
    property_table = payload.get("PropertyTable")
    if not isinstance(property_table, dict):
        raise ValueError(f"PubChem returned no property table for {identifier!r}")
    properties = property_table.get("Properties", [])
    if not isinstance(properties, list) or not properties:
        raise ValueError(f"PubChem returned no properties for {identifier!r}")
    first = properties[0]
    if not isinstance(first, dict):
        raise ValueError(f"invalid PubChem property record for {identifier!r}")
    smiles = first.get("ConnectivitySMILES") or first.get("CanonicalSMILES")
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError(f"PubChem returned no ConnectivitySMILES for {identifier!r}")
    return smiles.strip()


def extract_cas_from_jcamp(path: Path) -> str | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = CAS_RE.search(text)
    return match.group(1).strip() if match and match.group(1).strip() else None


def _existing_smiles(path: Path) -> str | None:
    if not path.is_file():
        return None
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    if len(values) != 1 or values[0].lower().endswith(".jdx"):
        return None
    if any(
        token in values[0].lower()
        for token in ("traceback", "client error", "pugrest")
    ):
        return None
    return values[0]


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.part")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def enrich_jcamp_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    client: JsonClient,
    overwrite: bool = False,
    strict: bool = False,
) -> dict[str, object]:
    paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jdx", ".dx"}
    )
    if not paths:
        raise FileNotFoundError(f"no JCAMP-DX files under {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, str | None] = {}
    records: list[dict[str, object]] = []
    first_error: Exception | None = None

    for path in paths:
        destination = output_dir / f"{path.stem}.txt"
        if not overwrite and _existing_smiles(destination) is not None:
            records.append({"file": path.name, "status": "existing"})
            continue
        try:
            cas = extract_cas_from_jcamp(path)
            if cas is None:
                records.append({"file": path.name, "status": "missing_cas"})
                continue
            if cas not in cache:
                cache[cas] = fetch_connectivity_smiles(cas, client=client)
            smiles = cache[cas]
            if smiles is None:
                records.append(
                    {"file": path.name, "cas": cas, "status": "not_found"}
                )
                continue
            _write_text_atomic(destination, f"{smiles}\n")
            records.append({"file": path.name, "cas": cas, "status": "written"})
        except (OSError, RuntimeError, ValueError) as exc:
            records.append(
                {
                    "file": path.name,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if first_error is None:
                first_error = exc
            if strict:
                break

    summary: dict[str, object] = {
        "source": "PubChem PUG REST",
        "input_files": len(paths),
        "written": sum(row["status"] == "written" for row in records),
        "existing": sum(row["status"] == "existing" for row in records),
        "not_found": sum(row["status"] == "not_found" for row in records),
        "missing_cas": sum(row["status"] == "missing_cas" for row in records),
        "failed": sum(row["status"] == "failed" for row in records),
        "records": records,
    }
    (output_dir / "pubchem_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if strict and first_error is not None:
        raise RuntimeError(
            "PubChem enrichment stopped after the first failure"
        ) from first_error
    return summary


def enrich_csv(
    input_csv: Path,
    output_csv: Path,
    *,
    client: JsonClient,
    cas_field: str = "CAS No",
    smiles_field: str = "SMILES",
    status_field: str = "PubChem Status",
    strict: bool = False,
) -> dict[str, object]:
    if input_csv.resolve() == output_csv.resolve():
        raise ValueError("input_csv and output_csv must be different paths")
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if cas_field not in fieldnames:
            raise ValueError(f"input CSV is missing {cas_field!r}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"input CSV is empty: {input_csv}")
    for field in (smiles_field, status_field):
        if field not in fieldnames:
            fieldnames.append(field)

    cache: dict[str, str | None] = {}
    counts = {
        "existing": 0,
        "written": 0,
        "not_found": 0,
        "missing_cas": 0,
        "failed": 0,
    }
    failures: list[dict[str, object]] = []
    for row_number, row in enumerate(rows, start=2):
        if (row.get(smiles_field) or "").strip():
            row[status_field] = "existing"
            counts["existing"] += 1
            continue
        cas = (row.get(cas_field) or "").strip()
        if not cas:
            row[status_field] = "missing_cas"
            counts["missing_cas"] += 1
            continue
        try:
            if cas not in cache:
                cache[cas] = fetch_connectivity_smiles(cas, client=client)
            smiles = cache[cas]
        except (RuntimeError, ValueError) as exc:
            row[status_field] = "failed"
            counts["failed"] += 1
            failures.append(
                {
                    "row": row_number,
                    "cas": cas,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if strict:
                raise
            continue
        if smiles is None:
            row[status_field] = "not_found"
            counts["not_found"] += 1
        else:
            row[smiles_field] = smiles
            row[status_field] = "written"
            counts["written"] += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_name(f".{output_csv.name}.part")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output_csv)
    finally:
        if temporary.exists():
            temporary.unlink()
    summary: dict[str, object] = {
        "source": "PubChem PUG REST",
        "rows": len(rows),
        **counts,
        "failures": failures,
    }
    manifest = output_csv.with_suffix(f"{output_csv.suffix}.manifest.json")
    manifest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _client(args: argparse.Namespace) -> RateLimitedClient:
    return RateLimitedClient(
        user_agent=args.user_agent,
        min_interval=args.min_interval,
        timeout=args.timeout,
        retries=args.retries,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    jcamp = subparsers.add_parser(
        "jcamp", help="Write one SMILES text file per JCAMP file"
    )
    jcamp.add_argument("--input-dir", type=Path, required=True)
    jcamp.add_argument("--output-dir", type=Path, required=True)
    jcamp.add_argument("--overwrite", action="store_true")
    jcamp.add_argument("--strict", action="store_true")

    csv_parser = subparsers.add_parser(
        "csv", help="Add SMILES to an aligned metadata CSV"
    )
    csv_parser.add_argument("--input-csv", type=Path, required=True)
    csv_parser.add_argument("--output-csv", type=Path, required=True)
    csv_parser.add_argument("--cas-field", default="CAS No")
    csv_parser.add_argument("--smiles-field", default="SMILES")
    csv_parser.add_argument("--status-field", default="PubChem Status")
    csv_parser.add_argument("--strict", action="store_true")

    for subparser in (jcamp, csv_parser):
        subparser.add_argument("--min-interval", type=float, default=0.25)
        subparser.add_argument("--timeout", type=float, default=30.0)
        subparser.add_argument("--retries", type=int, default=4)
        subparser.add_argument("--user-agent", default="UltraIR-data-preparation/1.0")

    args = parser.parse_args()
    if args.command == "jcamp":
        summary = enrich_jcamp_directory(
            args.input_dir,
            args.output_dir,
            client=_client(args),
            overwrite=args.overwrite,
            strict=args.strict,
        )
    else:
        summary = enrich_csv(
            args.input_csv,
            args.output_csv,
            client=_client(args),
            cas_field=args.cas_field,
            smiles_field=args.smiles_field,
            status_field=args.status_field,
            strict=args.strict,
        )
    public_summary = {
        key: value
        for key, value in summary.items()
        if key not in {"records", "failures"}
    }
    print(json.dumps(public_summary, indent=2))


if __name__ == "__main__":
    main()
