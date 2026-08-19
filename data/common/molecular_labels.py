"""Generate aligned molecular labels used by the molecular UltraIR tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RAW_PATTERNS = [
    ("Alkane", "[CX4;H3,H2]"), ("Methyl", "[CH3]"),
    ("Alkene", "[CX3]=[CX3]"), ("Alkyne", "[CX2]#[CX2]"),
    ("Alcohols", "[#6][OX2H]"), ("Amines", "[NX3;H2,H1,H0;!$(N[CX3](=O))]"),
    ("Nitriles", "[NX1]#[CX2]"), ("Aromatics", "[$([cX3](:*):*),$([cX2+](:*):*)]"),
    ("Alkyl halides", "[#6][F,Cl,Br,I]"), ("Esters", "[#6][CX3](=O)[OX2H0][#6]"),
    ("Ketones", "[#6][CX3](=O)[#6]"), ("Aldehydes", "[CX3H1](=O)[#6]"),
    ("Carboxylic acids", "[CX3](=O)[OX2H1]"), ("Ether", "[OD2]([#6;!$(C=O)])([#6;!$(C=O)])"),
    ("Acyl halides", "[CX3](=[OX1])[F,Cl,Br,I]"), ("Amides", "[NX3][CX3](=[OX1])[#6]"),
    ("Nitro", "[$([N+](=O)[O-]),$([NX3](=O)=O)][#6]"),
]
PROPERTY_NAMES = ["SAScore", "LogP", "TPSA", "NumHDonors", "NumHAcceptors",
                  "NumRotatableBonds", "FractionCSP3", "BertzCT", "QED",
                  "NumAromaticRings", "NumAliphaticRings"]


def _rdkit_tools():
    try:
        from rdkit import Chem, rdBase
        from rdkit.Chem import (
            Descriptors,
            GraphDescriptors,
            Lipinski,
            QED,
            rdFingerprintGenerator,
            rdMolDescriptors,
        )
        from rdkit.DataStructs import ConvertToNumpyArray
    except ImportError as exc:
        raise RuntimeError("molecular labels require RDKit") from exc
    try:
        from rdkit.Contrib.SA_Score import sascorer
    except ImportError as exc:
        raise RuntimeError(
            "molecular property generation requires rdkit.Contrib.SA_Score"
        ) from exc
    if not hasattr(sascorer, "calculateScore"):
        raise RuntimeError("RDKit SA_Score does not provide calculateScore")
    return (
        Chem,
        rdBase,
        Descriptors,
        GraphDescriptors,
        Lipinski,
        QED,
        rdFingerprintGenerator,
        rdMolDescriptors,
        ConvertToNumpyArray,
        sascorer.calculateScore,
    )


def _smiles_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict").strip()
    return str(value).strip()


def generate_labels(
    smiles: np.ndarray, radius: int = 2, n_bits: int = 2048
) -> dict[str, np.ndarray]:
    """Return labels and a valid mask in the original SMILES order."""
    if np.asarray(smiles).ndim != 1:
        raise ValueError(f"smiles must be 1D, got {np.asarray(smiles).shape}")
    if radius < 0 or n_bits < 1:
        raise ValueError(
            f"radius must be non-negative and n_bits positive, got {radius}, {n_bits}"
        )
    (
        Chem,
        rdBase,
        Descriptors,
        GraphDescriptors,
        Lipinski,
        QED,
        rdFingerprintGenerator,
        rdMolDescriptors,
        ConvertToNumpyArray,
        sascore,
    ) = _rdkit_tools()
    patterns = [(name, Chem.MolFromSmarts(pattern)) for name, pattern in RAW_PATTERNS]
    if any(pattern is None for _, pattern in patterns):
        raise ValueError("one of the functional-group SMARTS patterns is invalid")
    n = len(smiles)
    groups = np.zeros((n, len(patterns)), dtype=np.uint8)
    fingerprints = np.zeros((n, n_bits), dtype=np.uint8)
    properties = np.full((n, len(PROPERTY_NAMES)), np.nan, dtype=np.float32)
    formulas = [""] * n
    valid = np.zeros((n,), dtype=np.uint8)
    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits
    )
    funcs = [
        ("SAScore", sascore),
        ("LogP", Descriptors.MolLogP),
        ("TPSA", Descriptors.TPSA),
        ("NumHDonors", Lipinski.NumHDonors),
        ("NumHAcceptors", Lipinski.NumHAcceptors),
        ("NumRotatableBonds", Lipinski.NumRotatableBonds),
        ("FractionCSP3", Lipinski.FractionCSP3),
        ("BertzCT", GraphDescriptors.BertzCT),
        ("QED", QED.qed),
        ("NumAromaticRings", Lipinski.NumAromaticRings),
        ("NumAliphaticRings", Lipinski.NumAliphaticRings),
    ]
    for i, raw in enumerate(smiles):
        text = _smiles_text(raw)
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(text) if text else None
        if mol is None:
            continue
        groups[i] = [int(mol.HasSubstructMatch(pattern)) for _, pattern in patterns]
        bit_vector = fingerprint_generator.GetFingerprint(mol)
        ConvertToNumpyArray(bit_vector, fingerprints[i])
        row_values = []
        for name, function in funcs:
            try:
                row_values.append(float(function(mol)))
            except Exception as exc:
                raise RuntimeError(
                    f"{name} generation failed for row {i}: {text}"
                ) from exc
        row = np.asarray(row_values, dtype=np.float32)
        if not np.isfinite(row).all():
            raise ValueError(f"non-finite molecular properties for row {i}: {text}")
        properties[i] = row
        formulas[i] = rdMolDescriptors.CalcMolFormula(mol)
        valid[i] = 1
    return {"functional_groups": groups, "fingerprint": fingerprints,
            "properties": properties, "formula": np.asarray(formulas, dtype=str),
            "valid_mask": valid}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create UltraIR molecular labels from smiles.npy.")
    parser.add_argument("--smiles", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    args = parser.parse_args()
    smiles = np.load(args.smiles, allow_pickle=True)
    labels = generate_labels(smiles, radius=args.radius, n_bits=args.n_bits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in labels.items():
        np.save(args.output_dir / f"{name}.npy", values)
    (args.output_dir / "labels.json").write_text(json.dumps({
        "functional_group_names": [name for name, _ in RAW_PATTERNS],
        "property_names": PROPERTY_NAMES, "morgan_radius": args.radius,
        "morgan_bits": args.n_bits, "num_rows": int(len(smiles)),
        "invalid_smiles": int(len(smiles) - labels["valid_mask"].sum()),
    }, indent=2), encoding="utf-8")
    print(f"saved labels for {len(smiles)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
