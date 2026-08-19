#!/usr/bin/env python3
"""Generate fixed-charge molecular-dynamics IR spectra with OpenMM.

The module keeps optional chemistry dependencies lazy so that ``--help`` and
the FFT helper remain usable in a lightweight environment.  It reads a CSV
whose first column is SMILES and writes an aligned CSV with one spectrum per
row.  The parent process owns output files; worker processes only run
simulations and return arrays, which makes resume and failure logging safe.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SPEED_OF_LIGHT_CM_S = 2.99792458e10
DEBYE_PER_E_NM = 48.03204256


@dataclass(frozen=True)
class SimulationConfig:
    openff: str = "openff-2.2.1.offxml"
    temperature: float = 300.0
    friction: float = 1.0
    dt_fs: float = 2.0
    total_ps: float = 8.0
    equilibration_ps: float = 2.0
    sample_interval: int = 1
    wavenumber_min: int = 400
    wavenumber_max: int = 4000
    wavenumber_step: int = 2
    window: str = "hann"
    zero_padding: int = 4
    charge_method: str = "gasteiger"
    platform: str = "auto"
    seed: int = 2025


def _wavenumber_grid(config: SimulationConfig) -> np.ndarray:
    if config.wavenumber_step <= 0:
        raise ValueError("wavenumber step must be positive")
    if config.wavenumber_max <= config.wavenumber_min:
        raise ValueError("wavenumber max must be greater than min")
    return np.arange(
        config.wavenumber_min,
        config.wavenumber_max + config.wavenumber_step,
        config.wavenumber_step,
        dtype=np.float32,
    )


def read_smiles(path: Path) -> list[str]:
    """Read the first non-empty CSV column, accepting an optional header."""
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if row and any(cell.strip() for cell in row):
                rows.append(row)
    if not rows:
        raise ValueError(f"no rows found in {path}")
    if rows[0][0].strip().lower() in {
        "smiles",
        "canonical_smiles",
        "isomeric_smiles",
        "structure",
    }:
        rows = rows[1:]
    smiles = [row[0].strip() for row in rows if row and row[0].strip()]
    if not smiles:
        raise ValueError(f"no SMILES found in {path}")
    return smiles


def compute_ir_from_dipole(
    dipole: np.ndarray,
    dt_fs: float,
    wavenumber_min: int = 400,
    wavenumber_max: int = 4000,
    wavenumber_step: int = 2,
    window: str = "hann",
    zero_padding: int = 4,
) -> np.ndarray:
    """Convert a dipole trajectory ``[frames, 3]`` to a normalized IR array.

    The three Cartesian components are windowed and transformed separately.
    Their power spectra are summed, weighted by wavenumber, interpolated to
    the requested grid, and finally normalized by the grid maximum.
    """
    values = np.asarray(dipole, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"dipole must have shape [frames, 3], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("dipole contains non-finite values")
    if dt_fs <= 0:
        raise ValueError("dt_fs must be positive")
    if wavenumber_step <= 0 or wavenumber_max <= wavenumber_min:
        raise ValueError("wavenumber range must have max > min and a positive step")
    if zero_padding < 1:
        raise ValueError("zero_padding must be at least one")
    grid = np.arange(
        wavenumber_min,
        wavenumber_max + wavenumber_step,
        wavenumber_step,
        dtype=np.float32,
    )
    if values.shape[0] < 8:
        return np.zeros_like(grid)

    centered = values - values.mean(axis=0, keepdims=True)
    if window.lower() in {"hann", "hanning"}:
        weights = np.hanning(values.shape[0])
    elif window.lower() in {"none", "rectangular", "boxcar"}:
        weights = np.ones(values.shape[0], dtype=np.float64)
    else:
        raise ValueError("window must be 'hann' or 'none'")

    samples = centered * weights[:, None]
    n_fft = 1
    while n_fft < values.shape[0] * zero_padding:
        n_fft <<= 1
    transformed = np.fft.rfft(samples, n=n_fft, axis=0)
    power = np.sum(np.abs(transformed) ** 2, axis=1)
    frequencies = np.fft.rfftfreq(n_fft, d=dt_fs * 1.0e-15)
    wavenumbers = frequencies / SPEED_OF_LIGHT_CM_S
    valid = wavenumbers > 0
    wavenumbers = wavenumbers[valid]
    power = (power[valid] * wavenumbers).astype(np.float64, copy=False)
    spectrum = np.interp(grid.astype(np.float64), wavenumbers, power, left=0.0, right=0.0)
    peak = float(np.max(spectrum)) if spectrum.size else 0.0
    if peak > 0.0 and np.isfinite(peak):
        spectrum /= peak
    return np.asarray(spectrum, dtype=np.float32)


def _chemistry_backend() -> dict[str, Any]:
    """Import OpenMM/OpenFF/RDKit only when a simulation is requested."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from openmm import app, openmm, unit
        from openff.toolkit.topology import Molecule
        from openff.toolkit.typing.engines.smirnoff import ForceField
        from openff.units.openmm import to_openmm
    except ImportError as exc:
        raise RuntimeError(
            "MD generation requires RDKit, OpenMM, openff-toolkit, and openff-units; "
            "install them before running this command"
        ) from exc
    return {
        "Chem": Chem,
        "AllChem": AllChem,
        "app": app,
        "openmm": openmm,
        "unit": unit,
        "Molecule": Molecule,
        "ForceField": ForceField,
        "to_openmm": to_openmm,
    }


def smiles_to_rdkit_3d(smiles: str, seed: int, backend: dict[str, Any] | None = None) -> Any:
    """Generate and UFF-relax one explicit-hydrogen 3-D conformer."""
    backend = backend or _chemistry_backend()
    chem = backend["Chem"]
    all_chem = backend["AllChem"]
    molecule = chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    molecule = chem.AddHs(molecule)
    parameters = all_chem.ETKDGv3()
    parameters.randomSeed = int(seed)
    result = all_chem.EmbedMolecule(molecule, parameters)
    if result != 0:
        raise ValueError(f"RDKit could not embed a conformer for {smiles!r}")
    all_chem.UFFOptimizeMolecule(molecule, maxIters=200)
    return molecule


def build_openmm_system(
    molecule: Any,
    config: SimulationConfig,
    backend: dict[str, Any] | None = None,
) -> tuple[Any, Any, Any]:
    """Build a non-periodic OpenMM system with fixed atom-centered charges."""
    backend = backend or _chemistry_backend()
    off_molecule = backend["Molecule"].from_rdkit(
        molecule, allow_undefined_stereo=True
    )
    if config.charge_method == "gasteiger":
        off_molecule.assign_partial_charges("gasteiger")
    elif config.charge_method == "am1bcc":
        off_molecule.assign_partial_charges("am1bcc")
    else:
        raise ValueError("charge_method must be 'gasteiger' or 'am1bcc'")
    force_field = backend["ForceField"](config.openff)
    topology = off_molecule.to_topology()
    system = force_field.create_openmm_system(
        topology, charge_from_molecules=[off_molecule]
    )
    positions = backend["to_openmm"](off_molecule.conformers[0])
    return system, topology.to_openmm(), positions


def _select_platform(name: str, backend: dict[str, Any]) -> tuple[Any, dict[str, str]]:
    openmm = backend["openmm"]
    requested = name.lower()
    if requested == "auto":
        for candidate, properties in (
            ("CUDA", {"Precision": "mixed"}),
            ("OpenCL", {"Precision": "mixed"}),
            ("CPU", {}),
            ("Reference", {}),
        ):
            try:
                return openmm.Platform.getPlatformByName(candidate), properties
            except Exception:
                continue
        raise RuntimeError("OpenMM has no usable simulation platform")
    canonical = {"cuda": "CUDA", "opencl": "OpenCL", "cpu": "CPU", "reference": "Reference"}.get(requested)
    if canonical is None:
        raise ValueError("platform must be auto, CUDA, OpenCL, CPU, or Reference")
    properties = {"Precision": "mixed"} if canonical in {"CUDA", "OpenCL"} else {}
    return openmm.Platform.getPlatformByName(canonical), properties


def build_simulation(
    system: Any,
    topology: Any,
    positions: Any,
    config: SimulationConfig,
    backend: dict[str, Any] | None = None,
) -> Any:
    backend = backend or _chemistry_backend()
    unit = backend["unit"]
    openmm = backend["openmm"]
    app = backend["app"]
    platform, properties = _select_platform(config.platform, backend)
    integrator = openmm.LangevinMiddleIntegrator(
        config.temperature * unit.kelvin,
        config.friction / unit.picosecond,
        config.dt_fs * unit.femtosecond,
    )
    simulation = app.Simulation(topology, system, integrator, platform, properties)
    simulation.context.setPositions(positions)
    openmm.LocalEnergyMinimizer.minimize(simulation.context)
    simulation.context.setVelocitiesToTemperature(
        config.temperature * unit.kelvin, config.seed
    )
    return simulation


def fixed_charge_dipole(context: Any, backend: dict[str, Any] | None = None) -> np.ndarray:
    """Return the total fixed-charge dipole in Debye-like relative units."""
    backend = backend or _chemistry_backend()
    state = context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(backend["unit"].nanometer)
    system = context.getSystem()
    charges = None
    for force_index in range(system.getNumForces()):
        force = system.getForce(force_index)
        if isinstance(force, backend["openmm"].NonbondedForce):
            charges = np.asarray(
                [
                    force.getParticleParameters(atom_index)[0].value_in_unit(
                        backend["unit"].elementary_charge
                    )
                    for atom_index in range(force.getNumParticles())
                ],
                dtype=np.float64,
            )
            break
    if charges is None:
        raise RuntimeError("OpenMM system does not contain a NonbondedForce")
    return np.asarray((charges[:, None] * np.asarray(positions)).sum(axis=0) * DEBYE_PER_E_NM, dtype=np.float32)


def simulate_one(smiles: str, config: SimulationConfig, molecule_seed: int) -> np.ndarray:
    backend = _chemistry_backend()
    molecule = smiles_to_rdkit_3d(smiles, molecule_seed, backend)
    system, topology, positions = build_openmm_system(molecule, config, backend)
    simulation = build_simulation(system, topology, positions, config, backend)
    total_steps = int(round(config.total_ps * 1000.0 / config.dt_fs))
    equilibration_steps = int(round(config.equilibration_ps * 1000.0 / config.dt_fs))
    if total_steps <= equilibration_steps:
        raise ValueError("total_ps must be greater than equilibration_ps")
    production_steps = total_steps - equilibration_steps
    if config.sample_interval < 1:
        raise ValueError("sample_interval must be positive")
    frame_count = production_steps // config.sample_interval
    if frame_count < 8:
        raise ValueError("simulation produces fewer than eight sampled frames")
    if equilibration_steps:
        simulation.step(equilibration_steps)
    dipole = np.empty((frame_count, 3), dtype=np.float32)
    for frame in range(frame_count):
        simulation.step(config.sample_interval)
        dipole[frame] = fixed_charge_dipole(simulation.context, backend)
    return compute_ir_from_dipole(
        dipole,
        config.dt_fs * config.sample_interval,
        config.wavenumber_min,
        config.wavenumber_max,
        config.wavenumber_step,
        config.window,
        config.zero_padding,
    )


def _worker(
    task_queue: Any,
    result_queue: Any,
    config: SimulationConfig,
    gpu_id: int | None,
) -> None:
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    while True:
        item = task_queue.get()
        if item is None:
            return
        index, smiles = item
        try:
            spectrum = simulate_one(smiles, config, config.seed + int(index))
            result_queue.put((index, smiles, spectrum, ""))
        except Exception as exc:  # worker errors are reported, not swallowed
            result_queue.put((index, smiles, None, f"{type(exc).__name__}: {exc}"))


def _parse_existing_output(path: Path, expected_header: list[str]) -> set[str]:
    if not path.is_file():
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"existing output is empty: {path}")
    if rows[0] != expected_header:
        raise ValueError(f"existing output header does not match requested grid: {path}")
    completed: set[str] = set()
    for row in rows[1:]:
        if not row:
            continue
        if len(row) != len(expected_header):
            raise ValueError(f"malformed row in existing output: {path}")
        text = row[0].strip()
        if text:
            completed.add(text)
    return completed


def _open_output(path: Path, header: list[str]) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    handle = path.open("a", encoding="utf-8", newline="")
    if not exists:
        csv.writer(handle).writerow(header)
        handle.flush()
    return handle


def _gpu_ids(value: str) -> list[int | None]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    identifiers = [int(item) for item in values]
    if any(identifier < 0 for identifier in identifiers):
        raise ValueError("GPU IDs must be non-negative")
    return identifiers


def run(
    input_path: Path,
    output_path: Path,
    config: SimulationConfig,
    *,
    gpus: str = "",
    workers_per_gpu: int = 1,
    workers: int | None = None,
    failures_path: Path | None = None,
    log_every: int = 100,
) -> dict[str, int]:
    """Generate spectra, resuming rows already present in ``output_path``."""
    if workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be positive")
    smiles = read_smiles(input_path)
    grid = _wavenumber_grid(config)
    header = ["smiles"] + [str(int(value)) for value in grid]
    completed = _parse_existing_output(output_path, header)
    pending = [(index, value) for index, value in enumerate(smiles) if value not in completed]
    if failures_path is None:
        failures_path = output_path.with_suffix(output_path.suffix + ".failures.tsv")
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    failure_exists = failures_path.is_file() and failures_path.stat().st_size > 0
    failure_handle = failures_path.open("a", encoding="utf-8", newline="")
    failure_writer = csv.writer(failure_handle, delimiter="\t", lineterminator="\n")
    if not failure_exists:
        failure_writer.writerow(("index", "smiles", "error"))
    output_handle = _open_output(output_path, header)
    output_writer = csv.writer(output_handle)
    successes = 0
    failures = 0
    try:
        gpu_values = _gpu_ids(gpus)
        if workers is None:
            worker_count = max(1, len(gpu_values) * workers_per_gpu) if gpu_values else 1
        else:
            if workers < 1:
                raise ValueError("workers must be positive")
            worker_count = workers
        assignments: list[int | None]
        if gpu_values:
            assignments = [gpu_values[index % len(gpu_values)] for index in range(worker_count)]
        else:
            assignments = [None] * worker_count

        started = time.monotonic()
        if worker_count == 1:
            if assignments[0] is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(assignments[0])
            for index, value in pending:
                try:
                    spectrum = simulate_one(value, config, config.seed + index)
                    result = (index, value, spectrum, "")
                except Exception as exc:
                    result = (index, value, None, f"{type(exc).__name__}: {exc}")
                result_index, result_smiles, spectrum, error = result
                if error:
                    failures += 1
                    failure_writer.writerow((result_index, result_smiles, error))
                    failure_handle.flush()
                else:
                    assert spectrum is not None
                    output_writer.writerow([result_smiles] + [f"{value:.8g}" for value in spectrum])
                    output_handle.flush()
                    successes += 1
                processed = successes + failures
                if log_every > 0 and processed % log_every == 0:
                    elapsed = max(time.monotonic() - started, 1e-9)
                    print(f"processed {processed}/{len(pending)} ({processed / elapsed:.2f}/s)", flush=True)
        elif pending:
            context = mp.get_context("spawn")
            task_queue = context.Queue(maxsize=max(2, worker_count * 2))
            result_queue = context.Queue()
            processes = [
                context.Process(target=_worker, args=(task_queue, result_queue, config, gpu_id))
                for gpu_id in assignments
            ]
            for process in processes:
                process.start()
            for item in pending:
                task_queue.put(item)
            for _ in processes:
                task_queue.put(None)
            for processed in range(len(pending)):
                result_index, result_smiles, spectrum, error = result_queue.get()
                if error:
                    failures += 1
                    failure_writer.writerow((result_index, result_smiles, error))
                    failure_handle.flush()
                else:
                    assert spectrum is not None
                    output_writer.writerow([result_smiles] + [f"{value:.8g}" for value in spectrum])
                    output_handle.flush()
                    successes += 1
                if log_every > 0 and (processed + 1) % log_every == 0:
                    elapsed = max(time.monotonic() - started, 1e-9)
                    print(f"processed {processed + 1}/{len(pending)} ({(processed + 1) / elapsed:.2f}/s)", flush=True)
            for process in processes:
                process.join()
    finally:
        output_handle.close()
        failure_handle.close()
    print(f"completed={successes} failed={failures} skipped={len(completed)}", flush=True)
    return {"completed": successes, "failed": failures, "skipped": len(completed)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV with SMILES in its first column")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV with one IR row per SMILES")
    parser.add_argument("--failures", type=Path, help="TSV for rows that could not be simulated")
    parser.add_argument("--openff", default=SimulationConfig.openff)
    parser.add_argument("--temperature", type=float, default=SimulationConfig.temperature)
    parser.add_argument("--friction", type=float, default=SimulationConfig.friction)
    parser.add_argument("--dt-fs", type=float, default=SimulationConfig.dt_fs)
    parser.add_argument("--total-ps", type=float, default=SimulationConfig.total_ps)
    parser.add_argument("--equilibration-ps", type=float, default=SimulationConfig.equilibration_ps)
    parser.add_argument("--sample-interval", type=int, default=SimulationConfig.sample_interval)
    parser.add_argument("--wavenumber-min", type=int, default=SimulationConfig.wavenumber_min)
    parser.add_argument("--wavenumber-max", type=int, default=SimulationConfig.wavenumber_max)
    parser.add_argument("--wavenumber-step", type=int, default=SimulationConfig.wavenumber_step)
    parser.add_argument("--window", choices=("hann", "none"), default=SimulationConfig.window)
    parser.add_argument("--zero-padding", type=int, default=SimulationConfig.zero_padding)
    parser.add_argument("--charge-method", choices=("gasteiger", "am1bcc"), default=SimulationConfig.charge_method)
    parser.add_argument("--platform", choices=("auto", "CUDA", "OpenCL", "CPU", "Reference"), default=SimulationConfig.platform)
    parser.add_argument("--seed", type=int, default=SimulationConfig.seed)
    parser.add_argument("--gpus", default="", help="Comma-separated GPU IDs; empty uses CPU/auto")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--workers", type=int, help="Total worker count; overrides --workers-per-gpu")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = SimulationConfig(
        openff=args.openff,
        temperature=args.temperature,
        friction=args.friction,
        dt_fs=args.dt_fs,
        total_ps=args.total_ps,
        equilibration_ps=args.equilibration_ps,
        sample_interval=args.sample_interval,
        wavenumber_min=args.wavenumber_min,
        wavenumber_max=args.wavenumber_max,
        wavenumber_step=args.wavenumber_step,
        window=args.window,
        zero_padding=args.zero_padding,
        charge_method=args.charge_method,
        platform=args.platform,
        seed=args.seed,
    )
    run(
        args.input,
        args.output,
        config,
        gpus=args.gpus,
        workers_per_gpu=args.workers_per_gpu,
        workers=args.workers,
        failures_path=args.failures,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
