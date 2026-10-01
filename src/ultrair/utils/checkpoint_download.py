"""Acquire released pretrained encoders in the checkpoint directory."""

from pathlib import Path
from urllib.request import urlopen

from tqdm.auto import tqdm


PRETRAINED_FILES = {
    *(f"ultrair_pretraining_general_epoch{epoch}.pt" for epoch in range(1, 6)),
    "ultrair_pretraining_molstrelu_epoch5.pt",
}


def ensure_pretrained_checkpoint(path: Path) -> Path:
    """Reuse an encoder or download it atomically from the public release."""
    if path.name not in PRETRAINED_FILES:
        raise ValueError(f"Unknown released pretrained checkpoint: {path.name}")
    if path.is_file():
        print(f"[checkpoint] Reuse {path}", flush=True)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    url = f"https://huggingface.co/yusentan/UltraIR/resolve/main/checkpoints/pretraining/{path.name}"
    print(f"[checkpoint] Download {path.name} to {path}", flush=True)
    with urlopen(url, timeout=60) as response:
        size = response.headers.get("Content-Length")
        total = int(size) if size is not None else None
        with partial.open("wb") as output, tqdm(
            total=total, desc=path.name, unit="B", unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            while block := response.read(1024 * 1024):
                output.write(block)
                progress.update(len(block))
    if partial.stat().st_size == 0 or (
        total is not None and partial.stat().st_size != total
    ):
        raise IOError(f"Incomplete checkpoint download: {partial}")
    partial.replace(path)
    print(f"[checkpoint] Ready {path}", flush=True)
    return path
