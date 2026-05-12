from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DATA_ROOT = PROJECT_ROOT / "data"
KNOWAIR_ROOT = PROJECT_ROOT / "data_knowair"
GAGNN_ROOT = PROJECT_ROOT / "datagagnn"


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_path(*parts: str | Path) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def experiment_output_path(*parts: str | Path) -> Path:
    return ensure_dir(OUTPUT_ROOT.joinpath(*parts))


def normalize_run_dir(log_dir: str | Path) -> Path:
    """
    Move legacy `experiments/...` run directories to `outputs/experiments/...`.

    The old path is kept as a symlink so continuation scripts that still look
    under `experiments/` continue to work.
    """
    raw = Path(log_dir)
    if raw.is_absolute():
        try:
            relative = raw.relative_to(PROJECT_ROOT)
        except ValueError:
            return ensure_dir(raw)
    else:
        relative = Path(*raw.parts[1:]) if raw.parts and raw.parts[0] == "." else raw
    legacy = PROJECT_ROOT / relative

    if relative.parts and relative.parts[0] == "experiments":
        canonical = OUTPUT_ROOT / relative
        ensure_dir(canonical)
        if not legacy.exists():
            ensure_dir(legacy.parent)
            legacy.symlink_to(canonical, target_is_directory=True)
        return canonical

    return ensure_dir(legacy)
