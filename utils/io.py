"""I/O helper utilities."""

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist.

    Args:
        path: Directory path.

    Returns:
        Resolved Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: str | Path) -> None:
    """Save data as JSON.

    Args:
        data: JSON-serialisable object.
        path: Destination file path.
    """
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> Any:
    """Load JSON file.

    Args:
        path: Source file path.

    Returns:
        Parsed Python object.
    """
    with open(path, "r") as f:
        return json.load(f)


def save_csv(data: dict | list | pd.DataFrame, path: str | Path) -> None:
    """Save data as CSV.

    Args:
        data: DataFrame, list of dicts, or dict of lists.
        path: Destination file path.
    """
    ensure_dir(Path(path).parent)
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, dict):
        df = pd.DataFrame(data)
    else:
        df = pd.DataFrame(data)
    df.to_csv(path, index=False)


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load CSV file.

    Args:
        path: Source file path.

    Returns:
        pandas DataFrame.
    """
    return pd.read_csv(path)


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    """Save multiple numpy arrays to a compressed .npz file.

    Args:
        path: Destination file path (should end with .npz).
        **arrays: Named numpy arrays to save.
    """
    ensure_dir(Path(path).parent)
    np.savez_compressed(path, **arrays)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load a .npz file into a dict of arrays.

    Args:
        path: Source .npz file path.

    Returns:
        Dictionary mapping array names to numpy arrays.
    """
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}
