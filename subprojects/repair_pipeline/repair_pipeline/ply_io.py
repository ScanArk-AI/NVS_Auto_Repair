"""3DGS .ply reader, dependency-free.

The header's property list is parsed into a numpy structured dtype, so property
order and SH degree do not matter: a degree-3 delivery carries 45 ``f_rest_*``
fields on top of the three ``f_dc_*`` ones, and a fixed-stride reader would
silently misalign every row.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814

_PLY_DTYPES = {
    "float": np.float32, "float32": np.float32, "double": np.float64, "float64": np.float64,
    "uchar": np.uint8, "uint8": np.uint8, "char": np.int8, "int8": np.int8,
    "short": np.int16, "int16": np.int16, "ushort": np.uint16, "uint16": np.uint16,
    "int": np.int32, "int32": np.int32, "uint": np.uint32, "uint32": np.uint32,
}


@dataclass
class Points:
    """Just enough for reference selection."""

    centers: np.ndarray  # (N, 3) float32
    opacity: np.ndarray  # (N,) float32 in [0, 1]

    def __len__(self) -> int:
        return len(self.centers)


@dataclass
class Gaussians:
    """Full parameters, activated the way the 3DGS rasteriser expects."""

    centers: np.ndarray   # (N, 3)
    rgbs: np.ndarray      # (N, 3) in [0, 1], the DC term only
    opacities: np.ndarray  # (N, 1) in [0, 1]
    scales: np.ndarray    # (N, 3), already exp'd
    quats_wxyz: np.ndarray  # (N, 4), normalised
    sh: np.ndarray        # (N, K, 3), K = (degree + 1) ** 2

    def __len__(self) -> int:
        return int(self.centers.shape[0])

    @property
    def sh_degree(self) -> int:
        degree = int(round(int(self.sh.shape[1]) ** 0.5)) - 1
        return degree


def _open_memmap(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    path = Path(path)
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a ply file")
        fmt, count, props, in_vertex = None, 0, [], False
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no end_header")
            parts = line.decode("ascii", "replace").split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                if parts[1] == "list":
                    raise ValueError(f"{path}: list properties are not supported")
                props.append((parts[2], parts[1]))
            elif parts[0] == "end_header":
                offset = f.tell()
                break

    if fmt not in ("binary_little_endian", "binary_big_endian"):
        raise ValueError(f"{path}: only binary ply is supported, got {fmt!r}")
    endian = "<" if fmt == "binary_little_endian" else ">"
    dtype = np.dtype([(n, endian + np.dtype(_PLY_DTYPES[t]).str[1:]) for n, t in props])
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,)), {n: i for i, n in enumerate(dtype.names)}


def read_points(path: Path) -> Points:
    """Centres and alpha only -- what the frustum test needs."""
    raw, _ = _open_memmap(path)
    names = raw.dtype.names or ()
    missing = {"x", "y", "z", "opacity"} - set(names)
    if missing:
        raise ValueError(f"{path}: missing properties {sorted(missing)}")
    centers = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float32)
    opacity = 1.0 / (1.0 + np.exp(-raw["opacity"].astype(np.float32)))
    return Points(centers=centers, opacity=opacity)


def _normalise_quats(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n[n < 1e-12] = 1.0
    return (q / n).astype(np.float32)


def read_gaussians(path: Path) -> Gaussians:
    """Full parameter set, used by the alpha renderer."""
    raw, _ = _open_memmap(path)
    names = list(raw.dtype.names or ())

    def col(name: str) -> np.ndarray:
        return np.asarray(raw[name], dtype=np.float32)

    f_dc = np.stack([col("f_dc_0"), col("f_dc_1"), col("f_dc_2")], -1)

    # f_rest is channel-major: [R1..Rk, G1..Gk, B1..Bk] with k = K - 1.
    rest_names = sorted((n for n in names if n.startswith("f_rest_")),
                        key=lambda n: int(n.rsplit("_", 1)[1]))
    k = len(rest_names) // 3
    # K = (degree + 1)^2 must be a perfect square, else the f_rest block is not
    # a whole set of SH bands and is ignored (degree-0 fallback).
    if rest_names and len(rest_names) % 3 == 0 and int(round((k + 1) ** 0.5)) ** 2 == (k + 1):
        f_rest = np.stack([col(n) for n in rest_names], -1)
        rest = f_rest.reshape(f_rest.shape[0], 3, k).transpose(0, 2, 1)
        sh = np.concatenate([f_dc[:, None, :], rest], axis=1)
    else:
        sh = f_dc[:, None, :]

    return Gaussians(
        centers=np.stack([col("x"), col("y"), col("z")], -1),
        rgbs=np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0),
        opacities=(1.0 / (1.0 + np.exp(-col("opacity"))))[:, None],
        scales=np.exp(np.stack([col("scale_0"), col("scale_1"), col("scale_2")], -1)),
        quats_wxyz=_normalise_quats(np.stack([col("rot_0"), col("rot_1"), col("rot_2"), col("rot_3")], -1)),
        sh=sh,
    )
