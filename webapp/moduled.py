# moduled.py
"""
Module D / E hybrid — build a refined pseudo-backbone from spectral eigenvectors.

Input:
    eigenvectors : np.ndarray, shape (L, K>=3)
        Spectral eigenvectors from Module B.

Output:
    {
        "coords_3d": coords,    # (L, 3) refined CA coordinates
        "pdb_string": pdb_txt   # CA-only PDB string
    }

This version:
    - infers a 1D chain order using a nearest-neighbor path
    - builds a smooth path
    - resamples so mean CA–CA distance ~ 3.8 Å
    - does a few smoothing passes to avoid kinks
"""

import numpy as np


def _nearest_neighbor_order(coords: np.ndarray) -> np.ndarray:
    """
    Greedy nearest-neighbor path through all points.
    Gives us a 1D "chain order" for residues.
    """
    n = coords.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    remaining = set(range(n))
    # start from point with smallest x (arbitrary but stable)
    current = int(np.argmin(coords[:, 0]))
    order = [current]
    remaining.remove(current)

    while remaining:
        cur_pt = coords[current]
        rem_idx = np.fromiter(remaining, dtype=int)
        dists = np.linalg.norm(coords[rem_idx] - cur_pt, axis=1)
        j = rem_idx[np.argmin(dists)]
        order.append(j)
        remaining.remove(j)
        current = j

    return np.array(order, dtype=int)


def _resample_chain(coords: np.ndarray, target_len: float = 3.8) -> np.ndarray:
    """
    Given an ordered path of points (N, 3), interpolate along arc-length
    and resample so that points are roughly 'target_len' Å apart.
    """
    if coords.shape[0] < 2:
        return coords.copy()

    diffs = np.diff(coords, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg_len)))
    total = s[-1]

    if total < 1e-6:
        # everything collapsed at one point — just return as-is
        return coords.copy()

    n = coords.shape[0]
    # Uniform samples along the path
    s_target = np.linspace(0.0, total, n)

    new = np.zeros_like(coords)
    for d in range(3):
        new[:, d] = np.interp(s_target, s, coords[:, d])

    # Scale so mean CA–CA distance ≈ target_len
    diffs2 = np.diff(new, axis=0)
    mean_len = np.mean(np.linalg.norm(diffs2, axis=1))
    if mean_len > 1e-6:
        scale = target_len / mean_len
        new *= scale

    return new


def _smooth_chain(coords: np.ndarray, n_iters: int = 3) -> np.ndarray:
    """
    Simple Laplacian-like smoothing: each internal point becomes
    the average of itself and its neighbors. Endpoints fixed.
    """
    out = coords.copy()
    for _ in range(n_iters):
        out[1:-1] = (out[:-2] + out[1:-1] + out[2:]) / 3.0
    return out


def _coords_to_pdb(coords: np.ndarray, chain_id: str = "A") -> str:
    """
    Build a CA-only PDB string from (L,3) coordinates.
    """
    lines = []
    for i, (x, y, z) in enumerate(coords, start=1):
        line = (
            "ATOM  {atom_id:5d}  CA  GLY {chain:1s}{res_id:4d}    "
            "{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
        ).format(
            atom_id=i,
            chain=chain_id,
            res_id=i,
            x=x,
            y=y,
            z=z,
        )
        lines.append(line)
    lines.append("END")
    return "\n".join(lines)


def build_pseudo_backbone(eigenvectors: np.ndarray, target_ca_dist: float = 3.8) -> dict:
    """
    Main entry point called from app.py.

    Parameters
    ----------
    eigenvectors : np.ndarray
        (L, K>=3) array from Module B.
    target_ca_dist : float
        Desired average CA–CA spacing in Å.

    Returns
    -------
    dict with:
        "coords_3d"   : np.ndarray, shape (L, 3)
        "pdb_string"  : str
    """
    if eigenvectors is None:
        raise ValueError("eigenvectors is None")

    ev = np.asarray(eigenvectors, dtype=float)
    if ev.ndim != 2 or ev.shape[1] < 3:
        raise ValueError("eigenvectors must be (L, K>=3)")

    # Use first 3 eigenvectors as a 3D embedding
    coords_raw = ev[:, :3].copy()

    # Center to origin for numerical stability
    coords_raw -= coords_raw.mean(axis=0, keepdims=True)

    # 1) Infer chain order
    order = _nearest_neighbor_order(coords_raw)
    coords_ordered = coords_raw[order]

    # 2) Resample to get roughly uniform spacing
    coords_chain = _resample_chain(coords_ordered, target_len=target_ca_dist)

    # 3) Smooth curvature a bit (removes sharp kinks)
    coords_smooth = _smooth_chain(coords_chain, n_iters=4)

    # 4) Final recentre for nicer viewing
    coords_smooth -= coords_smooth.mean(axis=0, keepdims=True)

    pdb_txt = _coords_to_pdb(coords_smooth, chain_id="A")

    return {
        "coords_3d": coords_smooth,
        "pdb_string": pdb_txt,
    }
