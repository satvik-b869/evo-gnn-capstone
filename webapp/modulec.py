# modulec.py
"""
Module C — Geographic Graph Coordinates
---------------------------------------

Turns the spectral eigenvectors from Module B
into 2D or 3D coordinates for each residue.
"""

import numpy as np


def compute_coordinates(eigenvectors: np.ndarray, dim: int = 2):
    """
    Convert eigenvectors (L × k) into coordinates (L × dim).

    Parameters
    ----------
    eigenvectors : np.ndarray
        Eigenvectors from Module B, shape (L, k)
    dim : int
        Number of coordinate dimensions (2 or 3)

    Returns
    -------
    coords : np.ndarray
        Geographic coordinates of shape (L, dim)
    """
    if not isinstance(eigenvectors, np.ndarray):
        eigenvectors = np.array(eigenvectors)

    L, k = eigenvectors.shape

    if dim > k:
        raise ValueError(f"Requested {dim}D but only have {k} eigenvectors.")

    coords = eigenvectors[:, :dim]

    return coords
