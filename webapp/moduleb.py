# module_b.py
"""
Module B — Graph Laplacian + Spectral Embeddings

This module takes the Direct Information (DI) matrix from Module A
and produces:

1. Graph Laplacian        (L or L_norm)
2. Top-k eigenvectors     (spectral modes)
3. Per-residue embeddings (DI row + eigenvectors)

Usage:
------
from module_b import compute_graph_embeddings

result = compute_graph_embeddings(di_matrix, k=8)

L = result["laplacian"]
eigenvectors = result["eigenvectors"]
embeddings = result["embeddings"]

"""

import numpy as np


def compute_graph_embeddings(di_matrix: np.ndarray, k: int = 8):
    """
    Compute graph spectral embeddings from a DI matrix.

    Parameters
    ----------
    di_matrix : np.ndarray
        L x L matrix of DI scores (symmetric, diagonals = 0)
    k : int
        Number of top eigenvectors to extract

    Returns
    -------
    result : dict
        {
            "laplacian": L_norm,
            "eigenvalues": evals,
            "eigenvectors": evecs,        # shape (L, k)
            "embeddings": embeddings      # shape (L, L + k)
        }
    """

    if di_matrix is None:
        raise ValueError("DI matrix is None.")

    if not isinstance(di_matrix, np.ndarray):
        di_matrix = np.array(di_matrix)

    if di_matrix.ndim != 2 or di_matrix.shape[0] != di_matrix.shape[1]:
        raise ValueError("DI matrix must be a square L x L matrix.")

    L = di_matrix.shape[0]

    # -----------------------------------------
    # STEP B1: Adjacency matrix A
    # -----------------------------------------
    A = np.copy(di_matrix)
    np.fill_diagonal(A, 0.0)

    # -----------------------------------------
    # STEP B2: Degree matrix D
    # -----------------------------------------
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)

    # -----------------------------------------
    # STEP B3: Normalized Laplacian
    # L_norm = I - D^{-1/2} A D^{-1/2}
    # -----------------------------------------
    with np.errstate(divide='ignore'):
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-12))

    L_norm = np.eye(L) - D_inv_sqrt @ A @ D_inv_sqrt

    # -----------------------------------------
    # STEP B4: Compute eigenvectors (smallest k)
    # -----------------------------------------
    # Use symmetric eigendecomposition
    evals, evecs = np.linalg.eigh(L_norm)

    # Sort eigenvalues/eigenvectors
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Keep only the first k eigenvectors
    k = min(k, L - 1)
    evecs_k = evecs[:, :k]

    # -----------------------------------------
    # STEP B5: Build residue embeddings
    # embedding[i] = [DI_row(i), EV1[i], ..., EVk[i]]
    # -----------------------------------------
    embeddings = []

    for i in range(L):
        di_row = di_matrix[i]          # length L
        ev = evecs_k[i]                # length k
        embedding_vec = np.concatenate([di_row, ev])
        embeddings.append(embedding_vec)

    embeddings = np.array(embeddings)

    # -----------------------------------------
    # Return results
    # -----------------------------------------
    return {
        "laplacian": L_norm,
        "eigenvalues": evals[:k],
        "eigenvectors": evecs_k,
        "embeddings": embeddings
    }
