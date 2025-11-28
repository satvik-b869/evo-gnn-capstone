# compute_di.py
"""
Mean-field DCA-like pipeline to compute a DI-style contact score matrix
from an aligned sequence array, and build a DI graph.

Usage
-----
from compute_di import compute_di

result = compute_di(msa_array)
DI = result["di_matrix"]          # numpy array, shape (L, L)
edges = result["di_graph"]        # list of dicts: {"i", "j", "score"}

Notes
-----
- This uses a simplified mean-field DCA approach.
- The "DI" here is an APC-corrected Frobenius norm of couplings, which
  is widely used as a DCA contact score.
- Designed for small / medium MSAs (e.g., L <= ~150) for a capstone project.
"""

from __future__ import annotations

from typing import List, Dict, Any
import numpy as np


# -------------------------
# Alphabet and mapping
# -------------------------

# 20 standard amino acids + gap as '-'
ALPHABET = "ACDEFGHIKLMNPQRSTVWY-"
Q = len(ALPHABET)            # 21
GAP_IDX = ALPHABET.index("-")

AA_TO_IDX = {aa: i for i, aa in enumerate(ALPHABET)}


# -------------------------
# Public entry point
# -------------------------

def compute_di(
    msa_array: List[str],
    pseudocount: float = 0.2,
    reg_lambda: float = 0.01,
    top_k_edges: int = 200,
    min_seq_sep: int = 3,
) -> Dict[str, Any]:
    """
    Compute a DCA-like DI matrix and DI graph from an MSA (aligned sequences).

    Parameters
    ----------
    msa_array : list of str
        List of aligned sequences (all same length).
    pseudocount : float
        Pseudocount weight added to frequencies (default 0.2).
    reg_lambda : float
        Diagonal regularization added to covariance before inversion.
    top_k_edges : int
        Number of top-scoring residue-residue pairs to keep in DI graph.
    min_seq_sep : int
        Minimum |i-j| separation for edges (to ignore trivial neighbors).

    Returns
    -------
    result : dict
        {
          "di_matrix": np.ndarray (L, L),
          "di_graph": List[Dict[str, Any]]
        }
    """
    if not msa_array:
        raise ValueError("msa_array is empty.")

    # Ensure sequences are same length
    L = len(msa_array[0])
    for s in msa_array:
        if len(s) != L:
            raise ValueError("All sequences in msa_array must have the same length.")

    # Convert characters to numeric representation
    msa_numeric = _convert_to_numeric(msa_array)  # (N, L)

    # Compute single-site and pairwise frequencies with pseudocounts
    fi, fij = _compute_frequencies(msa_numeric, pseudocount=pseudocount)

    # Build and invert covariance matrix (mean-field DCA)
    C_inv = _invert_covariance(fi, fij, reg_lambda=reg_lambda)

    # Extract couplings and compute DI-like score (Frobenius norm + APC)
    di_matrix = _compute_di_from_couplings(C_inv, L)

    # Build a DI graph with top-scoring edges
    di_graph = _build_di_graph(di_matrix, top_k=top_k_edges, min_seq_sep=min_seq_sep)

    return {
        "di_matrix": di_matrix,
        "di_graph": di_graph,
    }


# -------------------------
# Helper functions
# -------------------------

def _convert_to_numeric(msa_array: List[str]) -> np.ndarray:
    """
    Convert aligned sequences (strings) to integer matrix (N, L) with values 0..Q-1.
    Unknown characters are treated as gaps.
    """
    N = len(msa_array)
    L = len(msa_array[0])
    msa_num = np.zeros((N, L), dtype=np.int32)

    for i, seq in enumerate(msa_array):
        seq = seq.upper()
        row = []
        for ch in seq:
            row.append(AA_TO_IDX.get(ch, GAP_IDX))  # unknown -> gap
        msa_num[i, :] = row

    return msa_num


def _compute_frequencies(
    msa_num: np.ndarray,
    pseudocount: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute single-site (fi) and pairwise (fij) frequencies with pseudocounts.

    Parameters
    ----------
    msa_num : np.ndarray
        Shape (N, L) with integer-coded amino acids.
    pseudocount : float
        Pseudocount fraction added to frequencies.

    Returns
    -------
    fi : np.ndarray
        Single-site frequencies, shape (L, Q).
    fij : np.ndarray
        Pairwise frequencies, shape (L, L, Q, Q).
    """
    N, L = msa_num.shape

    # Uniform weight for all sequences (can extend to sequence weights later)
    weights = np.ones(N, dtype=np.float64)
    Meff = weights.sum()

    # Single-site frequencies
    fi = np.zeros((L, Q), dtype=np.float64)
    for n in range(N):
        w = weights[n]
        for i in range(L):
            a = msa_num[n, i]
            fi[i, a] += w

    fi /= Meff  # normalize

    # Pairwise frequencies
    fij = np.zeros((L, L, Q, Q), dtype=np.float64)
    for n in range(N):
        w = weights[n]
        for i in range(L):
            ai = msa_num[n, i]
            for j in range(i + 1, L):
                aj = msa_num[n, j]
                fij[i, j, ai, aj] += w
                fij[j, i, aj, ai] += w  # symmetric

        # Diagonal blocks (i==j) can be defined as fi, but not strictly necessary
    fij /= Meff

    # Apply pseudocounts
    # fi_reg = (1 - λ) * fi + λ / Q
    # fij_reg = (1 - λ) * fij + λ / Q^2
    lam = pseudocount
    fi = (1.0 - lam) * fi + lam / Q
    fij = (1.0 - lam) * fij + lam / (Q * Q)

    return fi, fij


def _invert_covariance(
    fi: np.ndarray,
    fij: np.ndarray,
    reg_lambda: float = 0.01,
) -> np.ndarray:
    """
    Build and invert the covariance matrix using mean-field DCA.

    To avoid singularity, we use only Q-1 states per position (we drop the last
    state, typically the gap).

    Parameters
    ----------
    fi : np.ndarray
        Single-site frequencies, shape (L, Q).
    fij : np.ndarray
        Pairwise frequencies, shape (L, L, Q, Q).
    reg_lambda : float
        Regularization coefficient added on the diagonal of covariance.

    Returns
    -------
    C_inv : np.ndarray
        Inverse covariance matrix, shape (L*(Q-1), L*(Q-1)).
    """
    L, Q = fi.shape
    q = Q - 1  # use first Q-1 states

    dim = L * q
    C = np.zeros((dim, dim), dtype=np.float64)

    def idx(pos: int, state: int) -> int:
        """Map (position, state) -> flat index in [0, L*q)."""
        return pos * q + state

    for i in range(L):
        for j in range(L):
            # covariance: C_ij(a,b) = fij(i,j,a,b) - fi(i,a)*fi(j,b)
            for a in range(q):
                for b in range(q):
                    C[idx(i, a), idx(j, b)] = (
                        fij[i, j, a, b] - fi[i, a] * fi[j, b]
                    )

    # Regularization on diagonal
    C += reg_lambda * np.eye(dim)

    # Inverse covariance
    # For numerical stability, use pinv instead of inv
    C_inv = np.linalg.pinv(C)

    return C_inv


def _compute_di_from_couplings(C_inv: np.ndarray, L: int) -> np.ndarray:
    """
    Extract couplings from C_inv and compute a DI-like matrix.

    We compute:
    - J_ij(a,b) = -C_inv_ij(a,b)  for a,b in 0..q-1
    - Frobenius norm: F_ij = sqrt( sum_{a,b} J_ij(a,b)^2 )
    - APC correction: DI_ij = F_ij - (F_i * F_j) / F_mean

    Parameters
    ----------
    C_inv : np.ndarray
        Inverse covariance, shape (L*(Q-1), L*(Q-1)).
    L : int
        Sequence length.

    Returns
    -------
    di_apc : np.ndarray
        APC-corrected DI-like score matrix, shape (L, L).
    """
    q = C_inv.shape[0] // L  # Q-1
    dim = L * q
    assert dim == C_inv.shape[0]

    def idx(pos: int, state: int) -> int:
        return pos * q + state

    # Compute Frobenius norm of couplings for each pair
    F = np.zeros((L, L), dtype=np.float64)

    for i in range(L):
        for j in range(i + 1, L):
            # Block for positions (i, j)
            # J_ij = -C_inv_ij
            block = np.zeros((q, q), dtype=np.float64)
            for a in range(q):
                for b in range(q):
                    block[a, b] = -C_inv[idx(i, a), idx(j, b)]
            # Frobenius norm
            F_ij = np.sqrt(np.sum(block * block))
            F[i, j] = F_ij
            F[j, i] = F_ij

    # Average Product Correction (APC)
    row_mean = F.mean(axis=1)
    col_mean = F.mean(axis=0)
    global_mean = F.mean()

    # Avoid division by zero
    if global_mean == 0:
        return F

    apc = np.outer(row_mean, col_mean) / global_mean
    di_apc = F - apc

    # Set diagonal to 0
    np.fill_diagonal(di_apc, 0.0)

    return di_apc


def _build_di_graph(
    di_matrix: np.ndarray,
    top_k: int = 200,
    min_seq_sep: int = 3,
) -> List[Dict[str, Any]]:
    """
    Build a DI graph from a DI matrix.

    Parameters
    ----------
    di_matrix : np.ndarray
        DI-like scores, shape (L, L).
    top_k : int
        Number of top-scoring pairs to keep.
    min_seq_sep : int
        Minimum |i-j| to consider an edge (skip close neighbors).

    Returns
    -------
    edges : list of dict
        Each dict has keys:
        - "i": int (0-based residue index)
        - "j": int (0-based residue index)
        - "score": float (DI score)
    """
    L = di_matrix.shape[0]
    pairs = []

    for i in range(L):
        for j in range(i + 1, L):
            if abs(i - j) < min_seq_sep:
                continue
            score = di_matrix[i, j]
            pairs.append((score, i, j))

    # Sort by score descending
    pairs.sort(reverse=True, key=lambda x: x[0])

    if top_k is not None and top_k > 0:
        pairs = pairs[:top_k]

    edges: List[Dict[str, Any]] = []
    for score, i, j in pairs:
        edges.append({
            "i": int(i),
            "j": int(j),
            "score": float(score),
        })

    return edges
