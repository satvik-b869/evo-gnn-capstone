import numpy as np
from Bio import SeqIO
import os

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY-")  # standard 20 aa + gap

def load_msa(msa_path):
    sequences = []
    for rec in SeqIO.parse(msa_path, "fasta"):
        sequences.append(str(rec.seq))
    if len(sequences) == 0:
        raise ValueError("MSA file empty")
    L = len(sequences[0])
    for seq in sequences:
        if len(seq) != L:
            raise ValueError("MSA sequences not same length")
    return sequences

def compute_frequencies(msa):
    N = len(msa)
    L = len(msa[0])
    aa_to_index = {aa:i for i, aa in enumerate(AMINO_ACIDS)}
    counts = np.zeros((L, len(AMINO_ACIDS)))

    for seq in msa:
        for i, aa in enumerate(seq):
            idx = aa_to_index.get(aa, aa_to_index['-'])
            counts[i][idx] += 1

    freqs = counts / N
    return freqs

if __name__ == "__main__":
    msa_path = "data/msa/input.a3m"
    msa = load_msa(msa_path)
    freqs = compute_frequencies(msa)
    os.makedirs("data/di", exist_ok=True)
    np.save("data/di/p_single_freq.npy", freqs)
    print("Saved frequencies to data/di/p_single_freq.npy")
