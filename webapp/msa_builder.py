# msa_builder.py
import subprocess
import tempfile
import os

def run_msa(seq_array):
    """
    Takes a list of raw unaligned sequences and returns an aligned MSA in FASTA format.
    Uses MUSCLE v3.8.
    """

    if len(seq_array) < 2:
        return {"error": "Need at least 2 sequences to build an MSA"}

    # ---------------------------------------
    # 1. Create a temporary FASTA file
    # ---------------------------------------
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fasta") as tmp_in:
        for i, seq in enumerate(seq_array):
            clean = seq.strip().replace(" ", "").replace("\n", "")
            tmp_in.write(f">seq{i}\n{clean}\n")
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path + ".aligned.fasta"

    # ---------------------------------------
    # 2. Run MUSCLE v3 syntax
    # ---------------------------------------
    try:
        cmd = [
            "muscle",
            "-in", tmp_in_path,
            "-out", tmp_out_path
        ]

        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

    except subprocess.CalledProcessError as e:
        print("❌ MUSCLE ERROR:")
        print(e.stderr.decode("utf-8", errors="ignore"))
        return {
            "error": "MUSCLE failed",
            "details": e.stderr.decode("utf-8", errors="ignore")
        }

    # ---------------------------------------
    # 3. Verify output file exists and is non empty
    # ---------------------------------------
    if not os.path.exists(tmp_out_path):
        return {"error": "MUSCLE did not produce output"}
    if os.path.getsize(tmp_out_path) == 0:
        return {"error": "MUSCLE output file is empty"}

    # ---------------------------------------
    # 4. Return aligned MSA
    # ---------------------------------------
    with open(tmp_out_path, "r") as f:
        aligned = f.read()

    os.remove(tmp_in_path)
    os.remove(tmp_out_path)

    return {"aligned_fasta": aligned}
