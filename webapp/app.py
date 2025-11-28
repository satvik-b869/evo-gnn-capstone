
from flask import Flask, render_template, request, jsonify
import requests
import time
from msa_builder import run_msa
from compute_di import compute_di
from moduleb import compute_graph_embeddings
from modulec import compute_coordinates
from moduled import build_pseudo_backbone
import numpy as np


app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

NCBI_BLAST_URL = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"

# 🔥 Added: global storage for last seq_array
LAST_SEQ_ARRAY = None
LAST_DI_MATRIX = None
LAST_EIGENVECTORS = None




@app.route("/")
def index():
    return render_template("index.html")

@app.route("/step1")
def step1():
    return render_template("step1.html")

@app.route("/moduleb")
def moduleb():
    return render_template("moduleb.html")

@app.route("/modulec")
def modulec():
    return render_template("modulec.html")

@app.route("/moduled")
def moduled():
    return render_template("moduled.html")


@app.route("/generate_msa", methods=["POST"])
def generate_msa():
    global LAST_SEQ_ARRAY   # <— Added so we can assign to global variable

    print("🟢 Received request")

    start_time = time.time()
    print("⏳ Timer started...")

    seq = request.form.get("sequence", "").strip()
    if not seq:
        return jsonify({"error": "No sequence provided"}), 400

    # -----------------------------------------
    # 1. SUBMIT BLAST REQUEST
    # -----------------------------------------
    print("Submitting BLAST request...")

    params = {
        "CMD": "Put",
        "DATABASE": "nr",
        "PROGRAM": "blastp",
        "QUERY": seq
    }

    try:
        submit = requests.post(NCBI_BLAST_URL, data=params, timeout=20)
    except Exception as e:
        return jsonify({"error": f"NCBI connection failed: {e}"}), 500

    text = submit.text

    # Extract RID
    rid = None
    for line in text.splitlines():
        if "RID =" in line:
            rid = line.split("=")[-1].strip()

    if not rid:
        return jsonify({"error": "Failed to obtain RID from NCBI"}), 500

    print(f"RID: {rid}")

    # -----------------------------------------
    # 2. WAIT FOR RESULTS
    # -----------------------------------------
    print("Waiting for BLAST results...")

    while True:
        status_params = {
            "CMD": "Get",
            "RID": rid,
            "FORMAT_OBJECT": "SearchInfo"
        }

        r = requests.get(NCBI_BLAST_URL, params=status_params)

        if "Status=WAITING" in r.text:
            time.sleep(3)
            continue

        if "Status=FAILED" in r.text:
            return jsonify({"error": "NCBI BLAST failed"}), 500

        if "Status=READY" in r.text:
            break

    print("Results ready!")

    # -----------------------------------------
    # 3. DOWNLOAD ALIGNED SEQUENCES
    # -----------------------------------------
    print("Fetching alignments...")

    results_params = {
        "CMD": "Get",
        "RID": rid,
        "FORMAT_TYPE": "Alignment",
        "ALIGNMENT_VIEW": "3"
    }

    aligned = requests.get(NCBI_BLAST_URL, params=results_params).text

    print("\n===== RAW ALIGNED OUTPUT =====\n")
    print(aligned)
    print("\n===== END RAW OUTPUT =====\n")

    if ">" not in aligned:
        print("⚠ WARNING: BLAST output has no FASTA headers — using Sbjct parser.")

    # -----------------------------------------
    # 3B. Extract aligned sequences
    # -----------------------------------------
    seq_array = []
    current_seq = ""

    for line in aligned.splitlines():
        line = line.strip()

        if line.startswith(">"):
            if current_seq:
                seq_array.append(current_seq)
                current_seq = ""
            continue

        if line.startswith("Sbjct"):
            parts = line.split()
            if len(parts) >= 3:
                seq_fragment = parts[2]
                current_seq += seq_fragment

    if current_seq:
        seq_array.append(current_seq)

    print(f"Extracted {len(seq_array)} clean sequences.")
    print(seq_array)

    # 🔥 Save globally for Step 1 page to use later
    LAST_SEQ_ARRAY = seq_array

    msa_result = run_msa(seq_array)

    if "error" in msa_result:
        print("MSA error:", msa_result["error"])
        aligned_final = ""
    else:
        print("MSA finished!")
        aligned_final = msa_result["aligned_fasta"]
        print(aligned_final)

    # Timer end
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    print(f"⏱ Total BLAST+Parse Time: {duration} seconds")

    print("Returning MSA + sequence array")


    return jsonify({
        "status": "success",
        "runtime_seconds": duration,
        "msa": aligned,
        "msa_array": seq_array,
        "msa_muscle": aligned_final
    })


# --------------------------------------------------------
# 🔥 NEW ENDPOINT: Step1.html will call this on page load
# --------------------------------------------------------
@app.route("/compute_di_now", methods=["GET"])
def compute_di_now():
    global LAST_SEQ_ARRAY, LAST_DI_MATRIX

    if LAST_SEQ_ARRAY is None:
        return jsonify({"error": "No MSA stored. Run alignment first."}), 400

    try:
        # Compute DI
        di_result = compute_di(LAST_SEQ_ARRAY)

        # Save DI matrix globally for Module B
        LAST_DI_MATRIX = di_result["di_matrix"].tolist()

        # Return response to Step1.html
        return jsonify({
            "status": "success",
            "di_matrix": LAST_DI_MATRIX,
            "di_graph": di_result["di_graph"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/compute_moduleb")
def compute_moduleb():
    global LAST_DI_MATRIX, LAST_EIGENVECTORS

    if LAST_DI_MATRIX is None:
        return jsonify({"error": "No DI matrix available yet"}), 400

    try:
        di_mat = np.array(LAST_DI_MATRIX)
        result = compute_graph_embeddings(di_mat, k=8)

        # Save eigenvectors so Module C can access them
        LAST_EIGENVECTORS = result["eigenvectors"]

        return jsonify({
            "laplacian": result["laplacian"].tolist(),
            "eigenvalues": result["eigenvalues"].tolist(),
            "eigenvectors": result["eigenvectors"].tolist(),
            "embeddings": result["embeddings"].tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/compute_modulec")
def compute_modulec():
    global LAST_EIGENVECTORS

    if LAST_EIGENVECTORS is None:
        return jsonify({"error": "No eigenvectors available yet"}), 400

    try:
        eigenvectors = np.array(LAST_EIGENVECTORS)

        coords_2d = compute_coordinates(eigenvectors, dim=2)

        # Only compute 3D if eigenvectors have ≥ 3 components
        if eigenvectors.shape[1] >= 3:
            coords_3d = compute_coordinates(eigenvectors, dim=3)
            coords_3d_list = coords_3d.tolist()
        else:
            coords_3d_list = None

        return jsonify({
            "status": "success",
            "coords_2d": coords_2d.tolist(),
            "coords_3d": coords_3d_list
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/compute_moduled", methods=["GET"])
def compute_moduled():
    global LAST_EIGENVECTORS

    if LAST_EIGENVECTORS is None:
        return jsonify({"error": "No eigenvectors available. Run Module B first."}), 400

    try:
        evecs = np.array(LAST_EIGENVECTORS, dtype=float)
        result = build_pseudo_backbone(evecs, target_ca_dist=3.8)

        return jsonify({
            "status": "success",
            "coords_3d": result["coords_3d"].tolist(),
            "pdb": result["pdb_string"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500





if __name__ == "__main__":
    app.run(debug=True)
