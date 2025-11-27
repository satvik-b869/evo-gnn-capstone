from flask import Flask, render_template, request, jsonify
import requests
import time
from msa_builder import run_msa


app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

NCBI_BLAST_URL = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate_msa", methods=["POST"])
def generate_msa():
    print("🟢 Received request")

    start_time = time.time()  # TIMER START
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
    # 3B. Extract aligned sequences from BLAST pairwise output
    # -----------------------------------------
    seq_array = []
    current_seq = ""

    for line in aligned.splitlines():
        line = line.strip()

        # New sequence begins
        if line.startswith(">"):
            if current_seq:
                seq_array.append(current_seq)
                current_seq = ""
            continue

        # Extract Sbjct fragments
        if line.startswith("Sbjct"):
            parts = line.split()
            if len(parts) >= 3:
                seq_fragment = parts[2]
                current_seq += seq_fragment

    # add the final sequence
    if current_seq:
        seq_array.append(current_seq)

    print(f"Extracted {len(seq_array)} clean sequences.")
    print(seq_array)
    msa_result = run_msa(seq_array)

    if "error" in msa_result:
        print("MSA error:", msa_result["error"])
        aligned_final = ""
    else:
        print("MSA finished!")
        aligned_final = msa_result["aligned_fasta"]
        print(aligned_final)



    # -----------------------------------------
    # TIMER END
    # -----------------------------------------
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    print(f"⏱ Total BLAST+Parse Time: {duration} seconds")

    # -----------------------------------------
    # 4. RETURN EVERYTHING
    # -----------------------------------------
    print("Returning MSA + sequence array")

    return jsonify({
        "status": "success",
        "runtime_seconds": duration,
        "msa": aligned,
        "msa_array": seq_array,
        "msa_muscle": aligned_final
    })

if __name__ == "__main__":
    app.run(debug=True)
