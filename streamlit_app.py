import streamlit as st
import json
import time

from faiss_index_extractor import CMSDenialAnalyzer

analyzer = CMSDenialAnalyzer()

st.set_page_config(page_title="CMS Denial Analyzer", layout="wide")

st.title("🩺 CMS Denial Analyzer")
st.markdown("Paste a single claim or upload a batch file in `.json` or `.jsonl` format.")

# --- Manual Input ---
with st.expander("➕ Paste a Single Claim (JSON)", expanded=True):
    default = '''{
  "cpt_code": "99213",
  "diagnosis": "E11.9",
  "modifiers": ["25"],
  "payer": "Medicare"
}'''
    claim_input = st.text_area("Claim Input", value=default, height=200)

# --- Manual Analyze Button ---
if st.button("Analyze"):
    try:
        data = json.loads(claim_input.strip())
        output_placeholder = st.empty()
        status = st.empty()
        results = []
        start = time.time()

        if isinstance(data, list):
            total = len(data)
            progress_bar = st.progress(0)

            for i, claim in enumerate(data, start=1):
                result = analyzer.analyze_claim(claim)
                results.append(f"### Claim {i}\n```\n{result}\n```")
                progress_bar.progress(i / total)
                status.text(f"🔄 Processing claim {i}/{total}...")
                time.sleep(0.001)

            elapsed = time.time() - start
            progress_bar.empty()
            status.success(f"✅ Batch processed in {elapsed:.2f} seconds.")
            output_placeholder.markdown("\n\n".join(results))

        else:
            result = analyzer.analyze_claim(data)
            elapsed = time.time() - start
            status.success(f"✅ Analyzed in {elapsed:.2f} seconds.")
            output_placeholder.code(result)

    except Exception as e:
        st.error(f"Error: {e}")

# --- File Upload ---
st.markdown("---")
uploaded_file = st.file_uploader("📁 Or Upload Batch File (.json or .jsonl)", type=["json", "jsonl"])

if uploaded_file:
    try:
        content = uploaded_file.read().decode("utf-8")
        if uploaded_file.name.endswith(".jsonl"):
            claims = [json.loads(line) for line in content.splitlines()]
        else:
            claims = json.loads(content)

        output_placeholder = st.empty()
        results = []
        total = len(claims)
        progress_bar = st.progress(0)
        status = st.empty()
        start = time.time()

        for i, claim in enumerate(claims, start=1):
            result = analyzer.analyze_claim(claim)
            results.append(f"### Claim {i}\n```\n{result}\n```")
            progress_bar.progress(i / total)
            status.text(f"🔄 Processing claim {i}/{total}...")
            time.sleep(0.001)

        elapsed = time.time() - start
        progress_bar.empty()
        status.success(f"✅ File processed in {elapsed:.2f} seconds.")
        output_placeholder.markdown("\n\n".join(results))

    except Exception as e:
        st.error(f"Error processing file: {e}")
