import streamlit as st
import pandas as pd
import os, json
import openai

from extract_text import extract_text_from_pdf
from analyze import build_few_shot_prompt, call_chatgpt

openai_api_key = st.secrets["openai"]["api_key"]

# ----------------- UI Setup -----------------
st.set_page_config(
    page_title="Pitch Deck Extractor",
    layout="wide",
)

# Inject modern design with font control and hidden uploader files
def set_custom_styles():
    custom_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,YOUR_BASE64_IMAGE_HERE");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .block-container {{
        background-color: rgba(255, 255, 255, 0.92);
        border-radius: 1rem;
        padding: 2rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
    }}
    h1 {{
        font-size: 24px !important;
        font-weight: 700;
    }}
    .custom-subheader {{
        font-size: 16px;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 0.5rem;
    }}
    .uploaded-filename, .processing-msg, .success-msg, .extracted-title {{
        font-size: 14px;
        font-weight: 500;
        color: #222;
        margin: 0.2rem 0;
    }}
    .stButton>button {{
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        background-color: #3A86FF;
        color: white;
        font-weight: 600;
        transition: 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: #265DAB;
        transform: scale(1.02);
    }}
    .narrow-uploader {{
    max-width: 500px;
    margin-left: auto;
    margin-right: auto;
    }}
    /* Hide uploaded file display */
    div[data-testid="stFileUploader"] > div > div:nth-child(2) {{
        display: none !important;
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

set_custom_styles()

# ----------------- App Title -----------------
st.markdown('<h1>📊 Pitch Deck Analysis</h1>', unsafe_allow_html=True)
st.markdown("""
Upload one or more pitch-deck PDFs. This tool leverages AI & predefined heuristics to extract:
**Startup Name**, **Founders**, **Founding Year**, **Industry**, **Niche**, **USP**, **Funding Stage**, **Revenue**, **Market Size**, **Amount Raised**.
""")

# ----------------- File Upload -----------------
st.markdown('<div class="narrow-uploader">', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Drag & drop PDF(s) here (or click to browse)", 
    type=["pdf"],
    accept_multiple_files=True,
)
st.markdown('</div>', unsafe_allow_html=True)

# ----------------- Processing -----------------
if uploaded_files:
    with st.spinner("🔎 Analyzing pitch decks..."):
        all_results = []

        for pdf_file in uploaded_files:
            st.markdown(
                f'<div class="processing-msg">Processing: <code>{pdf_file.name}</code></div>',
                unsafe_allow_html=True
            )
            bytes_data = pdf_file.read()

            with open(f"temp_{pdf_file.name}", "wb") as f:
                f.write(bytes_data)

            try:
                deck_text = extract_text_from_pdf(f"temp_{pdf_file.name}")
                os.remove(f"temp_{pdf_file.name}")
                prompt = build_few_shot_prompt(deck_text)
                result = call_chatgpt(prompt, api_key=openai_api_key)
                result["__filename"] = pdf_file.name
                all_results.append(result)

            except Exception as e:
                st.error(f"❌ Error processing {pdf_file.name}: {e}")
                continue

    # ----------------- Display Results -----------------
    if all_results:
        st.markdown(
            '<div class="success-msg">✅ All pitch decks processed successfully!</div>',
            unsafe_allow_html=True
        )

        rows = []
        for rec in all_results:
            row = {
                "Filename": rec.get("__filename"),
                "Startup Name": rec.get("Startup Name"),
                "Founding Year": rec.get("Founding Year"),
                "Founders": "; ".join(rec.get("Founders") or []),
                "Industry": rec.get("Industry"),
                "Niche": rec.get("Niche"),
                "USP": rec.get("USP"),
                "Funding Stage": rec.get("Funding Stage"),
                "Current Revenue": rec.get("Current Revenue"),
                "Amount Raised": rec.get("Amount Raised"),
            }
            market = rec.get("Market") or {}
            row["TAM"] = market.get("TAM")
            row["SAM"] = market.get("SAM")
            row["SOM"] = market.get("SOM")
            rows.append(row)

        df = pd.DataFrame(rows)

        st.markdown('<div class="extracted-title">📑 Library</div>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

        # ----------------- Export Options -----------------
        json_str = json.dumps(all_results, indent=2)
        csv_str = df.to_csv(index=False).encode("utf-8")

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Export as JSON", 
                data=json_str, 
                file_name="All_decks.json",
                mime="application/json"
            )
        with col2:
            st.download_button(
                label="📊 Export as CSV", 
                data=csv_str,
                file_name="All_decks.csv",
                mime="text/csv"
            )
