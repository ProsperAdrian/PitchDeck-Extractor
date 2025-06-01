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

# Inject modern design
def set_background():
    page_bg_img = f"""
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
        font-size: 2.4rem;
        font-weight: 700;
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
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background()

# ----------------- App Title -----------------
st.title("📊 Pitch Deck Extractor")
st.markdown("""
Upload one or more pitch-deck PDFs. This tool leverages AI & predefined heuristics to extract:
**Startup Name**, **Founders**, **Founding Year**, **Industry**, **Funding Stage**, **Revenue**, **Market (TAM/SAM/SOM)**.
""")

# ----------------- File Upload -----------------
with st.container():
    st.subheader("📂 Upload Pitch Decks")
    uploaded_files = st.file_uploader(
        "Drag & drop PDF(s) here (or click to browse)", 
        type=["pdf"],
        accept_multiple_files=True,
    )

# ----------------- Processing -----------------
if uploaded_files:
    with st.spinner("🔎 Analyzing pitch decks..."):
        all_results = []

        for pdf_file in uploaded_files:
            st.markdown(f"#### Processing: `{pdf_file.name}`")
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
        st.success("✅ All pitch decks processed successfully!")

        rows = []
        for rec in all_results:
            row = {
                "Filename": rec.get("__filename"),
                "Startup Name": rec.get("StartupName"),
                "Founding Year": rec.get("FoundingYear"),
                "Founders": "; ".join(rec.get("Founders") or []),
                "Industry": rec.get("Industry"),
                "Niche": rec.get("Niche"),
                "USP": rec.get("USP"),
                "Funding Stage": rec.get("FundingStage"),
                "Current Revenue": rec.get("CurrentRevenue"),
                "Amount Raised": rec.get("AmountRaised"),
            }
            market = rec.get("Market") or {}
            row["TAM"] = market.get("TAM")
            row["SAM"] = market.get("SAM")
            row["SOM"] = market.get("SOM")
            rows.append(row)

        df = pd.DataFrame(rows)

        st.markdown("### 📑 Extracted Results")
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
