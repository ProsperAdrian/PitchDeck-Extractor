# ─────────────────────────────────────────────────────────────────────────────
# streamlit_app.py
#
# • Ensure st.set_page_config is first.
# • Use @st.cache_data to cache each PDF’s “parsed JSON” by filename+bytes.
# • Only call ChatGPT when a file is first uploaded (or when its bytes change).
# • Subsequent tab switches / filtering / sorting / re-drawing use cached data.
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import os
import json
import hashlib
import fitz                           # PyMuPDF, for rendering PDF pages as images
from openai import OpenAI

from extract_text import extract_text_from_pdf
from analyze import build_few_shot_prompt, call_chatgpt

# ─────────────────────────────────────────────────────────────────────────────
# 1) STREAMLIT PAGE CONFIG (MUST BE FIRST)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pitch Deck Extractor",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# 2) GET YOUR OPENAI KEY FROM STREAMLIT SECRETS
# ─────────────────────────────────────────────────────────────────────────────
openai_api_key = st.secrets["openai"]["api_key"]

# ─────────────────────────────────────────────────────────────────────────────
# 3) HIDE “use_column_width” DEPRECATION WARNINGS VIA CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Hide yellow‐box deprecation warnings about use_column_width */
        .stAlert, .stAlertWarning {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# 4) CUSTOM CSS FOR A SMOOTH LOOK
# ─────────────────────────────────────────────────────────────────────────────
def set_custom_styles():
    css = """
    <style>
    .stApp {
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .block-container {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 1rem;
        padding: 2rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
    }
    h1 {
        font-size: 24px !important;
        font-weight: 700;
    }
    .uploaded-filename, .processing-msg, .success-msg, .extracted-title {
        font-size: 14px;
        font-weight: 500;
        color: #222;
        margin-top: -0.1rem;
        margin-bottom: 2.5rem;
    }
    .stButton>button {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        background-color: #3A86FF;
        color: white;
        font-weight: 600;
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #265DAB;
        transform: scale(1.02);
    }
    .narrow-uploader {
        max-width: 500px;
        margin-left: auto;
        margin-right: auto;
    }
    div[data-testid="stFileUploader"] > div > div:nth-child(2),
    div[data-testid="stFileUploader"] ul,
    div[data-testid="stFileUploader"] li {
        display: none !important;
    }
    .success-msg-container {
        font-size: 14px;
        font-weight: 500;
        color: #222;
        margin-top: -0.5rem;
        margin-bottom: 3rem;
    }
    .extracted-title {
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_custom_styles()

# ─────────────────────────────────────────────────────────────────────────────
# 5) APP TITLE & DESCRIPTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<h1>📊 Pitch Deck Extractor</h1>', unsafe_allow_html=True)
st.markdown("""
Upload one or more pitch‐deck PDFs. This tool leverages AI & heuristics to extract:
**Startup Name**, **Founders**, **Founding Year**, **Industry**, **Niche**, **USP**, **Funding Stage**, **Current Revenue**, **Market Size**, and **Amount Raised**.
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 6) CREATE TWO TABS: LIBRARY VIEW & DASHBOARD VIEW
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["1️⃣ Library View", "2️⃣ Dashboard & Interactive Filtering"])


# ─────────────────────────────────────────────────────────────────────────────
# 7) CACHE FUNCTION: EXTRACT & CALL CHATGPT EXACTLY ONCE PER PDF CONTENT
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def parse_deck_with_chatgpt(pdf_bytes: bytes) -> dict:
    """
    Given raw bytes of a PDF, extract its text, build a few‐shot prompt, call ChatGPT,
    and return the parsed JSON. This is cached (keyed by pdf_bytes hash), so repeated
    calls with identical PDF bytes will reuse the cached result instead of sending
    a new request to OpenAI.
    """
    # 7a) Write the bytes to a temporary file (so extract_text can read it)
    tmp_folder = "temp_cache"
    os.makedirs(tmp_folder, exist_ok=True)
    # Use a hash of PDF bytes as the temporary filename
    digest = hashlib.sha256(pdf_bytes).hexdigest()
    temp_path = os.path.join(tmp_folder, f"{digest}.pdf")
    with open(temp_path, "wb") as f:
        f.write(pdf_bytes)

    # 7b) Extract all text from PDF
    deck_text = extract_text_from_pdf(temp_path)
    os.remove(temp_path)

    # 7c) Build the few‐shot prompt & call ChatGPT
    prompt = build_few_shot_prompt(deck_text)
    result = call_chatgpt(prompt, api_key=openai_api_key)

    # 7d) Normalize: ensure all expected keys exist
    expected_keys = [
        "StartupName", "Startup Name",
        "FoundingYear", "Founding Year",
        "Founders", "Industry", "Niche", "USP",
        "FundingStage", "Funding Stage",
        "CurrentRevenue", "Current Revenue",
        "Market", "AmountRaised", "Amount Raised"
    ]

    normalized = {}

    # We prefer the underscored key (e.g. “StartupName”) if present, else the spaced one.
    # Then we rename everything to our “Library View” column names.
    normalized["Startup Name"]    = result.get("StartupName") or result.get("Startup Name") or None
    normalized["Founding Year"]   = result.get("FoundingYear") or result.get("Founding Year") or None
    normalized["Founders"]        = result.get("Founders") or []
    normalized["Industry"]        = result.get("Industry") or None
    normalized["Niche"]           = result.get("Niche") or None
    normalized["USP"]             = result.get("USP") or None
    normalized["Funding Stage"]   = result.get("FundingStage") or result.get("Funding Stage") or None
    normalized["Current Revenue"] = result.get("CurrentRevenue") or result.get("Current Revenue") or None
    normalized["Amount Raised"]   = result.get("AmountRaised") or result.get("Amount Raised") or None

    # Normalize market sub‐keys
    m = result.get("Market") or {}
    normalized["Market"] = {
        "TAM": m.get("TAM") or None,
        "SAM": m.get("SAM") or None,
        "SOM": m.get("SOM") or None
    }

    # Keep the raw JSON as well
    normalized["__raw_json"] = result
    return normalized


# ─────────────────────────────────────────────────────────────────────────────
# 8) TAB 1: LIBRARY VIEW → UPLOAD + PARSE (CACHED) + RENDER
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    # 8a) FILE UPLOADER (centered)
    st.markdown('<div class="narrow-uploader">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drag & drop PDF(s) here (or click to browse)", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Prepare containers for results
    all_metadata = []         # List of dicts to build our “Library” DataFrame
    pdf_buffers    = {}       # Map filename → raw PDF bytes for later “key slide” preview

    if uploaded_files:
        with st.spinner("🔎 Analyzing pitch decks..."):
            for pdf_file in uploaded_files:
                filename = pdf_file.name
                st.markdown(
                    f'<div class="uploaded-filename">Processing <strong>{filename}</strong>…</div>',
                    unsafe_allow_html=True,
                )
                raw_bytes = pdf_file.read()

                # 8b) Invoke cached function parse_deck_with_chatgpt(raw_bytes)
                #     → if this exact PDF‐bytes has already been parsed, no new cost.
                metadata = parse_deck_with_chatgpt(raw_bytes)

                # Attach filename & store
                metadata["Filename"] = filename
                all_metadata.append(metadata)

                # Store PDF bytes for key‐slide preview
                pdf_buffers[filename] = raw_bytes

        # 8c) DISPLAY LIBRARY TABLE + EXPORT BUTTONS
        if all_metadata:
            st.markdown(
                """
                <div class="success-msg-container">
                    ✅ All pitch decks processed successfully!
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Build DataFrame for “Library View”
            rows = []
            for rec in all_metadata:
                row = {
                    "Filename": rec["Filename"],
                    "Startup Name": rec["Startup Name"],
                    "Founding Year": rec["Founding Year"],
                    "Founders": "; ".join(rec["Founders"]),
                    "Industry": rec["Industry"],
                    "Niche": rec["Niche"],
                    "USP": rec["USP"],
                    "Funding Stage": rec["Funding Stage"],
                    "Current Revenue": rec["Current Revenue"],
                    "Amount Raised": rec["Amount Raised"],
                    "TAM": rec["Market"]["TAM"],
                    "SAM": rec["Market"]["SAM"],
                    "SOM": rec["Market"]["SOM"],
                }
                rows.append(row)

            df = pd.DataFrame(rows)

            st.markdown('<div class="extracted-title">📑 Library</div>', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)

            # EXPORT BUTTONS
            json_str  = json.dumps([rec["__raw_json"] for rec in all_metadata], indent=2)
            csv_bytes = df.to_csv(index=False).encode("utf-8")

            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    label="📥 Export as JSON",
                    data=json_str,
                    file_name="All_decks.json",
                    mime="application/json",
                )
            with c2:
                st.download_button(
                    label="📊 Export as CSV",
                    data=csv_bytes,
                    file_name="All_decks.csv",
                    mime="text/csv",
                )

            st.markdown("---")

            # 8d) KEY SLIDE PREVIEW (CHATGPT‐DRIVEN PAGE NUMBERS)
            st.markdown("### 🔑 Key Slide Preview")
            st.markdown("Select a deck from the table above to preview its important slides (Team, Market, Traction).")

            selected_deck = st.selectbox(
                "❓ Which Deck would you like to preview?",
                options=df["Filename"].tolist()
            )

            if selected_deck:
                pdf_bytes = pdf_buffers[selected_deck]
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")

                # 8d‐i) Extract text for each page
                page_texts = [page.get_text() for page in doc]

                # 8d‐ii) Ask ChatGPT to identify key pages (Team, Market, Traction)
                key_info = identify_key_slide_pages(page_texts, api_key=openai_api_key)

                # Convert 1-indexed pages to 0-indexed
                raw_team     = key_info.get("TeamPage")
                raw_market   = key_info.get("MarketPage")
                raw_traction = key_info.get("TractionPage")
                team_idx     = (int(raw_team) - 1) if raw_team else None
                market_idx   = (int(raw_market) - 1) if raw_market else None
                traction_idx = (int(raw_traction) - 1) if raw_traction else None

                # Build a list of (label, page_index) for whichever exist
                key_slides = []
                if isinstance(team_idx, int) and 0 <= team_idx < doc.page_count:
                    key_slides.append((f"Team Slide (page {team_idx+1})", team_idx))
                if isinstance(market_idx, int) and 0 <= market_idx < doc.page_count:
                    key_slides.append((f"Market Slide (page {market_idx+1})", market_idx))
                if isinstance(traction_idx, int) and 0 <= traction_idx < doc.page_count:
                    key_slides.append((f"Traction Slide (page {traction_idx+1})", traction_idx))

                if not key_slides:
                    st.warning("⚠️ ChatGPT did not locate Team/Market/Traction slides in this deck.")
                else:
                    cols = st.columns(len(key_slides))
                    for col, (label, page_index) in zip(cols, key_slides):
                        page = doc[page_index]
                        pix = page.get_pixmap(dpi=100)
                        img_bytes = pix.tobytes("png")  # render as PNG
                        col.image(img_bytes, caption=label, use_container_width=True)

                doc.close()


# ─────────────────────────────────────────────────────────────────────────────
# 9) TAB 2: DASHBOARD & INTERACTIVE FILTERING
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("## 📊 Dashboard & Interactive Filtering")

    # If no PDF has been processed yet, prompt the user to go back to Tab 1
    if "all_metadata" not in locals() or not all_metadata:
        st.warning("Upload at least one PDF in the Library View first, then come here to see the Dashboard.")
    else:
        # Build a smaller DataFrame for filtering/charts
        rows2 = []
        for rec in all_metadata:
            startup_name  = rec["Startup Name"]
            # Convert founding year to int if possible
            fy = rec["Founding Year"]
            try:
                fy = int(fy)
            except:
                fy = None
            industry      = rec["Industry"]
            funding_stage = rec["Funding Stage"]

            rows2.append({
                "Filename": rec["Filename"],
                "Startup Name": startup_name,
                "Founding Year": fy,
                "Industry": industry,
                "Funding Stage": funding_stage
            })

        df2 = pd.DataFrame(rows2)

        # ── Sidebar Filters ──────────────────────────────────────────────────
        st.sidebar.header("🔎 Filters")

        all_industries = sorted([i for i in df2["Industry"].unique() if pd.notna(i)])
        sel_industries = st.sidebar.multiselect("Industry", options=all_industries, default=all_industries)

        valid_years = df2["Founding Year"].dropna().astype(int)
        if not valid_years.empty:
            min_year = int(valid_years.min())
            max_year = int(valid_years.max())
            sel_year_range = st.sidebar.slider(
                "Founding Year Range",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
            )
        else:
            sel_year_range = (0, 9999)

        all_stages = sorted([s for s in df2["Funding Stage"].unique() if pd.notna(s)])
        sel_stages = st.sidebar.multiselect("Funding Stage", options=all_stages, default=all_stages)

        # ── Apply Filters ────────────────────────────────────────────────────
        if valid_years.empty:
            filtered = df2[
                (df2["Industry"].isin(sel_industries)) &
                (df2["Funding Stage"].isin(sel_stages))
            ]
        else:
            filtered = df2[
                (df2["Industry"].isin(sel_industries)) &
                (df2["Funding Stage"].isin(sel_stages)) &
                (df2["Founding Year"].between(sel_year_range[0], sel_year_range[1]))
            ]

        st.markdown(f"#### 🔍 {filtered.shape[0]} startups match your filters")

        if not filtered.empty:
            # Industry Breakdown (bar chart)
            st.markdown("**Industry Breakdown**")
            industry_counts = filtered["Industry"].value_counts()
            st.bar_chart(industry_counts)

            # Founding Year Distribution (bar chart)
            st.markdown("**Founding Year Distribution**")
            year_counts = filtered["Founding Year"].value_counts().sort_index()
            st.bar_chart(year_counts)

            # Funding Stage Breakdown (bar chart)
            st.markdown("**Funding Stage Breakdown**")
            stage_counts = filtered["Funding Stage"].value_counts()
            st.bar_chart(stage_counts)

        st.markdown("---")
        st.markdown("### 💾 Filtered Results Table")
        st.dataframe(filtered, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 10) HELPER FUNCTION: ASK CHATGPT FOR KEY SLIDE PAGES
# ─────────────────────────────────────────────────────────────────────────────
def identify_key_slide_pages(page_texts: list[str], api_key: str) -> dict:
    """
    Given a list of page_texts, ask ChatGPT: "Which page # is Team / Market / Traction?"
    Returns e.g. {"TeamPage": 7, "MarketPage": 5, "TractionPage": 15} or nulls if no match.
    """
    prompt_lines = [
        "I will give you short text snippets from each slide of a pitch deck, one snippet per page. "
        "Identify EXACTLY which page number (1-indexed) is the Team slide, "
        "which page is the Market slide, and which page is the Traction slide. "
        "If you cannot find one of those categories, return null for that field. "
        "Answer in JSON format like:\n",
        "{\n"
        '  "TeamPage": 7,\n'
        '  "MarketPage": 5,\n'
        '  "TractionPage": 15\n'
        "}\n"
    ]

    # Append each page’s snippet (first 200 chars) to the prompt
    for i, text in enumerate(page_texts):
        snippet = text.replace("\n", " ").strip()[:200]
        prompt_lines.append(f"---\nPage {i+1}:\n{snippet}\n")

    final_prompt = "\n".join(prompt_lines)
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0.0,
        max_tokens=200,
    )
    content = response.choices[0].message.content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end != -1:
            try:
                return json.loads(content[start:end])
            except:
                pass
        return {"TeamPage": None, "MarketPage": None, "TractionPage": None}
