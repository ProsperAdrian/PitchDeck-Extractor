# streamlit_app.py

import streamlit as st
import pandas as pd
import os, json
import fitz  # PyMuPDF, to render PDF pages as images

from extract_text import extract_text_from_pdf
from analyze import build_few_shot_prompt, call_chatgpt

# --- Pull OpenAI key from Streamlit secrets ---
openai_api_key = st.secrets["openai"]["api_key"]

# ----------------- UI SETUP -----------------
st.set_page_config(
    page_title="Pitch Deck Extractor",
    layout="wide",
)

def set_custom_styles():
    custom_css = f"""
    <style>
    .stApp {{
        /* Optional: set a background image as base64 if you like */
        /* background-image: url("data:image/png;base64,YOUR_BASE64_HERE"); */
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
        margin-top: -0.1rem;
        margin-bottom: 2.5rem;
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
    div[data-testid="stFileUploader"] > div > div:nth-child(2),
    div[data-testid="stFileUploader"] ul,
    div[data-testid="stFileUploader"] li {{
        display: none !important;
    }}
    .success-msg-container {{
        font-size: 14px;
        font-weight: 500;
        color: #222;
        margin-top: -0.5rem;
        margin-bottom: 3rem;
    }}
    .extracted-title {{
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

set_custom_styles()

# ----------------- App Title and Description -----------------
st.markdown('<h1>📊 Pitch Deck Analysis</h1>', unsafe_allow_html=True)
st.markdown("""
Upload one or more pitch‐deck PDFs. This tool leverages AI & predefined heuristics to extract:<br>
**Startup Name**, **Founders**, **Founding Year**, **Industry**, **Niche**, **USP**, **Funding Stage**, **Revenue**, **Market Size**, and **Amount Raised**.
""", unsafe_allow_html=True)

# Create two tabs: Library View and Dashboard View
tab1, tab2 = st.tabs(["1️⃣ Library View", "2️⃣ Dashboard View"])

#
# ----------------- TAB 1: LIBRARY VIEW -----------------
#
with tab1:
    # Center the uploader
    st.markdown('<div class="narrow-uploader">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drag & drop PDF(s) here (or click to browse)", 
        type=["pdf"],
        accept_multiple_files=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    all_results = []
    pdf_buffers = {}   # Keep the raw bytes so we can preview key slides
    df = None          # Will hold the DataFrame once extraction is done

    if uploaded_files:
        with st.spinner("🔎 Analyzing pitch decks..."):
            # Ensure a temp folder exists
            os.makedirs("temp", exist_ok=True)

            for pdf_file in uploaded_files:
                st.markdown(
                    f'<div class="uploaded-filename">Processing <strong>{pdf_file.name}</strong>…</div>',
                    unsafe_allow_html=True
                )

                # Save file‐buffer to disk so extract_text can read it
                bytes_data = pdf_file.read()
                temp_path = os.path.join("temp", pdf_file.name)
                with open(temp_path, "wb") as f:
                    f.write(bytes_data)

                try:
                    # 1) Extract full text from PDF
                    deck_text = extract_text_from_pdf(temp_path)
                    os.remove(temp_path)

                    # 2) Build the few‐shot prompt & call GPT
                    prompt = build_few_shot_prompt(deck_text)
                    result = call_chatgpt(prompt, api_key=openai_api_key)
                    result["__filename"] = pdf_file.name
                    all_results.append(result)

                    # 3) Keep raw bytes for key‐slide previews
                    pdf_buffers[pdf_file.name] = bytes_data

                except Exception as e:
                    st.error(f"❌ Error processing **{pdf_file.name}**: {e}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    continue

        if all_results:
            st.markdown("""
            <div class="success-msg-container">
                ✅ All pitch decks processed successfully!
            </div>
            """, unsafe_allow_html=True)

            # Build a Pandas DataFrame of extracted results
            rows = []
            for rec in all_results:
                startup_name  = rec.get("StartupName") or rec.get("Startup Name")
                founding_year = rec.get("FoundingYear") or rec.get("Founding Year")
                founders      = rec.get("Founders") or []
                industry      = rec.get("Industry")
                niche         = rec.get("Niche")
                usp           = rec.get("USP")
                funding_stage = rec.get("FundingStage") or rec.get("Funding Stage")
                curr_rev      = rec.get("CurrentRevenue") or rec.get("Current Revenue")
                amt_raised    = rec.get("AmountRaised") or rec.get("Amount Raised")

                row = {
                    "Filename": rec.get("__filename"),
                    "Startup Name": startup_name,
                    "Founding Year": founding_year,
                    "Founders": "; ".join(founders),
                    "Industry": industry,
                    "Niche": niche,
                    "USP": usp,
                    "Funding Stage": funding_stage,
                    "Current Revenue": curr_rev,
                    "Amount Raised": amt_raised,
                }
                market = rec.get("Market") or {}
                row["TAM"] = market.get("TAM")
                row["SAM"] = market.get("SAM")
                row["SOM"] = market.get("SOM")
                rows.append(row)

            df = pd.DataFrame(rows)

            st.markdown('<div class="extracted-title">📑 Library</div>', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)

            # ----------------- Export Buttons -----------------
            json_str = json.dumps(all_results, indent=2)
            csv_str  = df.to_csv(index=False).encode("utf-8")

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

            st.markdown("---")

    #
    # ----------------- Key Slide Preview -----------------
    #
    if df is not None and not df.empty:
        st.markdown("### 🔑 Key Slide Preview")
        st.markdown(
            "Select a deck from the table above to preview its important slides (Team, Market, Traction)."
        )

        options = df["Filename"].tolist()
        selected_deck = st.selectbox(
            "❓ Which Deck would you like to preview?",
            options=options
        )

        if selected_deck:
            pdf_bytes = pdf_buffers[selected_deck]
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            # We will pick THREE distinct pages: Team, Market, Traction
            team_idx = None
            market_idx = None
            traction_idx = None

            # (1) First pass: find the Team slide: look for “team” or “founder”
            for i in range(doc.page_count):
                text = doc[i].get_text().lower()
                if ("team" in text or "founder" in text) and team_idx is None:
                    team_idx = i

            # (2) Next pass: find the Market slide: look for explicit “tam” / “market size” / “market opportunity” / “sam” / “som”
            for i in range(doc.page_count):
                if team_idx is not None and i == team_idx:
                    continue
                text = doc[i].get_text().lower()
                if (
                    "tam" in text
                    or "market size" in text
                    or "market opportunity" in text
                    or "sam" in text
                    or "som" in text
                ) and market_idx is None:
                    market_idx = i

            # (3) Final pass: find the Traction slide: look for “traction” or “gross revenue” or “revenue”
            for i in range(doc.page_count):
                if i == team_idx or i == market_idx:
                    continue
                text = doc[i].get_text().lower()
                if (
                    "traction" in text
                    or "gross revenue" in text
                    or "revenue" in text
                ) and traction_idx is None:
                    traction_idx = i

            key_slides = {}
            if team_idx is not None:
                key_slides["Team Slide"] = team_idx
            if market_idx is not None:
                key_slides["Market Slide"] = market_idx
            if traction_idx is not None:
                key_slides["Traction Slide"] = traction_idx

            if not key_slides:
                st.info("⚠️ Unable to auto‐detect Team/Market/Traction slides in this deck.")
            else:
                cols = st.columns(len(key_slides))
                for col, (label, idx) in zip(cols, key_slides.items()):
                    if idx < doc.page_count:
                        page = doc[idx]
                        pix = page.get_pixmap(dpi=100)
                        img_bytes = pix.tobytes("png")
                        col.image(
                            img_bytes,
                            caption=f"{label} (page {idx+1})",
                            use_container_width=True
                        )
                    else:
                        col.write(f"❌ {label} out of range")

            doc.close()

#
# ----------------- TAB 2: DASHBOARD VIEW -----------------
#
with tab2:
    st.markdown("## 📊 Dashboard & Interactive Filtering")

    if not all_results:
        st.warning("Upload at least one PDF in the Library View first, then come here to see the Dashboard.")
    else:
        # Reconstruct a small DataFrame for filtering & charts
        rows2 = []
        for rec in all_results:
            startup_name = rec.get("StartupName") or rec.get("Startup Name")
            fy_raw       = rec.get("FoundingYear") or rec.get("Founding Year")

            try:
                founding_year = int(fy_raw)
            except:
                founding_year = None

            industry      = rec.get("Industry")
            funding_stage = rec.get("FundingStage") or rec.get("Funding Stage")

            rows2.append({
                "Filename": rec["__filename"],
                "Startup Name": startup_name,
                "Founding Year": founding_year,
                "Industry": industry,
                "Funding Stage": funding_stage
            })

        df2 = pd.DataFrame(rows2)

        # ------- Sidebar Filters -------
        st.sidebar.header("🔎 Filters")

        # Industry filter
        all_industries = sorted([i for i in df2["Industry"].unique() if pd.notna(i)])
        sel_industries = st.sidebar.multiselect(
            "Industry",
            options=all_industries,
            default=all_industries
        )

        # Founding Year range (guard against missing/None)
        years_list = df2["Founding Year"].dropna().astype(int).tolist()
        if len(years_list) == 0:
            st.sidebar.info("No numeric founding‐year data available.")
            sel_year_range = (None, None)
        else:
            min_year = min(years_list)
            max_year = max(years_list)
            if min_year == max_year:
                st.sidebar.write(f"Founded in: {min_year}")
                sel_year_range = (min_year, max_year)
            else:
                sel_year_range = st.sidebar.slider(
                    "Founding Year Range",
                    min_value=min_year,
                    max_value=max_year,
                    value=(min_year, max_year)
                )

        # Funding Stage filter
        all_stages = sorted([s for s in df2["Funding Stage"].unique() if pd.notna(s)])
        sel_stages = st.sidebar.multiselect(
            "Funding Stage",
            options=all_stages,
            default=all_stages
        )

        # ------- Apply Filters -------
        mask = (
            df2["Industry"].isin(sel_industries)
            & df2["Funding Stage"].isin(sel_stages)
        )
        if sel_year_range[0] is not None and sel_year_range[1] is not None:
            yr_min, yr_max = sel_year_range
            mask &= df2["Founding Year"].between(yr_min, yr_max)

        filtered = df2[mask]

        st.markdown(f"#### 🔍 {filtered.shape[0]} startups match your filters")

        # ------- Summary Charts -------
        if not filtered.empty:
            # 1) Industry Breakdown (bar chart)
            st.markdown("**Industry Breakdown**")
            industry_counts = filtered["Industry"].value_counts()
            st.bar_chart(industry_counts)

            # 2) Founding Year Distribution (bar chart)
            st.markdown("**Founding Year Distribution**")
            year_counts = filtered["Founding Year"].value_counts().sort_index()
            st.bar_chart(year_counts)

            # 3) Funding Stage Breakdown (bar chart)
            st.markdown("**Funding Stage Breakdown**")
            stage_counts = filtered["Funding Stage"].value_counts()
            st.bar_chart(stage_counts)

        st.markdown("---")

        # ------- Display Filtered Table -------
        st.markdown("### 💾 Filtered Results Table")
        st.dataframe(filtered, use_container_width=True)
