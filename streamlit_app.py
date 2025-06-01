# streamlit_app.py

import streamlit as st
import pandas as pd
import os, json, io, warnings
import fitz  # PyMuPDF, used to render PDF pages as images
import matplotlib.pyplot as plt  # for the Dashboard pie chart

from extract_text import extract_text_from_pdf
from analyze import build_few_shot_prompt, call_chatgpt

# --- Pull OpenAI key from Streamlit secrets ---
openai_api_key = st.secrets["openai"]["api_key"]

# ----------------- SUPPRESS STREAMLIT DEPRECATION WARNINGS -----------------
# We switch from `use_column_width=` to `use_container_width=` for images, so
# the warnings should go away. Just in case, silence other minor warnings:
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------- UI SETUP -----------------
st.set_page_config(
    page_title="Pitch Deck Extractor",
    layout="wide",
)

def set_custom_styles():
    custom_css = f"""
    <style>
    .stApp {{
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
    # Narrow the uploader to center
    st.markdown('<div class="narrow-uploader">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drag & drop PDF(s) here (or click to browse)", 
        type=["pdf"],
        accept_multiple_files=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    all_results = []
    pdf_buffers = {}  # Keep raw bytes of each PDF so we can render key slides

    if uploaded_files:
        with st.spinner("🔎 Analyzing pitch decks..."):
            for pdf_file in uploaded_files:
                # Show filename being processed
                st.markdown(f'<div class="uploaded-filename">Processing <strong>{pdf_file.name}</strong>…</div>', unsafe_allow_html=True)

                # Save buffer temporarily to disk so extract_text can read it
                bytes_data = pdf_file.read()
                temp_folder = "temp"
                os.makedirs(temp_folder, exist_ok=True)
                temp_path = os.path.join(temp_folder, pdf_file.name)
                with open(temp_path, "wb") as f:
                    f.write(bytes_data)

                try:
                    # 1) Extract all slide text
                    deck_text = extract_text_from_pdf(temp_path)
                    os.remove(temp_path)

                    # 2) Build and run our few‐shot prompt
                    prompt = build_few_shot_prompt(deck_text)
                    result = call_chatgpt(prompt, api_key=openai_api_key)
                    result["__filename"] = pdf_file.name
                    all_results.append(result)

                    # 3) Store the raw PDF bytes for “Key Slide Preview”
                    pdf_buffers[pdf_file.name] = bytes_data

                except Exception as e:
                    st.error(f"❌ Error processing **{pdf_file.name}**: {e}")
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
                # The JSON keys might be either "StartupName" or "Startup Name" depending on your analyze.py
                startup_name = rec.get("StartupName") or rec.get("Startup Name")
                founding_year = rec.get("FoundingYear") or rec.get("Founding Year")
                founders = rec.get("Founders") or []
                industry = rec.get("Industry")
                niche = rec.get("Niche")
                usp = rec.get("USP")
                funding_stage = rec.get("FundingStage") or rec.get("Funding Stage")
                current_revenue = rec.get("CurrentRevenue") or rec.get("Current Revenue")
                amount_raised = rec.get("AmountRaised") or rec.get("Amount Raised")

                row = {
                    "Filename": rec.get("__filename"),
                    "Startup Name": startup_name,
                    "Founding Year": founding_year,
                    "Founders": "; ".join(founders),
                    "Industry": industry,
                    "Niche": niche,
                    "USP": usp,
                    "Funding Stage": funding_stage,
                    "Current Revenue": current_revenue,
                    "Amount Raised": amount_raised,
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

            st.markdown("---")

            #
            # ----------------- Key Slide Preview Picker -----------------
            #
            st.markdown("### 🔑 Key Slide Preview")
            st.markdown(
                "Select a deck from the table above to preview its important slides (Team, Market, Traction)."
            )

            # Because df and pdf_buffers exist, we can safely build the selectbox
            selected_deck = st.selectbox(
                "❓ Which Deck would you like to preview?",
                options=df["Filename"].tolist()
            )

            if selected_deck:
                pdf_bytes = pdf_buffers[selected_deck]
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")

                # 1) Scan every page’s extracted text to find “TEAM”, “MARKET”, “TRACTION”
                page_texts = []
                for pg in doc:
                    text = pg.get_text().upper()  # uppercase for case‐insensitive search
                    page_texts.append(text)

                def find_first_index(keyword):
                    keyword = keyword.upper()
                    for idx, content in enumerate(page_texts):
                        if keyword in content:
                            return idx
                    return None

                team_idx = find_first_index("TEAM")
                market_idx = find_first_index("MARKET")
                traction_idx = find_first_index("TRACTION")

                key_slides = []
                if team_idx is not None:
                    key_slides.append(("Team Slide (page {})".format(team_idx + 1), team_idx))
                if market_idx is not None:
                    key_slides.append(("Market Slide (page {})".format(market_idx + 1), market_idx))
                if traction_idx is not None:
                    key_slides.append(("Traction Slide (page {})".format(traction_idx + 1), traction_idx))

                # If none of those were found, show a message
                if not key_slides:
                    st.warning("⚠️ Could not find any “Team,” “Market,” or “Traction” slides in this deck.")
                else:
                    cols = st.columns(len(key_slides))
                    for col, (label, page_index) in zip(cols, key_slides):
                        if page_index < doc.page_count:
                            page = doc[page_index]
                            pix = page.get_pixmap(dpi=100)
                            img_bytes = pix.tobytes("png")  # use_container_width → True
                            col.image(
                                img_bytes,
                                caption=label,
                                use_container_width=True
                            )
                        else:
                            col.write(f"❌ {label} not found (out of range)")
                doc.close()

    # If no files have been uploaded yet, we skip all of the above and show nothing.


#
# ----------------- TAB 2: DASHBOARD VIEW -----------------
#
with tab2:
    st.markdown("## 📊 Dashboard & Interactive Filtering")

    # We only proceed if “df” (the DataFrame) exists and has at least one row
    if "df" not in locals() or df.empty:
        st.warning("Upload at least one PDF in the Library View first, then come here to see the Dashboard.")
    else:
        # Re‐construct a smaller DataFrame for filtering/charts
        rows2 = []
        for rec in all_results:
            startup_name = rec.get("StartupName") or rec.get("Startup Name")
            founding_year = rec.get("FoundingYear") or rec.get("Founding Year")
            # Convert to int if possible
            try:
                founding_year = int(founding_year)
            except:
                founding_year = None

            industry = rec.get("Industry")
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
            "Industry", options=all_industries, default=all_industries
        )

        # Founding Year range
        non_na_years = df2["Founding Year"].dropna()
        if not non_na_years.empty:
            min_year = int(non_na_years.min())
            max_year = int(non_na_years.max())
        else:
            min_year = None
            max_year = None

        if min_year is not None and max_year is not None:
            sel_year_range = st.sidebar.slider(
                "Founding Year Range", 
                min_year, 
                max_year, 
                (min_year, max_year)
            )
        else:
            sel_year_range = (None, None)

        # Funding Stage filter
        all_stages = sorted([s for s in df2["Funding Stage"].unique() if pd.notna(s)])
        sel_stages = st.sidebar.multiselect(
            "Funding Stage", options=all_stages, default=all_stages
        )

        # ------- Apply Filters -------
        if min_year is not None and max_year is not None:
            filtered = df2[
                (df2["Industry"].isin(sel_industries)) &
                (df2["Funding Stage"].isin(sel_stages)) &
                (df2["Founding Year"].between(sel_year_range[0], sel_year_range[1]))
            ]
        else:
            # If no numeric founding years, just filter by industry & funding stage
            filtered = df2[
                (df2["Industry"].isin(sel_industries)) &
                (df2["Funding Stage"].isin(sel_stages))
            ]

        st.markdown(f"#### 🔍 {filtered.shape[0]} startups match your filters")

        # ------- Summary Charts -------
        if not filtered.empty:
            # Industry Breakdown
            st.markdown("**Industry Breakdown**")
            industry_counts = filtered["Industry"].value_counts()
            st.bar_chart(industry_counts)

            # Founding Year Distribution
            st.markdown("**Founding Year Distribution**")
            if "Founding Year" in filtered.columns and filtered["Founding Year"].notna().any():
                year_counts = filtered["Founding Year"].value_counts().sort_index()
                st.bar_chart(year_counts)

            # Funding Stage Pie Chart (via Matplotlib)
            st.markdown("**Funding Stage Breakdown**")
            stage_counts = filtered["Funding Stage"].value_counts()
            fig, ax = plt.subplots(figsize=(4, 4))
            stage_counts.plot.pie(
                autopct="%.0f%%", 
                ylabel="", 
                legend=False, 
                ax=ax
            )
            st.pyplot(fig)

        st.markdown("---")

        # ------- Display Filtered Table -------
        st.markdown("### 💾 Filtered Results Table")
        st.dataframe(filtered, use_container_width=True)
