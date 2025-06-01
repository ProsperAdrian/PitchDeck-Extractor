# streamlit_app.py

import streamlit as st
import pandas as pd
import os
import json
import fitz  # PyMuPDF
import matplotlib.pyplot as plt

from extract_text import extract_text_from_pdf
from analyze import build_few_shot_prompt, call_chatgpt
from openai import OpenAI

# ----------------------------------------
# 1) PULL YOUR OPENAI KEY FROM STREAMLIT SECRETS
# ----------------------------------------
openai_api_key = st.secrets["openai"]["api_key"]

# ----------------------------------------
# 2) HIDE DEPRECATION WARNINGS FOR use_column_width
# ----------------------------------------
# This CSS rule will suppress any yellow‐banner “use_column_width is deprecated” messages.
HIDE_WARNING_STYLE = """
<style>
    .stAlert, .stAlertWarning {
        display: none;
    }
</style>
"""
st.markdown(HIDE_WARNING_STYLE, unsafe_allow_html=True)

# ----------------------------------------
# 3) STREAMLIT PAGE CONFIG
# ----------------------------------------
st.set_page_config(
    page_title="Pitch Deck Extractor",
    layout="wide",
)

# ----------------------------------------
# 4) CUSTOM CSS TO POLISH STYLING
# ----------------------------------------
def set_custom_styles():
    custom_css = f"""
    <style>
    .stApp {{
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .block-container {{
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 1rem;
        padding: 2rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
    }}
    h1 {{
        font-size: 24px !important;
        font-weight: 700;
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

# ----------------------------------------
# 5) APP TITLE & DESCRIPTION
# ----------------------------------------
st.markdown('<h1>📊 Pitch Deck Analysis</h1>', unsafe_allow_html=True)
st.markdown("""
Upload one or more pitch‐deck PDFs. This tool leverages AI + heuristics to extract:
**Startup Name**, **Founders**, **Founding Year**, **Industry**, **Niche**, **USP**, **Funding Stage**, **Revenue**, **Market Size**, and **Amount Raised**.
""", unsafe_allow_html=True)

# ----------------------------------------
# 6) LAY OUT TWO TABS: LIBRARY VIEW & DASHBOARD VIEW
# ----------------------------------------
tab1, tab2 = st.tabs(["1️⃣ Library View", "2️⃣ Dashboard & Interactive Filtering"])


# ----------------------------------------
# 7) HELPER FUNCTION: ASK CHATGPT TO IDENTIFY KEY SLIDE PAGES
# ----------------------------------------
def identify_key_slide_pages(page_texts: list[str], api_key: str) -> dict:
    """
    Given a list of page texts (0-indexed), ask ChatGPT which page numbers are
    the Team slide, the Market slide, and the Traction slide.
    Returns a dict: {"TeamPage": X, "MarketPage": Y, "TractionPage": Z}
    (all 1-indexed). If not found, returns None for that key.
    """
    prompt_lines = [
        "I will give you text snippets from each slide of a pitch deck (one snippet per page). "
        "Please identify EXACTLY which page number (1-indexed) is the Team slide, "
        "which page number is the Market slide, and which page number is the Traction slide. "
        "If you cannot find one of those categories, return null for that field. "
        "Answer in JSON format with keys \"TeamPage\", \"MarketPage\", \"TractionPage\".\n"
    ]

    # Append each page’s first ~200 characters of text
    for i, full_text in enumerate(page_texts):
        snippet = full_text.replace("\n", " ").strip()[:200]
        prompt_lines.append(f"---\nPage {i+1}:\n{snippet}\n")

    prompt_lines.append(
        "\nFormat your answer exactly like:\n"
        "{\n"
        '  "TeamPage": 7,\n'
        '  "MarketPage": 5,\n'
        '  "TractionPage": 15\n'
        "}\n"
    )

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
            return json.loads(content[start : end])
        return {"TeamPage": None, "MarketPage": None, "TractionPage": None}


# ----------------------------------------
# 8) TAB 1: LIBRARY VIEW (UPLOAD + EXTRACT + KEY SLIDE PREVIEW)
# ----------------------------------------
with tab1:
    # 8a) FILE UPLOADER (centered width)
    st.markdown('<div class="narrow-uploader">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drag & drop PDF(s) here (or click to browse)", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    all_results = []
    pdf_buffers = {}

    if uploaded_files:
        with st.spinner("🔎 Analyzing pitch decks..."):
            # Process each uploaded PDF
            for pdf_file in uploaded_files:
                st.markdown(
                    f'<div class="uploaded-filename">Processing <strong>{pdf_file.name}</strong>…</div>',
                    unsafe_allow_html=True,
                )

                # Write the PDF bytes to a temporary file so extract_text can read it
                raw_bytes = pdf_file.read()
                temp_folder = "temp"
                os.makedirs(temp_folder, exist_ok=True)
                temp_path = os.path.join(temp_folder, pdf_file.name)
                with open(temp_path, "wb") as f:
                    f.write(raw_bytes)

                try:
                    # (1) TEXT EXTRACTION -> FEW‐SHOT PROMPT -> CHATGPT CALL
                    deck_text = extract_text_from_pdf(temp_path)
                    os.remove(temp_path)

                    prompt = build_few_shot_prompt(deck_text)
                    result = call_chatgpt(prompt, api_key=openai_api_key)
                    result["__filename"] = pdf_file.name
                    all_results.append(result)

                    # (2) KEEP THE ORIGINAL BYTES FOR LATER SLIDE PREVIEW
                    pdf_buffers[pdf_file.name] = raw_bytes

                except Exception as e:
                    st.error(f"❌ Error processing **{pdf_file.name}**: {e}")
                    continue

        # 8b) IF ANY RESULTS OVERALL, SHOW A SUCCESS BANNER + TABLE + EXPORT BUTTONS
        if all_results:
            st.markdown(
                """
                <div class="success-msg-container">
                    ✅ All pitch decks processed successfully!
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Build a DataFrame for the “Library” table
            rows = []
            for rec in all_results:
                startup_name   = rec.get("StartupName") or rec.get("Startup Name")
                founding_year  = rec.get("FoundingYear") or rec.get("Founding Year")
                founders       = rec.get("Founders") or []
                industry       = rec.get("Industry")
                niche          = rec.get("Niche")
                usp            = rec.get("USP")
                funding_stage  = rec.get("FundingStage") or rec.get("Funding Stage")
                current_rev    = rec.get("CurrentRevenue") or rec.get("Current Revenue")
                amount_raised  = rec.get("AmountRaised") or rec.get("Amount Raised")

                row = {
                    "Filename": rec.get("__filename"),
                    "Startup Name": startup_name,
                    "Founding Year": founding_year,
                    "Founders": "; ".join(founders),
                    "Industry": industry,
                    "Niche": niche,
                    "USP": usp,
                    "Funding Stage": funding_stage,
                    "Current Revenue": current_rev,
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

            # EXPORT BUTTONS (JSON + CSV)
            json_str = json.dumps(all_results, indent=2)
            csv_bytes = df.to_csv(index=False).encode("utf-8")

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Export as JSON",
                    data=json_str,
                    file_name="All_decks.json",
                    mime="application/json",
                )
            with col2:
                st.download_button(
                    label="📊 Export as CSV",
                    data=csv_bytes,
                    file_name="All_decks.csv",
                    mime="text/csv",
                )

            st.markdown("---")

            # 8c) KEY SLIDE PREVIEW (CHATGPT‐DRIVEN)
            st.markdown("### 🔑 Key Slide Preview")
            st.markdown("Select a deck from the table above to preview its important slides (Team, Market, Traction).")

            selected_deck = st.selectbox(
                "❓ Which Deck would you like to preview?",
                options=df["Filename"].tolist()
            )

            if selected_deck:
                pdf_bytes = pdf_buffers[selected_deck]
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")

                # (1) Extract full text from each page
                page_texts = []
                for pg in doc:
                    page_texts.append(pg.get_text())

                # (2) Ask ChatGPT which pages correspond to Team/Market/Traction
                key_info = identify_key_slide_pages(page_texts, api_key=openai_api_key)

                raw_team    = key_info.get("TeamPage")
                raw_market  = key_info.get("MarketPage")
                raw_traction= key_info.get("TractionPage")

                # Convert GPT’s 1-indexed values to 0-indexed
                team_idx     = (int(raw_team) - 1) if raw_team else None
                market_idx   = (int(raw_market) - 1) if raw_market else None
                traction_idx = (int(raw_traction) - 1) if raw_traction else None

                # Collect only valid slides
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
                        img_bytes = pix.tobytes("png")
                        col.image(img_bytes, caption=label, use_container_width=True)

                doc.close()


# ----------------------------------------
# 9) TAB 2: DASHBOARD & INTERACTIVE FILTERING
# ----------------------------------------
with tab2:
    st.markdown("## 📊 Dashboard & Interactive Filtering")

    if not all_results:
        st.warning("Upload at least one PDF in the Library View first, then come here to see the Dashboard.")
    else:
        # Build a smaller DataFrame with only the fields needed for filtering & charts
        rows2 = []
        for rec in all_results:
            startup_name  = rec.get("StartupName") or rec.get("Startup Name")
            founding_year = rec.get("FoundingYear") or rec.get("Founding Year")
            try:
                founding_year = int(founding_year)
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

        # If there are no valid founding years (all null), skip the slider entirely
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

        # Apply filters
        filtered = df2.copy()
        if valid_years.empty:
            # If no numeric founding years exist, ignore that filter
            filtered = filtered[
                (filtered["Industry"].isin(sel_industries)) &
                (filtered["Funding Stage"].isin(sel_stages))
            ]
        else:
            filtered = filtered[
                (filtered["Industry"].isin(sel_industries)) &
                (filtered["Funding Stage"].isin(sel_stages)) &
                (filtered["Founding Year"].between(sel_year_range[0], sel_year_range[1]))
            ]

        st.markdown(f"#### 🔍 {filtered.shape[0]} startups match your filters")

        if not filtered.empty:
            # Industry Breakdown Bar Chart
            st.markdown("**Industry Breakdown**")
            industry_counts = filtered["Industry"].value_counts()
            st.bar_chart(industry_counts)

            # Founding Year Distribution Bar Chart
            st.markdown("**Founding Year Distribution**")
            year_counts = filtered["Founding Year"].value_counts().sort_index()
            st.bar_chart(year_counts)

            # Funding Stage Pie Chart
            st.markdown("**Funding Stage Breakdown**")
            stage_counts = filtered["Funding Stage"].value_counts()
            fig, ax = plt.subplots()
            stage_counts.plot.pie(autopct="%.0f%%", ylabel="", legend=False, ax=ax)
            st.pyplot(fig)

        st.markdown("---")

        # Display the filtered table at the bottom
        st.markdown("### 💾 Filtered Results Table")
        st.dataframe(filtered, use_container_width=True)

