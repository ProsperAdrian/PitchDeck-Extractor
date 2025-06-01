# streamlit_app.py

import streamlit as st
import pandas as pd
import os
import io
import json
import fitz  # PyMuPDF, used to render PDF pages as images
from openai import OpenAI

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
        /* (Optional) You can embed a local base64 image, e.g. YOUR_BASE64_IMAGE_HERE */
        /* background-image: url("data:image/png;base64,YOUR_BASE64_IMAGE_HERE"); */
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

# ----------------------- Helper: Identify Key Slide Pages via ChatGPT -----------------------
def identify_key_slide_pages(page_texts: list[str], api_key: str) -> dict:
    """
    Given a list of page texts (0-indexed), ask ChatGPT to tell us:
      - which page is the Team slide
      - which page is the Market slide
      - which page is the Traction slide

    Returns a dict like: {"TeamPage": 7, "MarketPage": 5, "TractionPage": 15}
    (all 1-indexed). If GPT cannot find one, returns null for that key.
    """
    prompt_lines = [
        "I’m going to give you the text from each slide of a pitch deck, one by one.  "
        "Please tell me exactly which page number (1-indexed) is the Team slide, "
        "which page number is the Market slide, and which page number is the Traction slide.  "
        "Format your answer exactly as JSON with keys \"TeamPage\", \"MarketPage\", \"TractionPage\".  "
        "If you can’t find any of them, put null for that field.\n"
    ]

    # Include a snippet (first ~200 chars) for each page
    for i, full_text in enumerate(page_texts):
        snippet = full_text.strip().replace("\n", " ")[:200]
        prompt_lines.append(f"---\nPage {i+1}:\n{snippet}\n")

    prompt_lines.append(
        "\nAnswer in JSON, for example:\n"
        "{\n"
        '  "TeamPage": 7,\n'
        '  "MarketPage": 5,\n'
        '  "TractionPage": 15\n'
        "}\n"
    )

    prompt = "\n".join(prompt_lines)

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200
    )
    content = response.choices[0].message.content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end != -1:
            return json.loads(content[start:end])
        else:
            return {"TeamPage": None, "MarketPage": None, "TractionPage": None}

# ----------------- TAB 1: LIBRARY VIEW -----------------
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
                st.markdown(f'<div class="uploaded-filename">Processing <strong>{pdf_file.name}</strong>…</div>', unsafe_allow_html=True)

                # Save buffer temporarily to disk so extract_text can read it
                bytes_data = pdf_file.read()
                temp_dir = "temp"
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, pdf_file.name)
                with open(temp_path, "wb") as f:
                    f.write(bytes_data)

                try:
                    deck_text = extract_text_from_pdf(temp_path)
                    os.remove(temp_path)
                    prompt = build_few_shot_prompt(deck_text)
                    result = call_chatgpt(prompt, api_key=openai_api_key)
                    result["__filename"] = pdf_file.name
                    all_results.append(result)
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
                startup_name   = rec.get("StartupName") or rec.get("Startup Name")
                founding_year  = rec.get("FoundingYear") or rec.get("Founding Year")
                founders       = rec.get("Founders") or []
                industry       = rec.get("Industry")
                niche          = rec.get("Niche")
                usp            = rec.get("USP")
                funding_stage  = rec.get("FundingStage") or rec.get("Funding Stage")
                current_revenue= rec.get("CurrentRevenue") or rec.get("Current Revenue")
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

            # ----------------- Key Slide Preview Picker (GPT-driven) -----------------
            st.markdown("### 🔑 Key Slide Preview")
            st.markdown("Select a deck from the table above to preview its important slides (Team, Market, Traction).")

            selected_deck = st.selectbox(
                "❓ Which Deck would you like to preview?",
                options=df["Filename"].tolist()
            )

            if selected_deck:
                pdf_bytes = pdf_buffers[selected_deck]
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")

                # 1) Extract full text from each page
                page_texts = []
                for pg in doc:
                    text = pg.get_text()
                    page_texts.append(text)

                # 2) Ask ChatGPT which pages correspond to Team/Market/Traction
                key_info = identify_key_slide_pages(page_texts, api_key=openai_api_key)

                # 3) Convert GPT’s 1-indexed answers to 0-indexed
                raw_team   = key_info.get("TeamPage")
                raw_market = key_info.get("MarketPage")
                raw_traction = key_info.get("TractionPage")

                team_idx    = (int(raw_team) - 1) if raw_team else None
                market_idx  = (int(raw_market) - 1) if raw_market else None
                traction_idx= (int(raw_traction) - 1) if raw_traction else None

                # 4) Build a list of exactly those slides GPT pointed to
                key_slides = []
                if isinstance(team_idx, int) and 0 <= team_idx < doc.page_count:
                    key_slides.append((f"Team Slide (page {team_idx+1})", team_idx))
                if isinstance(market_idx, int) and 0 <= market_idx < doc.page_count:
                    key_slides.append((f"Market Slide (page {market_idx+1})", market_idx))
                if isinstance(traction_idx, int) and 0 <= traction_idx < doc.page_count:
                    key_slides.append((f"Traction Slide (page {traction_idx+1})", traction_idx))

                # 5) If GPT gave us nothing useful, warn; otherwise, render images
                if not key_slides:
                    st.warning("⚠️ ChatGPT did not locate Team/Market/Traction slides in this deck.")
                else:
                    cols = st.columns(len(key_slides))
                    for col, (label, page_index) in zip(cols, key_slides):
                        page = doc[page_index]
                        pix = page.get_pixmap(dpi=100)
                        img_bytes = pix.tobytes("png")
                        col.image(
                            img_bytes,
                            caption=label,
                            use_container_width=True
                        )

                doc.close()

    # If no decks uploaded, skip this entire block.

# ----------------- TAB 2: DASHBOARD VIEW -----------------
with tab2:
    st.markdown("## 📊 Dashboard & Interactive Filtering")

    if not all_results:
        st.warning("Upload at least one PDF in the Library View first, then come here to see the Dashboard.")
    else:
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

        # Sidebar Filters
        st.sidebar.header("🔎 Filters")

        all_industries = sorted([i for i in df2["Industry"].unique() if pd.notna(i)])
        sel_industries = st.sidebar.multiselect(
            "Industry", options=all_industries, default=all_industries
        )

        min_year = int(df2["Founding Year"].dropna().min()) if not df2["Founding Year"].dropna().empty else 0
        max_year = int(df2["Founding Year"].dropna().max()) if not df2["Founding Year"].dropna().empty else 0
        sel_year_range = st.sidebar.slider(
            "Founding Year Range",
            min_year,
            max_year,
            (min_year, max_year)
        )

        all_stages = sorted([s for s in df2["Funding Stage"].unique() if pd.notna(s)])
        sel_stages = st.sidebar.multiselect(
            "Funding Stage", options=all_stages, default=all_stages
        )

        filtered = df2[
            (df2["Industry"].isin(sel_industries)) &
            (df2["Funding Stage"].isin(sel_stages)) &
            (df2["Founding Year"].between(sel_year_range[0], sel_year_range[1]))
        ]

        st.markdown(f"#### 🔍 {filtered.shape[0]} startups match your filters")

        if not filtered.empty:
            st.markdown("**Industry Breakdown**")
            industry_counts = filtered["Industry"].value_counts()
            st.bar_chart(industry_counts)

            st.markdown("**Founding Year Distribution**")
            year_counts = filtered["Founding Year"].value_counts().sort_index()
            st.bar_chart(year_counts)

            import matplotlib.pyplot as plt
            st.markdown("**Funding Stage Breakdown**")
            stage_counts = filtered["Funding Stage"].value_counts()
            fig, ax = plt.subplots()
            stage_counts.plot.pie(
                autopct="%.0f%%",
                ylabel="",
                legend=False,
                ax=ax
            )
            st.pyplot(fig)

        st.markdown("---")

        st.markdown("### 💾 Filtered Results Table")
        st.dataframe(filtered, use_container_width=True)
