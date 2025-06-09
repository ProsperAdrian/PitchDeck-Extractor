# ─────────────────────────────────────────────────────────────────────────────
# streamlit_app.py
#
# • st.set_page_config must be the first Streamlit command.
# • Matplotlib removed (uses st.bar_chart instead).
# • Deprecation warnings hidden via CSS.
# • ChatGPT picks actual “Team/Market/Traction” pages.
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import os
import json
import fitz                                # PyMuPDF, for rendering PDF pages
from openai import OpenAI

from extract_text import extract_text_from_pdf
from analyze import (build_few_shot_prompt, call_chatgpt, build_insight_prompt, call_chatgpt_insight)
from analyze_scoring import build_structured_scoring_prompt, call_structured_pitch_scorer


# ─────────────────────────────────────────────────────────────────────────────
# 1) SET PAGE CONFIG (MUST BE THE FIRST STREAMLIT CALL)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pitch Deck Extractor",
    layout="wide",
)

# DO NOT DELETE session state on rerun.
# Instead, keep it persistent for new uploads to work.
if "all_results" not in st.session_state:
    st.session_state.all_results = []

if "insights_cache" not in st.session_state:
    st.session_state.insights_cache = {}

# Local alias for ease of use
all_results = st.session_state.all_results


# ─────────────────────────────────────────────────────────────────────────────
# 2) PULL YOUR OPENAI KEY FROM STREAMLIT SECRETS
# ─────────────────────────────────────────────────────────────────────────────
openai_api_key = st.secrets["openai"]["api_key"]

# ─────────────────────────────────────────────────────────────────────────────
# 3) HIDE DEPRECATION WARNINGS FOR use_column_width
# ─────────────────────────────────────────────────────────────────────────────
HIDE_WARNING_STYLE = """
<style>
    /* Hide any yellow-box warnings about deprecated use_column_width */
    .stAlert, .stAlertWarning {
        display: none;
    }
</style>
"""
st.markdown(HIDE_WARNING_STYLE, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 4) CUSTOM CSS FOR A CLEANER LOOK
# ─────────────────────────────────────────────────────────────────────────────
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
        font-size: 28px !important;
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
        margin-bottom: 1rem;
    }}
    .extracted-title {{
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

set_custom_styles()

st.markdown("""
    <style>
    /* Narrow dropdowns in the main view (not sidebar) */
    div[data-baseweb="select"] {
        max-width: 400px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Reduce width of file uploader */
    div[data-testid="stFileUploader"] {
        max-width: 600px;
        margin-left: 0;  /* optional: align left */
    }
    </style>
""", unsafe_allow_html=True)



# ─────────────────────────────────────────────────────────────────────────────
# 5) APPLICATION TITLE & DESCRIPTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<h1>❖ Pitch Deck Analysis</h1>', unsafe_allow_html=True)
st.markdown("""
Upload one or more pitch‐deck PDFs. This tool leverages AI + heuristics to extract:
**Startup Name**, **Founders**, **Founding Year**, **Industry**, **Niche**, **USP**, **Funding Stage**, **Revenue**, **Market Size**, and **Amount Raised**.
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 6) CREATE TWO TABS: LIBRARY VIEW & DASHBOARD VIEW
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Library View", "Dashboard", "AI Insights"])


# ─────────────────────────────────────────────────────────────────────────────
# 7) HELPER: ASK CHATGPT TO PICK KEY SLIDE PAGE NUMBERS
# ─────────────────────────────────────────────────────────────────────────────
def identify_key_slide_pages(page_texts: list[str], api_key: str) -> dict:
    """
    Given a list of page texts (0-indexed), ask ChatGPT which page numbers
    correspond to the Team, Market, and Traction slides. Returns a dict:
      { "TeamPage": <int or null>, "MarketPage": <int or null>, "TractionPage": <int or null> }
    Page numbers are 1-indexed. If ChatGPT cannot find a category, it returns null.
    """
    prompt_lines = [
        "I will give you text snippets from each slide of a pitch deck, one snippet per page. "
        "Identify EXACTLY which page number (1-indexed) is the Team slide, "
        "which page number is the Market slide, and which page number is the Traction slide. "
        "If you cannot find one of those categories, return null. "
        "Answer in JSON format with keys \"TeamPage\", \"MarketPage\", \"TractionPage\".\n"
    ]

    for i, full_text in enumerate(page_texts):
        # Use just the first 200 characters as a “snippet” to keep the prompt concise.
        snippet = full_text.replace("\n", " ").strip()[:200]
        prompt_lines.append(f"---\nPage {i+1}:\n{snippet}\n")

    prompt_lines.append(
        "\nRespond exactly like:\n"
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
        # Fallback: find the `{…}` block and parse it
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end != -1:
            try:
                return json.loads(content[start:end])
            except:
                pass
        # If parsing fails, return all nulls
        return {"TeamPage": None, "MarketPage": None, "TractionPage": None}


# ─────────────────────────────────────────────────────────────────────────────
# 8) TAB 1: LIBRARY VIEW → UPLOAD + EXTRACT + KEY SLIDE PREVIEW
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

    all_results = []      # Will hold the JSON‐extracted metadata for each deck
    if "all_results" not in st.session_state:
        st.session_state.all_results = []

    if "insights_cache" not in st.session_state:
        st.session_state.insights_cache = {}

        # Replace your list with the session one
    all_results = st.session_state.all_results

    pdf_buffers = {}      # Will hold raw PDF bytes for later “Key Slide Preview”

    if uploaded_files:
        with st.spinner("🔎 Analyzing pitch decks..."):
            for pdf_file in uploaded_files:

                raw_bytes = pdf_file.read()
                temp_folder = "temp"
                os.makedirs(temp_folder, exist_ok=True)
                temp_path = os.path.join(temp_folder, pdf_file.name)
                with open(temp_path, "wb") as f:
                    f.write(raw_bytes)

                try:
                    # 1) Extract all text from PDF
                    deck_text = extract_text_from_pdf(temp_path)
                    
                    # FIRST build the result dict
                    prompt = build_few_shot_prompt(deck_text)
                    result = call_chatgpt(prompt, api_key=openai_api_key)
                    
                    # THEN add FullText and filename
                    result["FullText"] = deck_text
                    result["__filename"] = pdf_file.name

                    os.remove(temp_path)

                    # 2) Build few‐shot prompt and call ChatGPT
                    prompt = build_few_shot_prompt(deck_text)
                    result = call_chatgpt(prompt, api_key=openai_api_key)
                    result["__filename"] = pdf_file.name
                    
                    # 👇 If you run AI insight generation here too:
                    # from analyze import build_insight_prompt, 
                    
                    # Structured pitch scoring
                    try:
                        scoring_prompt = build_structured_scoring_prompt(deck_text)
                        scoring_result = call_structured_pitch_scorer(scoring_prompt, api_key=openai_api_key)
                        result["Section Scores"] = scoring_result.get("sections", [])
                        result["Pitch Score"] = scoring_result.get("total_score", None)
                        # Only override if a new summary was returned
                        structured_summary = scoring_result.get("summary", "").strip()
                        if structured_summary:
                            result["Summary Insight"] = structured_summary

                    except Exception as e:
                        result["Section Scores"] = []
                        result["Pitch Score"] = None
                        result["Summary Insight"] = "Could not generate structured insight."

                    
                    # 👇 Store in Streamlit session state
                    st.session_state.all_results.append(result)


                    # 3) Store PDF bytes so we can render pages later
                    pdf_buffers[pdf_file.name] = raw_bytes

                except Exception as e:
                    st.error(f"❌ Error processing **{pdf_file.name}**: {e}")
                    continue

        # 8b) AFTER PROCESSING, SHOW THE “LIBRARY” TABLE + EXPORT BUTTONS
        if all_results:
            st.markdown(
                """
                <div class="success-msg-container">
                    ✅ All pitch decks processed successfully!
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("---")
            # Build a Pandas DataFrame of extracted fields
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
            
            st.markdown('<div class="extracted-title">Library</div>', unsafe_allow_html=True)

            # 🔻 INSERT THIS BLOCK RIGHT HERE
            startup_names = df["Startup Name"].dropna().unique().tolist()
            
            startups_to_remove = st.multiselect(
                "Select startups to remove:",
                options=startup_names,
                help="Use this to filter out any startups you don't want included in the results or exports."
            )
            
            if startups_to_remove:
                df = df[~df["Startup Name"].isin(startups_to_remove)]
                all_results = [
                    rec for rec in all_results
                    if (rec.get("StartupName") or rec.get("Startup Name")) not in startups_to_remove
                ]
            # 🔺 END OF INSERTED BLOCK

            
            st.dataframe(df, use_container_width=True)

            # EXPORT BUTTONS: JSON + CSV
            json_str  = json.dumps(all_results, indent=2)
            csv_bytes = df.to_csv(index=False).encode("utf-8")

            col1, col2, _ = st.columns([1, 1, 6])
            with col1:
                st.download_button(
                    label="Export JSON ➜",
                    data=json_str,
                    file_name="All_decks.json",
                    mime="application/json",
                )
            with col2:
                st.download_button(
                    label="Export CSV ➜",
                    data=csv_bytes,
                    file_name="All_decks.csv",
                    mime="text/csv",
                )

            st.markdown("---")

            # 8c) KEY SLIDE PREVIEW: CHATGPT‐DRIVEN
            st.markdown("##### Key Slide Preview")
            st.markdown("Select a deck from the table above to preview its important slides (Team, Market, Traction).")

            selected_deck = st.selectbox(
                "❓ Which Deck would you like to preview?",
                options=df["Filename"].tolist()
            )

            if selected_deck:
                pdf_bytes = pdf_buffers[selected_deck]
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")

                # Extract full text from each page to feed ChatGPT
                page_texts = [page.get_text() for page in doc]

                # Ask ChatGPT to tell us the page numbers
                key_info = identify_key_slide_pages(page_texts, api_key=openai_api_key)
                raw_team     = key_info.get("TeamPage")
                raw_market   = key_info.get("MarketPage")
                raw_traction = key_info.get("TractionPage")

                team_idx     = (int(raw_team) - 1) if raw_team else None
                market_idx   = (int(raw_market) - 1) if raw_market else None
                traction_idx = (int(raw_traction) - 1) if raw_traction else None

                # Build a list of whichever key pages exist
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


# ─────────────────────────────────────────────────────────────────────────────
# 9) TAB 2: DASHBOARD & INTERACTIVE FILTERING
# ─────────────────────────────────────────────────────────────────────────────
# ----------------- TAB 2: DASHBOARD VIEW -----------------
#
# … your code above remains unchanged …

with tab2:
    st.markdown("##### Dashboard")

    if not all_results:
        st.warning(
            "Upload at least one PDF in the Library View first, then come here to see the Dashboard."
        )
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

        # 1) Industry filter
        all_industries = sorted([i for i in df2["Industry"].unique() if pd.notna(i)])
        sel_industries = st.sidebar.multiselect(
            "Industry",
            options=all_industries,
            default=all_industries
        )

        # 2) Founding Year range (guard against missing/None)
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

        # 3) Funding Stage filter
        all_stages = sorted([s for s in df2["Funding Stage"].unique() if pd.notna(s)])
        sel_stages = st.sidebar.multiselect(
            "Funding Stage",
            options=all_stages,
            default=all_stages
        )

        # ------- Apply Filters (allow NaN so “missing” decks are not dropped) -------
        # Start with all True, then AND in each condition:
        mask = pd.Series(True, index=df2.index)

        # ● Industry filter: keep row if (industry is in sel_industries) OR industry is NaN
        mask &= (df2["Industry"].isin(sel_industries) | df2["Industry"].isna())

        # ● Funding Stage filter: similarly allow NaN
        mask &= (df2["Funding Stage"].isin(sel_stages) | df2["Funding Stage"].isna())

        # ● Founding Year filter: only apply if we actually built a slider
        if sel_year_range[0] is not None and sel_year_range[1] is not None:
            yr_min, yr_max = sel_year_range
            mask &= (
                df2["Founding Year"].between(yr_min, yr_max)
                | df2["Founding Year"].isna()
            )

        filtered = df2[mask]

        st.markdown(f"###### 🔍 {filtered.shape[0]} startups match your filters")

        # ------- Summary Charts -------
        if not filtered.empty:
            # 1) Industry Breakdown (bar chart)
            st.markdown("**Industry Breakdown**")
            industry_counts = filtered["Industry"].value_counts(dropna=True)
            st.bar_chart(industry_counts)

            # 2) Founding Year Distribution (bar chart)
            st.markdown("**Founding Year Distribution**")
            year_counts = (
                filtered["Founding Year"]
                .dropna()
                .astype(int)
                .value_counts()
                .sort_index()
            )
            st.bar_chart(year_counts)

            # 3) Funding Stage Breakdown (bar chart)
            st.markdown("**Funding Stage Breakdown**")
            stage_counts = filtered["Funding Stage"].value_counts(dropna=True)
            st.bar_chart(stage_counts)

        st.markdown("---")

        # ------- Display Filtered Table -------
        st.markdown("##### 💾 Filtered Results Table")
        st.dataframe(filtered, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# 10) TAB 3: AI INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🧠 AI-Generated Startup Insights")
    st.markdown("Below is an AI assessment of each pitch deck, based on team, traction, market, and clarity.")

    if not all_results:
        st.info("Please upload and process decks in the Library View first.")
    else:
        for rec in all_results:
            st.markdown("---")
            st.subheader(f"🚀 {rec.get('Startup Name', 'Unnamed Startup')}")

            # Metrics Row
            col1, col2 = st.columns([1, 2])
            with col1:
                pitch_score = rec.get("Pitch Score")
                st.metric("Pitch Quality Score", f"{pitch_score}/100" if pitch_score is not None else "N/A")

            with col2:
                filename = rec.get("__filename")
            
                # Use cached insight if available, otherwise generate and store
                if filename not in st.session_state.insights_cache:
                    try:
                        insight_prompt = build_insight_prompt(rec.get("FullText", ""))
                        insight_result = call_chatgpt_insight(insight_prompt, api_key=openai_api_key)
                        st.session_state.insights_cache[filename] = insight_result
                    except:
                        st.session_state.insights_cache[filename] = {
                            "Red Flags": [],
                            "Summary Insight": "Could not generate insight."
                        }
            
                # Retrieve from cache and show
                insight = st.session_state.insights_cache[filename]
                summary = insight.get("Summary Insight", "").strip()
            
                if summary:
                    st.markdown("**💡 Summary Insight:**")
                    st.success(summary)
                else:
                    st.markdown("**💡 Summary Insight:**")
                    st.warning("No insight available.")
            
                # Also show red flags from insight
                red_flags = insight.get("Red Flags", [])
                if red_flags:
                    st.markdown("**⚠️ Red Flags:**")
                    for flag in red_flags:
                        st.markdown(f"- {flag}")



            # Section Scores - 3-column Table Format
            section_scores = rec.get("Section Scores", [])
            if section_scores:
                st.markdown("**📊 Section-wise Breakdown:**")
                section_table = pd.DataFrame([
                    {
                        "Key Section": sec.get("name", ""),
                        "Rating (out of 10)": sec.get("score", "N/A"),
                        "Comment": sec.get("reason", "")
                    }
                    for sec in section_scores
                ])
                st.dataframe(section_table, use_container_width=True)


            # Red Flags
            red_flags = rec.get("Red Flags", [])
            if red_flags:
                st.markdown("**⚠️ Red Flags:**")
                for flag in red_flags:
                    st.markdown(f"- {flag}")
