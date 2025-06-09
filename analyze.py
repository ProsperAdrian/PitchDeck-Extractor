# analyze_scoring.py

import json
from openai import OpenAI

# Define your rubric
SCORING_RUBRIC = [
    {"name": "Team", "weight": 15, "aliases": ["team", "leadership", "founders"]},
    {"name": "Problem", "weight": 10, "aliases": ["problem", "pain point"]},
    {"name": "Solution", "weight": 10, "aliases": ["solution", "product", "offering"]},
    {"name": "Revenue Model", "weight": 10, "aliases": ["revenue", "monetization", "how we make money"]},
    {"name": "Market Size", "weight": 10, "aliases": ["market", "tam", "sam", "som"]},
    {"name": "Traction", "weight": 10, "aliases": ["traction", "metrics", "growth"]},
    {"name": "Go-to-Market", "weight": 10, "aliases": ["go-to-market", "gtm", "sales", "acquisition"]},
    {"name": "Competition", "weight": 10, "aliases": ["competition", "competitive", "landscape"]},
    {"name": "Business Model Fit", "weight": 10, "aliases": ["business model", "scaling", "product-market fit"]},
    {"name": "Ask & Use of Funds", "weight": 5, "aliases": ["ask", "funds", "use of proceeds"]}
]


# Prompt template
SCORING_PROMPT = """
You are a VC analyst scoring the quality of a startup pitch deck based on 10 essential sections. 
Each section has a weight. Based on the full slide text, evaluate each section's presence and quality (0–10), then compute a weighted total score.

Return JSON with this structure:
{
  "sections": [
    {"name": "Team", "score": 8, "comment": "Founders listed with roles and past experience."},
    {"name": "Problem", "score": 6, "comment": "Problem is implied but not well articulated."},
    ...
  ],
  "total_score": 72,
  "summary": "Solid pitch overall with strong team and traction, but weak go-to-market clarity."
}

--- START OF SLIDES ---
{deck_text}
--- END ---
"""

def build_structured_scoring_prompt(deck_text: str) -> str:
    return f"""
You are a world-class venture capital analyst evaluating startup pitch decks. Your task is to score the quality of a pitch based on **exactly these 10 sections**:

1. Team
2. Problem
3. Solution
4. Business Model
5. Market Size
6. Product
7. Traction
8. Competition
9. Financials
10. Ask

🔒 **Important Rules**:
- Only score a section if the content *directly* addresses it in the pitch. Do not assume or infer.
- If a section is **missing**, vague, or superficial, give it a **score of 0 to 3** and say why.
- Never award 10/10 unless the content is clear, complete, and convincing.
- You MUST include a brief reason for each score (1 sentence max).
- Return total score (sum of all 10 section scores) as total_score.

🛑 If a section is not present, do not guess—penalize.

Return your output as **strict JSON**:

json
{{
  "sections": [
    {{ "name": "Team", "score": 7, "reason": "Experienced founders but lacks depth on roles" }},
    ...
  ],
  "total_score": 65,
  "summary": "Strong traction and product, but team details and financials are lacking."
}}
--- BEGIN SLIDE TEXT ---
{deck_text}
--- END SLIDE TEXT ---
"""

def call_structured_pitch_scorer(prompt: str, api_key: str, model="gpt-4") -> dict:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1000,
    )
    content = response.choices[0].message.content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end != -1:
            return json.loads(content[start:end])
        raise ValueError(f"Could not parse structured scoring JSON:\n{content}")


analysis.py:
# scripts/analyze.py

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from extract_text import extract_text_from_pdf
from analyze_insight import build_insight_prompt, call_chatgpt_insight
from analyze_scoring import build_structured_scoring_prompt, call_structured_pitch_scorer



# Folder paths
INPUT_FOLDER = "input_decks"
OUTPUT_FOLDER = "parsed_entities"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---- FEW‐SHOT EXAMPLES (hardcoded) ----
EXAMPLE_1_TEXT = """
----- Slide 1 -----
Yabscore
----- Slide 2 -----
Founded in Oct 2019, we are a sport-tech startup focused on mobile sports betting in Nigeria.
----- Slide 3 -----
Team:
IK Ezekwelu – Co-Founder
Dapo Arowa – Co-Founder
Adewale Adeleke – Creative Head
----- Slide 4 -----
Unique Selling Proposition:
Yabscore is the first fully mobile sports-betting platform tailored to Nigerian football fans, offering in-play wagering and live performance stats.
----- Slide 7 -----
Market Size:
TAM: $95 Billion +
SAM: $2.2 Billion +
Market Opp.: $193 Million +
----- Slide 12 -----
Traction:
Gross Revenues in 2020: $3.1k
"""

EXAMPLE_1_JSON = {
  "Startup Name": "Yabscore",
  "Founding Year": "2019",
  "Founders": ["IK Ezekwelu", "Dapo Arowa"],
  "Industry": "Sporttech",
  "Niche": "Mobile sports betting",
  "USP": "Yabscore is the first fully mobile sports-betting platform tailored to Nigerian football fans, offering in-play wagering and live performance stats.",
  "Funding Stage": None,
  "Current Revenue": "$3.1k",
  "Market": { "TAM": "$95B", "SAM": "$2.2B", "SOM": "$193B" },
  "Amount Raised": "$0"
}

EXAMPLE_2_TEXT = """
----- Slide 1 -----
Quidax

----- Slide 2 -----
Founded in August 2018, Quidax is a fintech “cryptocurrency enabler” that lets individuals and businesses across Africa buy, sell, save and spend crypto in their local currency through an exchange, OTC desk and a single, full-stack crypto API 
.

----- Slide 3 -----
Team:
Buchi Okoro – Co-Founder & CEO
Uzo Awili – Co-Founder & CTO
Morris Ebieroma – Co-Founder & CIO 

----- Slide 4 -----
Unique Selling Proposition:
An Africa-focused, all-in-one crypto platform offering:
• Seamless fiat on/off-ramps and 1,200+ trading pairs.
• A single API that lets banks, fintechs and gaming apps embed custody, trading and payments in days.
• “African Proximity Advantage” – deep local rails, faster support and lower switching costs than global rivals .

----- Slide 7 -----
Market Opportunity:
• 575 million+ global crypto users as of Dec 2024; 65 million in Africa, with Nigeria ranked #2 worldwide for adoption 
.
(The deck does not state dollar TAM/SAM/SOM figures.)

----- Slide 12 -----
Traction:
• Crossed $10 million ARR and 700 k sign-ups in 2023 
.
• Surpassed $100 million cumulative trading volume by Oct 2020 and now processes ~$25 million monthly 
.
• Serves 2,000+ business API clients across digital banking, gaming and fintech .

(No fundraising ask, Series round or formal TAM/SAM/SOM numbers are disclosed in the deck.)
"""

EXAMPLE_2_JSON = {
  "Startup Name": "Quidax",
  "Founding Year": "2018",
  "Founders": ["Buchi Okoro", "Uzo Awili", "Morris Ebieroma"],
  "Industry": "FinTech",
  "Niche": "Cryptocurrency exchange",
  "USP": "All-in-one platform with seamless fiat on/off ramps and a single API enabling African users and businesses to access 1,200+ crypto pairs securely",
  "Funding Stage": "null",
  "Current Revenue": "$10.2m",
  "Market": { "TAM": "null", "SAM": "null", "SOM": "null" },
  "Amount Raised": "$0"
}



PROMPT_PREFIX = """
You are an expert at extracting structured data from investor pitch decks. For each deck, I will present the slide text. Return exactly one JSON object with these ten fields:
{
  "Startup Name": string or null,  # what is the most likely startup name? likely a single name most repeated in deck used to describe the company, not a sentence or contain hashtag #
  "Founding Year": string or null, # If no explicit “Founded in YYYY” appears, Scan all content for founding-year clues, including: • timeline or roadmap dates, • traction graphs captions, • team-bio phrasing, • funding-history dates. Determine the most probable calendar year in which the company was founded. If multiple plausible years appear, choose the earliest one that has at least one direct or indirect supporting signal.
  "Founders": [string, ...] or null, # Who are the likely founders of this startup?
  "Industry": string or null,       # one of: Fintech, Insurtech, Regtech, Healthtech, Medtech, Biotech, Pharmatech, Femtech, Eldertech, Proptech, Contech, Agtech, Foodtech, RestaurantTech, ClimateTech, CleanTech, EnergyTech, Greentech, Edtech, HRtech, Worktech, Martech, Adtech, RetailTech, Ecommerce, Marketplace, MobilityTech, Autotech, TransportTech, LogisticsTech, SupplyChainTech, TravelTech, SpaceTech, AerospaceTech, DefenceTech, SportTech, GamingTech, eSportsTech, MediaTech, StreamingTech, MusicTech, CreatorEconomyTech, SocialTech, Cybersecurity, AI, MachineLearning, BigData, AnalyticsTech, CloudTech, SaaS, DevOps, IoT, Robotics, HardwareTech, WearablesTech, 3DPrinting, AR/VR/XR, Metaverse, Web3, Blockchain, Crypto, NFT, QuantumTech, LegalTech, Govtech, CivicTech, NonprofitTech, ProductivityTech, CollaborationTech, PetTech, ElderCareTech etc.
  "Niche": string or null,          # free-text description e.g. “crypto exchange”, “mobile betting”, “AI tutoring”
  "USP": string or null,            # a single sentence from the deck that states the unique selling proposition
  "Funding Stage": string or null,   # If no explicit round is mentioned, Scan the deck for the following signals: • capital sought, • traction metrics (users, revenue, growth), • product maturity, • team size & seniority, • prior funding, • planned use of funds, • target investors, • implied valuation. Using these signals and standard VC heuristics, decide the most probable funding round (Pre-seed, Seed, Series A, Series B, Series C or later).
  "Current Revenue": string or null, # What is the revenue corresponding to the latest actual year in the financials, as opposed to future forecasts?
  "Market": { "TAM": string or null, "SAM": string or null, "SOM": string or null } or null,
  "Amount Raised": string or null,  # How much funds have this startup previously raised from investors since its inception? do not include the amount they want to raise in future
}
If any field is not present, set it to null.

---- EXAMPLE 1 ----
Slide texts:
""" + EXAMPLE_1_TEXT + """
JSON answer:
""" + json.dumps(EXAMPLE_1_JSON, indent=2) + """

---- EXAMPLE 2 ----
Slide texts:
""" + EXAMPLE_2_TEXT + """
JSON answer:
""" + json.dumps(EXAMPLE_2_JSON, indent=2) + """

---- NOW PROCESS THIS NEW DECK ----
Slide texts:
"""

# ----------------------------------------

def build_few_shot_prompt(deck_slide_text):
    """
    Concatenate the prompt prefix (with Examples 1 & 2) and the new deck's slide text.
    """
    return PROMPT_PREFIX + deck_slide_text + "\nJSON answer:"

def call_chatgpt(prompt, api_key, model="gpt-3.5-turbo"):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=800
    )

    content = response.choices[0].message.content.strip()

    # Parse out the JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end != -1:
            return json.loads(content[start:end])
        raise ValueError(f"Could not parse JSON from response:\n{content}")


if __name__ == "__main__":
    for fname in os.listdir(INPUT_FOLDER):
        if not fname.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(INPUT_FOLDER, fname)
        # print(f"Processing {fname}…")

        # 1) Extract slide text
        deck_text = extract_text_from_pdf(pdf_path)

        # 2) Build few-shot prompt
        prompt = build_few_shot_prompt(deck_text)

        # 3) Call ChatGPT
        try:
            result = call_chatgpt(prompt)
        except Exception as e:
            print(f"  Error calling ChatGPT for {fname}: {e}")
            continue

        # 4) Write output JSON
                # 4) Post-process & write output JSON
        # Ensure all ten fields exist; if missing, set to null
        expected_keys = [
            "Startup Name", "Founding Year", "Founders", "Industry", 
            "Niche", "USP", "Funding Stage", "Current Revenue", "Market","Amount Raised"
        ]
        normalized = {}
        for key in expected_keys:
            normalized[key] = result.get(key, None)
        # Ensure Market itself has TAM/SAM/SOM
        if isinstance(normalized["Market"], dict):
            for sub in ["TAM", "SAM", "SOM"]:
                if sub not in normalized["Market"]:
                    normalized["Market"][sub] = None
        else:
            normalized["Market"] = {"TAM": None, "SAM": None, "SOM": None}

        # 4a) Generate Structured Scores
        structured_summary = ""
        scoring_prompt = build_structured_scoring_prompt(deck_text)
        try:
            scoring_result = call_structured_pitch_scorer(scoring_prompt, api_key)
            result["Section Scores"] = scoring_result.get("sections", [])
            result["Pitch Score"] = scoring_result.get("total_score", None)
            structured_summary = scoring_result.get("summary", "").strip()
        except Exception as e:
            result["Section Scores"] = []
            result["Pitch Score"] = None
            structured_summary = ""
        
        # 4b) Generate Red Flags + Questions + Fallback Summary
        fallback_summary = ""
        insight_prompt = build_insight_prompt(deck_text)
        try:
            insight_result = call_chatgpt_insight(insight_prompt, api_key)
            result["Red Flags"] = insight_result.get("Red Flags", [])
            result["Suggested Questions"] = insight_result.get("Suggested Questions", [])
            fallback_summary = insight_result.get("Summary Insight", "").strip()
        except Exception as e:
            result["Red Flags"] = []
            result["Suggested Questions"] = []
            fallback_summary = ""
        
        # 4c) Final Summary Insight (prefer structured)
        result["Summary Insight"] = structured_summary or fallback_summary or "No summary insight available."

        # Merge extracted fields
        result.update(normalized)
        
        # 5) Save full result
        base = os.path.splitext(fname)[0]
        out_path = os.path.join(OUTPUT_FOLDER, f"{base}_parsed.json")
        with open(out_path, "w", encoding="utf-8") as fout:
            json.dump(result, fout, indent=2)
        print(f"  → Saved {base}_parsed.json\n")

