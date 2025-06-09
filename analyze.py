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


# ---------- INSIGHT GENERATION BLOCK ----------


def build_insight_prompt(deck_slide_text: str) -> str:
    """
    Build a high-quality prompt for qualitative pitch evaluation based on deck content.
    This prompt will be sent to OpenAI to generate Pitch Score, Red Flags, Suggested Questions, and Summary Insight.
    """
    prompt = """
You are a world-class venture capital analyst. Given the slide text from a startup's pitch deck, evaluate the deck's quality and investment readiness.

Only flag information as missing if it is genuinely absent or unclear. For example:
- If a deck has a slide titled similar to “Business Model” or “Revenue Model” or a calculation of revenue/volume/sales is made, do not say the revenue model is missing.
- If a slide shows customer acquisition plan or channels or funnels, do not say the strategy is unclear.
- If team roles and bios are listed, do not flag them as incomplete.

Be generous but fair in your assessment.


Return exactly one JSON object with the following keys:
- "Pitch Score": integer (0 to 100) — overall quality of the pitch, based on clarity, traction, team, market, and completeness.
- "Red Flags": list of strings — weaknesses, missing slides, unclear metrics, unrealistic claims, etc.
- "Suggested Questions": list of strings — what an investor should ask in a meeting to probe the deck further.
- "Summary Insight": one or two sentences summarizing the investment potential.

If information is missing, penalize the score and flag it clearly.

--- EXAMPLE 1 ---
Slide text:
"We are an AI platform helping students revise smarter using personalized flashcards. The product is live with 2k monthly users. Team: Janet (Founder, ex-Edmodo), Kunle (CTO, Oxford PhD). Monetization TBD."

JSON Output:
{
  "Pitch Score": 68,
  "Red Flags": [
    "No clear monetization strategy",
    "Limited traction data (only user count mentioned)"
  ],
  "Suggested Questions": [
    "What are your revenue projections for the next 12 months?",
    "Who is your paying customer (schools, parents, students)?"
  ],
  "Summary Insight": "The founding team has strong credentials and early traction, but monetization and go-to-market strategy remain unclear."
}

--- EXAMPLE 2 ---
Slide text:
"Our SaaS platform automates logistics for mid-size retailers. $150k ARR in 6 months, with 95% retention. Team includes ex-Amazon logistics head. Raising $1M Seed to scale."

JSON Output:
{
  "Pitch Score": 90,
  "Red Flags": [],
  "Suggested Questions": [
    "What’s your CAC and LTV?",
    "How do you plan to scale customer acquisition?"
  ],
  "Summary Insight": "This is a high-quality deck with strong traction and a credible team in a clear market."
}

--- NOW EVALUATE THIS DECK ---
Slide text:
""" + deck_slide_text.strip() + """

JSON Output:
"""
    return prompt


def call_chatgpt_insight(prompt: str, api_key: str, model="gpt-3.5-turbo") -> dict:
    from openai import OpenAI
    import json
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=800,
    )

    content = response.choices[0].message.content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end != -1:
            return json.loads(content[start:end])
        raise ValueError(f"Could not parse JSON from insight response:\n{content}")








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

        # 4) Generate VC insight
        scoring_prompt = build_structured_scoring_prompt(deck_text)
        try:
            scoring_result = call_structured_pitch_scorer(scoring_prompt, api_key)
            result["Section Scores"] = scoring_result.get("sections", [])
            result["Pitch Score"] = scoring_result.get("total_score", None)
            result["Summary Insight"] = scoring_result.get("summary", "")
        except Exception as e:
            print(f"  ⚠️ Failed to generate structured scoring for {fname}: {e}")
            result["Section Scores"] = []
            result["Pitch Score"] = None
            result["Summary Insight"] = "Could not generate structured insight."

        # 4b) Red Flags + Questions
        insight_prompt = build_insight_prompt(deck_text)
        try:
            insight_result = call_chatgpt_insight(insight_prompt, api_key)
            result["Red Flags"] = insight_result.get("Red Flags", [])
            result["Suggested Questions"] = insight_result.get("Suggested Questions", [])
        except Exception as e:
            print(f"  ⚠️ Failed to generate red flags/questions for {fname}: {e}")
            result["Red Flags"] = []
            result["Suggested Questions"] = []

      
        # Merge extracted fields
        result.update(normalized)
        
        # 5) Save full result
        base = os.path.splitext(fname)[0]
        out_path = os.path.join(OUTPUT_FOLDER, f"{base}_parsed.json")
        with open(out_path, "w", encoding="utf-8") as fout:
            json.dump(result, fout, indent=2)
        print(f"  → Saved {base}_parsed.json\n")
