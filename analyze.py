# scripts/analyze.py

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from extract_text import extract_text_from_pdf  # our Phase 2 function

# Load the OpenAI API key from .env
load_dotenv()
client = OpenAI(api_key="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")

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
Gross Revenues in 2020: $3,100
"""

EXAMPLE_1_JSON = {
  "StartupName": "Yabscore",
  "FoundingYear": "2019",
  "Founders": ["IK Ezekwelu", "Dapo Arowa"],
  "Industry": "Sporttech",
  "Niche": "Mobile sports betting",
  "USP": "Yabscore is the first fully mobile sports-betting platform tailored to Nigerian football fans, offering in-play wagering and live performance stats.",
  "FundingStage": None,
  "CurrentRevenue": "$3,100",
  "Market": { "TAM": "$95 Billion", "SAM": "$2.2 Billion", "SOM": "$193 Million" },
  "AmountRaised": "0"
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
  "StartupName": "Quidax",
  "FoundingYear": "2018",
  "Founders": ["Buchi Okoro", "Uzo Awili", "Morris Ebieroma"],
  "Industry": "FinTech",
  "Niche": "Cryptocurrency exchange",
  "USP": "All-in-one platform with seamless fiat on/off ramps and a single API enabling African users and businesses to access 1,200+ crypto pairs securely",
  "FundingStage": "null",
  "CurrentRevenue": "$10m ARR",
  "Market": { "TAM": "null", "SAM": "null", "SOM": "null" },
  "AmountRaised": "0"
}



PROMPT_PREFIX = """
You are an expert at extracting structured data from investor pitch decks. For each deck, I will present the slide text. Return exactly one JSON object with these ten fields:
{
  "StartupName": string or null,  # what is the likely startup name?
  "FoundingYear": string or null, # If no explicit “Founded in YYYY” appears, Scan all content for founding-year clues, including: • timeline or roadmap dates, • traction graphs captions, • team-bio phrasing, • funding-history dates. Determine the most probable calendar year in which the company was founded. If multiple plausible years appear, choose the earliest one that has at least one direct or indirect supporting signal.
  "Founders": [string, ...] or null,
  "Industry": string or null,       # one of: Fintech, Insurtech, Regtech, Healthtech, Medtech, Biotech, Pharmatech, Femtech, Eldertech, Proptech, Contech, Agtech, Foodtech, RestaurantTech, ClimateTech, CleanTech, EnergyTech, Greentech, Edtech, HRtech, Worktech, Martech, Adtech, RetailTech, Ecommerce, Marketplace, MobilityTech, Autotech, TransportTech, LogisticsTech, SupplyChainTech, TravelTech, SpaceTech, AerospaceTech, DefenceTech, SportTech, GamingTech, eSportsTech, MediaTech, StreamingTech, MusicTech, CreatorEconomyTech, SocialTech, Cybersecurity, AI, MachineLearning, BigData, AnalyticsTech, CloudTech, SaaS, DevOps, IoT, Robotics, HardwareTech, WearablesTech, 3DPrinting, AR/VR/XR, Metaverse, Web3, Blockchain, Crypto, NFT, QuantumTech, LegalTech, Govtech, CivicTech, NonprofitTech, ProductivityTech, CollaborationTech, PetTech, ElderCareTech etc.
  "Niche": string or null,          # free-text description e.g. “crypto exchange”, “mobile betting”, “AI tutoring”
  "USP": string or null,            # a single sentence from the deck that states the unique selling proposition
  "FundingStage": string or null,   # If no explicit round is mentioned, Scan the deck for the following signals: • capital sought, • traction metrics (users, revenue, growth), • product maturity, • team size & seniority, • prior funding, • planned use of funds, • target investors, • implied valuation. Using these signals and standard VC heuristics, decide the most probable funding round (Pre-seed, Seed, Series A, Series B, Series C or later).
  "CurrentRevenue": string or null,
  "Market": { "TAM": string or null, "SAM": string or null, "SOM": string or null } or null,
  "AmountRaised": string or null,  # what is the likely cummulative amount the startup has raised from inception?
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

def call_chatgpt(prompt, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=800
    )
    # Access `.content` instead of indexing as a dict
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
        print(f"Processing {fname}…")

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
            "StartupName", "FoundingYear", "Founders", "Industry", 
            "Niche", "USP", "FundingStage", "CurrentRevenue", "Market","AmountRaised"
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

        base = os.path.splitext(fname)[0]
        out_path = os.path.join(OUTPUT_FOLDER, f"{base}_parsed.json")
        with open(out_path, "w", encoding="utf-8") as fout:
            json.dump(normalized, fout, indent=2)
        print(f"  → Saved {base}_parsed.json\n")