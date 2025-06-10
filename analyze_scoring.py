# analyze_scoring.py

import json
from openai import OpenAI

# Define your rubric
SCORING_RUBRIC = [
    { "name": "Team",                           "weight": 0.15, "aliases": ["team", "leadership", "founders"] },
    { "name": "Problem & Opportunity",          "weight": 0.15, "aliases": ["problem", "pain point", "opportunity"] },
    { "name": "Solution & Product",             "weight": 0.20, "aliases": ["solution", "product", "offering"] },
    { "name": "Market Size & Competitive Landscape", "weight": 0.15, "aliases": ["market", "tam", "sam", "som", "competition", "competitive", "landscape"] },
    { "name": "Business Model & Financials",    "weight": 0.15, "aliases": ["business model", "revenue model", "financials", "projections", "revenue"] },
    { "name": "Traction",                       "weight": 0.15, "aliases": ["traction", "metrics", "growth"] },
    { "name": "Ask & Use of Proceeds",          "weight": 0.05, "aliases": ["ask", "funds", "use of proceeds"] }
]

def apply_weights(sections: list[dict]) -> int:
    """
    Given a list of {"name":…, "score":…, "comment":…}, compute the weighted total (0–100).
    Weights are fractions (e.g., 0.15 for 15%) and sum to 1.0.
    """
    weight_map = {sec["name"]: sec["weight"] for sec in SCORING_RUBRIC}
    total = sum(sec["score"] * weight_map.get(sec["name"], 0) * 10 for sec in sections)
    return round(total)

def build_structured_scoring_prompt(deck_text: str) -> str:
    # Dynamically list sections + weights
    section_lines = "\n".join(
        f"{i+1}. {sec['name']} (weight {int(sec['weight']*100)}%)"
        for i, sec in enumerate(SCORING_RUBRIC)
    )
    return f"""
You are a world-class venture capital analyst evaluating startup pitch decks. Your task is to score the quality of a pitch based on **exactly these {len(SCORING_RUBRIC)} sections**:

{section_lines}

🔒 **Important Rules**:
- Assign a score from **0 to 10** for each section (0 = poor/absent, 10 = exceptional).
- Only score a section if the content *directly* addresses it in the pitch. Do not assume or infer.
- If a section is **missing**, vague, or superficial, give it a **score of 0 to 3** and explain why.
- Never award 10/10 unless the content is clear, complete, compelling, and tailored to the startup.
- Include a brief reason for each score (1 sentence max) using the key `"comment"`.
- Compute the weighted total score (0-100) using the formula:
  total_score = (Team × {SCORING_RUBRIC[0]['weight']} + Problem & Opportunity × {SCORING_RUBRIC[1]['weight']} + Solution & Product × {SCORING_RUBRIC[2]['weight']} + Market Size & Competitive Landscape × {SCORING_RUBRIC[3]['weight']} + Business Model & Financials × {SCORING_RUBRIC[4]['weight']} + Traction × {SCORING_RUBRIC[5]['weight']} + Ask & Use of Proceeds × {SCORING_RUBRIC[6]['weight']}) × 10
- Round the total_score to the nearest integer.

Return your output as **strict JSON**:
{{
  "sections": [
    {{ "name": "Team", "score": 7, "comment": "Experienced founders but lacks technical expertise" }},
    ...
  ],
  "total_score": <weighted total (0-100)>
}}

--- BEGIN SLIDE TEXT ---
{deck_text.strip()}
--- END SLIDE TEXT ---
"""

def call_structured_pitch_scorer(prompt: str, api_key: str, model="gpt-3.5-turbo") -> dict:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1000,
    )
    content = response.choices[0].message.content.strip()

    # Parse JSON
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end != -1:
            result = json.loads(content[start:end])
        else:
            raise ValueError(f"Could not parse JSON from scoring response:\n{content}")

    # Validate section scores
    sections = result.get("sections", [])
    for section in sections:
        score = section.get("score", 0)
        if not isinstance(score, (int, float)) or not 0 <= score <= 10:
            raise ValueError(f"Invalid score {score} for section {section.get('name')}; must be an integer between 0 and 10")

    # Recompute the weighted total
    weighted_score = apply_weights(sections)

    # Validate total_score
    if not 0 <= weighted_score <= 100:
        raise ValueError(f"Invalid total_score {weighted_score}; must be between 0 and 100")

    return {
        "sections": sections,
        "total_score": weighted_score
    }
