# analyze_scoring.py

import json
from openai import OpenAI

# Define your rubric
SCORING_RUBRIC = [
    { "name": "Team",                           "weight": 15, "aliases": ["team", "leadership", "founders"] },
    { "name": "Problem & Opportunity",          "weight": 15, "aliases": ["problem", "pain point", "opportunity"] },
    { "name": "Solution & Product",             "weight": 20, "aliases": ["solution", "product", "offering"] },
    { "name": "Market Size & Competitive Landscape", "weight": 15, "aliases": ["market", "tam", "sam", "som", "competition", "competitive", "landscape"] },
    { "name": "Business Model & Financials",    "weight": 15, "aliases": ["business model", "revenue model", "financials", "projections", "revenue"] },
    { "name": "Traction",                       "weight": 15, "aliases": ["traction", "metrics", "growth"] },
    { "name": "Ask & Use of Proceeds",          "weight":  5, "aliases": ["ask", "funds", "use of proceeds"] }
]

# 2) Add this function immediately below:
def apply_weights(sections: list[dict]) -> int:
    """
    Given a list of {"name":…, "score":…}, compute the weighted total
    (0–100 normalized).
    """
    weight_map = {sec["name"]: sec["weight"] for sec in SCORING_RUBRIC}
    total      = sum(sec["score"] * weight_map.get(sec["name"], 0) for sec in sections)
    max_total  = sum(weight * 10 for weight in weight_map.values())
    return round(total / max_total * 100)

def build_structured_scoring_prompt(deck_text: str) -> str:
    # Dynamically list your sections + weights
    section_lines = "\n".join(
        f"{i+1}. {sec['name']} (weight {sec['weight']})"
        for i, sec in enumerate(SCORING_RUBRIC)
    )
    return f"""
You are a world-class venture capital analyst evaluating startup pitch decks. Your task is to score the quality of a pitch based on **exactly these {len(SCORING_RUBRIC)} sections**:

{section_lines}

🔒 **Important Rules**:
- Only score a section if the content *directly* addresses it in the pitch. Do not assume or infer.
- If a section is **missing**, vague, or superficial, give it a **score of 0 to 3** and say why.
- Never award 10/10 unless the content is clear, complete, and convincing.
- You MUST include a brief reason for each score (1 sentence max). Use the key `"comment"` for it.

- **Do** compute a weighted sum (score × weight) and return that as `total_score`.

Return your output as **strict JSON**:
{{
  "sections": [
    {{ "name": "...", "score": 7, "comment": "..." }},
    …
  ],
  "total_score": <your weighted total>
}}
--- BEGIN SLIDE TEXT ---
{deck_text}
--- END SLIDE TEXT ---
"""


def call_structured_pitch_scorer(prompt: str, api_key: str, model="gpt-4") -> dict:
    client   = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1000,
    )
    content = response.choices[0].message.content.strip()

    # 1) Parse JSON
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        start  = content.find("{")
        end    = content.rfind("}") + 1
        result = json.loads(content[start:end])

    # 2) Recompute the weighted total
    sections      = result.get("sections", [])
    weighted_score = apply_weights(sections)

    # 3) Return only what you need
    return {
        "sections":    sections,
        "total_score": weighted_score
    }
