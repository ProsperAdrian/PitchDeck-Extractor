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
- Return total score (sum of all 10 section scores) as `total_score`.

🛑 If a section is not present, do not guess—penalize.

Return your output as **strict JSON**:

```json
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
