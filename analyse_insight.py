# analyse_insight.py
# ---------- INSIGHT GENERATION BLOCK ----------


def build_insight_prompt(deck_slide_text: str) -> str:
    """
    Build a high-quality prompt for qualitative pitch evaluation based on deck content.
    This prompt will be sent to OpenAI to generate Pitch Score, Red Flags, and Summary Insight.
    """
    prompt = """
You are a world-class venture capital analyst. Given the slide text from a startup's pitch deck, evaluate the deck's quality and investment readiness.

Return exactly one JSON object with the following keys:
- "Pitch Score": integer (0 to 100) — overall quality of the pitch, based on clarity, traction, team, market, and completeness.
- "Red Flags": list of strings — weaknesses, missing slides, unclear metrics, unrealistic claims, etc.
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
  "Summary Insight": "The founding team has strong credentials and early traction, but monetization and go-to-market strategy remain unclear."
}

--- EXAMPLE 2 ---
Slide text:
"Our SaaS platform automates logistics for mid-size retailers. $150k ARR in 6 months, with 95% retention. Team includes ex-Amazon logistics head. Raising $1M Seed to scale."

JSON Output:
{
  "Pitch Score": 90,
  "Red Flags": [],
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
