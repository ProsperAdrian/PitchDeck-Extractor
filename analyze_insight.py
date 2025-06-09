# analyze_insight.py
# ---------- INSIGHT GENERATION BLOCK ----------


def build_insight_prompt(deck_slide_text: str) -> str:
    """
    Build a high-quality prompt for qualitative pitch evaluation based on deck content.
    This prompt will be sent to OpenAI to generate Pitch Score, Red Flags, Suggested Questions, and Summary Insight.
    """
    prompt = """
You are a world-class venture capital analyst. Given the slide text from a startup's pitch deck, evaluate the deck's quality and investment readiness.

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


extract_text.py:

# scripts/extract_text.py

import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """
    Returns a single string with each slide labeled:
      "----- Slide 1 -----\n<slide 1 text>\n\n----- Slide 2 -----\n<slide 2 text>\n\n..."
    """
    doc = fitz.open(pdf_path)
    lines = []
    for page in doc:
        text = page.get_text().strip()
        lines.append(f"----- Slide {page.number+1} -----\n{text}\n")
    return "\n".join(lines)


if __name__ == "__main__":
    # If run directly, process all PDFs in input_decks/ and write .txt to ground_truth/
    import os
    INPUT_FOLDER = "input_decks"
    OUTPUT_FOLDER = "ground_truth"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for fname in os.listdir(INPUT_FOLDER):
        if not fname.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(INPUT_FOLDER, fname)
        plain = extract_text_from_pdf(pdf_path)
        base = os.path.splitext(fname)[0]
        out_txt = os.path.join(OUTPUT_FOLDER, f"{base}.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(plain)
        print(f"Extracted text for {fname} → {base}.txt")
