# Pitch Deck Extractor (AI‐Backed + Heuristics)

## Overview
This tool ingests investor pitch‐deck PDFs, sends slide text to a few‐shot ChatGPT prompt, and outputs a JSON summarizing:
- `Startup Name`
- `Founding Year`
- `Founders` (array)
- `Industry`
- `Niche`
- `USP`
- `Funding Stage`
- `Current Revenue`
- `Market` (`TAM`, `SAM`, `SOM`)
- `Amount Raised`

The results are presented in a searchable table, can be exported (CSV/JSON), and preview key slides (Team, Market, Traction) using GPT-based heuristics.

## Features
- Few-shot GPT prompting to extract structured JSON from unstructured pitch deck text.
- PDF rendering with PyMuPDF to preview selected slides.
- Interactive dashboard with filters by industry, funding stage, and founding year.
- Streamlit UI with two views: Library and Dashboard.
- Export results as CSV or JSON.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/pitch-deck-extractor.git
cd pitch-deck-extractor
```

### 2. Install Dependencies

```pip install -r requirements.txt```

### 3. Set Your OpenAI API Key

```[openai]
api_key = "your_openai_api_key"
```

## Usage

### Run the Streamlit App

```streamlit run streamlit_app.py```

### Analyze Pitch Decks

- pload one or more PDFs in the Library View
- View extracted fields in a table
- Preview key slides (Team, Market, Traction)
- Filter and analyze startups in the Dashboard View
- Export to CSV or JSON


## Project Structure

├── input_decks/               # Drop your PDF files here (batch mode)
├── parsed_entities/           # JSON output files
├── scripts/
│   └── analyze.py             # Core GPT-powered extraction logic
├── extract_text.py            # PDF-to-text utility
├── streamlit_app.py           # Streamlit app frontend
├── .streamlit/secrets.toml    # Store OpenAI API key here
├── requirements.txt
└── README.md

## Fields Extracted

| Field               | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| **Startup Name**    | Most likely company name                                     |
| **Founding Year**   | Inferred from slides or context                              |
| **Founders**        | Names listed on the team slide                               |
| **Industry**        | Categorized into common verticals (e.g. FinTech, HealthTech) |
| **Niche**           | Brief description like "crypto exchange", "mobile betting"   |
| **USP**             | Unique selling proposition (verbatim from deck)              |
| **Funding Stage**   | Inferred from revenue, team, or milestones                   |
| **Current Revenue** | Most recent actual revenue (not forecast)                    |
| **Market**          | TAM / SAM / SOM if available                                 |
| **Amount Raised**   | Total historical funding raised                              |


 ## Example Output

 {
  "Startup Name": "Yabscore",
  "Founding Year": "2019",
  "Founders": ["IK Ezekwelu", "Dapo Arowa"],
  "Industry": "Sporttech",
  "Niche": "Mobile sports betting",
  "USP": "Yabscore is the first fully mobile sports-betting platform tailored to Nigerian football fans...",
  "Funding Stage": "Seed",
  "Current Revenue": "$3.1k",
  "Market": {
    "TAM": "$95B",
    "SAM": "$2.2B",
    "SOM": "$193M"
  },
  "Amount Raised": "$10m"
}

## Roadmap

- OCR support for scanned/image-based decks
- PDF export of summary reports
- Multi-language pitch deck parsing
- DocSend/Notion pitch link conversion
- Domain-specific LLM fine-tuning (optional)

## Contributing

Contributions, ideas, and feedback are welcome!
Feel free to fork the repo and submit a pull request.

