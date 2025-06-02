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
-Few-shot GPT prompting to extract structured JSON from unstructured pitch deck text.
-PDF rendering with PyMuPDF to preview selected slides.
-Interactive dashboard with filters by industry, funding stage, and founding year.
-Streamlit UI with two views: Library and Dashboard.
-Export results as CSV or JSON.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/pitch-deck-extractor.git
cd pitch-deck-extractor

### 2. Install Dependencies

