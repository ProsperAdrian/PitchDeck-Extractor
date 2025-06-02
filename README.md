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
