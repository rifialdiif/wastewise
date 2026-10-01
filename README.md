# WasteWise

AI-based waste classification (MobileNetV2) and Knowledge-Base-grounded treatment recommendation API (FastAPI + Gemini).

> Work in progress. Full documentation comes in a later phase.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env        # then fill in GEMINI_API_KEY
```

## Run

```bash
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs

## Test

```bash
pytest
```
