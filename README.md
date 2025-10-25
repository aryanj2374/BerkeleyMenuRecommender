## Berkeley Dining RAG Toolkit

This repository provides everything needed to scrape UC Berkeley dining hall menus, build a retrieval-ready dataset, and power a chat assistant that recommends meals to students.

### Components

- **`scraper.py`** – pulls the live dining menus, enriches each item with nutrition, allergen, icon legend, and location metadata, then writes it to `menus.json`.
- **`menu_recommender.py`** – loads `menus.json`, builds a TF‑IDF index, layers nutrition/dietary heuristics, and returns top dishes for any request.
- **`chat_agent.py`** – wraps the recommender and (optionally) an OpenAI-compatible LLM so you can answer user queries in natural language.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Regenerate the menu data (requires network access):

```bash
python3 scraper.py
```

Run the recommender directly:

```bash
python3 menu_recommender.py "high protein vegetarian meal"
```

The first run might take a few seconds while the TF‑IDF index is built; results are
cached under `.cache/recommender` so subsequent queries start almost instantly.

### Chat assistant (Ollama)

Start Ollama and pull a local model (for example `llama3.1`):

```bash
ollama pull llama3.1
```

You can override the host or model with `OLLAMA_HOST` and `--model`.

```bash
python3 chat_agent.py "What should I eat before a workout?"
```

> Tip: Older Ollama builds (pre `/api/chat`) still work—the helper automatically
> falls back to `/api/generate`, and if that is missing, it shells out to the
> `ollama run` CLI. You don’t have to upgrade immediately, but ensuring the CLI is
> on PATH keeps the final fallback available.

If the Ollama service is unavailable the assistant falls back to a rules-based summary of the top matches.

For multi-turn chats, just run:

```bash
python3 chat_agent.py
```

### HTTP API

Serve the assistant over HTTP (suitable for web or mobile front ends):

```bash
uvicorn app:app --reload
```

Environment variables:

- `MENUS_JSON` – path to the menu dataset (defaults to `menus.json`).
- `OLLAMA_MODEL` – override the model name (defaults to `llama3.1`).
- `OLLAMA_HOST` – Ollama base URL (defaults to `http://localhost:11434`).

Example request:

```bash
curl -X POST http://127.0.0.1:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"query":"Need a vegetarian high protein dinner","top_k":5}'
```

Response fields:

- `used_llm` – whether an LLM generated the prose reply.
- `response` – the natural language answer you can show directly.
- `items` – array of structured cards with:
  - `name`
- `serving` → location, meal period, and hours
- `serving.crowdedness` → estimated wait level (`Low/Medium/High` + score)
  - `dietary` → choices/tags (e.g., vegan, halal)
  - `nutrition` → key macros
  - `blurb` → short item description
  - `metadata` → category, relevance score, and source reference

The response contains the LLM-formatted reply, whether the LLM was used, and the structured recommendation list.

### Next steps

- Wire the chat agent into your front-end or messaging interface.
- Swap the TF‑IDF retriever for embeddings/vector-db once the dataset or queries become more complex.
- Extend the heuristics to incorporate wait-time signals or Grubhub availability when those systems are ready.
