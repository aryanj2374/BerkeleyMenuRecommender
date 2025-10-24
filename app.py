"""FastAPI service exposing the dining chat assistant."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

from chat_agent import ChatResult, DiningChatAgent, DEFAULT_MODEL, load_agent

app = FastAPI(title="Cal Dining Assistant")

AGENT: Optional[DiningChatAgent] = None


class ChatRequest(BaseModel):
    query: str = Field(..., description="Natural language request from the user.")
    top_k: int = Field(5, ge=1, le=10, description="Maximum number of dishes to return.")


class RecommendationItem(BaseModel):
    name: str
    serving: Dict[str, Any]
    dietary: Dict[str, Any]
    nutrition: Dict[str, Any]
    blurb: str
    metadata: Dict[str, Any]


class ChatResponse(BaseModel):
    used_llm: bool
    response: str
    items: list[RecommendationItem]


@app.on_event("startup")
def startup() -> None:
    """Initialize the chat agent once when the API starts."""
    global AGENT
    menus_path = Path(os.getenv("MENUS_JSON", "menus.json"))
    if not menus_path.exists():
        raise RuntimeError(f"menus.json not found at {menus_path}. Run scraper.py first.")

    model = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
    AGENT = load_agent(menus_path, model=model)


@app.get("/health", tags=["meta"])
def health() -> Dict[str, Any]:
    """Simple health check."""
    return {
        "status": "ok",
        "model": os.getenv("OLLAMA_MODEL", DEFAULT_MODEL),
        "menus_cached": bool(Path(os.getenv("MENUS_JSON", "menus.json")).exists()),
        "ollama_host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    }


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(request: ChatRequest) -> ChatResponse:
    """Return grounded recommendations for the provided query."""
    if AGENT is None:
        raise HTTPException(status_code=503, detail="Assistant is still loading. Try again shortly.")

    result: ChatResult = AGENT.respond(request.query, top_k=request.top_k)
    formatted = [_format_item_for_api(item) for item in result.recommendations]
    return ChatResponse(
        used_llm=result.used_llm,
        response=result.response,
        items=formatted,
    )


def _format_item_for_api(item: Dict[str, Any]) -> RecommendationItem:
    serving = {
        "location": item.get("location"),
        "meal": item.get("meal"),
        "hours": item.get("hours") or [],
        "hours_structured": item.get("hours_structured") or [],
    }
    dietary = {
        "choices": item.get("dietary_choices") or [],
        "tags": item.get("tags") or [],
    }
    metadata = {
        "category": item.get("category"),
        "score": item.get("score"),
        "menu_reference": item.get("menu_reference"),
    }
    return RecommendationItem(
        name=item.get("name", "Unknown Item"),
        serving=serving,
        dietary=dietary,
        nutrition=item.get("nutrition") or {},
        blurb=item.get("blurb") or "",
        metadata=metadata,
    )
