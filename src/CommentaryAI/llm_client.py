"""LLM sağlayıcı soyutlaması: Google Gemini / şablon yedeği.

Hiçbir SDK'ya sert bağımlılık yok; Gemini saf HTTP (httpx) ile çağrılır,
böylece kurulum hafif kalır. Anahtar yoksa otomatik şablon anlatımına düşülür.
"""

from __future__ import annotations

from typing import Any

import httpx

from CommentaryAI.prompts.commentary_prompt import (
    SYSTEM_PROMPT,
    build_user_prompt,
    events_to_lines,
)


class CommentaryResult:
    def __init__(self, text: str, provider: str) -> None:
        self.text = text
        self.provider = provider


def generate_commentary(
    events: list[dict[str, Any]],
    provider: str,
    *,
    gemini_api_key: str = "",
    gemini_model: str = "gemini-2.0-flash",
    target_seconds: float | None = None,
) -> CommentaryResult:
    if not events:
        return CommentaryResult("Maçta kayda değer bir pas tespit edilemedi.", "template")

    try:
        if provider == "gemini" and gemini_api_key:
            return CommentaryResult(_call_gemini(events, gemini_api_key, gemini_model, target_seconds), "gemini")
    except Exception as exc:  # ağ/anahtar hatasında demoyu bozma
        return CommentaryResult(_template_commentary(events), f"template (fallback: {exc.__class__.__name__})")

    return CommentaryResult(_template_commentary(events), "template")


def _call_gemini(events: list[dict[str, Any]], api_key: str, model: str,
                 target_seconds: float | None = None) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    resp = httpx.post(
        url,
        params={"key": api_key},
        json={
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": build_user_prompt(events, target_seconds)}]}],
            "generationConfig": {"temperature": 0.9},
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def _template_commentary(events: list[dict[str, Any]]) -> str:
    """LLM yokken kural tabanlı, yine de coşkulu bir anlatım üret."""

    parts: list[str] = ["Ve maçımız tüm hızıyla devam ediyor sevgili izleyiciler!"]
    for ev in events:
        etype = ev.get("event_type")
        frm = ev.get("from_player_id")
        to = ev.get("to_player_id")
        if etype == "completed_pass":
            parts.append(f"İşte {frm} numara topla buluştu ve {to} numaraya nefis bir pas! Müthiş bir kombinasyon!")
        elif etype == "intercepted_pass":
            parts.append(f"Ama dikkat! {frm} numaranın pasını {to} numara araya girip KAPTI! İnanılmaz bir okuma!")
        else:
            parts.append(f"{frm} numaranın ayağında top karıştı, kim alacak derken oyun sürüyor!")
    parts.append("Ne maç ama, kenarda heyecan dorukta!")
    return " ".join(parts)
