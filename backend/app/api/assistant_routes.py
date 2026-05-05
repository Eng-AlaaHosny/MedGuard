"""
LLM orchestration layer (Anthropic) with MedGuard as a tool.

Server-side only: set ANTHROPIC_API_KEY. The model calls `medguard_analyze_pair`
for structured DDI output; the assistant must not contradict tool JSON.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.routes import run_pair_inference

router = APIRouter(prefix="/assistant", tags=["assistant"])

DEFAULT_ANTHROPIC_MODEL = os.environ.get(
    "ANTHROPIC_MODEL",
    "claude-3-5-haiku-20241022",
)

MAX_TOOL_ROUNDS = 6
MAX_TOKENS = 4096

MEDGUARD_TOOL = {
    "name": "medguard_analyze_pair",
    "description": (
        "Run the MedGuard Bio_ClinicalBERT pipeline on two explicit drug names: "
        "NER on the inference text, DDI interaction type, severity, KG context, "
        "and Lipinski features. Use this whenever the user asks about interactions, "
        "safety of combining two medications, or similar. Pass normalized generic names "
        "when possible (e.g. warfarin, ibuprofen). "
        "Optional clinical_context is a short user sentence passed through to the model."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "drug_a": {
                "type": "string",
                "description": "First drug name (ingredient preferred).",
            },
            "drug_b": {
                "type": "string",
                "description": "Second drug name (ingredient preferred).",
            },
            "clinical_context": {
                "type": "string",
                "description": "Optional short clinical sentence for context.",
            },
        },
        "required": ["drug_a", "drug_b"],
    },
}

SYSTEM_PROMPT = """You are the conversational layer for the MedGuard academic demo.

Rules:
1) For any question about drug–drug interactions, combining two medications, or whether two drugs are safe together: call the tool `medguard_analyze_pair` with concrete drug names. You may call it more than once if the user compares several pairs.
2) When the tool returns JSON, your natural-language answer MUST be consistent with that JSON. Do not lower or raise severity or change interaction_type compared to the tool output. Quote or paraphrase severity_label, interaction_type, plain_guidance, and interaction_reason faithfully.
3) For general educational questions (e.g. what warfarin does, what a blood thinner is) you may answer without the tool, clearly as general information, not personalized medical advice.
4) Always remind the user this is an educational demonstration, not clinical advice. Never tell them to start, stop, or change a medication without a licensed clinician.

If drug names are vague, ask a brief clarifying question or call the tool with the best specific ingredient names you can infer."""


class AssistantMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AssistantChatRequest(BaseModel):
    messages: List[AssistantMessage] = Field(
        ...,
        min_length=1,
        description="Conversation history ending with the latest user message.",
    )


class ToolTraceItem(BaseModel):
    tool: str
    input: Dict[str, Any]
    ok: bool
    error: Optional[str] = None


class AssistantChatResponse(BaseModel):
    reply: str
    model: str
    tool_trace: List[ToolTraceItem] = []
    stop_reason: Optional[str] = None


def _ddi_to_tool_json(drug_a: str, drug_b: str, clinical_context: Optional[str]) -> Dict[str, Any]:
    ctx = (clinical_context or "").strip()
    res = run_pair_inference(drug_a.strip(), drug_b.strip(), ctx or None)
    return res.model_dump(mode="json")


def _execute_tool(name: str, tool_input: Dict[str, Any]) -> tuple[str, bool, Optional[str]]:
    if name != "medguard_analyze_pair":
        err = f"Unknown tool: {name}"
        return json.dumps({"error": err}), False, err
    drug_a = tool_input.get("drug_a") or ""
    drug_b = tool_input.get("drug_b") or ""
    clinical_context = tool_input.get("clinical_context")
    if not drug_a.strip() or not drug_b.strip():
        err = "drug_a and drug_b are required non-empty strings."
        return json.dumps({"error": err}), False, err
    try:
        payload = _ddi_to_tool_json(drug_a, drug_b, clinical_context)
        return json.dumps(payload), True, None
    except Exception as exc:  # noqa: BLE001 — surface to model as tool error
        err = str(exc)
        return json.dumps({"error": err, "detail": err}), False, err


def _anthropic_client():
    try:
        import anthropic
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="Anthropic SDK not installed. Run: pip install -r requirements-llm.txt",
        ) from exc
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY is not set. Add it to the environment for the assistant endpoint.",
        )
    return anthropic.Anthropic(api_key=key)


@router.get("/status")
def assistant_status():
    key_ok = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    try:
        import anthropic  # noqa: F401
        sdk_ok = True
    except ImportError:
        sdk_ok = False
    return {
        "anthropic_sdk_installed": sdk_ok,
        "anthropic_api_key_configured": key_ok,
        "default_model": DEFAULT_ANTHROPIC_MODEL,
    }


@router.post("/chat", response_model=AssistantChatResponse)
async def assistant_chat(request: AssistantChatRequest):
    """
    Anthropic Messages API with tool use; MedGuard runs only inside tool execution.
    """
    client = _anthropic_client()
    model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL).strip() or DEFAULT_ANTHROPIC_MODEL

    messages: List[Dict[str, Any]] = [
        {"role": m.role, "content": m.content} for m in request.messages
    ]

    tool_trace: List[ToolTraceItem] = []
    stop_reason: Optional[str] = None
    final_text_parts: List[str] = []
    last_response: Any = None

    def _one_turn(msgs: List[Dict[str, Any]]):
        return client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=[MEDGUARD_TOOL],
            messages=msgs,
        )

    for _ in range(MAX_TOOL_ROUNDS):
        response = await asyncio.to_thread(_one_turn, messages)
        last_response = response
        stop_reason = response.stop_reason

        assistant_blocks = response.content
        messages.append({"role": "assistant", "content": assistant_blocks})

        tool_use_blocks = [b for b in assistant_blocks if getattr(b, "type", None) == "tool_use"]
        if not tool_use_blocks:
            for block in assistant_blocks:
                if getattr(block, "type", None) == "text":
                    final_text_parts.append(block.text)
            break

        tool_result_payloads: List[Dict[str, Any]] = []
        for block in tool_use_blocks:
            raw_input = getattr(block, "input", {}) or {}
            if not isinstance(raw_input, dict):
                raw_input = {}
            out_str, ok, err = await asyncio.to_thread(
                _execute_tool, block.name, raw_input
            )
            tool_trace.append(
                ToolTraceItem(
                    tool=block.name,
                    input=dict(raw_input),
                    ok=ok,
                    error=err,
                )
            )
            tool_result_payloads.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": out_str,
                }
            )
        messages.append({"role": "user", "content": tool_result_payloads})

    reply = "\n\n".join(t for t in final_text_parts if t).strip()
    if not reply and last_response is not None:
        for block in last_response.content:
            if getattr(block, "type", None) == "text":
                reply += block.text
        reply = reply.strip()
    if not reply:
        reply = (
            "No assistant text was returned. "
            "If a tool ran, check tool_trace in the JSON response for model outputs."
        )

    return AssistantChatResponse(
        reply=reply,
        model=model,
        tool_trace=tool_trace,
        stop_reason=stop_reason,
    )
