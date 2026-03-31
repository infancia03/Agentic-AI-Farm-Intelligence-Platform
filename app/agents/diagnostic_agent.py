"""
AgriFarm — Diagnostic Agent
YOUR original agentic tool-calling loop (kept intact).
Added: RAG knowledge-base tool + improved system prompt.
"""

from __future__ import annotations
import json
import time
from typing import Any

from sqlalchemy.orm import Session
from loguru import logger

from app.llm_client import get_client, FREE_MODELS
from app.tools.sensor_tools import SensorTools
import os


SYSTEM_PROMPT = """You are an expert agricultural diagnostic AI agent for Indian farms.

Your responsibilities:
1. Analyse farm IoT sensor data (temperature, moisture, pH, NPK) to identify crop health issues
2. Detect anomalies and correlate multiple factors (e.g., temp + humidity = disease risk)
3. Search the knowledge base for proven treatment protocols
4. Provide evidence-based diagnostic reports ranked by severity

Rules:
- ALWAYS call sensor tools to gather data BEFORE making any diagnosis
- Use lookup_knowledge_base for treatment advice on diseases or deficiencies
- Consider Indian farming context (Tamil Nadu / Maharashtra / Punjab)
- Prioritise issues as Critical > High > Medium > Low
- Keep final response under 400 words and actionable
"""


class DiagnosticAgent:
    """
    Diagnostic Agent — iterative LLM tool-calling loop.
    The LLM decides which tools to call; the agent executes them
    and feeds results back until the LLM produces a final text answer.
    """

    def __init__(self):
        self.tools = SensorTools()
        self.tool_schemas = SensorTools.TOOL_SCHEMAS

    # ── Tool dispatch ─────────────────────────────────────────
    def _run_tool(self, name: str, args: dict, db: Session) -> Any:
        dispatch = {
            "get_latest_sensor_data":      self.tools.get_latest_sensor_data,
            "analyze_soil_health":         self.tools.analyze_soil_health,
            "check_irrigation_efficiency": self.tools.check_irrigation_efficiency,
            "detect_pest_patterns":        self.tools.detect_pest_patterns,
            "get_field_history":           self.tools.get_field_history,
            "get_active_alerts":           self.tools.get_active_alerts,
            "lookup_knowledge_base":       self.tools.lookup_knowledge_base,
        }
        fn = dispatch.get(name)
        if fn:
            return fn(db, **args)
        return {"error": f"Unknown tool: {name}"}

    # ── Main agentic loop (your original logic, preserved) ────
    def diagnose(self, query: str, db: Session, field_id: str | None = None) -> dict:
        start      = time.time()
        tools_used: list[str] = []
        messages = [{"role": "user", "content": f"[Field: {field_id}] {query}" if field_id else query}]

        client = get_client()
        model  = FREE_MODELS[0]

        for iteration in range(6):    # max 6 tool-call rounds
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                    tools=self.tool_schemas,
                    tool_choice="auto",
                    max_tokens=1024,
                    temperature=0.3,
                )
                msg = resp.choices[0].message

                # ── LLM wants to call tools ───────────────────
                if msg.tool_calls:
                    # Append assistant turn with tool_calls
                    messages.append({
                        "role":       "assistant",
                        "content":    msg.content or "",
                        "tool_calls": [
                            {
                                "id":       tc.id,
                                "type":     "function",
                                "function": {
                                    "name":      tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    })

                    for tc in msg.tool_calls:
                        fn_name = tc.function.name
                        fn_args = json.loads(tc.function.arguments)
                        result  = self._run_tool(fn_name, fn_args, db)
                        tools_used.append(fn_name)
                        logger.debug(f"Tool {fn_name} → {str(result)[:80]}")

                        messages.append({
                            "role":         "tool",
                            "tool_call_id": tc.id,
                            "content":      json.dumps(result),
                        })

                else:
                    # ── Final text answer ─────────────────────
                    return {
                        "response":       msg.content or "",
                        "tools_used":     list(set(tools_used)),
                        "execution_time": round(time.time() - start, 2),
                        "iterations":     iteration + 1,
                    }

            except Exception as e:
                logger.error(f"DiagnosticAgent error (iter {iteration}): {e}")
                return {
                    "response":       f"Diagnostic error: {e}",
                    "tools_used":     tools_used,
                    "execution_time": round(time.time() - start, 2),
                    "error":          True,
                }

        return {
            "response":       "Analysis complete (max iterations reached).",
            "tools_used":     list(set(tools_used)),
            "execution_time": round(time.time() - start, 2),
            "iterations":     6,
        }
