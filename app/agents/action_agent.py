"""
AgriFarm — Action Agent
YOUR original action/remediation agent — kept intact.
The LLM chooses which farm actions to execute via tool calling.
"""

from __future__ import annotations
import json
import time
from typing import Any

from sqlalchemy.orm import Session
from loguru import logger

from app.llm_client import get_client, FREE_MODELS
from app.database import RemediationLog


ACTION_SYSTEM_PROMPT = """You are an agricultural action-planning AI agent for Indian farms.

Your job:
1. Read the diagnostic report for a field
2. Decide which remediation actions to execute (in priority order)
3. Call the appropriate action tools
4. Return a cost summary and execution plan

Available actions and their typical costs (INR):
- trigger_irrigation      ₹0.05 / litre
- apply_fertilizer        ₹45 / kg (avg across NPK types)
- send_farmer_alert       ₹0.50 / SMS
- adjust_ph_level         ₹25 / kg
- activate_cooling_system ₹15 / hour

Rules:
- Prioritise critical issues first
- Never over-irrigate (max 500L per trigger)
- If unsure, send_farmer_alert instead of taking irreversible action
- Always consider cost-effectiveness
"""

ACTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "trigger_irrigation",
            "description": "Activate drip/sprinkler irrigation for a field",
            "parameters": {
                "type": "object",
                "properties": {
                    "field_id":              {"type": "string"},
                    "water_amount_liters":   {"type": "number"},
                    "duration_minutes":      {"type": "number"},
                },
                "required": ["field_id", "water_amount_liters"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_fertilizer",
            "description": "Schedule fertiliser application for nutrient deficiency",
            "parameters": {
                "type": "object",
                "properties": {
                    "field_id":        {"type": "string"},
                    "fertilizer_type": {"type": "string", "enum": ["nitrogen", "phosphorus", "potassium", "npk_balanced"]},
                    "amount_kg":       {"type": "number"},
                },
                "required": ["field_id", "fertilizer_type", "amount_kg"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_farmer_alert",
            "description": "Send SMS/WhatsApp alert to the farmer",
            "parameters": {
                "type": "object",
                "properties": {
                    "field_id": {"type": "string"},
                    "message":  {"type": "string"},
                    "urgency":  {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                },
                "required": ["field_id", "message", "urgency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_ph_level",
            "description": "Apply lime (raise pH) or sulphur (lower pH) to correct soil imbalance",
            "parameters": {
                "type": "object",
                "properties": {
                    "field_id":        {"type": "string"},
                    "adjustment_type": {"type": "string", "enum": ["increase_ph_lime", "decrease_ph_sulfur"]},
                    "amount_kg":       {"type": "number"},
                },
                "required": ["field_id", "adjustment_type", "amount_kg"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "activate_cooling_system",
            "description": "Deploy shade nets or misting to reduce heat stress",
            "parameters": {
                "type": "object",
                "properties": {
                    "field_id":       {"type": "string"},
                    "cooling_method": {"type": "string", "enum": ["shade_net", "misting", "both"]},
                    "duration_hours": {"type": "number"},
                },
                "required": ["field_id", "cooling_method"],
            },
        },
    },
]

# Cost lookup (INR per unit)
ACTION_COSTS = {
    "trigger_irrigation":      lambda a: a.get("water_amount_liters", 0) * 0.05,
    "apply_fertilizer":        lambda a: a.get("amount_kg", 0) * 45,
    "send_farmer_alert":       lambda a: 0.50,
    "adjust_ph_level":         lambda a: a.get("amount_kg", 0) * 25,
    "activate_cooling_system": lambda a: a.get("duration_hours", 2) * 15,
}

ACTION_DESCRIPTIONS = {
    "trigger_irrigation":      lambda a: f"Irrigation triggered: {a.get('water_amount_liters')}L for {a.get('duration_minutes', 30)} min",
    "apply_fertilizer":        lambda a: f"Fertiliser scheduled: {a.get('fertilizer_type')} {a.get('amount_kg')}kg",
    "send_farmer_alert":       lambda a: f"Alert sent [{a.get('urgency')}]: {str(a.get('message', ''))[:60]}",
    "adjust_ph_level":         lambda a: f"pH adjustment: {a.get('adjustment_type')} {a.get('amount_kg')}kg",
    "activate_cooling_system": lambda a: f"Cooling activated: {a.get('cooling_method')} for {a.get('duration_hours', 2)}h",
}


class ActionAgent:
    """
    Action Agent — LLM picks which farm actions to execute via tool calling.
    All actions are simulated (logged to DB) for demo purposes.
    """

    def _execute_action(self, name: str, args: dict, db: Session) -> dict:
        cost = round(ACTION_COSTS.get(name, lambda a: 0)(args), 2)
        desc = ACTION_DESCRIPTIONS.get(name, lambda a: name)(args)

        log = RemediationLog(
            alert_id=args.get("alert_id", 0),
            field_id=args.get("field_id", ""),
            action_type=name,
            action_details=json.dumps(args),
            success=True,
            cost_estimate=cost,
        )
        db.add(log)
        db.commit()

        return {"success": True, "action": desc, "cost_inr": cost}

    def plan_and_execute(
        self,
        diagnostic_info: str,
        field_id: str,
        db: Session,
        auto_execute: bool = True,
    ) -> dict:
        start          = time.time()
        actions_taken: list[dict] = []

        query = (
            f"Field {field_id} diagnostic report:\n\n{diagnostic_info}\n\n"
            f"Auto-execution is {'ENABLED' if auto_execute else 'DISABLED'}. "
            "Analyse and execute appropriate remediation actions."
        )
        messages = [{"role": "user", "content": query}]
        client   = get_client()
        model    = FREE_MODELS[0]

        for iteration in range(6):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": ACTION_SYSTEM_PROMPT}] + messages,
                    tools=ACTION_TOOLS,
                    tool_choice="auto",
                    max_tokens=1024,
                    temperature=0.2,
                )
                msg = resp.choices[0].message

                if msg.tool_calls and auto_execute:
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
                        fn_args.setdefault("field_id", field_id)
                        result  = self._execute_action(fn_name, fn_args, db)
                        actions_taken.append({
                            "action":     fn_name,
                            "parameters": fn_args,
                            "result":     result,
                        })
                        messages.append({
                            "role":         "tool",
                            "tool_call_id": tc.id,
                            "content":      json.dumps(result),
                        })
                else:
                    total_cost = sum(a["result"].get("cost_inr", 0) for a in actions_taken)
                    return {
                        "response":       msg.content or "",
                        "actions_taken":  actions_taken,
                        "total_actions":  len(actions_taken),
                        "total_cost_inr": round(total_cost, 2),
                        "execution_time": round(time.time() - start, 2),
                    }

            except Exception as e:
                logger.error(f"ActionAgent error: {e}")
                return {
                    "response":       f"Action error: {e}",
                    "actions_taken":  actions_taken,
                    "execution_time": round(time.time() - start, 2),
                    "error":          True,
                }

        total_cost = sum(a["result"].get("cost_inr", 0) for a in actions_taken)
        return {
            "response":       "Remediation complete.",
            "actions_taken":  actions_taken,
            "total_actions":  len(actions_taken),
            "total_cost_inr": round(total_cost, 2),
            "execution_time": round(time.time() - start, 2),
        }
