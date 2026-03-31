"""
AgriFarm — Orchestrator Agent
YOUR original two-phase orchestrator (Diagnostic → Action) extended with
Disease, Yield, and Market agents. Intent classification routes to the
right specialist automatically.
"""

from __future__ import annotations
import json
import time
from typing import Optional

from sqlalchemy.orm import Session
from loguru import logger

from app.database import AgentLog
from app.agents.diagnostic_agent import DiagnosticAgent
from app.agents.action_agent import ActionAgent
from app.llm_client import quick_ask, detailed_ask

# Trigger keywords for specialist routing
_DISEASE_KW   = {"disease","blight","rust","mold","mildew","spot","wilt","leaf","infected","virus","fungus","pest","scan","photo","image","upload"}
_YIELD_KW     = {"yield","forecast","harvest","production","tonnes","predict","crop output"}
_MARKET_KW    = {"price","market","sell","mandi","apmc","revenue","profit","per quintal","commodity"}
_IRRIGATION_KW= {"irrigat","water","moisture","drip","sprinkler","rainfall","drought"}


def _classify(query: str) -> str:
    q = query.lower()
    if any(k in q for k in _DISEASE_KW):    return "disease"
    if any(k in q for k in _YIELD_KW):      return "yield"
    if any(k in q for k in _MARKET_KW):     return "market"
    # Fall through → sensor diagnostic
    return "diagnostic"


class OrchestratorAgent:
    """
    Phase 1 — Classify intent
    Phase 2 — Route to specialist agent (Disease / Yield / Market)
              OR run your original Diagnostic → Action pipeline
    Phase 3 — Log and return comprehensive response
    """

    def __init__(self):
        self.diagnostic = DiagnosticAgent()
        self.action     = ActionAgent()

    # ── Public entry point ────────────────────────────────────
    def process_query(
        self,
        query:          str,
        db:             Session,
        field_id:       Optional[str] = None,
        auto_remediate: bool          = True,
        farm_context:   dict          = {},
    ) -> dict:
        start  = time.time()
        intent = _classify(query)
        logger.info(f"Orchestrator → intent={intent} field={field_id}")

        # ── Disease detection (needs image bytes in farm_context) ─
        if intent == "disease":
            result = self._run_disease(query, farm_context, db)

        # ── Yield forecasting ─────────────────────────────────
        elif intent == "yield":
            result = self._run_yield(query, farm_context)

        # ── Market advisory ───────────────────────────────────
        elif intent == "market":
            result = self._run_market(query, farm_context)

        # ── Sensor diagnostic + remediation (YOUR original flow) ─
        else:
            result = self._run_diagnostic_action(query, db, field_id, auto_remediate)

        total = round(time.time() - start, 2)
        result["total_execution_time"] = total
        result["intent"]               = intent
        result["field_id"]             = field_id

        # Log to DB
        self._log(db, intent, query, json.dumps(result, default=str), [], total)
        return result

    # ── Your original 2-phase pipeline ───────────────────────
    def _run_diagnostic_action(
        self, query: str, db: Session,
        field_id: Optional[str], auto_remediate: bool,
    ) -> dict:
        # Phase 1: Diagnostic
        diag = self.diagnostic.diagnose(query, db, field_id)
        self._log(db, "diagnostic", query, diag.get("response",""), diag.get("tools_used",[]), diag.get("execution_time",0))

        # Phase 2: Action (if issues detected)
        action_result = None
        diag_text     = diag.get("response", "").lower()
        needs_action  = any(w in diag_text for w in [
            "deficient","low","high","critical","urgent","risk","issue",
            "problem","excess","insufficient","needed","alert",
        ])

        if needs_action and auto_remediate and field_id:
            action_result = self.action.plan_and_execute(
                diagnostic_info=diag.get("response",""),
                field_id=field_id,
                db=db,
                auto_execute=True,
            )
            self._log(db, "action", f"Remediation for {field_id}",
                      action_result.get("response",""),
                      [a["action"] for a in action_result.get("actions_taken",[])],
                      action_result.get("execution_time",0))

        return {
            "orchestrator_summary": self._summary(diag, action_result),
            "diagnostic_phase":     {
                "response":       diag.get("response",""),
                "tools_used":     diag.get("tools_used",[]),
                "execution_time": diag.get("execution_time",0),
                "iterations":     diag.get("iterations",0),
            },
            "action_phase": action_result or {"message": "No remediation needed"},
        }

    def _run_disease(self, query: str, ctx: dict, db: Session) -> dict:
        from app.agents.disease_agent import run as disease_run
        image_bytes = ctx.get("image_bytes")
        if image_bytes:
            import base64
            if isinstance(image_bytes, str):
                image_bytes = base64.b64decode(image_bytes)
            r = disease_run(image_bytes, farm_context=ctx)
        else:
            # Text-only disease question
            answer = detailed_ask(
                query,
                system="You are a plant pathologist for Indian farms. Answer this crop disease question.",
            )
            r = {"llm_advice": answer, "agent": "disease_agent"}
        return {"orchestrator_summary": r.get("llm_advice",""), **r}

    def _run_yield(self, query: str, ctx: dict) -> dict:
        from app.agents.yield_market_agent import run_yield
        crop  = ctx.get("crop", "Tomato")
        state = ctx.get("state", "Tamil Nadu")
        r     = run_yield(crop, state)
        return {"orchestrator_summary": r.get("narrative",""), **r}

    def _run_market(self, query: str, ctx: dict) -> dict:
        from app.agents.yield_market_agent import run_market
        crop = ctx.get("crop", "Tomato")
        qty  = float(ctx.get("quantity_quintals", 10))
        r    = run_market(crop, qty)
        return {"orchestrator_summary": r.get("advisory",""), **r}

    def _summary(self, diag: dict, action: Optional[dict]) -> str:
        s = f"**Diagnostic Summary:**\n{diag.get('response','')[:300]}…\n\n"
        if action:
            s += f"**Actions Executed:** {action.get('total_actions',0)} | "
            s += f"**Cost:** ₹{action.get('total_cost_inr',0)}\n"
            for a in action.get("actions_taken",[]):
                s += f"  • {a['action']}: {a['result'].get('action','')}\n"
        return s

    def _log(self, db: Session, agent_type: str, query: str,
             response: str, tools: list, exec_time: float):
        try:
            db.add(AgentLog(
                agent_type=agent_type, query=query[:500],
                response=response[:1000],
                tools_used=json.dumps(tools),
                execution_time=exec_time,
            ))
            db.commit()
        except Exception as e:
            logger.warning(f"Log write failed: {e}")

    def get_field_recommendations(self, field_id: str, db: Session) -> dict:
        return self.process_query(
            query=f"Provide a comprehensive health assessment for field {field_id}. "
                  "Include soil health, irrigation status, pest risks, and nutrient levels.",
            db=db, field_id=field_id, auto_remediate=False,
        )
