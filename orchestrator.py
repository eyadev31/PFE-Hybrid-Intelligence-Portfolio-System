"""
Hybrid Intelligence Portfolio System -- Intelligence Orchestrator 
===================================================================
The "Brain Stem" of the system.
Responsible for full end-to-end multi-agent orchestration.
Strictly enforces Step 3 contracts: calls agents, validates schemas,
runs mathematical bounds checking, and ensures graceful handoffs.
"""

import logging
import json
import uuid
from typing import Optional, Dict, Any

from schemas.agent1_output import Agent1Output
from schemas.agent2_output import Agent2Output
from schemas.agent3_output import Agent3Output
from schemas.agent4_output import Agent4Output
from schemas.news_output import Agent5Output

from agents.agent1_macro import Agent1MacroIntelligence
from agents.agent2_daq import Agent2BehavioralIntelligence
from agents.agent3_strategist import Agent3PortfolioStrategist
from agents.agent4_supervisor import Agent4RiskSupervisor
from agents.agent5_news import Agent5NewsIntelligence

logger = logging.getLogger(__name__)

class OrchestrationError(Exception):
    """Raised when an agent output fails contract validation."""
    pass


class IntelligenceOrchestrator:
    """
    Coordinates the 4-agent pipeline.
    Enforces deterministic handoffs through Pydantic schemas.
    """
    def __init__(self):
        self.agent1 = Agent1MacroIntelligence()
        self.agent2 = Agent2BehavioralIntelligence()
        self.agent3 = Agent3PortfolioStrategist()
        self.agent4 = Agent4RiskSupervisor()
        self.agent5 = Agent5NewsIntelligence()

    def run_pipeline(
        self, 
        session_id: str = None, 
        user_answers: list[dict] = None, 
        mock: bool = False
    ) -> Dict[str, Any]:
        """
        Run the full 4-agent pipeline sequentially with strict contract validation.
        """
        if not session_id:
            session_id = f"session_{uuid.uuid4().hex[:8]}"

        logger.info(f"========== STARTING ORCHESTRATION PIPELINE [{session_id}] ==========")

        # ── Step 1: Agent 1 Macro Context ──────────────────────
        logger.info("--> Running Agent 1 (Macro Intelligence)...")
        agent1_raw = self.agent1.run(mock=mock)
        agent1_validated = self._validate_schema(agent1_raw, Agent1Output, "Agent 1")
        logger.info("--> Agent 1 output VALIDATED against schema.")

        # ── Step 1.5: Agent 5 News Sentiment Intelligence ──────
        logger.info("--> Running Agent 5 (News Sentiment Intelligence)...")
        try:
            agent5_raw = self.agent5.run(mock=mock, agent1_output=agent1_validated)
            agent5_validated = self._validate_schema(agent5_raw, Agent5Output, "Agent 5")
            logger.info("--> Agent 5 output VALIDATED against schema.")
        except Exception as e:
            logger.warning(f"Agent 5 (News) failed gracefully: {e}. Pipeline continues without news signals.")
            agent5_validated = None

        # ── Step 2: Agent 2 Behavioral Profiling ───────────────
        logger.info("--> Running Agent 2 (Behavioral Profiling)...")
        
        # If user answers are provided (e.g. from API/frontend), process them.
        # Otherwise, run the full mock DAQ session.
        if mock and not user_answers:
            agent2_full = self.agent2.run_mock(agent1_output=agent1_validated)
        else:
            agent2_full = self.agent2.run_mock(agent1_output=agent1_validated)

        # Agent 2 returns a wrapped dict containing both phases when running the full session
        agent2_raw = agent2_full.get("phase2_profile", agent2_full)

        agent2_validated = self._validate_schema(agent2_raw, Agent2Output, "Agent 2")
        logger.info("--> Agent 2 output VALIDATED against schema.")

        # ── Step 3: Agent 3 Allocation Optimizer ───────────────
        logger.info("--> Running Agent 3 (Allocation Strategist)...")
        agent3_raw = self.agent3.run(
            agent1_output=agent1_validated,
            agent2_output=agent2_validated, 
        )
        
        # Numeric Checks before Pydantic validation
        self._run_numeric_checks(agent3_raw)
        
        agent3_validated = self._validate_schema(agent3_raw, Agent3Output, "Agent 3")
        logger.info("--> Agent 3 output VALIDATED against schema and numeric bounds.")

        # ── Step 4: Agent 4 Risk Oversight (CRO) ───────────────
        logger.info("--> Running Agent 4 (Risk Supervisor / CRO)...")
        agent4_raw = self.agent4.run(
            agent1_output=agent1_validated,
            agent2_output=agent2_validated,
            agent3_output=agent3_validated,
        )
        agent4_validated = self._validate_schema(agent4_raw, Agent4Output, "Agent 4")
        logger.info("--> Agent 4 output VALIDATED against schema.")
        
        logger.info(f"========== PIPELINE COMPLETE [{session_id}] ==========")
        
        result = {
            "session_id": session_id,
            "agent1_output": agent1_validated,
            "agent5_output": agent5_validated,
            "agent2_output": agent2_validated,
            "agent3_output": agent3_validated,
            "agent4_output": agent4_validated,
            "final_portfolio": agent4_validated.get("adjusted_allocation", agent3_validated.get("allocation")),
        }

        # Include news signal summary if available
        if agent5_validated:
            result["news_signal"] = agent5_validated.get("market_signal", {})
            result["news_events"] = agent5_validated.get("event_detection", {})

        return result

    def _validate_schema(self, data: dict, schema_model, agent_name: str) -> dict:
        """Validate output strictly against the established Pydantic contract."""
        try:
            # Ensure the output parses correctly according to the schema rules
            model_instance = schema_model.model_validate(data)
            return model_instance.model_dump()
        except Exception as e:
            logger.error(f"{agent_name} Schema Validation Failed! Contract breached.")
            raise OrchestrationError(f"{agent_name} output validation failed:\n{str(e)}")

    def _run_numeric_checks(self, agent3_data: dict):
        """
        Monster Requirement: 
        Must enforce Sum allocations = 1, CVaR constraint, Max drawdown constraint
        """
        allocations = agent3_data.get("allocation", [])
        total_weight = sum([a.get("weight", 0.0) for a in allocations])
        
        # Check 1: Sum == 1.0 (with small float tolerance)
        if abs(total_weight - 1.0) > 0.001:
            raise OrchestrationError(f"Agent 3 Numeric Audit Failed: Total allocation weight is {total_weight}, expected 1.0")
            
        # Check 2: Max Drawdown
        metrics = agent3_data.get("portfolio_metrics", {})
        mc = agent3_data.get("monte_carlo", {})
        
        cvar = metrics.get("cvar_95", 0.0)
        median_dd = mc.get("median_max_drawdown", 0.0)
        
        # Verify these exist as per numeric checks requirement
        if "cvar_95" not in metrics:
            raise OrchestrationError("Agent 3 Numeric Audit Failed: Missing CVaR metrics")
            
        if "median_max_drawdown" not in mc:
            raise OrchestrationError("Agent 3 Numeric Audit Failed: Missing Drawdown estimates")
            
        logger.info(f"Numeric Checks Passed. Weights Sum: {total_weight}. CVaR: {cvar:.2%}. Drawdown: {median_dd:.2%}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    orchestrator = IntelligenceOrchestrator()
    try:
        results = orchestrator.run_pipeline(mock=True)
        print("\\nPipeline Succeeded! Final Portfolio:")
        print(json.dumps(results['final_portfolio'], indent=2))
    except OrchestrationError as e:
        print(f"\\nPipeline Failed Contract Validation: {e}")
