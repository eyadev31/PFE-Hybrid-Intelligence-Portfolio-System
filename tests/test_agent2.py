"""
Agent 2 -- Cognitive & Behavioral Profiling Agent Tests
==========================================================
Comprehensive tests covering:
  - Question calibration engine
  - Behavioral consistency analyzer
  - Adaptive risk classifier
  - Dynamic question generator
  - Schema validation
  - Full pipeline integration (mock mode)
"""

import pytest
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ================================================================
#  FIXTURES
# ================================================================

@pytest.fixture
def mock_agent1_output():
    """Realistic Agent 1 output for testing."""
    return {
        "timestamp": "2026-02-24T22:00:00Z",
        "data_freshness": "2026-02-24T21:55:00Z",
        "market_regime": {
            "primary_regime": "bear_high_vol",
            "confidence": 0.82,
            "hmm_regime": "bear_high_vol",
            "rf_regime": "bear_high_vol",
            "models_agree": True,
            "regime_duration_days": 14,
            "transition_probability": 0.30,
            "description": "Bearish regime with elevated volatility",
        },
        "volatility_state": {
            "current_state": "elevated",
            "vix_level": 28.5,
            "realized_vol_percentile": 75.0,
            "vol_trend": "increasing",
            "vol_of_vol": "elevated",
            "term_structure": "backwardation",
        },
        "macro_environment": {
            "macro_regime": "contraction",
            "monetary_policy": "tightening",
            "inflation_state": "above_target",
            "growth_state": "slowing",
            "liquidity": "tight",
            "composite_score": -0.4,
            "key_indicators": {
                "fed_funds_rate": 5.25,
                "treasury_10y": 4.5,
                "unemployment": 4.1,
                "gdp_growth": 0.8,
            },
            "yield_curve": {"inverted": True, "signal": "warning"},
        },
        "systemic_risk": {
            "overall_risk_level": 0.55,
            "risk_category": "elevated",
            "risk_signals": {},
            "risk_assessment": "Elevated risk environment",
            "recommended_caution": True,
        },
        "cross_asset_analysis": {
            "correlation_state": "increasing",
            "median_correlation": 0.55,
            "risk_appetite_index": 0.35,
            "key_correlations": {},
        },
    }


@pytest.fixture
def calm_agent1_output():
    """Calm market Agent 1 output."""
    return {
        "market_regime": {
            "primary_regime": "bull_low_vol",
            "confidence": 0.90,
            "models_agree": True,
            "regime_duration_days": 60,
            "transition_probability": 0.10,
            "description": "Bull market with low volatility",
        },
        "volatility_state": {
            "current_state": "low",
            "vix_level": 14.0,
            "realized_vol_percentile": 20.0,
            "vol_trend": "stable",
        },
        "systemic_risk": {
            "overall_risk_level": 0.10,
            "risk_category": "low",
        },
        "macro_environment": {
            "macro_regime": "stable_growth",
            "key_indicators": {},
            "yield_curve": {"inverted": False},
        },
    }


@pytest.fixture
def sample_questions():
    """Sample generated questions for testing."""
    return [
        {
            "question_id": "q_test_1",
            "category": "loss_aversion",
            "difficulty": 0.7,
            "scenario": "Your portfolio dropped 20% in 2 weeks.",
            "question_text": "What do you do?",
            "choices": [
                {"id": "A", "text": "Sell everything", "risk_signal": 0.05, "behavioral_tag": "panic"},
                {"id": "B", "text": "Reduce exposure 30%", "risk_signal": 0.30, "behavioral_tag": "cautious"},
                {"id": "C", "text": "Hold positions", "risk_signal": 0.60, "behavioral_tag": "neutral"},
                {"id": "D", "text": "Buy the dip", "risk_signal": 0.90, "behavioral_tag": "contrarian"},
            ],
            "behavioral_insight": "Tests panic behavior",
        },
        {
            "question_id": "q_test_2",
            "category": "overconfidence",
            "difficulty": 0.5,
            "scenario": "You beat the market 3 times in a row.",
            "question_text": "How do you adjust your strategy?",
            "choices": [
                {"id": "A", "text": "Stay the course", "risk_signal": 0.20, "behavioral_tag": "disciplined"},
                {"id": "B", "text": "Slightly increase risk", "risk_signal": 0.50, "behavioral_tag": "moderate"},
                {"id": "C", "text": "Significantly increase risk", "risk_signal": 0.75, "behavioral_tag": "overconfident"},
                {"id": "D", "text": "Use leverage", "risk_signal": 0.95, "behavioral_tag": "extreme"},
            ],
            "behavioral_insight": "Tests overconfidence",
        },
        {
            "question_id": "q_test_3",
            "category": "herd_behavior",
            "difficulty": 0.4,
            "scenario": "Everyone is buying crypto.",
            "question_text": "Do you join in?",
            "choices": [
                {"id": "A", "text": "Ignore it", "risk_signal": 0.10, "behavioral_tag": "independent"},
                {"id": "B", "text": "Research first", "risk_signal": 0.40, "behavioral_tag": "analytical"},
                {"id": "C", "text": "Small position", "risk_signal": 0.65, "behavioral_tag": "moderate"},
                {"id": "D", "text": "Go all in", "risk_signal": 0.95, "behavioral_tag": "herd"},
            ],
            "behavioral_insight": "Tests herd following",
        },
        {
            "question_id": "q_test_4",
            "category": "time_pressure",
            "difficulty": 0.8,
            "scenario": "Flash crash, 10 minutes to decide.",
            "question_text": "What's your move?",
            "choices": [
                {"id": "A", "text": "Sell immediately", "risk_signal": 0.05, "behavioral_tag": "flight"},
                {"id": "B", "text": "Sell half", "risk_signal": 0.35, "behavioral_tag": "cautious"},
                {"id": "C", "text": "Do nothing", "risk_signal": 0.55, "behavioral_tag": "freeze"},
                {"id": "D", "text": "Buy more", "risk_signal": 0.90, "behavioral_tag": "fight"},
            ],
            "behavioral_insight": "Tests time pressure decisions",
        },
    ]


@pytest.fixture
def conservative_answers():
    """Conservative investor mock answers."""
    return [
        {"question_id": "q_test_1", "selected_choice_id": "A", "response_time_seconds": 35.0, "changed_answer": False},
        {"question_id": "q_test_2", "selected_choice_id": "A", "response_time_seconds": 28.0, "changed_answer": False},
        {"question_id": "q_test_3", "selected_choice_id": "A", "response_time_seconds": 22.0, "changed_answer": False},
        {"question_id": "q_test_4", "selected_choice_id": "A", "response_time_seconds": 8.0, "changed_answer": True},
    ]


@pytest.fixture
def aggressive_answers():
    """Aggressive investor mock answers."""
    return [
        {"question_id": "q_test_1", "selected_choice_id": "D", "response_time_seconds": 5.0, "changed_answer": False},
        {"question_id": "q_test_2", "selected_choice_id": "D", "response_time_seconds": 3.0, "changed_answer": False},
        {"question_id": "q_test_3", "selected_choice_id": "D", "response_time_seconds": 4.0, "changed_answer": False},
        {"question_id": "q_test_4", "selected_choice_id": "D", "response_time_seconds": 2.0, "changed_answer": False},
    ]


@pytest.fixture
def mixed_answers():
    """Mixed/inconsistent investor answers."""
    return [
        {"question_id": "q_test_1", "selected_choice_id": "A", "response_time_seconds": 40.0, "changed_answer": True},
        {"question_id": "q_test_2", "selected_choice_id": "D", "response_time_seconds": 5.0, "changed_answer": False},
        {"question_id": "q_test_3", "selected_choice_id": "B", "response_time_seconds": 30.0, "changed_answer": True},
        {"question_id": "q_test_4", "selected_choice_id": "D", "response_time_seconds": 3.0, "changed_answer": False},
    ]


# ================================================================
#  QUESTION CALIBRATOR TESTS
# ================================================================

class TestQuestionCalibrator:
    def test_high_stress_calibration(self, mock_agent1_output):
        from ml.question_engine import QuestionCalibrator
        cal = QuestionCalibrator.calibrate(mock_agent1_output)

        assert cal["stress_multiplier"] >= 0.5
        assert cal["regime_used"] == "bear_high_vol"
        assert len(cal["categories"]) == 4

    def test_low_stress_calibration(self, calm_agent1_output):
        from ml.question_engine import QuestionCalibrator
        cal = QuestionCalibrator.calibrate(calm_agent1_output)

        assert cal["stress_multiplier"] <= 0.4
        assert cal["regime_used"] == "bull_low_vol"

    def test_stress_affects_difficulty(self, mock_agent1_output, calm_agent1_output):
        from ml.question_engine import QuestionCalibrator
        high = QuestionCalibrator.calibrate(mock_agent1_output)
        low = QuestionCalibrator.calibrate(calm_agent1_output)

        high_diff_range = high["difficulty_range"]
        low_diff_range = low["difficulty_range"]
        assert high_diff_range[0] >= low_diff_range[0]

    def test_categories_no_duplicates(self, mock_agent1_output):
        from ml.question_engine import QuestionCalibrator
        cal = QuestionCalibrator.calibrate(mock_agent1_output)
        names = [c["name"] for c in cal["categories"]]
        assert len(names) == len(set(names))

    def test_scenario_params(self, mock_agent1_output):
        from ml.question_engine import QuestionCalibrator
        cal = QuestionCalibrator.calibrate(mock_agent1_output)
        params = cal["scenario_params"]

        assert "drawdown_range_pct" in params
        assert "current_regime" in params
        assert params["current_regime"] == "bear_high_vol"


# ================================================================
#  BEHAVIORAL ANALYZER TESTS
# ================================================================

class TestBehavioralAnalyzer:
    def test_conservative_profile(self, sample_questions, conservative_answers):
        from ml.behavioral_analyzer import BehavioralConsistencyAnalyzer
        result = BehavioralConsistencyAnalyzer.analyze(
            sample_questions, conservative_answers, market_stress=0.7
        )

        assert result["consistency_score"] >= 0.5
        assert result["emotional_stability"] in ("stable", "moderate", "volatile", "highly_volatile")
        assert "stress_response_pattern" in result
        assert "decision_speed_profile" in result

    def test_aggressive_profile(self, sample_questions, aggressive_answers):
        from ml.behavioral_analyzer import BehavioralConsistencyAnalyzer
        result = BehavioralConsistencyAnalyzer.analyze(
            sample_questions, aggressive_answers, market_stress=0.3
        )

        assert result["consistency_score"] >= 0.5  # consistent in being aggressive

    def test_mixed_detects_contradictions(self, sample_questions, mixed_answers):
        from ml.behavioral_analyzer import BehavioralConsistencyAnalyzer
        result = BehavioralConsistencyAnalyzer.analyze(
            sample_questions, mixed_answers, market_stress=0.5
        )

        assert len(result["contradiction_flags"]) >= 1
        assert result["consistency_score"] < 0.9

    def test_decision_speed(self, sample_questions, conservative_answers):
        from ml.behavioral_analyzer import BehavioralConsistencyAnalyzer
        result = BehavioralConsistencyAnalyzer.analyze(
            sample_questions, conservative_answers, market_stress=0.5
        )
        assert result["decision_speed_profile"] in ("deliberate", "moderate", "impulsive", "unknown")


# ================================================================
#  RISK CLASSIFIER TESTS
# ================================================================

class TestRiskClassifier:
    def test_conservative_classification(self, sample_questions, conservative_answers):
        from ml.behavioral_analyzer import BehavioralConsistencyAnalyzer
        from ml.risk_classifier import AdaptiveRiskClassifier

        profile = BehavioralConsistencyAnalyzer.analyze(
            sample_questions, conservative_answers, market_stress=0.5
        )
        result = AdaptiveRiskClassifier.classify(
            sample_questions, conservative_answers, profile, market_stress=0.5
        )

        assert result["risk_score"] < 0.35
        assert "conservative" in result["behavioral_type"]
        assert result["liquidity_preference"] == "high"

    def test_aggressive_classification(self, sample_questions, aggressive_answers):
        from ml.behavioral_analyzer import BehavioralConsistencyAnalyzer
        from ml.risk_classifier import AdaptiveRiskClassifier

        profile = BehavioralConsistencyAnalyzer.analyze(
            sample_questions, aggressive_answers, market_stress=0.5
        )
        result = AdaptiveRiskClassifier.classify(
            sample_questions, aggressive_answers, profile, market_stress=0.5
        )

        assert result["risk_score"] > 0.65
        assert "aggressive" in result["behavioral_type"] or "growth" in result["behavioral_type"]

    def test_market_adjustment(self, sample_questions, aggressive_answers):
        from ml.behavioral_analyzer import BehavioralConsistencyAnalyzer
        from ml.risk_classifier import AdaptiveRiskClassifier

        profile = BehavioralConsistencyAnalyzer.analyze(
            sample_questions, aggressive_answers, market_stress=0.2
        )
        result = AdaptiveRiskClassifier.classify(
            sample_questions, aggressive_answers, profile, market_stress=0.2
        )

        assert result["market_adjusted"] is True
        assert result["risk_score_raw"] != result["risk_score"]  # adjustment applied

    def test_max_drawdown_estimation(self, sample_questions, conservative_answers):
        from ml.behavioral_analyzer import BehavioralConsistencyAnalyzer
        from ml.risk_classifier import AdaptiveRiskClassifier

        profile = BehavioralConsistencyAnalyzer.analyze(
            sample_questions, conservative_answers, market_stress=0.5
        )
        result = AdaptiveRiskClassifier.classify(
            sample_questions, conservative_answers, profile, market_stress=0.5
        )

        assert 0.02 <= result["max_acceptable_drawdown"] <= 0.50

    def test_confidence_scoring(self, sample_questions, mixed_answers):
        from ml.behavioral_analyzer import BehavioralConsistencyAnalyzer
        from ml.risk_classifier import AdaptiveRiskClassifier

        profile = BehavioralConsistencyAnalyzer.analyze(
            sample_questions, mixed_answers, market_stress=0.5
        )
        result = AdaptiveRiskClassifier.classify(
            sample_questions, mixed_answers, profile, market_stress=0.5
        )

        assert 0.2 <= result["confidence"] <= 0.98


# ================================================================
#  SCHEMA VALIDATION TESTS
# ================================================================

class TestSchemas:
    def test_question_set_schema(self):
        from schemas.agent2_output import QuestionSetOutput
        data = {
            "session_id": "test_001",
            "timestamp": "2026-02-24T22:00:00Z",
            "questions": [{
                "question_id": "q1",
                "category": "loss_aversion",
                "difficulty": 0.5,
                "scenario": "Test scenario",
                "question_text": "Test question?",
                "choices": [
                    {"id": "A", "text": "Option A", "risk_signal": 0.1},
                    {"id": "B", "text": "Option B", "risk_signal": 0.5},
                    {"id": "C", "text": "Option C", "risk_signal": 0.9},
                ],
            }],
            "market_calibration": {
                "stress_multiplier": 0.5,
            },
        }
        output = QuestionSetOutput(**data)
        assert output.session_id == "test_001"
        assert len(output.questions) == 1

    def test_agent2_output_schema(self):
        from schemas.agent2_output import Agent2Output
        data = {
            "session_id": "test_001",
            "timestamp": "2026-02-24T22:00:00Z",
            "risk_classification": {
                "risk_score": 0.71,
                "risk_score_raw": 0.75,
                "behavioral_type": "growth_seeker_with_volatility_sensitivity",
                "confidence": 0.88,
                "liquidity_preference": "medium",
            },
            "behavioral_profile": {
                "consistency_score": 0.82,
                "emotional_stability": "moderate",
                "emotional_stability_score": 0.65,
            },
            "llm_narrative": {
                "investor_narrative": "Test narrative",
            },
        }
        output = Agent2Output(**data)
        assert output.risk_classification.risk_score == 0.71

    def test_invalid_risk_score_rejected(self):
        from schemas.agent2_output import Agent2Output
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Agent2Output(
                session_id="test",
                timestamp="2026-02-24T22:00:00Z",
                risk_classification={
                    "risk_score": 1.5,  # Invalid: > 1.0
                    "risk_score_raw": 0.75,
                    "behavioral_type": "test",
                    "confidence": 0.5,
                    "liquidity_preference": "medium",
                },
                behavioral_profile={
                    "consistency_score": 0.5,
                    "emotional_stability": "stable",
                    "emotional_stability_score": 0.5,
                },
                llm_narrative={},
            )


# ================================================================
#  QUESTION GENERATOR TESTS
# ================================================================

class TestQuestionGenerator:
    def test_fallback_bank(self, mock_agent1_output):
        from ml.question_engine import QuestionCalibrator
        from llm.question_generator import DynamicQuestionGenerator

        cal = QuestionCalibrator.calibrate(mock_agent1_output)
        gen = DynamicQuestionGenerator()
        questions = gen._generate_fallback(cal, 4, "test_session")

        assert len(questions) == 4
        for q in questions:
            assert "question_id" in q
            assert "choices" in q
            assert len(q["choices"]) >= 3


# ================================================================
#  FULL PIPELINE INTEGRATION
# ================================================================

class TestAgent2Pipeline:
    def test_mock_pipeline_runs(self):
        from agents.agent2_daq import Agent2BehavioralIntelligence
        agent = Agent2BehavioralIntelligence()
        result = agent.run_mock()

        assert "phase1_questions" in result
        assert "phase2_profile" in result

    def test_mock_produces_valid_questions(self):
        from agents.agent2_daq import Agent2BehavioralIntelligence
        agent = Agent2BehavioralIntelligence()
        result = agent.run_mock()

        questions = result["phase1_questions"]["questions"]
        assert len(questions) >= 1
        for q in questions:
            assert "question_id" in q
            assert "choices" in q

    def test_mock_produces_risk_score(self):
        from agents.agent2_daq import Agent2BehavioralIntelligence
        agent = Agent2BehavioralIntelligence()
        result = agent.run_mock()

        profile = result["phase2_profile"]
        risk = profile["risk_classification"]
        assert 0.0 <= risk["risk_score"] <= 1.0
        assert risk["behavioral_type"] != ""
        assert 0.0 <= risk["confidence"] <= 1.0

    def test_mock_produces_behavioral_profile(self):
        from agents.agent2_daq import Agent2BehavioralIntelligence
        agent = Agent2BehavioralIntelligence()
        result = agent.run_mock()

        profile = result["phase2_profile"]["behavioral_profile"]
        assert 0.0 <= profile["consistency_score"] <= 1.0
        assert profile["emotional_stability"] in ("stable", "moderate", "volatile", "highly_volatile")

    def test_pipeline_with_custom_agent1(self, mock_agent1_output):
        from agents.agent2_daq import Agent2BehavioralIntelligence
        agent = Agent2BehavioralIntelligence()
        result = agent.run_mock(agent1_output=mock_agent1_output)

        assert result["phase2_profile"]["market_regime_at_assessment"] == "bear_high_vol"
