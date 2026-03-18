"""
Hybrid Intelligence Portfolio System — Agent 5 Test Suite
===========================================================
Comprehensive tests for Agent 5 (News Sentiment Intelligence Agent).

Tests cover:
  1. Mock pipeline end-to-end
  2. Schema validation
  3. News collector (mock mode)
  4. News processor (cleaning, entity extraction, topic classification)
  5. Sentiment engine (rule-based fallback)
  6. Impact scorer (multi-factor formula)
  7. Event detector (keyword matching, severity)
  8. Temporal aggregator (momentum computation)
  9. Market signal generation
"""

import pytest
import json
from datetime import datetime, timezone, timedelta


# ═══════════════════════════════════════════════════════
#  TEST 1: FULL MOCK PIPELINE
# ═══════════════════════════════════════════════════════

class TestAgent5Pipeline:
    """End-to-end mock pipeline tests."""

    def test_mock_pipeline_runs_successfully(self):
        """The full Agent 5 mock pipeline must run without errors."""
        from agents.agent5_news import Agent5NewsIntelligence

        agent = Agent5NewsIntelligence()
        output = agent.run_mock()

        assert output is not None
        assert isinstance(output, dict)
        assert "timestamp" in output
        assert "articles" in output
        assert "temporal_sentiment" in output
        assert "event_detection" in output
        assert "market_signal" in output
        assert "agent_metadata" in output

    def test_mock_pipeline_with_agent1_context(self):
        """Pipeline should work with Agent 1 context provided."""
        from agents.agent5_news import Agent5NewsIntelligence

        mock_agent1 = {
            "market_regime": {"primary_regime": "bull_low_vol", "confidence": 0.85},
            "systemic_risk": {"risk_category": "low"},
        }

        agent = Agent5NewsIntelligence()
        output = agent.run(mock=True, agent1_output=mock_agent1)

        assert output is not None
        assert output["market_signal"]["signal_type"] in ("bullish", "bearish", "neutral")

    def test_output_has_articles(self):
        """Mock pipeline must produce processed articles."""
        from agents.agent5_news import Agent5NewsIntelligence

        agent = Agent5NewsIntelligence()
        output = agent.run_mock()

        assert output["article_count"] > 0
        assert len(output["articles"]) > 0

    def test_articles_have_required_fields(self):
        """Each article must have all required fields."""
        from agents.agent5_news import Agent5NewsIntelligence

        agent = Agent5NewsIntelligence()
        output = agent.run_mock()

        for article in output["articles"]:
            assert "article_id" in article
            assert "title" in article
            assert "source" in article
            assert "sentiment" in article
            assert "impact" in article


# ═══════════════════════════════════════════════════════
#  TEST 2: SCHEMA VALIDATION
# ═══════════════════════════════════════════════════════

class TestAgent5Schema:
    """Pydantic schema validation tests."""

    def test_output_validates_against_schema(self):
        """Complete output must pass Agent5Output validation."""
        from agents.agent5_news import Agent5NewsIntelligence
        from schemas.news_output import Agent5Output

        agent = Agent5NewsIntelligence()
        output = agent.run_mock()

        # This should not raise
        validated = Agent5Output.model_validate(output)
        assert validated.timestamp != ""
        assert validated.market_signal.signal_type in ("bullish", "bearish", "neutral")

    def test_validate_news_output_function(self):
        """The validate_news_output helper must work."""
        from agents.agent5_news import Agent5NewsIntelligence
        from schemas.news_output import validate_news_output

        agent = Agent5NewsIntelligence()
        output = agent.run_mock()

        is_valid, error = validate_news_output(output)
        assert is_valid, f"Validation failed: {error}"

    def test_invalid_output_fails_validation(self):
        """An incomplete output must fail validation."""
        from schemas.news_output import validate_news_output

        bad_output = {"timestamp": "2026-01-01", "articles": []}
        is_valid, error = validate_news_output(bad_output)
        assert not is_valid


# ═══════════════════════════════════════════════════════
#  TEST 3: NEWS COLLECTOR
# ═══════════════════════════════════════════════════════

class TestNewsCollector:
    """News collector tests."""

    def test_mock_collection(self):
        """Mock collector must return articles."""
        from ml.news_collector import NewsCollector

        collector = NewsCollector()
        articles = collector.collect_mock()

        assert len(articles) > 0
        for article in articles:
            assert "title" in article
            assert "source" in article
            assert "published_at" in article

    def test_deduplication(self):
        """Collector must deduplicate articles by title."""
        from ml.news_collector import NewsCollector

        collector = NewsCollector()
        articles = [
            {"title": "Bitcoin rises 10%", "source": "reuters", "published_at": "2026-01-01"},
            {"title": "Bitcoin rises 10%", "source": "cnbc", "published_at": "2026-01-01"},
            {"title": "Gold falls sharply", "source": "reuters", "published_at": "2026-01-01"},
        ]
        unique = collector._deduplicate(articles)
        assert len(unique) == 2

    def test_stats_populated(self):
        """Collection stats must be populated after mock collection."""
        from ml.news_collector import NewsCollector

        collector = NewsCollector()
        collector.collect_mock()
        stats = collector.stats

        assert stats["total_raw"] > 0
        assert "sources_queried" in stats


# ═══════════════════════════════════════════════════════
#  TEST 4: NEWS PROCESSOR
# ═══════════════════════════════════════════════════════

class TestNewsProcessor:
    """News processor tests."""

    def test_text_cleaning(self):
        """HTML and boilerplate must be removed."""
        from ml.news_processor import NewsProcessor

        processor = NewsProcessor()
        dirty = "<p>Bitcoin <b>surges</b> 10%. Click here to read more</p>"
        clean = processor._clean_text(dirty)

        assert "<p>" not in clean
        assert "<b>" not in clean
        assert "Click here to read more" not in clean
        assert "Bitcoin" in clean

    def test_relevance_scoring(self):
        """Financial text should score higher than non-financial."""
        from ml.news_processor import NewsProcessor

        processor = NewsProcessor()
        high_relevance = "Federal Reserve cuts interest rates, stocks rally, Bitcoin surges"
        low_relevance = "Local weather forecast shows rain tomorrow"

        high_score = processor._score_relevance(high_relevance)
        low_score = processor._score_relevance(low_relevance)

        assert high_score > low_score

    def test_entity_extraction(self):
        """Financial entities must be detected."""
        from ml.news_processor import NewsProcessor

        processor = NewsProcessor()
        text = "Bitcoin surges as Federal Reserve cuts rates, Gold declines"
        entities, asset_classes, organizations = processor._extract_entities(text)

        assert "Bitcoin" in entities
        assert "Gold" in entities
        assert "Federal Reserve" in entities or "Federal Reserve" in organizations

    def test_topic_classification(self):
        """Topics must be correctly classified."""
        from ml.news_processor import NewsProcessor

        assert NewsProcessor._classify_topic("Federal Reserve cuts interest rates") in ("interest_rates", "monetary_policy")
        assert NewsProcessor._classify_topic("Bitcoin ETF approved by SEC") in ("crypto", "regulation", "etf_approval")
        assert NewsProcessor._classify_topic("Oil prices spike on Middle East tensions") in ("commodities", "geopolitical")

    def test_batch_processing(self):
        """Batch processing must produce output articles."""
        from ml.news_collector import NewsCollector
        from ml.news_processor import NewsProcessor

        collector = NewsCollector()
        processor = NewsProcessor()

        raw = collector.collect_mock()
        processed = processor.process_batch(raw)

        assert len(processed) > 0
        for article in processed:
            assert "article_id" in article
            assert "relevance_score" in article
            assert "entities" in article
            assert "topic" in article


# ═══════════════════════════════════════════════════════
#  TEST 5: SENTIMENT ENGINE
# ═══════════════════════════════════════════════════════

class TestSentimentEngine:
    """Sentiment engine tests (rule-based fallback)."""

    def test_rule_based_bullish(self):
        """Bullish text should produce positive sentiment."""
        from ml.sentiment_engine import SentimentEngine

        engine = SentimentEngine()
        result = engine._rule_based_sentiment("Stocks surge to record high, rally continues")

        assert result["label"] == "positive"
        assert result["score"] > 0

    def test_rule_based_bearish(self):
        """Bearish text should produce negative sentiment."""
        from ml.sentiment_engine import SentimentEngine

        engine = SentimentEngine()
        result = engine._rule_based_sentiment("Market crash, stocks plunge amid crisis fears")

        assert result["label"] == "negative"
        assert result["score"] < 0

    def test_neutral_result(self):
        """Empty text should produce neutral sentiment."""
        from ml.sentiment_engine import SentimentEngine

        engine = SentimentEngine()
        result = engine._neutral_result()

        assert result["label"] == "neutral"
        assert result["score"] == 0.0

    def test_batch_analysis(self):
        """Batch analysis must process all articles."""
        from ml.sentiment_engine import SentimentEngine

        engine = SentimentEngine()
        articles = [
            {"combined_text": "Bitcoin rallies sharply", "title": "test"},
            {"combined_text": "Stocks crash in crisis", "title": "test2"},
        ]
        results = engine.analyze_batch(articles)

        assert len(results) == 2
        for r in results:
            assert "sentiment" in r
            assert r["sentiment"]["label"] in ("positive", "neutral", "negative")


# ═══════════════════════════════════════════════════════
#  TEST 6: IMPACT SCORER
# ═══════════════════════════════════════════════════════

class TestImpactScorer:
    """Impact scoring tests."""

    def test_reuters_higher_than_blog(self):
        """Reuters (tier 1) should score higher than unknown blog."""
        from ml.impact_scorer import ImpactScorer

        scorer = ImpactScorer()
        reuters_weight = scorer._get_source_weight("reuters")
        blog_weight = scorer._get_source_weight("random_blog")

        assert reuters_weight > blog_weight

    def test_recent_article_higher_impact(self):
        """Recent articles should have higher recency score."""
        from ml.impact_scorer import ImpactScorer

        scorer = ImpactScorer()
        now = datetime.now(timezone.utc)
        recent = scorer._calculate_recency(now.isoformat())
        old = scorer._calculate_recency((now - timedelta(days=5)).isoformat())

        assert recent > old

    def test_cluster_boost(self):
        """Clustered articles should get boost."""
        from ml.impact_scorer import ImpactScorer

        assert ImpactScorer._calculate_cluster_boost(1) == 1.0
        assert ImpactScorer._calculate_cluster_boost(3) > 1.0
        assert ImpactScorer._calculate_cluster_boost(5) > ImpactScorer._calculate_cluster_boost(2)

    def test_batch_scoring(self):
        """Batch scoring must produce impact for all articles."""
        from ml.impact_scorer import ImpactScorer

        scorer = ImpactScorer()
        articles = [
            {
                "article_id": "test1",
                "source": "reuters",
                "sentiment": {"score": 0.8, "confidence": 0.9},
                "topic": "interest_rates",
                "entities": ["Federal Reserve"],
                "asset_classes": ["bonds", "equities"],
                "published_at": datetime.now(timezone.utc).isoformat(),
            },
        ]
        results = scorer.score_batch(articles)

        assert len(results) == 1
        assert "impact" in results[0]
        assert 0 <= results[0]["impact"]["impact_score"] <= 1


# ═══════════════════════════════════════════════════════
#  TEST 7: EVENT DETECTOR
# ═══════════════════════════════════════════════════════

class TestEventDetector:
    """Event detection tests."""

    def test_fed_event_detection(self):
        """Fed-related articles should trigger fed_meeting event."""
        from ml.event_detector import EventDetector

        detector = EventDetector()
        articles = [{
            "title": "Federal Reserve FOMC rate decision: rates held steady",
            "combined_text": "The Federal Reserve FOMC kept interest rates unchanged at the latest rate decision meeting.",
            "sentiment": {"score": 0.1, "label": "neutral"},
            "source": "reuters",
        }]
        result = detector.detect_events(articles)

        assert result["event_count"] > 0
        event_types = [e["type"] for e in result["events_detected"]]
        assert "fed_meeting" in event_types

    def test_black_swan_detection(self):
        """Crisis keywords should trigger black_swan event."""
        from ml.event_detector import EventDetector

        detector = EventDetector()
        articles = [{
            "title": "Major bank collapse triggers market crash, contagion fears spread",
            "combined_text": "A major bank collapse has triggered a market crash with contagion fears spreading to other institutions.",
            "sentiment": {"score": -0.9, "label": "negative"},
            "source": "reuters",
        }]
        result = detector.detect_events(articles)

        assert result["event_count"] > 0
        assert result["has_critical_event"] or result["highest_severity"] in ("major", "critical", "black_swan")

    def test_no_events_in_generic_news(self):
        """Generic news should not trigger events."""
        from ml.event_detector import EventDetector

        detector = EventDetector()
        articles = [{
            "title": "Company announces new product line",
            "combined_text": "A technology company announced a new product line at its annual conference.",
            "sentiment": {"score": 0.3, "label": "positive"},
            "source": "cnbc",
        }]
        result = detector.detect_events(articles)

        assert result["event_count"] == 0


# ═══════════════════════════════════════════════════════
#  TEST 8: TEMPORAL AGGREGATOR
# ═══════════════════════════════════════════════════════

class TestTemporalAggregator:
    """Temporal aggregation tests."""

    def test_momentum_computation(self):
        """Momentum should be computed from temporal data."""
        from ml.temporal_aggregator import TemporalAggregator

        overall = {"1h": 0.5, "6h": 0.3, "24h": 0.1, "3d": 0.0, "7d": -0.1}
        direction, strength = TemporalAggregator._compute_momentum(overall)

        assert direction in ("accelerating_bullish", "bullish", "stable", "bearish", "accelerating_bearish")
        assert -1.0 <= strength <= 1.0

    def test_regime_classification(self):
        """Regime should be classified based on sentiment."""
        from ml.temporal_aggregator import TemporalAggregator

        bullish = {"1h": 0.6, "6h": 0.5, "24h": 0.4, "3d": 0.3, "7d": 0.2}
        bearish = {"1h": -0.6, "6h": -0.5, "24h": -0.4, "3d": -0.3, "7d": -0.2}

        assert TemporalAggregator._classify_regime(bullish) in ("bullish", "very_bullish")
        assert TemporalAggregator._classify_regime(bearish) in ("bearish", "very_bearish")


# ═══════════════════════════════════════════════════════
#  TEST 9: MARKET SIGNAL
# ═══════════════════════════════════════════════════════

class TestMarketSignal:
    """Market signal generation tests."""

    def test_signal_has_required_fields(self):
        """Market signal must have all required fields."""
        from agents.agent5_news import Agent5NewsIntelligence

        agent = Agent5NewsIntelligence()
        output = agent.run_mock()
        signal = output["market_signal"]

        assert "signal_type" in signal
        assert "confidence" in signal
        assert "affected_assets" in signal
        assert "signal_strength" in signal
        assert "recommended_bias" in signal
        assert signal["signal_type"] in ("bullish", "bearish", "neutral")
        assert 0 <= signal["confidence"] <= 1

    def test_metadata_populated(self):
        """Agent metadata must be fully populated."""
        from agents.agent5_news import Agent5NewsIntelligence

        agent = Agent5NewsIntelligence()
        output = agent.run_mock()
        meta = output["agent_metadata"]

        assert meta["agent_id"] == "agent5_news_intelligence"
        assert meta["execution_time_ms"] > 0
        assert meta["articles_collected"] > 0
        assert meta["data_quality"] == "mock"
        assert len(meta["models_used"]) > 0
