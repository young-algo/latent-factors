"""
Unit tests for factor_naming module.

Tests cover:
- FactorName dataclass creation and validation
- Name cleaning and normalization
- Quality scoring
- Tag detection
- Prompt generation
- Response parsing
- Quality report generation
"""

import pytest
from src import (
    FactorName,
    validate_name,
    score_quality,
    detect_tags,
    generate_name_prompt,
    parse_name_response,
    generate_quality_report,
    BANNED_WORDS
)


class TestFactorName:
    """Test suite for FactorName dataclass."""

    def test_basic_creation(self):
        """Test basic FactorName creation."""
        name = FactorName(
            short_name="Regional Banks vs Tech",
            description="Traditional lenders vs technology companies",
            theme="Sector Rotation",
            confidence="high"
        )

        assert name.short_name == "Regional Banks vs Tech"
        assert name.confidence == "high"
        assert name.quality_score == 0.0

    def test_markdown_cleaning(self):
        """Test that markdown is removed from names."""
        name = FactorName(
            short_name="**Factor Name:** Regional Banks",
            description="**Description:** Traditional lenders"
        )

        assert "**" not in name.short_name
        assert "**" not in name.description
        assert "Factor Name:" not in name.short_name
        assert name.short_name == "Regional Banks"

    def test_confidence_normalization(self):
        """Test confidence level normalization."""
        name1 = FactorName(short_name="Test", confidence="HIGH")
        assert name1.confidence == "high"

        name2 = FactorName(short_name="Test", confidence="invalid")
        assert name2.confidence == "medium"

    def test_display_label_truncation(self):
        """Test display label truncation."""
        long_name = "A" * 50
        name = FactorName(short_name=long_name)

        label = name.display_label()
        assert len(label) == 30

    def test_to_dict_roundtrip(self):
        """Test dictionary serialization roundtrip."""
        original = FactorName(
            short_name="Test Name",
            description="Test description",
            confidence="high",
            quality_score=85.0
        )

        data = original.to_dict()
        restored = FactorName.from_dict(data)

        assert restored.short_name == original.short_name
        assert restored.quality_score == original.quality_score


class TestValidation:
    """Test suite for name validation."""

    def test_valid_name(self):
        """Test validation passes for good name."""
        name = FactorName(
            short_name="Regional Banks vs Tech",
            description="Traditional lenders vs growth companies"
        )

        issues = validate_name(name)
        assert len(issues) == 0

    def test_short_name_too_long(self):
        """Test validation catches long short_name."""
        name = FactorName(
            short_name="A" * 31
        )

        issues = validate_name(name)
        assert any("too long" in i for i in issues)

    def test_short_name_too_short(self):
        """Test validation catches short short_name."""
        name = FactorName(
            short_name="Hi"
        )

        issues = validate_name(name)
        assert any("too short" in i for i in issues)

    def test_banned_words(self):
        """Test validation catches banned words."""
        for banned in ["beta", "factor", "dynamics"]:
            name = FactorName(short_name=f"Test {banned} name")
            issues = validate_name(name)
            assert any(banned in i for i in issues)

    def test_markdown_cleaned_in_init(self):
        """Test markdown is cleaned during initialization."""
        name = FactorName(
            short_name="Test **bold** name",
            description="Some **bold** text"
        )

        # Markdown should be cleaned in __post_init__, not flagged
        assert "**" not in name.short_name
        assert "**" not in name.description

    def test_similar_high_low_descriptions(self):
        """Test validation catches similar high/low descriptions."""
        name = FactorName(
            short_name="Test Name",
            high_exposure_desc="These are growth stocks",
            low_exposure_desc="These are growth stocks"
        )

        issues = validate_name(name)
        assert any("similar" in i for i in issues)


class TestQualityScoring:
    """Test suite for quality scoring."""

    def test_ideal_name_scores_high(self):
        """Test ideal name gets high score."""
        name = FactorName(
            short_name="Regional Banks vs Tech",
            description="Traditional lenders vs growth companies",
            confidence="high",
            theme="Sector Rotation",
            high_exposure_desc="Banks",
            low_exposure_desc="Tech"
        )

        score = score_quality(name)
        assert score > 70

    def test_generic_name_scores_low(self):
        """Test generic name gets low score."""
        name = FactorName(
            short_name="Beta Factor Dynamics",
            confidence="low"
        )

        score = score_quality(name)
        assert score < 50

    def test_length_penalties(self):
        """Test length-based penalties."""
        too_long = FactorName(short_name="A" * 35)
        too_short = FactorName(short_name="Hi")
        just_right = FactorName(short_name="Regional Banks")

        assert score_quality(too_long) < score_quality(just_right)
        assert score_quality(too_short) < score_quality(just_right)

    def test_distinctiveness_bonus(self):
        """Test distinctiveness from existing names."""
        name = FactorName(short_name="Unique Factor Name")
        existing = ["Other Factor", "Different Name"]

        score_with_others = score_quality(name, existing)

        # Should get bonus for being distinct
        assert score_with_others > score_quality(name, ["Unique Factor Name"])


class TestTagDetection:
    """Test suite for tag detection."""

    def test_sector_tags(self):
        """Test sector tag detection."""
        name = FactorName(
            short_name="Regional Banks vs Biotech",
            description="Financial companies vs healthcare"
        )

        tags = detect_tags(name)
        assert "sector" in tags

    def test_value_growth_tags(self):
        """Test value/growth tag detection."""
        name = FactorName(
            short_name="Value vs Growth",
            theme="Value vs Growth"
        )

        tags = detect_tags(name)
        assert "value_growth" in tags

    def test_multiple_tags(self):
        """Test multiple tag detection."""
        name = FactorName(
            short_name="Small Cap Value",
            description="Momentum in small companies"
        )

        tags = detect_tags(name)
        assert "size" in tags
        assert "value_growth" in tags
        assert "momentum" in tags


class TestPromptGeneration:
    """Test suite for prompt generation."""

    def test_prompt_contains_required_elements(self):
        """Test prompt includes all required context."""
        prompt = generate_name_prompt(
            factor_code="F1",
            top_stocks=["JPM", "BAC", "WFC"],
            bottom_stocks=["AAPL", "MSFT", "GOOGL"],
            sector_analysis={"top_sectors": [{"sector": "Financials", "avg_loading": 0.5}]},
            existing_names=["Other Factor"]
        )

        assert "F1" in prompt
        assert "JPM" in prompt
        assert "AAPL" in prompt
        assert "Financials" in prompt
        assert "Other Factor" in prompt
        assert "short_name" in prompt
        assert "JSON" in prompt

    def test_prompt_handles_missing_sector_analysis(self):
        """Test prompt generation without sector analysis."""
        prompt = generate_name_prompt(
            factor_code="F1",
            top_stocks=["AAPL"],
            bottom_stocks=["JPM"],
            sector_analysis={}
        )

        assert "F1" in prompt
        assert "AAPL" in prompt


class TestResponseParsing:
    """Test suite for response parsing."""

    def test_valid_json_parsing(self):
        """Test parsing valid JSON response."""
        response = '''
        {
          "short_name": "Regional Banks vs Tech",
          "description": "Traditional lenders vs technology companies",
          "theme": "Sector Rotation",
          "high_exposure_desc": "Banks",
          "low_exposure_desc": "Tech companies",
          "confidence": "high"
        }
        '''

        name = parse_name_response(response)
        assert name.short_name == "Regional Banks vs Tech"
        assert name.confidence == "high"

    def test_json_with_markdown_code_block(self):
        """Test parsing JSON inside markdown code block."""
        response = '''```json
        {
          "short_name": "Test Name",
          "confidence": "medium"
        }
        ```'''

        name = parse_name_response(response)
        assert name.short_name == "Test Name"

    def test_invalid_json_raises_error(self):
        """Test that invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_name_response("not valid json")


class TestQualityReport:
    """Test suite for quality report generation."""

    def test_report_contains_all_factors(self):
        """Test report includes all factors."""
        factors = {
            "F1": FactorName(short_name="Good Name", quality_score=90),
            "F2": FactorName(short_name="Bad Name", quality_score=30)
        }

        report = generate_quality_report(factors)
        assert "F1" in report
        assert "F2" in report
        assert "Good Name" in report
        assert "Bad Name" in report

    def test_report_shows_issues(self):
        """Test report shows validation issues."""
        factors = {
            "F1": FactorName(short_name="A" * 35)  # Too long
        }

        report = generate_quality_report(factors)
        assert "too long" in report

    def test_report_has_summary(self):
        """Test report includes summary statistics."""
        factors = {
            "F1": FactorName(short_name="Name 1", quality_score=80),
            "F2": FactorName(short_name="Name 2", quality_score=60)
        }

        report = generate_quality_report(factors)
        assert "Summary Statistics" in report
        assert "Average quality" in report


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
