"""
Factor Naming Framework - Core naming infrastructure for latent factors.

Provides standardized naming with validation, quality scoring, and formatting.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import re
from difflib import SequenceMatcher


@dataclass
class FactorName:
    """Standardized factor name with metadata.

    Attributes:
        short_name: 2-4 word label for charts/UI (max 30 chars)
        description: 1-sentence explanation (max 100 chars)
        theme: High-level theme (e.g., "Value vs Growth", "Sector Rotation")
        high_exposure_desc: What high-exposure stocks have in common
        low_exposure_desc: What low-exposure stocks have in common
        confidence: high, medium, or low
        tags: List of category tags (e.g., ["sector", "value_growth"])
        quality_score: Computed quality score (0-100)
        approved: Whether name has been human-approved
        edited: Whether name has been manually edited
        generated_at: ISO timestamp of generation
        edited_at: ISO timestamp of last edit (if any)
    """
    short_name: str
    description: str = ""
    theme: str = ""
    high_exposure_desc: str = ""
    low_exposure_desc: str = ""
    confidence: str = "medium"
    tags: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    approved: bool = False
    edited: bool = False
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    edited_at: Optional[str] = None

    def __post_init__(self):
        """Validate and clean fields after initialization."""
        self.short_name = self._clean_short_name(self.short_name)
        self.description = self._clean_description(self.description)
        self.confidence = self.confidence.lower()
        if self.confidence not in ("high", "medium", "low"):
            self.confidence = "medium"

    def _clean_short_name(self, name: str) -> str:
        """Remove markdown and normalize short name."""
        # Remove markdown formatting
        name = re.sub(r'\*\*', '', name)
        name = re.sub(r'^Factor Name:\s*', '', name, flags=re.IGNORECASE)
        name = re.sub(r'^\d+\.\s*', '', name)
        # Normalize whitespace
        name = ' '.join(name.split())
        return name.strip()

    def _clean_description(self, desc: str) -> str:
        """Remove markdown and normalize description."""
        desc = re.sub(r'\*\*', '', desc)
        desc = re.sub(r'^\*\*.*?\*\*\s*', '', desc)
        desc = ' '.join(desc.split())
        return desc.strip()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FactorName":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def display_label(self) -> str:
        """Get display label for charts and UI."""
        return self.short_name[:30] if self.short_name else "Unnamed Factor"

    def full_label(self) -> str:
        """Get full label with description for tooltips."""
        if self.description:
            return f"{self.short_name}: {self.description}"
        return self.short_name


# Banned words that indicate generic naming
BANNED_WORDS = {
    "beta", "factor", "dynamic", "dynamics", "balance", "balanced",
    "mixed", "diverse", "blend", "composite", "complex", "multifactor",
    "general", "generic", "broad", "overall", "market", "systematic"
}

# Theme categories for tagging
THEME_CATEGORIES = {
    "sector": ["technology", "healthcare", "financial", "energy", "industrial",
               "consumer", "utilities", "materials", "reits", "telecom"],
    "size": ["large cap", "small cap", "mid cap", "micro cap", "mega cap"],
    "value_growth": ["value", "growth", "income", "dividend", "yield"],
    "quality": ["quality", "profitability", "earnings", "margins"],
    "momentum": ["momentum", "trend", "acceleration"],
    "volatility": ["volatility", "risk", "defensive", "stable"],
    "macro": ["interest rate", "inflation", "credit", "liquidity"]
}


def validate_name(name: FactorName) -> List[str]:
    """Validate a factor name and return list of issues.

    Returns:
        List of validation issues (empty list if valid)
    """
    issues = []

    # Check short name length
    if len(name.short_name) > 30:
        issues.append(f"Short name too long ({len(name.short_name)} chars, max 30)")

    if len(name.short_name) < 5:
        issues.append(f"Short name too short ({len(name.short_name)} chars, min 5)")

    # Check description length
    if len(name.description) > 100:
        issues.append(f"Description too long ({len(name.description)} chars, max 100)")

    # Check for banned words
    short_lower = name.short_name.lower()
    for banned in BANNED_WORDS:
        if banned in short_lower:
            issues.append(f"Contains banned word: '{banned}'")

    # Check for markdown remnants
    if "**" in name.short_name or "**" in name.description:
        issues.append("Contains markdown formatting (**)")

    # Check theme consistency
    if name.high_exposure_desc and name.low_exposure_desc:
        # High and low should be opposites, not identical
        high_lower = name.high_exposure_desc.lower()
        low_lower = name.low_exposure_desc.lower()
        similarity = SequenceMatcher(None, high_lower, low_lower).ratio()
        if similarity > 0.8:
            issues.append("High and low exposure descriptions are too similar")

    return issues


def score_quality(name: FactorName, existing_names: List[str] = None) -> float:
    """Score the quality of a factor name (0-100).

    Args:
        name: The FactorName to score
        existing_names: List of other factor names for distinctiveness check

    Returns:
        Quality score from 0-100
    """
    score = 50.0  # Start at neutral

    # Length scoring
    short_len = len(name.short_name)
    if 10 <= short_len <= 25:
        score += 15  # Ideal length
    elif 5 <= short_len < 10:
        score += 10  # Good but brief
    elif short_len > 30:
        score -= 20  # Too long

    desc_len = len(name.description)
    if 20 <= desc_len <= 80:
        score += 10
    elif desc_len > 100:
        score -= 10

    # Penalize banned words
    short_lower = name.short_name.lower()
    banned_count = sum(1 for b in BANNED_WORDS if b in short_lower)
    score -= banned_count * 15

    # Reward specificity indicators
    specificity_markers = ["vs", "versus", "-", ":", ">"]
    if any(m in short_lower for m in specificity_markers):
        score += 10  # Has contrast indicator

    # Check for sector keywords
    for category, keywords in THEME_CATEGORIES.items():
        if any(kw in short_lower for kw in keywords):
            score += 5  # Has thematic content
            break

    # Confidence bonus/penalty
    if name.confidence == "high":
        score += 5
    elif name.confidence == "low":
        score -= 10

    # Distinctiveness check against existing names
    if existing_names:
        similarities = []
        for existing in existing_names:
            sim = SequenceMatcher(None, short_lower, existing.lower()).ratio()
            similarities.append(sim)

        max_sim = max(similarities) if similarities else 0
        if max_sim > 0.8:
            score -= 20  # Very similar to existing
        elif max_sim > 0.6:
            score -= 10  # Somewhat similar
        elif max_sim < 0.4:
            score += 10  # Very distinct

    # Completeness bonus
    if name.high_exposure_desc and name.low_exposure_desc:
        score += 5
    if name.theme:
        score += 3

    # Clamp to 0-100
    return max(0.0, min(100.0, score))


def detect_tags(name: FactorName) -> List[str]:
    """Auto-detect thematic tags from name content.

    Args:
        name: The FactorName to analyze

    Returns:
        List of detected tags
    """
    text = f"{name.short_name} {name.description} {name.theme}".lower()
    tags = []

    for category, keywords in THEME_CATEGORIES.items():
        for keyword in keywords:
            if keyword in text:
                tags.append(category)
                break

    return tags


def generate_name_prompt(factor_code: str, top_stocks: List[str],
                         bottom_stocks: List[str],
                         sector_analysis: Dict[str, Any],
                         existing_names: List[str] = None) -> str:
    """Generate structured prompt for LLM factor naming.

    Args:
        factor_code: Factor identifier (e.g., "F1")
        top_stocks: Top 10 stocks by factor exposure
        bottom_stocks: Bottom 10 stocks by factor exposure
        sector_analysis: Sector composition analysis
        existing_names: Other factor names to avoid similarity

    Returns:
        Formatted prompt string
    """
    # Build sector context
    sector_text = ""
    if sector_analysis:
        top_sectors = sector_analysis.get('top_sectors', [])
        if top_sectors:
            sector_text = "Top sectors by exposure: " + ", ".join(
                f"{s['sector']} ({s['avg_loading']:.2f})" for s in top_sectors[:3]
            )

    # Build existing names warning
    existing_text = ""
    if existing_names:
        existing_text = f"\nExisting factor names to avoid similarity with: {', '.join(existing_names)}"

    prompt = f"""Name this financial factor based on the stock characteristics.

Factor: {factor_code}

High Exposure Stocks (positive loadings):
{chr(10).join(f"  - {s}" for s in top_stocks)}

Low Exposure Stocks (negative loadings):
{chr(10).join(f"  - {s}" for s in bottom_stocks)}

{sector_text}{existing_text}

Provide a concise, distinctive name that captures what distinguishes high-exposure from low-exposure stocks.

Respond ONLY with valid JSON in this exact format:
{{
  "short_name": "2-4 word label (max 30 chars)",
  "description": "1 sentence explaining the factor (max 100 chars)",
  "theme": "High-level theme like 'Value vs Growth' or 'Sector Rotation'",
  "high_exposure_desc": "What high-exposure stocks have in common",
  "low_exposure_desc": "What low-exposure stocks have in common",
  "confidence": "high|medium|low"
}}

Guidelines:
- Use specific terms, not generic ones like "Beta", "Factor", "Dynamic"
- Include contrast indicators like "vs" or "-" when comparing groups
- Reference actual sectors or characteristics observed in the data
- Keep short_name under 30 characters for chart labels
- Avoid the pattern "X Earners vs Y Innovators" - be more specific about business models"""

    return prompt


def parse_name_response(response_text: str) -> FactorName:
    """Parse LLM JSON response into FactorName.

    Args:
        response_text: Raw LLM response text

    Returns:
        FactorName object

    Raises:
        ValueError: If response cannot be parsed
    """
    # Try to extract JSON from response
    response_text = response_text.strip()

    # Handle markdown code blocks
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in response: {e}")

    # Map to FactorName fields
    return FactorName(
        short_name=data.get("short_name", ""),
        description=data.get("description", ""),
        theme=data.get("theme", ""),
        high_exposure_desc=data.get("high_exposure_desc", ""),
        low_exposure_desc=data.get("low_exposure_desc", ""),
        confidence=data.get("confidence", "medium"),
        tags=[]  # Will be populated by detect_tags
    )


def generate_quality_report(factors: Dict[str, FactorName]) -> str:
    """Generate a human-readable quality report for factor names.

    Args:
        factors: Dictionary mapping factor codes to FactorName objects

    Returns:
        Formatted report string
    """
    lines = ["=" * 70, "Factor Naming Quality Report", "=" * 70, ""]

    # Collect all short names for distinctiveness checks
    existing_names = [f.short_name for f in factors.values()]

    for code, name in sorted(factors.items()):
        # Update quality score
        other_names = [n for n in existing_names if n != name.short_name]
        name.quality_score = score_quality(name, other_names)
        name.tags = detect_tags(name)

        # Get validation issues
        issues = validate_name(name)

        lines.append(f"\n{code}: \"{name.short_name}\"")
        lines.append(f"  Quality Score: {name.quality_score:.0f}/100")
        lines.append(f"  Confidence: {name.confidence}")
        lines.append(f"  Tags: {', '.join(name.tags) if name.tags else 'none'}")

        if name.description:
            lines.append(f"  Description: {name.description[:60]}...")

        if issues:
            lines.append("  Issues:")
            for issue in issues:
                lines.append(f"    ⚠ {issue}")
        else:
            lines.append("  ✓ No validation issues")

        if name.approved:
            lines.append("  ✓ Human approved")
        elif name.edited:
            lines.append("  ✎ Human edited (pending approval)")

    # Summary statistics
    scores = [f.quality_score for f in factors.values()]
    approved_count = sum(1 for f in factors.values() if f.approved)

    lines.append("\n" + "=" * 70)
    lines.append("Summary Statistics")
    lines.append("=" * 70)
    lines.append(f"  Total factors: {len(factors)}")
    lines.append(f"  Average quality: {sum(scores)/len(scores):.1f}/100")
    lines.append(f"  High quality (>70): {sum(1 for s in scores if s > 70)}")
    lines.append(f"  Low quality (<50): {sum(1 for s in scores if s < 50)}")
    lines.append(f"  Human approved: {approved_count}")
    lines.append("")

    return "\n".join(lines)
