"""
Factor Naming Framework - Core naming infrastructure for latent factors.

Provides standardized naming with validation, quality scoring, and formatting.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Sequence
from datetime import datetime
import json
import re
from difflib import SequenceMatcher

import pandas as pd
import numpy as np


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
        rationale: Detailed reasoning for the name choice
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
    rationale: str = ""
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


# =============================================================================
# Enrichment Helper Functions
# =============================================================================

def calculate_style_attribution(
    factor_returns: pd.Series,
    all_returns: pd.DataFrame
) -> Dict[str, float]:
    """Regress factor against style proxies and return attribution.

    Uses the first few factors as proxies for known risk factors:
    F1 ~ Market, F2 ~ Value/Growth, F3 ~ Momentum, etc.

    Args:
        factor_returns: Time series of returns for this factor
        all_returns: DataFrame of all factor returns

    Returns:
        Dictionary mapping style names to attribution weights
    """
    styles = {}

    if all_returns is None or len(all_returns.columns) < 4:
        return {'Idiosyncratic': 1.0}

    style_names = ['Market', 'Value', 'Momentum', 'Size', 'Quality']

    for i, style in enumerate(style_names[:min(5, len(all_returns.columns))]):
        if all_returns.columns[i] != factor_returns.name:
            corr = factor_returns.corr(all_returns.iloc[:, i])
            if not np.isnan(corr):
                styles[style] = abs(corr)

    # Normalize
    total = sum(styles.values())
    if total > 0:
        styles = {k: v/total for k, v in styles.items()}

    # Calculate idiosyncratic
    styles['Idiosyncratic'] = max(0, 1 - sum(styles.values()))

    return styles


def calculate_sector_exposure_from_fundamentals(
    loadings: pd.Series,
    fundamentals: pd.DataFrame
) -> Dict[str, float]:
    """Calculate actual sector exposure weighted by factor loadings.

    Weights each stock's sector by its factor loading to get net exposure.
    Positive loadings contribute positively, negative loadings negatively.

    Args:
        loadings: Factor loadings series (ticker -> loading)
        fundamentals: DataFrame with 'Sector' column, indexed by ticker

    Returns:
        Dictionary mapping sectors to net exposure weights
    """
    if fundamentals is None or fundamentals.empty or 'Sector' not in fundamentals.columns:
        return {}

    sector_exposure = {}

    for ticker in loadings.index:
        if ticker in fundamentals.index:
            sector = fundamentals.loc[ticker].get('Sector')
            if sector and not pd.isna(sector):
                # Use signed loading for net exposure
                loading = loadings[ticker]
                sector_exposure[sector] = sector_exposure.get(sector, 0) + loading

    # Sort by absolute exposure
    return dict(sorted(sector_exposure.items(), key=lambda x: abs(x[1]), reverse=True))


def analyze_factor_characteristics(loadings: pd.Series) -> Dict[str, Any]:
    """Deep analysis of factor characteristics for naming.

    Args:
        loadings: Factor loadings series

    Returns:
        Dictionary with structural statistics about the factor
    """
    if loadings is None or loadings.empty:
        return {}

    pos_count = (loadings > 0).sum()
    neg_count = (loadings < 0).sum()

    pos_95 = loadings[loadings > 0].quantile(0.95) if pos_count > 0 else 0
    neg_5 = loadings[loadings < 0].quantile(0.05) if neg_count > 0 else 0

    # Calculate concentration (Herfindahl index)
    loadings_abs = loadings.abs()
    total = loadings_abs.sum()
    if total > 0:
        hhi = ((loadings_abs / total) ** 2).sum()
    else:
        hhi = 0

    return {
        'pos_count': int(pos_count),
        'neg_count': int(neg_count),
        'pos_95': float(pos_95),
        'neg_5': float(neg_5),
        'concentration': float(hhi),
        'skew': float(loadings.skew()) if len(loadings) > 2 else 0,
        'dispersion': float(pos_95 - neg_5)
    }


def format_stock_profile(
    ticker: str,
    loading: float,
    fundamentals: pd.DataFrame
) -> str:
    """Format a single stock's profile for the LLM prompt.

    Args:
        ticker: Stock ticker symbol
        loading: Factor loading for this stock
        fundamentals: DataFrame with fundamental data, indexed by ticker

    Returns:
        Formatted string with stock profile
    """
    profile = f"  - {ticker} (loading: {loading:.3f})"

    if fundamentals is None or fundamentals.empty or ticker not in fundamentals.index:
        return profile

    f = fundamentals.loc[ticker]

    # Company name and sector
    name = f.get('Name', '')
    sector = f.get('Sector', '')
    if name and not pd.isna(name):
        profile += f"\n      Company: {name}"
    if sector and not pd.isna(sector):
        industry = f.get('Industry', '')
        if industry and not pd.isna(industry):
            profile += f" | Sector: {sector} / {industry}"
        else:
            profile += f" | Sector: {sector}"

    # Market cap
    mcap = f.get('MarketCapitalization')
    if mcap and not pd.isna(mcap):
        try:
            mcap = float(mcap)
            if mcap > 1e12:
                profile += f"\n      Market Cap: ${mcap/1e12:.1f}T"
            elif mcap > 1e9:
                profile += f"\n      Market Cap: ${mcap/1e9:.1f}B"
            else:
                profile += f"\n      Market Cap: ${mcap/1e6:.0f}M"
        except (ValueError, TypeError):
            pass

    # Valuation metrics
    valuations = []
    for metric, label in [('PERatio', 'P/E'), ('PriceToBookRatio', 'P/B'),
                          ('PriceToSalesRatioTTM', 'P/S')]:
        val = f.get(metric)
        if val and not pd.isna(val):
            try:
                val = float(val)
                if val > 0:
                    valuations.append(f"{label}: {val:.1f}")
            except (ValueError, TypeError):
                pass
    if valuations:
        profile += f"\n      Valuation: {', '.join(valuations)}"

    # Profitability
    profitability = []
    roe = f.get('ROE') or f.get('ReturnOnEquityTTM')
    margin = f.get('ProfitMargin')
    if roe and not pd.isna(roe):
        try:
            roe = float(roe)
            profitability.append(f"ROE: {roe*100:.1f}%")
        except (ValueError, TypeError):
            pass
    if margin and not pd.isna(margin):
        try:
            margin = float(margin)
            profitability.append(f"Margin: {margin*100:.1f}%")
        except (ValueError, TypeError):
            pass
    if profitability:
        profile += f"\n      Profitability: {', '.join(profitability)}"

    # Dividend yield
    div_yield = f.get('DividendYield')
    if div_yield and not pd.isna(div_yield):
        try:
            div_yield = float(div_yield)
            if div_yield > 0:
                profile += f"\n      Dividend Yield: {div_yield*100:.1f}%"
        except (ValueError, TypeError):
            pass

    return profile


# =============================================================================
# Prompt Generation
# =============================================================================

def generate_name_prompt(
    factor_code: str,
    top_stocks: List[str],
    bottom_stocks: List[str],
    sector_analysis: Dict[str, Any] = None,
    existing_names: List[str] = None,
    # Enrichment parameters (optional)
    fundamentals: pd.DataFrame = None,
    loadings: pd.Series = None,
    style_attribution: Dict[str, float] = None,
    sector_exposure: Dict[str, float] = None,
    factor_stats: Dict[str, Any] = None
) -> str:
    """Generate structured prompt for LLM factor naming.

    Creates a hedge-fund-style prompt that encourages names like:
    - Fama-French: "Value", "Momentum", "Size", "Quality", "Low Vol"
    - Hedge fund baskets: "Unprofitable Tech", "Dividend Aristocrats",
      "Zombies", "R&D Giants", "High Beta Junk", "Space vs Defense"

    Args:
        factor_code: Factor identifier (e.g., "F1")
        top_stocks: Top 10 stocks by positive factor exposure
        bottom_stocks: Top 10 stocks by negative factor exposure
        sector_analysis: Legacy sector composition analysis (optional)
        existing_names: Other factor names to avoid similarity
        fundamentals: DataFrame with fundamental data for enriched profiles
        loadings: Full factor loadings series for enriched profiles
        style_attribution: Pre-calculated style attribution dict
        sector_exposure: Pre-calculated sector exposure dict
        factor_stats: Pre-calculated factor statistics dict

    Returns:
        Formatted prompt string for LLM
    """
    sections = []

    # Header with naming style guide
    sections.append("""You are a quantitative hedge fund analyst naming latent factors extracted from stock returns.

NAMING STYLE - Create names like professional quant funds use:
- Fama-French classics: "Value", "Momentum", "Size", "Quality", "Low Volatility"
- Hedge fund baskets: "Unprofitable Tech", "Dividend Aristocrats", "Zombies",
  "R&D Giants", "High Beta Junk", "Asset Heavy vs Asset Light Energy"
- Thematic: "Space vs Defense", "AI Infrastructure", "Rate Sensitive Financials"

Capture: market anomalies, sentiment patterns, positioning themes, structural shifts,
sector tilts, quality gradients, business model differences, behavioral biases.""")

    sections.append(f"\nFACTOR: {factor_code}")

    # Style attribution section (if available)
    if style_attribution:
        style_lines = []
        # Sort by attribution value descending
        sorted_styles = sorted(style_attribution.items(), key=lambda x: x[1], reverse=True)
        for style, weight in sorted_styles:
            if weight > 0.05:  # Only show significant attributions
                style_lines.append(f"  - {style}: {weight:.0%}")
        if style_lines:
            sections.append(f"\nSTYLE ATTRIBUTION (correlation to known factors):\n" + "\n".join(style_lines))

    # Sector exposure section (if available)
    if sector_exposure:
        sector_lines = []
        for sector, weight in list(sector_exposure.items())[:6]:
            sign = "+" if weight > 0 else ""
            sector_lines.append(f"  - {sector}: {sign}{weight:.1%}")
        if sector_lines:
            sections.append(f"\nSECTOR EXPOSURE (loading-weighted):\n" + "\n".join(sector_lines))

    # Factor statistics (if available)
    if factor_stats:
        stats_lines = []
        if 'pos_count' in factor_stats:
            stats_lines.append(f"  - Long positions: {factor_stats['pos_count']}")
        if 'neg_count' in factor_stats:
            stats_lines.append(f"  - Short positions: {factor_stats['neg_count']}")
        if 'concentration' in factor_stats:
            stats_lines.append(f"  - Concentration (HHI): {factor_stats['concentration']:.3f}")
        if 'skew' in factor_stats:
            stats_lines.append(f"  - Loading skewness: {factor_stats['skew']:.2f}")
        if stats_lines:
            sections.append(f"\nFACTOR STRUCTURE:\n" + "\n".join(stats_lines))

    # Long positions with enriched profiles
    if fundamentals is not None and loadings is not None and not fundamentals.empty:
        # Build enriched profiles
        long_profiles = []
        for ticker in top_stocks[:10]:
            if ticker in loadings.index:
                loading = loadings[ticker]
                profile = format_stock_profile(ticker, loading, fundamentals)
                long_profiles.append(profile)
        if long_profiles:
            sections.append(f"\nLONG POSITIONS (top 10):\n" + "\n".join(long_profiles))
        else:
            sections.append(f"\nLONG POSITIONS (top 10):\n" + "\n".join(f"  - {s}" for s in top_stocks[:10]))
    else:
        # Simple ticker list
        sections.append(f"\nLONG POSITIONS (positive loadings):\n" + "\n".join(f"  - {s}" for s in top_stocks[:10]))

    # Short positions with enriched profiles
    if fundamentals is not None and loadings is not None and not fundamentals.empty:
        short_profiles = []
        for ticker in bottom_stocks[:10]:
            if ticker in loadings.index:
                loading = loadings[ticker]
                profile = format_stock_profile(ticker, loading, fundamentals)
                short_profiles.append(profile)
        if short_profiles:
            sections.append(f"\nSHORT POSITIONS (top 10):\n" + "\n".join(short_profiles))
        else:
            sections.append(f"\nSHORT POSITIONS (top 10):\n" + "\n".join(f"  - {s}" for s in bottom_stocks[:10]))
    else:
        sections.append(f"\nSHORT POSITIONS (negative loadings):\n" + "\n".join(f"  - {s}" for s in bottom_stocks[:10]))

    # Legacy sector analysis (if no enriched data)
    if sector_analysis and not sector_exposure:
        top_sectors = sector_analysis.get('top_sectors', [])
        if top_sectors:
            sector_text = ", ".join(f"{s['sector']} ({s.get('avg_loading', 0):.2f})" for s in top_sectors[:3])
            sections.append(f"\nTOP SECTORS: {sector_text}")

    # Existing names to avoid
    if existing_names:
        sections.append(f"\nEXISTING FACTOR NAMES (avoid similarity): {', '.join(existing_names[:10])}")

    # Instructions
    sections.append("""
ANALYSIS INSTRUCTIONS:
Identify the economic theme this factor captures:
- VALUE vs GROWTH? Look for P/E, P/B differences between longs and shorts
- MOMENTUM? Recent winners vs losers, trend-following patterns
- SIZE? Large cap vs small cap split
- QUALITY vs JUNK? Profitability, ROE, margins, debt levels
- SECTOR ROTATION? Tech vs Energy? Healthcare vs Financials? Cyclical vs Defensive?
- BUSINESS MODEL? Asset-light vs asset-heavy? Subscription vs transactional?
- PROFITABILITY? Profitable vs unprofitable? High-margin vs low-margin?

Respond with JSON:
{
  "short_name": "2-4 word hedge fund style name (max 30 chars)",
  "description": "1 sentence economic rationale (max 100 chars)",
  "theme": "High-level theme: Value, Momentum, Size, Quality, Sector, etc.",
  "high_exposure_desc": "What unifies the long positions",
  "low_exposure_desc": "What unifies the short positions",
  "confidence": "high|medium|low",
  "rationale": "2-3 sentences explaining your reasoning based on the data"
}

NAMING RULES:
✗ AVOID: "Beta Factor", "Dynamic Mix", "Balanced Exposure", ticker-based names like "AAPL Factor"
✗ AVOID: Generic terms: "Factor", "Dynamic", "Composite", "Multi", "Systematic"
✓ USE: Specific themes: "Unprofitable Tech", "Dividend Aristocrats", "Quality Growth"
✓ USE: Contrast patterns: "Growth vs Value", "Large vs Small", "Cyclical vs Defensive"
✓ USE: Hedge fund language: "High Beta Junk", "Momentum Winners", "Deep Value Traps\"""")

    return "\n".join(sections)


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
        rationale=data.get("rationale", ""),
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
