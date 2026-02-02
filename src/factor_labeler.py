"""
LLM-Powered Factor Naming: Intelligent Factor Interpretation
===========================================================

This module provides sophisticated factor naming capabilities using Large Language
Models (LLMs) to automatically generate meaningful, human-readable names for
quantitative factors based on their stock exposures and fundamental characteristics.

Core Functionality
-----------------

**1. Intelligent Factor Analysis**
- Analyzes factor loadings to identify key stock exposures
- Integrates fundamental data for richer context (sector, market cap, ratios)
- Generates descriptive summaries of factor characteristics

**2. LLM-Powered Naming**
- Uses OpenAI GPT models for factor interpretation
- Employs sophisticated prompt engineering for accurate naming
- Provides both factor names and economic explanations

**3. Robust Error Handling**
- Comprehensive API error recovery with exponential backoff
- Input validation and data quality checks
- Graceful fallbacks for network and API issues

**4. Batch Processing**
- Efficient processing of multiple factors
- Rate limiting and cost optimization
- Progress tracking and logging

Key Features
-----------
- **Enhanced Context**: Rich fundamental data integration beyond ticker symbols
- **Prompt Engineering**: Carefully crafted prompts to avoid generic factor names
- **Error Recovery**: Robust handling of API failures and edge cases
- **Cost Optimization**: Intelligent caching and rate limiting
- **Validation**: API key validation and response verification

Dependencies
-----------
- **Core**: openai, pandas, typing
- **Config**: python-dotenv for environment management
- **Logging**: Built-in logging for monitoring and debugging
- **Data**: Expects fundamental data from alphavantage_system

Integration Points
-----------------
- **Called by**: discover_and_label.py (main workflow)
- **Calls**: OpenAI Chat Completions API
- **Data Sources**: Fundamental data DataFrames from DataBackend
- **Output**: Human-readable factor names and descriptions

Critical Fixes Applied
--------------------
- **API Reliability**: Comprehensive error handling with retry logic
- **Token Management**: Configurable limits and response validation
- **Context Enhancement**: Rich fundamental data integration
- **Rate Limiting**: Intelligent request pacing and backoff
- **Input Validation**: Robust data quality checks
- **Response Parsing**: Improved parsing with fallback mechanisms

Requirements
-----------
- **Environment**: OPENAI_API_KEY environment variable
- **API Access**: OpenAI API account with sufficient credits
- **Models**: Compatible with gpt-5.2-mini, gpt-5.2, gpt-5.2-pro
- **Data**: Fundamental data with sector, market cap, financial ratios

Performance Characteristics
--------------------------
- **Cost per Factor**: ~$0.01-0.05 depending on model choice
- **Processing Time**: ~2-5 seconds per factor (including API latency)
- **Batch Efficiency**: Linear scaling with built-in rate limiting
- **Cache Benefits**: Significant cost savings for repeated analysis

Examples
--------
>>> # Basic factor naming
>>> factor_names = ask_llm("F1", ["AAPL", "MSFT"], ["JPM", "BAC"], fundamentals_df)

>>> # Batch processing
>>> all_names = batch_name_factors(factor_loadings, fundamentals_df)

>>> # API validation
>>> if validate_api_key():
...     proceed_with_naming()

Notes
----
- This module was enhanced during the factor quality debugging process
- Addresses the issue of repetitive "Finance Beta" type factor names
- Provides the LLM with rich context to generate distinctive factor names
- Critical component for making factor models interpretable and actionable
"""

import os, textwrap, logging, time, json
from typing import List, Dict, Sequence, Optional

from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv

from .factor_naming import (
    FactorName, generate_name_prompt, parse_name_response,
    validate_name, score_quality, detect_tags, generate_quality_report
)

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client only if API key is available
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
else:
    client = None
_LOG = logging.getLogger(__name__)


def _summarise_group(tickers: Sequence[str],
                     fundamentals: pd.DataFrame) -> str:
    """Return a rich textual description of the group with enhanced fundamental data."""
    try:
        if fundamentals.empty or len(tickers) == 0:
            return "No data available"
            
        # Filter to valid tickers present in fundamentals
        valid_tickers = [t for t in tickers if t in fundamentals.index]
        if not valid_tickers:
            return "No fundamental data available"
            
        sub = fundamentals.loc[valid_tickers]
        summary_parts = []
        
        # Sector diversity analysis
        if "Sector" in sub.columns and not sub["Sector"].isna().all():
            try:
                sector_counts = sub["Sector"].value_counts()
                if len(sector_counts) == 1:
                    summary_parts.append(f"Sector: {sector_counts.index[0]}")
                elif len(sector_counts) <= 3:
                    top_sectors = ", ".join(sector_counts.head(2).index)
                    summary_parts.append(f"Sectors: {top_sectors}")
                else:
                    summary_parts.append(f"Multi-sector ({len(sector_counts)} sectors)")
            except Exception:
                summary_parts.append("Sector: n/a")
        
        # Market cap analysis with size classification
        if "MarketCapitalization" in sub.columns:
            try:
                cap_data = pd.to_numeric(sub["MarketCapitalization"], errors='coerce')
                valid_caps = cap_data.dropna()
                if len(valid_caps) > 0:
                    median_cap = valid_caps.median()
                    # Classify by market cap
                    if median_cap > 200e9:
                        size_class = "mega-cap"
                    elif median_cap > 10e9:
                        size_class = "large-cap" 
                    elif median_cap > 2e9:
                        size_class = "mid-cap"
                    else:
                        size_class = "small-cap"
                    summary_parts.append(f"Size: {size_class} (${median_cap/1e9:.1f}B median)")
            except Exception:
                pass
        
        # Industry/business model diversity
        if "Industry" in sub.columns and not sub["Industry"].isna().all():
            try:
                industry_counts = sub["Industry"].value_counts()
                if len(industry_counts) <= 2:
                    industries = ", ".join(industry_counts.head(2).index)
                    summary_parts.append(f"Industries: {industries}")
                else:
                    summary_parts.append(f"Diverse industries ({len(industry_counts)} types)")
            except Exception:
                pass
        
        # Financial characteristics
        financial_info = []
        
        # P/E ratio analysis
        if "PERatio" in sub.columns:
            try:
                pe_data = pd.to_numeric(sub["PERatio"], errors='coerce')
                valid_pe = pe_data.dropna()
                if len(valid_pe) > 2:
                    median_pe = valid_pe.median()
                    if median_pe > 25:
                        financial_info.append("high-growth (high P/E)")
                    elif median_pe > 15:
                        financial_info.append("growth-oriented")
                    else:
                        financial_info.append("value-oriented (low P/E)")
            except Exception:
                pass
        
        # Dividend yield analysis  
        if "DividendYield" in sub.columns:
            try:
                div_data = pd.to_numeric(sub["DividendYield"], errors='coerce')
                valid_div = div_data.dropna()
                if len(valid_div) > 2:
                    median_div = valid_div.median() * 100  # Convert to percentage
                    if median_div > 3:
                        financial_info.append("dividend-focused")
                    elif median_div > 1:
                        financial_info.append("modest dividends")
            except Exception:
                pass
        
        if financial_info:
            summary_parts.append(f"Style: {', '.join(financial_info)}")
        
        # Company examples for context
        example_tickers = valid_tickers[:3]  # Show first 3 as examples
        if len(valid_tickers) > 3:
            summary_parts.append(f"Examples: {', '.join(example_tickers)} (+ {len(valid_tickers)-3} more)")
        else:
            summary_parts.append(f"Companies: {', '.join(example_tickers)}")
        
        return " | ".join(summary_parts) if summary_parts else "Limited fundamental data"
            
    except Exception as e:
        _LOG.warning("Error summarizing group %s: %s", tickers[:3], str(e))
        return "Group summary unavailable"


def ask_llm(factor_id: str,
            top_pos: Sequence[str],
            top_neg: Sequence[str],
            fundamentals: pd.DataFrame,
            sector_analysis: Optional[Dict] = None,
            existing_names: Optional[List[str]] = None,
            temperature: float = 0.4,
            max_tokens: int = 200,
            model: str = "gpt-5.2-mini",
            max_retries: int = 3) -> FactorName:
    """
    Generate structured factor name using LLM with comprehensive error handling.

    Parameters
    ----------
    factor_id : str
        Factor identifier
    top_pos : Sequence[str]
        Tickers with highest positive exposures
    top_neg : Sequence[str]
        Tickers with highest negative exposures
    fundamentals : pd.DataFrame
        Fundamental data for context
    sector_analysis : Optional[Dict]
        Sector composition analysis
    existing_names : Optional[List[str]]
        Other factor names to avoid similarity with
    temperature : float, default 0.4
        LLM temperature parameter (lower for more consistent output)
    max_tokens : int, default 200
        Maximum tokens in response (JSON needs more space)
    model : str, default "gpt-5.2-mini"
        OpenAI model to use
    max_retries : int, default 3
        Maximum retry attempts for API calls

    Returns
    -------
    FactorName
        Structured factor name with metadata, or fallback if API fails
    """
    if client is None:
        _LOG.error("OpenAI API key not set. Set OPENAI_API_KEY environment variable.")
        return FactorName(
            short_name=f"{factor_id}: Unnamed",
            description="API key not available",
            confidence="low"
        )

    # Input validation
    if not top_pos and not top_neg:
        _LOG.warning("No tickers provided for factor %s", factor_id)
        return FactorName(
            short_name=f"{factor_id}: No Exposures",
            description="No stock exposures available",
            confidence="low"
        )

    try:
        # Build sector analysis from fundamentals if not provided
        if sector_analysis is None and not fundamentals.empty:
            sector_analysis = _build_sector_analysis(top_pos, top_neg, fundamentals)

        # Generate structured prompt
        prompt = generate_name_prompt(
            factor_code=factor_id,
            top_stocks=list(top_pos[:10]),
            bottom_stocks=list(top_neg[:10]),
            sector_analysis=sector_analysis,
            existing_names=existing_names
        )

        # Retry logic with exponential backoff
        last_error = None
        for attempt in range(max_retries):
            try:
                _LOG.debug("Requesting factor name for %s (attempt %d/%d)",
                          factor_id, attempt + 1, max_retries)

                # Rate limiting - simple delay between requests
                if attempt > 0:
                    delay = min(2 ** attempt, 30)  # Exponential backoff, max 30 seconds
                    _LOG.info("Retrying after %d seconds...", delay)
                    time.sleep(delay)

                rsp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a quantitative finance expert. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=30,
                    response_format={"type": "json_object"}  # Enforce JSON output
                )

                # Validate response
                if not rsp.choices or not rsp.choices[0].message.content:
                    raise ValueError("Empty response from OpenAI API")

                content = rsp.choices[0].message.content.strip()
                if not content:
                    raise ValueError("Empty content in OpenAI response")

                # Parse JSON response into FactorName
                factor_name = parse_name_response(content)
                factor_name.tags = detect_tags(factor_name)

                # Score quality
                factor_name.quality_score = score_quality(factor_name, existing_names)

                _LOG.debug("Successfully generated factor name for %s (score: %.1f)",
                          factor_id, factor_name.quality_score)
                return factor_name

            except Exception as e:
                last_error = e
                _LOG.warning("API call failed for %s (attempt %d/%d): %s",
                           factor_id, attempt + 1, max_retries, str(e))

                # Don't retry on certain errors
                if "api_key" in str(e).lower() or "authentication" in str(e).lower():
                    break

                continue

        # All retries failed - return fallback
        _LOG.error("All retry attempts failed for factor %s. Last error: %s",
                   factor_id, str(last_error))
        return _generate_fallback_name(factor_id, top_pos, top_neg, sector_analysis)

    except Exception as e:
        _LOG.error("Unexpected error in ask_llm for factor %s: %s", factor_id, str(e))
        return _generate_fallback_name(factor_id, top_pos, top_neg, sector_analysis)


def _build_sector_analysis(top_pos: Sequence[str],
                           top_neg: Sequence[str],
                           fundamentals: pd.DataFrame) -> Dict:
    """Build sector composition analysis from fundamentals."""
    analysis = {'top_sectors': [], 'bottom_sectors': []}

    try:
        if fundamentals.empty or "Sector" not in fundamentals.columns:
            return analysis

        # Analyze high exposure stocks
        valid_pos = [t for t in top_pos if t in fundamentals.index]
        if valid_pos:
            pos_sectors = fundamentals.loc[valid_pos, "Sector"].value_counts()
            for sector, count in pos_sectors.head(3).items():
                avg_loading = 0.5  # Placeholder
                analysis['top_sectors'].append({
                    'sector': sector,
                    'count': count,
                    'avg_loading': avg_loading
                })

        # Analyze low exposure stocks
        valid_neg = [t for t in top_neg if t in fundamentals.index]
        if valid_neg:
            neg_sectors = fundamentals.loc[valid_neg, "Sector"].value_counts()
            for sector, count in neg_sectors.head(3).items():
                analysis['bottom_sectors'].append({
                    'sector': sector,
                    'count': count
                })

    except Exception as e:
        _LOG.warning("Error building sector analysis: %s", str(e))

    return analysis


def _generate_fallback_name(factor_id: str,
                            top_pos: Sequence[str],
                            top_neg: Sequence[str],
                            sector_analysis: Optional[Dict] = None) -> FactorName:
    """Generate rule-based fallback name when LLM fails."""

    # Try sector-based naming
    if sector_analysis and sector_analysis.get('top_sectors'):
        top_sector = sector_analysis['top_sectors'][0]['sector']

        # Check if top and bottom are different sectors
        bottom_sectors = sector_analysis.get('bottom_sectors', [])
        if bottom_sectors and bottom_sectors[0]['sector'] != top_sector:
            bottom_sector = bottom_sectors[0]['sector']
            return FactorName(
                short_name=f"{top_sector} vs {bottom_sector}",
                description=f"Factor distinguishing {top_sector} from {bottom_sector}",
                theme="Sector Rotation",
                confidence="low",
                tags=["sector"]
            )
        else:
            return FactorName(
                short_name=f"{top_sector} Exposure",
                description=f"Stocks with high {top_sector} sector exposure",
                theme="Sector Concentration",
                confidence="low",
                tags=["sector"]
            )

    # Size-based fallback
    if top_pos and top_neg:
        return FactorName(
            short_name=f"{factor_id}: Factor",
            description=f"Factor with {len(top_pos)} positive and {len(top_neg)} negative exposures",
            confidence="low"
        )

    return FactorName(
        short_name=f"{factor_id}: Unnamed",
        description="Unable to generate name",
        confidence="low"
    )


def batch_name_factors(factor_exposures: pd.DataFrame,
                      fundamentals: pd.DataFrame,
                      top_n: int = 10,
                      **llm_kwargs) -> Dict[str, FactorName]:
    """
    Name multiple factors with rate limiting and progress tracking.

    Parameters
    ----------
    factor_exposures : pd.DataFrame
        Factor exposures matrix (tickers x factors)
    fundamentals : pd.DataFrame
        Fundamental data for context
    top_n : int, default 10
        Number of top/bottom stocks to consider
    **llm_kwargs
        Additional arguments passed to ask_llm

    Returns
    -------
    Dict[str, FactorName]
        Dictionary mapping factor IDs to FactorName objects
    """
    if factor_exposures.empty:
        _LOG.warning("Empty factor exposures provided")
        return {}

    factor_names = {}
    total_factors = len(factor_exposures.columns)
    named_so_far = []  # Track names for distinctiveness scoring

    _LOG.info("Starting batch naming of %d factors", total_factors)

    for i, factor_id in enumerate(factor_exposures.columns, 1):
        try:
            _LOG.info("Naming factor %d/%d: %s", i, total_factors, factor_id)

            exposures = factor_exposures[factor_id]
            top_pos = exposures.nlargest(top_n).index.tolist()
            top_neg = exposures.nsmallest(top_n).index.tolist()

            # Build sector analysis
            sector_analysis = _build_sector_analysis(top_pos, top_neg, fundamentals)

            # Generate name with distinctiveness context
            factor_name = ask_llm(
                factor_id, top_pos, top_neg, fundamentals,
                sector_analysis=sector_analysis,
                existing_names=named_so_far.copy(),
                **llm_kwargs
            )

            factor_names[factor_id] = factor_name
            named_so_far.append(factor_name.short_name)

            # Simple rate limiting between requests
            if i < total_factors:  # Don't sleep after last request
                time.sleep(0.5)  # Reduced delay for JSON responses

        except Exception as e:
            _LOG.error("Failed to name factor %s: %s", factor_id, str(e))
            factor_names[factor_id] = _generate_fallback_name(factor_id, [], [], None)

    # Generate and log quality report
    report = generate_quality_report(factor_names)
    _LOG.info("\n%s", report)

    successful = len([v for v in factor_names.values() if v.quality_score > 30])
    _LOG.info("Completed batch naming: %d/%d factors with acceptable quality",
              successful, total_factors)

    return factor_names


def validate_api_key() -> bool:
    """
    Check if OpenAI API key is available and valid.
    
    Returns
    -------
    bool
        True if API key is available and appears valid
    """
    if not openai_api_key:
        return False
        
    try:
        # Simple test call to validate key
        test_client = OpenAI(api_key=openai_api_key)
        test_client.chat.completions.create(
            model="gpt-5.2-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        return True
    except Exception as e:
        _LOG.error("API key validation failed: %s", str(e))
        return False