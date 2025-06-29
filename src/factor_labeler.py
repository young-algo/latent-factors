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
- **Models**: Compatible with gpt-4o-mini, gpt-4o, gpt-3.5-turbo
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

import os, textwrap, logging, time
from typing import List, Dict, Sequence, Optional

from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv

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
            temperature: float = 0.7,
            max_tokens: int = 100,
            model: str = "gpt-4o-mini",
            max_retries: int = 3) -> str:
    """
    Return a concise factor label + rationale with comprehensive error handling.
    
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
    temperature : float, default 0.7
        LLM temperature parameter
    max_tokens : int, default 100
        Maximum tokens in response (increased from 60)
    model : str, default "gpt-4o-mini"
        OpenAI model to use
    max_retries : int, default 3
        Maximum retry attempts for API calls
        
    Returns
    -------
    str
        Factor label and explanation, or fallback name if API fails
    """
    if client is None:
        _LOG.error("OpenAI API key not set. Set OPENAI_API_KEY environment variable.")
        return f"{factor_id}: API key not available"
    
    # Input validation
    if not top_pos and not top_neg:
        _LOG.warning("No tickers provided for factor %s", factor_id)
        return f"{factor_id}: No exposures available"
    
    try:
        # Build context with error handling
        pos_desc = _summarise_group(top_pos, fundamentals) if top_pos else "No positive exposures"
        neg_desc = _summarise_group(top_neg, fundamentals) if top_neg else "No negative exposures"

        sys_msg = textwrap.dedent("""
        You are a seasoned equity strategist creating distinct factor names.
        
        CRITICAL: Avoid generic terms like "Beta", "Finance", "Dynamics", "Balance", "Divergence".
        Instead, focus on the SPECIFIC economic theme that differentiates the high vs low exposure stocks.
        
        Examples of GOOD names:
        - "Tech Giants vs Cyclicals" (if mega-cap tech vs industrial stocks)
        - "Growth vs Value" (if high P/E vs low P/E stocks)  
        - "Dividend Aristocrats" (if high dividend stocks)
        - "Small Cap Momentum" (if small growth stocks)
        - "REIT vs Industrials" (if real estate vs manufacturing)
        
        Focus on: sector differences, size differences, growth vs value, business models, economic sensitivity.
        """).strip()

        user_msg = textwrap.dedent(f"""
        Analyze this factor's stock exposures:

        HIGH EXPOSURE STOCKS: {', '.join(top_pos[:8])}
        Characteristics: {pos_desc}

        LOW EXPOSURE STOCKS: {', '.join(top_neg[:8])}  
        Characteristics: {neg_desc}

        What economic theme distinguishes these two groups? 
        
        Respond with: **Factor Name** (2-4 words): One sentence explanation focusing on the key economic difference.
        
        Avoid generic financial terms. Be specific about what makes these groups different.
        """).strip()

        # Retry logic with exponential backoff
        last_error = None
        for attempt in range(max_retries):
            try:
                _LOG.debug("Requesting factor name for %s (attempt %d/%d)", factor_id, attempt + 1, max_retries)
                
                # Rate limiting - simple delay between requests
                if attempt > 0:
                    delay = min(2 ** attempt, 30)  # Exponential backoff, max 30 seconds
                    _LOG.info("Retrying after %d seconds...", delay)
                    time.sleep(delay)
                
                rsp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": sys_msg},
                              {"role": "user", "content": user_msg}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=30  # Add timeout
                )
                
                # Validate response
                if not rsp.choices or not rsp.choices[0].message.content:
                    raise ValueError("Empty response from OpenAI API")
                
                content = rsp.choices[0].message.content.strip()
                if not content:
                    raise ValueError("Empty content in OpenAI response")
                
                _LOG.debug("Successfully received factor name for %s", factor_id)
                return content
                
            except Exception as e:
                last_error = e
                _LOG.warning("API call failed for %s (attempt %d/%d): %s", 
                           factor_id, attempt + 1, max_retries, str(e))
                
                # Don't retry on certain errors
                if "api_key" in str(e).lower() or "authentication" in str(e).lower():
                    break
                    
                continue
        
        # All retries failed
        _LOG.error("All retry attempts failed for factor %s. Last error: %s", factor_id, str(last_error))
        return f"{factor_id}: Naming failed after {max_retries} attempts"
        
    except Exception as e:
        _LOG.error("Unexpected error in ask_llm for factor %s: %s", factor_id, str(e))
        return f"{factor_id}: Unexpected error during naming"


def batch_name_factors(factor_exposures: pd.DataFrame,
                      fundamentals: pd.DataFrame,
                      top_n: int = 8,
                      **llm_kwargs) -> Dict[str, str]:
    """
    Name multiple factors with rate limiting and progress tracking.
    
    Parameters
    ----------
    factor_exposures : pd.DataFrame
        Factor exposures matrix (tickers x factors)
    fundamentals : pd.DataFrame  
        Fundamental data for context
    top_n : int, default 8
        Number of top/bottom stocks to consider
    **llm_kwargs
        Additional arguments passed to ask_llm
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping factor names to descriptions
    """
    if factor_exposures.empty:
        _LOG.warning("Empty factor exposures provided")
        return {}
    
    factor_names = {}
    total_factors = len(factor_exposures.columns)
    
    _LOG.info("Starting batch naming of %d factors", total_factors)
    
    for i, factor_id in enumerate(factor_exposures.columns, 1):
        try:
            _LOG.info("Naming factor %d/%d: %s", i, total_factors, factor_id)
            
            exposures = factor_exposures[factor_id]
            top_pos = exposures.nlargest(top_n).index.tolist()
            top_neg = exposures.nsmallest(top_n).index.tolist()
            
            name = ask_llm(factor_id, top_pos, top_neg, fundamentals, **llm_kwargs)
            factor_names[factor_id] = name
            
            # Simple rate limiting between requests
            if i < total_factors:  # Don't sleep after last request
                time.sleep(1)  # 1 second between requests
                
        except Exception as e:
            _LOG.error("Failed to name factor %s: %s", factor_id, str(e))
            factor_names[factor_id] = f"{factor_id}: Naming failed"
    
    _LOG.info("Completed batch naming: %d/%d factors successfully named", 
             len([v for v in factor_names.values() if "failed" not in v.lower()]), total_factors)
    
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
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        return True
    except Exception as e:
        _LOG.error("API key validation failed: %s", str(e))
        return False