# Decision Synthesizer Design

## Problem

The equity-factors system generates sophisticated signals (regime detection, factor momentum, cross-sectional rankings, extreme value alerts) but lacks a "last mile" that translates these into actionable trading decisions. Users see "F3 is bullish" but don't know what to *do* with that information.

**Core pain points:**
- Signals feel too abstract ("F3 is bullish" vs. "buy these stocks")
- No clear triggers for when to act vs. wait
- Multiple signals sometimes conflict with no resolution

## Solution

A **Decision Synthesizer** module that sits above existing signal generators and produces:
1. **Morning Briefing** - daily structured summary of market state and recommendations
2. **Decision Framework** - conviction scoring and conflict resolution
3. **Actionable Recommendations** - concrete trade suggestions with sizing

## User Context

- **Trading frequency**: Weekly core rebalancing + opportunistic trades when signals are strong
- **Output format**: Morning briefing document + structured decision framework
- **Priority**: Regime dominates (risk management first, then alpha)

---

## Morning Briefing Structure

### 1. Market State Summary

```
═══════════════════════════════════════════════════════
MORNING BRIEFING - {date}
═══════════════════════════════════════════════════════

REGIME: {regime_name} ({confidence}% confidence, {days} days in current state)
TREND:  {trend_description}

FACTOR MOMENTUM (7-day):
  {for each factor: symbol + return + strength}

EXTREMES DETECTED: {list or "None today"}
```

### 2. Signal Alignment Score

A single 1-10 score showing whether signals agree or conflict:
- **9-10**: All signals pointing same direction → act with confidence
- **6-8**: Mostly aligned, minor conflicts → act with normal sizing
- **4-5**: Mixed signals → reduce position sizes or wait
- **1-3**: Signals contradicting → stay flat or defensive

### 3. Recommended Actions

Categorized by urgency:
- **OPPORTUNISTIC** - act today if possible
- **WEEKLY REBALANCE** - incorporate into next scheduled rebalance
- **WATCH** - not actionable yet, but monitor

---

## Decision Framework

### Conviction Scoring

Each potential action scored on three dimensions:

| Dimension | Source | Weight |
|-----------|--------|--------|
| Signal Strength | How extreme is the reading? (z-score > 1.5 = strong) | 40% |
| Signal Agreement | Do multiple signals confirm? (regime + momentum + cross-section) | 35% |
| Regime Fit | Does this action make sense in current regime? | 25% |

**Conviction = weighted sum → mapped to High / Medium / Low**

### Action Triggers

| Trigger Type | Condition | Action Category |
|--------------|-----------|-----------------|
| Regime Shift | New regime with >70% confidence, held for 2+ days | OPPORTUNISTIC |
| Extreme Reading | Any factor z-score > 2.0 (or < -2.0) | OPPORTUNISTIC |
| Factor Momentum Flip | Factor crosses from negative to positive (or vice versa) over 5-day MA | WEEKLY REBALANCE |
| Cross-Sectional Divergence | Top decile vs bottom decile spread > 1 std dev | WEEKLY REBALANCE |
| Gradual Drift | Slow changes accumulating over time | WATCH |

### Conflict Resolution Hierarchy

When signals disagree:
1. **Regime dominates** - if regime says "risk-off," don't chase momentum
2. **Confirmation required** - need 2 of 3 signal types agreeing to act
3. **Default to smaller size** - conflicting signals = half position

---

## Recommendation Output Format

### OPPORTUNISTIC Actions

```
┌─────────────────────────────────────────────────────────────────┐
│ ACTION: {action_description}                                    │
├─────────────────────────────────────────────────────────────────┤
│ Conviction: {HIGH|MEDIUM|LOW} ({score}/10)                      │
│                                                                 │
│ WHY:                                                            │
│  • Regime: {regime_rationale}                                   │
│  • Signal: {primary_signal_description}                         │
│  • Confirmation: {confirming_signals}                           │
│                                                                 │
│ CONFLICTS: {conflicts_or_none}                                  │
│                                                                 │
│ SUGGESTED EXPRESSION:                                           │
│  • Simple: {etf_trade}                                          │
│  • Targeted: {stock_list_with_sizes}                            │
│                                                                 │
│ SIZING LOGIC: {sizing_rationale}                                │
│ STOP CONSIDERATION: {exit_trigger}                              │
└─────────────────────────────────────────────────────────────────┘
```

### WATCH Items

```
┌─────────────────────────────────────────────────────────────────┐
│ WATCH: {observation_title}                                      │
├─────────────────────────────────────────────────────────────────┤
│ Conviction: LOW ({score}/10) - not actionable yet               │
│                                                                 │
│ OBSERVATION:                                                    │
│  • {observation_details}                                        │
│                                                                 │
│ TRIGGER TO UPGRADE: {conditions_for_actionability}              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Position Sizing

### Base Sizing by Conviction

| Conviction Level | Base Size | Modifier |
|------------------|-----------|----------|
| HIGH (8-10) | 5% of portfolio | Full size |
| MEDIUM (5-7) | 3% of portfolio | Reduced |
| LOW (3-4) | 1% of portfolio | Toe-hold only |
| CONFLICTED | 0% | Wait for clarity |

### Portfolio Risk Guardrails

```
PORTFOLIO RISK CHECK:
  Current factor exposures vs. limits:
  • {factor_name}: {current}% (limit: {limit}%) {status}

  If recommendation would breach limit:
  → Downgrade to WATCH or suggest reducing existing position first
```

---

## Implementation Architecture

### New Module: `src/decision_synthesizer.py`

```python
class DecisionSynthesizer:
    """Translates signals into actionable trading decisions."""

    def collect_all_signals(self, universe: str) -> SignalState:
        """Gather current state from all signal sources."""

    def score_conviction(self, signals: SignalState, action: Action) -> float:
        """Apply decision framework to score potential action."""

    def resolve_conflicts(self, signals: SignalState) -> ConflictResolution:
        """Apply hierarchy-based conflict resolution."""

    def generate_recommendations(self, signals: SignalState) -> List[Recommendation]:
        """Produce actionable items with conviction scores."""

    def render_briefing(self, recommendations: List[Recommendation]) -> str:
        """Format as morning briefing report."""
```

### New CLI Command

```bash
# Generate morning briefing
uv run python -m src briefing --universe VTHR

# With specific output format
uv run python -m src briefing --universe VTHR --format markdown
uv run python -m src briefing --universe VTHR --format json
```

### Dashboard Integration

Add "Morning Briefing" tab to Alpha Command Center:
- Live signal alignment score (big number, color-coded)
- Today's recommendations (sorted by conviction)
- Historical briefing accuracy tracking

---

## Tracking & Feedback Loop

To improve over time, track:
- **Hit rate**: Did HIGH conviction recommendations outperform?
- **Regime accuracy**: Were regime calls correct in hindsight?
- **Signal lead time**: How early did signals fire before moves?

This data feeds back into tuning conviction weights and trigger thresholds.

---

## Dependencies

Existing modules required:
- `regime_detection.py` - current regime state and confidence
- `trading_signals.py` - factor momentum, RSI, MACD, extremes
- `cross_sectional.py` - stock rankings by factor exposure
- `signal_aggregator.py` - meta-model predictions
- `latent_factors.py` - factor definitions and loadings

---

## Open Questions

1. **Historical tracking storage**: SQLite table or separate file?
2. **Notification system**: Email/Slack integration for OPPORTUNISTIC alerts?
3. **Paper trading mode**: Simulated execution tracking before live use?
