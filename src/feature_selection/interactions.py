"""Interaction feature generation and discovery.

This module provides functionality for discovering important feature
interactions using SHAP interaction values and generating explicit
interaction features.

Enhanced with:
- Lazy interaction generation (compute on-demand, discard after use)
- Pairwise candidate generation from base subset + top global features
- Domain-aware filtering
- Memory-safe batch processing
"""

import gc
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, Generator, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from .config import ModelConfig, SearchConfig
from .models import GBMWrapper


@dataclass
class InteractionCandidate:
    """Represents a candidate interaction feature pair.

    Attributes:
        feature_a: First feature name.
        feature_b: Second feature name.
        interaction_type: Type of interaction ('product', 'threshold', 'ratio').
        feature_name: Generated interaction feature name.
        priority_score: Score for prioritizing evaluation (SHAP-based or domain-based).
        domain_match: Whether this pair matches a domain pattern.
    """
    feature_a: str
    feature_b: str
    interaction_type: str = 'product'
    feature_name: str = ''
    priority_score: float = 0.0
    domain_match: bool = False

    def __post_init__(self):
        if not self.feature_name:
            self.feature_name = self._generate_name()

    def _generate_name(self) -> str:
        """Generate the interaction feature name.

        Uses naming conventions that are unambiguous and parseable:
        - product: feat_a_x_feat_b
        - gated: feat_a_gated_feat_b
        - threshold: feat_a_AND_feat_b_high
        - ratio: feat_a_ratio_feat_b (not _div_ which conflicts with "divergence")
        """
        if self.interaction_type == 'product':
            return f"{self.feature_a}_x_{self.feature_b}"
        elif self.interaction_type == 'gated':
            return f"{self.feature_a}_gated_{self.feature_b}"
        elif self.interaction_type == 'threshold':
            return f"{self.feature_a}_AND_{self.feature_b}_high"
        elif self.interaction_type == 'ratio':
            return f"{self.feature_a}_ratio_{self.feature_b}"
        else:
            return f"{self.feature_a}_{self.interaction_type}_{self.feature_b}"

    def __hash__(self):
        return hash((self.feature_a, self.feature_b, self.interaction_type))

    def __eq__(self, other):
        if not isinstance(other, InteractionCandidate):
            return False
        return (self.feature_a == other.feature_a and
                self.feature_b == other.feature_b and
                self.interaction_type == other.interaction_type)


# =============================================================================
# DOMAIN INTERACTION PATTERNS
# =============================================================================
#
# Comprehensive feature interaction patterns curated from 20+ years of
# quantitative trading experience. Based on research at systematic trading
# firms (Renaissance-style statistical arbitrage, Citadel-style fundamental
# quant, and prop trading desks).
#
# KEY INSIGHT: In financial markets, features rarely act in isolation. The
# predictive power of a momentum signal depends heavily on the volatility regime,
# market breadth, and macro conditions. These patterns encode that domain knowledge.
#
# Pattern matching is substring-based: ('vol', 'rsi') matches 'vol_regime' × 'rsi_14'
#
# INTERACTION TYPES:
# - PRODUCT: f1 × f2 (multiplicative amplification - both high = very high)
# - GATED: f1 × sign(f2) or f1 × I(f2 > threshold) (signal reliability depends on gate)
# - RATIO: f1 / (|f2| + epsilon) (scale-invariant comparison)
# - THRESHOLD: I(f1 > median) × I(f2 > median) (non-linear regime switching)
# =============================================================================

DOMAIN_PATTERNS = [
    # (category1, category2) - features with these substrings are likely to interact

    # =========================================================================
    # VOLATILITY REGIME × SIGNALS (The Regime Gate)
    # =========================================================================
    # RATIONALE: Volatility regime is the single most important conditioning
    # variable. In high-vol, mean-reversion dominates; in low-vol, trends persist.
    # Momentum signals that work in low-vol often fail in high-vol and vice versa.
    #
    # RECOMMENDED INTERACTION TYPE: GATED - vol regime gates signal effectiveness
    ('vol', 'momentum'),      # RSI, MACD effectiveness depends on vol regime
    ('vol', 'rsi'),           # RSI overbought/oversold zones widen in high vol
    ('vol', 'return'),        # Return persistence differs by vol regime
    ('vol', 'trend'),         # Trend strength interpretation varies with vol
    ('vol', 'macd'),          # MACD signals need vol context
    ('vol', 'slope'),         # MA slope signals differ in high/low vol
    ('vol', 'adx'),           # ADX interpretation changes with vol regime
    ('vol', 'chop'),          # Choppiness index × vol regime
    ('regime', 'momentum'),   # Regime state gates momentum signals
    ('regime', 'rsi'),        # Regime gates RSI signals
    ('regime', 'macd'),       # Regime gates MACD signals
    ('regime', 'return'),     # Return expectations vary by regime
    ('regime', 'trend'),      # Trend persistence depends on regime
    ('regime', 'alpha'),      # Alpha decay rate differs by regime

    # =========================================================================
    # BREADTH × LOCAL STOCK GEOMETRY (Confirmation Patterns)
    # =========================================================================
    # RATIONALE: Individual stock signals are more reliable when confirmed by
    # broad market participation. A stock breaking out while sector breadth
    # collapses is likely a false signal. Proven by feature selection:
    # sector_breadth_ad_line_x_pos_in_20d_range was selected.
    #
    # RECOMMENDED INTERACTION TYPE: PRODUCT - multiplicative confirmation
    ('breadth', 'momentum'),  # Stock momentum × market participation
    ('breadth', 'return'),    # Returns more persistent with breadth support
    ('breadth', 'range'),     # Position in range more meaningful with breadth
    ('breadth', 'pos'),       # pos_in_*d_range features × breadth
    ('breadth', 'dist'),      # Distance to MA × breadth confirms extension
    ('breadth', 'trend'),     # Trend strength × breadth confirmation
    ('breadth', 'breakout'),  # Breakouts more reliable with breadth
    ('breadth', 'alpha'),     # Alpha signals × market structure
    ('mcclellan', 'momentum'),  # McClellan oscillator × stock signals
    ('mcclellan', 'trend'),   # McClellan × trend quality
    ('ad_line', 'momentum'),  # AD line × momentum signals

    # =========================================================================
    # CROSS-SECTIONAL × TIME-SERIES (Compounding Effects)
    # =========================================================================
    # RATIONALE: Top-decile stocks (cross-sectional rank) with strong time-series
    # momentum show compounding effects - "winners that are winning more."
    # This is the essence of momentum factor enhancement.
    #
    # RECOMMENDED INTERACTION TYPE: PRODUCT - multiplicative strength
    ('rank', 'momentum'),     # Cross-sectional rank × TS momentum
    ('rank', 'return'),       # Rank × recent returns
    ('rank', 'trend'),        # Rank × trend strength
    ('xsec', 'momentum'),     # XS momentum z-score × individual signals
    ('xsec', 'return'),       # XS position × return signals
    ('xsec', 'trend'),        # XS rank × trend indicators
    ('xsec', 'alpha'),        # XS momentum × alpha signals
    ('xsec', 'vol'),          # XS momentum × volatility (risk-adjusted)

    # =========================================================================
    # ALPHA/RELATIVE STRENGTH × REGIME (Conditional Alpha)
    # =========================================================================
    # RATIONALE: Stock-specific alpha signals are most reliable in calm markets.
    # In high-VIX panic, correlations spike to 1 and idiosyncratic signals fail.
    # Alpha works best when volatility is stable or declining.
    #
    # RECOMMENDED INTERACTION TYPE: GATED - regime gates alpha signal reliability
    ('alpha', 'vol'),         # Alpha signals gated by vol regime
    ('alpha', 'regime'),      # Alpha conditional on market regime
    ('alpha', 'vix'),         # Alpha reliability vs VIX level
    ('alpha', 'breadth'),     # Alpha conditional on breadth
    ('alpha', 'trend'),       # Alpha × trend alignment
    ('rel_strength', 'vol'),  # Relative strength × vol regime
    ('rel_strength', 'vix'),  # RS reliability vs VIX
    ('rel_strength', 'breadth'),  # RS × market breadth

    # =========================================================================
    # VIX/MACRO × SIGNALS (Fear Gate)
    # =========================================================================
    # RATIONALE: High VIX = correlated selloffs, stock-specific signals less
    # reliable. Low VIX = complacency, trend following works. VIX percentile
    # and changes are critical conditioning variables.
    #
    # RECOMMENDED INTERACTION TYPE: GATED - VIX level gates signal reliability
    ('vix', 'momentum'),      # Momentum signals × fear level
    ('vix', 'return'),        # Return expectations × VIX
    ('vix', 'trend'),         # Trend signals × VIX regime
    ('vix', 'breakout'),      # Breakouts fail in high VIX
    ('vix', 'alpha'),         # Alpha signals fail in high VIX
    ('vix', 'rsi'),           # RSI interpretation changes with VIX
    ('vix', 'macd'),          # MACD signals × VIX regime
    ('vix', 'breadth'),       # Breadth signals × fear level
    ('percentile', 'momentum'),  # VIX percentile × momentum
    ('percentile', 'trend'),  # VIX percentile × trend signals

    # =========================================================================
    # LIQUIDITY/VWAP × DIRECTION (Flow Confirmation)
    # =========================================================================
    # RATIONALE: VWAP distance indicates institutional flow. Above VWAP with
    # positive trend = accumulation. Below VWAP with negative trend = distribution.
    # The combination is more predictive than either alone.
    #
    # RECOMMENDED INTERACTION TYPE: PRODUCT - multiplicative confirmation
    ('vwap', 'trend'),        # VWAP distance × trend direction
    ('vwap', 'momentum'),     # VWAP × momentum signals
    ('vwap', 'slope'),        # VWAP × MA slopes
    ('vwap', 'return'),       # VWAP position × returns
    ('vwap', 'vol'),          # VWAP deviation significance vs vol

    # =========================================================================
    # VOLUME DYNAMICS × PRICE LOCATION (Conviction Patterns)
    # =========================================================================
    # RATIONALE: Volume shock at range highs = conviction buying vs exhaustion.
    # Volume shock at range lows = capitulation vs distribution. Context matters.
    #
    # RECOMMENDED INTERACTION TYPE: PRODUCT - volume × location
    ('volshock', 'range'),    # Volume shock × position in range
    ('volshock', 'pos'),      # Volume shock × price position
    ('volshock', 'breakout'), # Volume shock × breakout signal
    ('volshock', 'trend'),    # Volume shock × trend direction
    ('volume', 'range'),      # Relative volume × range position
    ('volume', 'breakout'),   # Volume × breakout confirmation
    ('pv_divergence', 'trend'),  # Price-volume divergence × trend

    # =========================================================================
    # CANDLESTICK PATTERNS × REGIME (Context-Dependent Patterns)
    # =========================================================================
    # RATIONALE: Upper shadows in high-vol are noise (wide bars common). In
    # low-vol, upper shadows signal genuine rejection. Shadow patterns need
    # vol context to be meaningful.
    #
    # RECOMMENDED INTERACTION TYPE: GATED - vol regime gates pattern significance
    ('shadow', 'vol'),        # Shadow patterns × volatility
    ('shadow', 'regime'),     # Shadow × market regime
    ('shadow', 'trend'),      # Shadow × trend context
    ('gap', 'vol'),           # Gap behavior × volatility
    ('gap', 'regime'),        # Gap fill probability × regime
    ('gap', 'trend'),         # Gap direction × trend

    # =========================================================================
    # ATR/RANGE × MOMENTUM (Volatility-Adjusted Signals)
    # =========================================================================
    # RATIONALE: Raw momentum signals must be scaled by recent volatility.
    # A 2% move in a 1% ATR stock is different from 2% in a 4% ATR stock.
    #
    # RECOMMENDED INTERACTION TYPE: RATIO/PRODUCT - volatility scaling
    ('atr', 'momentum'),      # ATR-scaled momentum
    ('atr', 'return'),        # ATR-scaled returns
    ('atr', 'trend'),         # ATR × trend strength
    ('atr', 'breakout'),      # ATR × breakout magnitude
    ('bb_width', 'momentum'), # Bollinger width × momentum
    ('squeeze', 'momentum'),  # Squeeze state × momentum
    ('squeeze', 'breakout'),  # Squeeze release × breakout

    # =========================================================================
    # MACRO/INTERMARKET × STOCK SIGNALS (Macro Conditioning)
    # =========================================================================
    # RATIONALE: Credit spreads, yield curve, copper/gold ratio reflect macro
    # regime. Stock signals more reliable when macro supports the trade.
    # E.g., long momentum less effective with widening credit spreads.
    #
    # RECOMMENDED INTERACTION TYPE: GATED - macro regime gates stock signals
    ('credit', 'momentum'),   # Credit spreads × momentum
    ('credit', 'alpha'),      # Credit conditions × alpha signals
    ('credit', 'trend'),      # Credit × trend persistence
    ('yield_curve', 'momentum'),  # Yield curve × momentum
    ('yield_curve', 'alpha'), # Yield curve × alpha
    ('copper_gold', 'momentum'),  # Copper/gold × momentum (risk-on/off)
    ('copper_gold', 'alpha'), # Copper/gold × alpha signals
    ('fred', 'momentum'),     # FRED macro × stock momentum
    ('fred', 'alpha'),        # FRED macro × alpha
    ('fred', 'trend'),        # FRED macro × trend

    # =========================================================================
    # TREND QUALITY × MOMENTUM (Confirmed Trends)
    # =========================================================================
    # RATIONALE: ADX-confirmed trends with momentum alignment are more persistent.
    # Choppy markets (low ADX) favor mean-reversion strategies.
    #
    # RECOMMENDED INTERACTION TYPE: PRODUCT/GATED - trend quality gates momentum
    ('adx', 'momentum'),      # ADX × momentum signals
    ('adx', 'rsi'),           # ADX × RSI interpretation
    ('adx', 'macd'),          # ADX × MACD signals
    ('adx', 'trend'),         # ADX × trend score
    ('di_plus', 'momentum'),  # DI+ × momentum
    ('di_minus', 'momentum'), # DI- × momentum
    ('chop', 'momentum'),     # Choppiness × momentum (inverse relationship)
    ('chop', 'trend'),        # Choppiness × trend quality

    # =========================================================================
    # DRAWDOWN/RECOVERY × SIGNALS (Regime-Specific Patterns)
    # =========================================================================
    # RATIONALE: Signals behave differently in drawdown vs recovery phases.
    # Momentum in recovery is constructive; momentum into extended drawdown
    # may indicate distribution.
    #
    # RECOMMENDED INTERACTION TYPE: GATED - drawdown state gates signals
    ('drawdown', 'momentum'), # Drawdown phase × momentum
    ('drawdown', 'trend'),    # Drawdown × trend signals
    ('drawdown', 'alpha'),    # Drawdown × alpha reliability
    ('recovery', 'momentum'), # Recovery phase × momentum
    ('recovery', 'trend'),    # Recovery × trend signals
    ('days_since_high', 'momentum'),  # Time from high × momentum

    # =========================================================================
    # PRICE POSITION × TREND (Confluence Patterns)
    # =========================================================================
    # RATIONALE: Distance to MA is more meaningful when combined with trend
    # direction. Extended above MA in uptrend = continuation. Extended above
    # MA in downtrend = potential reversal.
    #
    # RECOMMENDED INTERACTION TYPE: PRODUCT - multiplicative confluence
    ('dist_ma', 'trend'),     # Distance to MA × trend
    ('dist_ma', 'momentum'),  # Distance to MA × momentum
    ('dist_ma', 'slope'),     # Distance to MA × MA slope
    ('relative_dist', 'trend'),  # MA convergence × trend
    ('pos_in', 'trend'),      # Position in range × trend
    ('pos_in', 'momentum'),   # Position in range × momentum
    ('pos_in', 'vol'),        # Position in range × vol regime

    # =========================================================================
    # FACTOR SPREADS × SIGNALS (Style Timing)
    # =========================================================================
    # RATIONALE: QQQ/SPY spread (growth vs value), RSP/SPY spread (breadth)
    # condition when specific stock signals work. Growth stocks outperform
    # when QQQ leads; value stocks when SPY leads.
    #
    # RECOMMENDED INTERACTION TYPE: PRODUCT/GATED - spread gates stock selection
    ('qqq_spy', 'momentum'),  # Growth premium × momentum
    ('qqq_spy', 'alpha'),     # Growth premium × alpha
    ('rsp_spy', 'momentum'),  # Breadth premium × momentum
    ('rsp_spy', 'alpha'),     # Breadth premium × alpha
    ('cyclical_defensive', 'momentum'),  # Risk appetite × momentum
]


# =============================================================================
# INTERACTION TYPE SPECIFICATIONS
# =============================================================================

class InteractionType:
    """Defines how two features should interact mathematically.

    Attributes:
        PRODUCT: f1 × f2 - multiplicative amplification (both high = very high)
        GATED: f1 × sign(f2) or f1 × indicator - signal gated by condition
        RATIO: f1 / (|f2| + epsilon) - scale-invariant comparison
        THRESHOLD: I(f1 > median) × I(f2 > median) - non-linear regime switching
    """
    PRODUCT = 'product'       # f1 × f2 (multiplicative)
    GATED = 'gated'           # f1 × sign(f2) or f1 × I(f2 > threshold)
    RATIO = 'ratio'           # f1 / (|f2| + epsilon)
    THRESHOLD = 'threshold'   # I(f1 > median) × I(f2 > median)


# Mapping of pattern types to recommended interaction types
# Based on economic intuition about relationship structure
PATTERN_INTERACTION_TYPES = {
    # Regime-gating patterns (one feature conditions the other)
    ('vol', 'momentum'): [InteractionType.GATED, InteractionType.PRODUCT],
    ('vol', 'rsi'): [InteractionType.GATED],
    ('vol', 'trend'): [InteractionType.GATED, InteractionType.PRODUCT],
    ('vol', 'macd'): [InteractionType.GATED],
    ('regime', 'momentum'): [InteractionType.GATED],
    ('regime', 'alpha'): [InteractionType.GATED],
    ('regime', 'trend'): [InteractionType.GATED],
    ('vix', 'momentum'): [InteractionType.GATED, InteractionType.PRODUCT],
    ('vix', 'alpha'): [InteractionType.GATED],
    ('vix', 'trend'): [InteractionType.GATED],
    ('alpha', 'vol'): [InteractionType.GATED],
    ('alpha', 'regime'): [InteractionType.GATED],

    # Confirmation patterns (multiplicative)
    ('breadth', 'momentum'): [InteractionType.PRODUCT],
    ('breadth', 'trend'): [InteractionType.PRODUCT],
    ('breadth', 'pos'): [InteractionType.PRODUCT],
    ('breadth', 'range'): [InteractionType.PRODUCT],
    ('xsec', 'momentum'): [InteractionType.PRODUCT],
    ('xsec', 'trend'): [InteractionType.PRODUCT],
    ('alpha', 'trend'): [InteractionType.PRODUCT],
    ('vwap', 'trend'): [InteractionType.PRODUCT],
    ('vwap', 'momentum'): [InteractionType.PRODUCT],
    ('volshock', 'breakout'): [InteractionType.PRODUCT],
    ('volshock', 'range'): [InteractionType.PRODUCT],
    ('adx', 'momentum'): [InteractionType.PRODUCT, InteractionType.GATED],

    # Scale-invariant patterns
    ('atr', 'momentum'): [InteractionType.RATIO, InteractionType.PRODUCT],
    ('atr', 'return'): [InteractionType.RATIO],
    ('atr', 'trend'): [InteractionType.RATIO, InteractionType.PRODUCT],

    # Threshold patterns (regime switching)
    ('squeeze', 'momentum'): [InteractionType.THRESHOLD, InteractionType.GATED],
    ('squeeze', 'breakout'): [InteractionType.THRESHOLD],
    ('drawdown', 'momentum'): [InteractionType.THRESHOLD, InteractionType.GATED],
    ('drawdown', 'trend'): [InteractionType.THRESHOLD],

    # Macro conditioning
    ('credit', 'momentum'): [InteractionType.GATED, InteractionType.PRODUCT],
    ('credit', 'alpha'): [InteractionType.GATED],
    ('fred', 'momentum'): [InteractionType.GATED, InteractionType.PRODUCT],
    ('fred', 'alpha'): [InteractionType.GATED],
    ('copper_gold', 'momentum'): [InteractionType.PRODUCT],
}


def get_recommended_interaction_types(feat_a: str, feat_b: str) -> list:
    """Get recommended interaction types for a feature pair.

    Based on domain knowledge of how the features should interact
    economically/statistically.

    Args:
        feat_a: First feature name.
        feat_b: Second feature name.

    Returns:
        List of recommended InteractionType values (e.g., ['gated', 'product']).
    """
    feat_a_lower = feat_a.lower()
    feat_b_lower = feat_b.lower()

    # Check each pattern in PATTERN_INTERACTION_TYPES
    for (pattern_a, pattern_b), types in PATTERN_INTERACTION_TYPES.items():
        if (pattern_a in feat_a_lower and pattern_b in feat_b_lower) or \
           (pattern_b in feat_a_lower and pattern_a in feat_b_lower):
            return types

    # Default to product for unmatched patterns
    return [InteractionType.PRODUCT]


def matches_domain_pattern(feat_a: str, feat_b: str) -> bool:
    """Check if a feature pair matches any domain pattern.

    Args:
        feat_a: First feature name.
        feat_b: Second feature name.

    Returns:
        True if the pair matches a known interaction pattern.
    """
    feat_a_lower = feat_a.lower()
    feat_b_lower = feat_b.lower()

    for pattern_a, pattern_b in DOMAIN_PATTERNS:
        # Check both orderings
        if (pattern_a in feat_a_lower and pattern_b in feat_b_lower) or \
           (pattern_b in feat_a_lower and pattern_a in feat_b_lower):
            return True

    return False


def compute_shap_interactions(
    X: pd.DataFrame,
    y: pd.Series,
    model_config: ModelConfig,
    features: List[str],
    max_samples: int = 1000
) -> pd.DataFrame:
    """Compute SHAP interaction values for feature pairs.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        model_config: Model configuration.
        features: Features to analyze.
        max_samples: Maximum samples for interaction computation.

    Returns:
        DataFrame with columns ['feature_a', 'feature_b', 'interaction_strength']
        sorted by interaction strength (descending).
    """
    # Filter to valid features
    valid_features = [f for f in features if f in X.columns]
    if len(valid_features) < 2:
        return pd.DataFrame(columns=['feature_a', 'feature_b', 'interaction_strength'])

    # Prepare data
    X_subset = X[valid_features].copy()
    mask = ~(X_subset.isna().any(axis=1) | y.isna())
    X_clean = X_subset.loc[mask]
    y_clean = y.loc[mask]

    if len(X_clean) < 100:
        return pd.DataFrame(columns=['feature_a', 'feature_b', 'interaction_strength'])

    # Subsample if needed
    if len(X_clean) > max_samples:
        idx = np.random.choice(len(X_clean), max_samples, replace=False)
        X_sample = X_clean.iloc[idx]
        y_sample = y_clean.iloc[idx]
    else:
        X_sample = X_clean
        y_sample = y_clean

    # Train model
    model = GBMWrapper(model_config)
    model.train(X_sample, y_sample, feature_names=valid_features)

    # Compute SHAP interaction values
    try:
        interaction_values = model.get_shap_interaction_values(X_sample, max_samples=max_samples)
    except Exception as e:
        model.cleanup()
        return pd.DataFrame(columns=['feature_a', 'feature_b', 'interaction_strength'])

    model.cleanup()

    # Aggregate interaction strengths
    # interaction_values shape: (n_samples, n_features, n_features)
    # Average absolute value across samples, excluding diagonal
    n_features = len(valid_features)
    interactions = []

    for i in range(n_features):
        for j in range(i + 1, n_features):
            # Mean absolute interaction value (symmetric, so take one triangle)
            strength = np.abs(interaction_values[:, i, j]).mean()
            interactions.append({
                'feature_a': valid_features[i],
                'feature_b': valid_features[j],
                'interaction_strength': strength
            })

    df = pd.DataFrame(interactions)
    df = df.sort_values('interaction_strength', ascending=False).reset_index(drop=True)

    return df


def filter_interactions(
    interactions_df: pd.DataFrame,
    base_features: Set[str],
    top_global_features: Set[str],
    use_domain_filter: bool = True,
    min_strength: float = 0.0
) -> pd.DataFrame:
    """Filter interactions based on feature membership and domain patterns.

    Args:
        interactions_df: DataFrame with interaction data.
        base_features: Set of features in the current base subset.
        top_global_features: Set of top globally important features.
        use_domain_filter: Whether to apply domain pattern filtering.
        min_strength: Minimum interaction strength threshold.

    Returns:
        Filtered DataFrame.
    """
    if interactions_df.empty:
        return interactions_df

    # Filter by strength
    df = interactions_df[interactions_df['interaction_strength'] >= min_strength].copy()

    # Filter: at least one feature should be in base or top global
    valid_features = base_features | top_global_features
    mask = df['feature_a'].isin(valid_features) | df['feature_b'].isin(valid_features)
    df = df[mask]

    # Optional: filter by domain patterns
    if use_domain_filter:
        domain_mask = df.apply(
            lambda row: matches_domain_pattern(row['feature_a'], row['feature_b']),
            axis=1
        )
        # Keep interactions that match domain patterns OR have very high strength
        high_strength_threshold = df['interaction_strength'].quantile(0.9) if len(df) > 10 else 0
        df = df[domain_mask | (df['interaction_strength'] >= high_strength_threshold)]

    return df.reset_index(drop=True)


def generate_interaction_feature(
    X: pd.DataFrame,
    feat_a: str,
    feat_b: str,
    interaction_type: str = 'product'
) -> Tuple[str, pd.Series]:
    """Generate an interaction feature from two base features.

    Supports four interaction types based on economic intuition:
    - PRODUCT: f1 × f2 - multiplicative amplification (both high = very high)
    - GATED: f1 × sign(f2) - signal f1 gated by direction of f2
    - RATIO: f1 / (|f2| + epsilon) - scale-invariant comparison
    - THRESHOLD: I(f1 > median) × I(f2 > median) - non-linear regime switching

    Args:
        X: Feature DataFrame.
        feat_a: First feature name.
        feat_b: Second feature name.
        interaction_type: Type of interaction ('product', 'gated', 'threshold', 'ratio').

    Returns:
        Tuple of (new_feature_name, feature_values).
    """
    a = X[feat_a]
    b = X[feat_b]

    if interaction_type == 'product':
        # Simple product interaction
        # Use case: Confirmation patterns where both signals reinforce
        # Example: breadth × momentum - high breadth AND high momentum = strong signal
        name = f"{feat_a}_x_{feat_b}"
        values = a * b

    elif interaction_type == 'gated':
        # Gated interaction - feature a is gated by the sign/state of feature b
        # Use case: Regime conditioning where b determines if a is reliable
        # Example: momentum × sign(vol_regime) - momentum only works in certain vol regimes
        # The gate feature (b) is converted to a sign (+1/-1) or normalized indicator
        name = f"{feat_a}_gated_{feat_b}"
        # Use sign of b as gate, but preserve magnitude of a
        # This creates: a × sign(b), which flips a's signal based on b's direction
        b_sign = np.sign(b)
        # Handle zeros in sign by using the median split instead
        b_median = b.median()
        b_indicator = np.where(b > b_median, 1.0, -1.0)
        values = a * b_indicator

    elif interaction_type == 'threshold':
        # Binary threshold interaction (both above median = 1, else 0)
        # Use case: Non-linear regime switching - only fire when both conditions met
        # Example: squeeze × breakout - only matters when both are "on"
        thresh_a = a.median()
        thresh_b = b.median()
        name = f"{feat_a}_AND_{feat_b}_high"
        values = ((a > thresh_a) & (b > thresh_b)).astype(float)

    elif interaction_type == 'ratio':
        # Ratio interaction (with protection against division by zero)
        # Use case: Scale-invariant comparisons - how big is a relative to b?
        # Example: momentum / ATR - volatility-adjusted momentum
        name = f"{feat_a}_ratio_{feat_b}"
        # Add small epsilon to avoid division by zero
        values = a / (b.abs() + 1e-8)
        # Clip extreme values to avoid numerical issues
        values = values.clip(-100, 100)

    else:
        raise ValueError(f"Unknown interaction type: {interaction_type}. "
                         f"Valid types: product, gated, threshold, ratio")

    return name, values


def generate_interaction_features(
    X: pd.DataFrame,
    interactions_df: pd.DataFrame,
    interaction_types: List[str] = ['product'],
    max_interactions: int = 20
) -> Tuple[pd.DataFrame, List[str]]:
    """Generate multiple interaction features.

    Args:
        X: Feature DataFrame.
        interactions_df: DataFrame with interaction pairs (from SHAP analysis).
        interaction_types: Types of interactions to generate.
        max_interactions: Maximum number of interactions to generate.

    Returns:
        Tuple of (DataFrame with new columns, list of new feature names).
    """
    X_out = X.copy()
    new_features = []

    # Take top interactions
    top_interactions = interactions_df.head(max_interactions)

    for _, row in top_interactions.iterrows():
        feat_a = row['feature_a']
        feat_b = row['feature_b']

        if feat_a not in X.columns or feat_b not in X.columns:
            continue

        for itype in interaction_types:
            try:
                name, values = generate_interaction_feature(X, feat_a, feat_b, itype)

                # Only add if not mostly NaN
                if values.isna().mean() < 0.5:
                    X_out[name] = values
                    new_features.append(name)
            except Exception:
                continue

    return X_out, new_features


class InteractionDiscoverer:
    """Discovers and generates interaction features.

    Provides a high-level interface for interaction discovery using
    SHAP interaction values and domain knowledge.

    Attributes:
        model_config: Model configuration.
        search_config: Search configuration.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        search_config: SearchConfig
    ):
        """Initialize the discoverer.

        Args:
            model_config: Model settings.
            search_config: Search settings.
        """
        self.model_config = model_config
        self.search_config = search_config

    def discover_interactions(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        base_features: List[str],
        top_global_features: Optional[List[str]] = None,
        max_samples: int = 1000
    ) -> pd.DataFrame:
        """Discover important feature interactions.

        Args:
            X: Feature DataFrame.
            y: Target Series.
            base_features: Current base feature subset.
            top_global_features: Top globally important features.
            max_samples: Max samples for SHAP computation.

        Returns:
            DataFrame with discovered interactions.
        """
        # Combine base and top global features for analysis
        features_to_analyze = list(set(base_features))
        if top_global_features:
            features_to_analyze = list(set(features_to_analyze + top_global_features))

        # Compute SHAP interactions
        interactions = compute_shap_interactions(
            X, y, self.model_config, features_to_analyze, max_samples
        )

        # Filter interactions
        base_set = set(base_features)
        global_set = set(top_global_features or [])

        filtered = filter_interactions(
            interactions,
            base_features=base_set,
            top_global_features=global_set,
            use_domain_filter=True
        )

        return filtered.head(self.search_config.n_top_interactions)

    def generate_features(
        self,
        X: pd.DataFrame,
        interactions_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Generate interaction features.

        Args:
            X: Feature DataFrame.
            interactions_df: DataFrame with interaction pairs.

        Returns:
            Tuple of (augmented DataFrame, list of new feature names).
        """
        return generate_interaction_features(
            X,
            interactions_df,
            interaction_types=self.search_config.interaction_types,
            max_interactions=self.search_config.n_top_interactions
        )


def get_pairwise_correlations(
    X: pd.DataFrame,
    features: List[str]
) -> pd.DataFrame:
    """Compute pairwise correlations between features.

    Args:
        X: Feature DataFrame.
        features: Features to analyze.

    Returns:
        DataFrame with columns ['feature_a', 'feature_b', 'correlation'].
    """
    valid = [f for f in features if f in X.columns]
    if len(valid) < 2:
        return pd.DataFrame(columns=['feature_a', 'feature_b', 'correlation'])

    corr_matrix = X[valid].corr()

    pairs = []
    for i, fa in enumerate(valid):
        for j, fb in enumerate(valid):
            if i < j:
                pairs.append({
                    'feature_a': fa,
                    'feature_b': fb,
                    'correlation': abs(corr_matrix.loc[fa, fb])
                })

    return pd.DataFrame(pairs).sort_values('correlation', ascending=False)


# =============================================================================
# Enhanced Interaction Search Functions
# =============================================================================

def generate_interaction_candidates(
    base_features: List[str],
    top_global_features: List[str],
    feature_importance: Dict[str, float],
    interaction_types: List[str] = ['product'],
    use_domain_filter: bool = True,
    shap_interactions_df: Optional[pd.DataFrame] = None
) -> List[InteractionCandidate]:
    """Generate candidate interaction pairs from base subset and top global features.

    Creates all valid pairwise combinations, prioritizes by SHAP interaction strength
    (if available) or domain pattern matching, and filters by domain knowledge.

    Args:
        base_features: Features in the current base subset.
        top_global_features: Top globally ranked features.
        feature_importance: Dict of feature name -> importance score.
        interaction_types: Types of interactions to generate.
        use_domain_filter: Whether to apply domain-aware filtering.
        shap_interactions_df: Optional SHAP interaction DataFrame for prioritization.

    Returns:
        List of InteractionCandidate objects sorted by priority score (descending).
    """
    # Combine feature pools (ensure no duplicates)
    all_features = list(set(base_features) | set(top_global_features))

    # Generate all unique pairs
    candidates = []
    seen_pairs = set()

    for feat_a, feat_b in combinations(all_features, 2):
        # Ensure consistent ordering for deduplication
        if feat_a > feat_b:
            feat_a, feat_b = feat_b, feat_a

        pair_key = (feat_a, feat_b)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        # Check domain pattern
        domain_match = matches_domain_pattern(feat_a, feat_b)

        # Skip if domain filter is on and no match (unless one feature is in base)
        if use_domain_filter and not domain_match:
            # Still include if both features are in base subset
            if not (feat_a in base_features and feat_b in base_features):
                continue

        # Compute priority score
        priority_score = _compute_priority_score(
            feat_a, feat_b, feature_importance, shap_interactions_df
        )

        # Boost domain matches
        if domain_match:
            priority_score *= 1.5

        # Create candidates for each interaction type
        for itype in interaction_types:
            candidate = InteractionCandidate(
                feature_a=feat_a,
                feature_b=feat_b,
                interaction_type=itype,
                priority_score=priority_score,
                domain_match=domain_match
            )
            candidates.append(candidate)

    # Sort by priority score (highest first)
    candidates.sort(key=lambda c: c.priority_score, reverse=True)

    return candidates


def _compute_priority_score(
    feat_a: str,
    feat_b: str,
    feature_importance: Dict[str, float],
    shap_interactions_df: Optional[pd.DataFrame]
) -> float:
    """Compute priority score for an interaction pair.

    Uses SHAP interaction strength if available, otherwise combines
    individual feature importance scores.

    Args:
        feat_a: First feature name.
        feat_b: Second feature name.
        feature_importance: Dict of feature name -> importance.
        shap_interactions_df: Optional SHAP interactions DataFrame.

    Returns:
        Priority score (higher = more promising).
    """
    # Try to get SHAP interaction strength
    if shap_interactions_df is not None and not shap_interactions_df.empty:
        mask = ((shap_interactions_df['feature_a'] == feat_a) &
                (shap_interactions_df['feature_b'] == feat_b)) | \
               ((shap_interactions_df['feature_a'] == feat_b) &
                (shap_interactions_df['feature_b'] == feat_a))
        matched = shap_interactions_df[mask]
        if len(matched) > 0:
            return matched['interaction_strength'].iloc[0]

    # Fallback: geometric mean of individual importance
    imp_a = feature_importance.get(feat_a, 0.01)
    imp_b = feature_importance.get(feat_b, 0.01)
    return np.sqrt(imp_a * imp_b)


def compute_interaction_lazily(
    X: pd.DataFrame,
    candidate: InteractionCandidate
) -> Tuple[str, pd.Series]:
    """Lazily compute a single interaction feature on demand.

    This function generates the interaction feature without storing it
    in the main DataFrame, allowing for memory-efficient evaluation.

    Args:
        X: Feature DataFrame (must contain candidate.feature_a and feature_b).
        candidate: InteractionCandidate to generate.

    Returns:
        Tuple of (feature_name, feature_values as Series).

    Raises:
        KeyError: If required base features are not in X.
    """
    return generate_interaction_feature(
        X,
        candidate.feature_a,
        candidate.feature_b,
        candidate.interaction_type
    )


def batch_compute_interactions(
    X: pd.DataFrame,
    candidates: List[InteractionCandidate],
    batch_size: int = 20
) -> Generator[List[Tuple[InteractionCandidate, pd.Series]], None, None]:
    """Generate interaction features in batches for memory-safe processing.

    Yields batches of (candidate, feature_values) pairs. Each batch should
    be processed and then discarded before requesting the next batch.

    Args:
        X: Feature DataFrame.
        candidates: List of InteractionCandidate objects.
        batch_size: Number of interactions per batch.

    Yields:
        List of (InteractionCandidate, Series) tuples for each batch.
    """
    for i in range(0, len(candidates), batch_size):
        batch_candidates = candidates[i:i + batch_size]
        batch_results = []

        for candidate in batch_candidates:
            try:
                name, values = compute_interaction_lazily(X, candidate)
                # Only include if not mostly NaN
                if values.isna().mean() < 0.5:
                    batch_results.append((candidate, values))
            except Exception:
                continue

        yield batch_results

        # Cleanup after each batch
        gc.collect()


def filter_candidates_by_domain(
    candidates: List[InteractionCandidate],
    domain_only: bool = True
) -> List[InteractionCandidate]:
    """Filter candidates to only those matching domain patterns.

    Args:
        candidates: List of InteractionCandidate objects.
        domain_only: If True, keep only domain-matching pairs.

    Returns:
        Filtered list of candidates.
    """
    if not domain_only:
        return candidates
    return [c for c in candidates if c.domain_match]


def filter_candidates_by_features(
    candidates: List[InteractionCandidate],
    required_features: Set[str],
    require_both: bool = False
) -> List[InteractionCandidate]:
    """Filter candidates requiring specific features.

    Args:
        candidates: List of InteractionCandidate objects.
        required_features: Set of feature names that must be involved.
        require_both: If True, both features must be in required_features.
                      If False, at least one must be.

    Returns:
        Filtered list of candidates.
    """
    if not required_features:
        return candidates

    filtered = []
    for c in candidates:
        in_a = c.feature_a in required_features
        in_b = c.feature_b in required_features
        if require_both:
            if in_a and in_b:
                filtered.append(c)
        else:
            if in_a or in_b:
                filtered.append(c)

    return filtered


class PairwiseInteractionSearch:
    """Manages pairwise interaction candidate generation and evaluation.

    Provides lazy evaluation, batch processing, and integration with
    the existing search framework.

    Attributes:
        search_config: Search configuration.
        candidates: Generated interaction candidates.
        evaluated_candidates: Dict of feature_name -> (delta_improvement, SubsetResult).
        selected_interactions: List of selected interaction feature names.
    """

    def __init__(self, search_config: SearchConfig):
        """Initialize the pairwise interaction search.

        Args:
            search_config: Search configuration with interaction parameters.
        """
        self.search_config = search_config
        self.candidates: List[InteractionCandidate] = []
        self.evaluated_candidates: Dict[str, Tuple[float, Any]] = {}
        self.selected_interactions: List[str] = []
        self._shap_interactions_df: Optional[pd.DataFrame] = None

    def generate_candidates(
        self,
        base_features: List[str],
        top_global_features: List[str],
        feature_importance: Dict[str, float],
        X: pd.DataFrame,
        y: pd.Series,
        model_config: ModelConfig,
        compute_shap: bool = True
    ) -> List[InteractionCandidate]:
        """Generate and prioritize interaction candidates.

        Args:
            base_features: Current base feature subset.
            top_global_features: Top globally ranked features.
            feature_importance: Feature importance scores.
            X: Feature DataFrame.
            y: Target Series.
            model_config: Model configuration.
            compute_shap: Whether to compute SHAP interaction values.

        Returns:
            List of prioritized InteractionCandidate objects.
        """
        # Optionally compute SHAP interactions for better prioritization
        if compute_shap:
            try:
                self._shap_interactions_df = compute_shap_interactions(
                    X, y, model_config, base_features,
                    max_samples=1000
                )
            except Exception:
                self._shap_interactions_df = None

        # Generate candidates
        self.candidates = generate_interaction_candidates(
            base_features=base_features,
            top_global_features=top_global_features,
            feature_importance=feature_importance,
            interaction_types=self.search_config.interaction_types,
            use_domain_filter=self.search_config.use_domain_filter_interactions,
            shap_interactions_df=self._shap_interactions_df
        )

        return self.candidates

    def get_candidates_for_evaluation(
        self,
        max_candidates: Optional[int] = None
    ) -> List[InteractionCandidate]:
        """Get candidates ready for evaluation.

        Args:
            max_candidates: Maximum number to return (defaults to n_top_interactions).

        Returns:
            List of top candidates for evaluation.
        """
        n = max_candidates or self.search_config.n_top_interactions
        return self.candidates[:n]

    def record_evaluation(
        self,
        candidate: InteractionCandidate,
        delta_improvement: float,
        result: Any
    ):
        """Record the evaluation result for a candidate.

        Args:
            candidate: Evaluated interaction candidate.
            delta_improvement: Improvement over baseline (positive = better).
            result: SubsetResult from evaluation.
        """
        self.evaluated_candidates[candidate.feature_name] = (delta_improvement, result)

    def get_top_evaluated(
        self,
        n: int = 5,
        min_improvement: float = 0.0
    ) -> List[Tuple[InteractionCandidate, float]]:
        """Get top evaluated interactions by improvement.

        Args:
            n: Number of top interactions to return.
            min_improvement: Minimum improvement threshold.

        Returns:
            List of (candidate, delta_improvement) tuples sorted by improvement.
        """
        # Build list of (candidate, improvement)
        results = []
        for candidate in self.candidates:
            if candidate.feature_name in self.evaluated_candidates:
                delta, _ = self.evaluated_candidates[candidate.feature_name]
                if delta >= min_improvement:
                    results.append((candidate, delta))

        # Sort by improvement (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n]

    def select_interaction(self, candidate: InteractionCandidate):
        """Mark an interaction as selected.

        Args:
            candidate: Selected interaction candidate.
        """
        if candidate.feature_name not in self.selected_interactions:
            self.selected_interactions.append(candidate.feature_name)

    def get_unevaluated_candidates(self) -> List[InteractionCandidate]:
        """Get candidates that haven't been evaluated yet.

        Returns:
            List of unevaluated candidates.
        """
        return [c for c in self.candidates
                if c.feature_name not in self.evaluated_candidates]

    def get_candidate_by_name(self, name: str) -> Optional[InteractionCandidate]:
        """Look up a candidate by feature name.

        Args:
            name: Interaction feature name.

        Returns:
            InteractionCandidate or None if not found.
        """
        for c in self.candidates:
            if c.feature_name == name:
                return c
        return None

    def cleanup(self):
        """Clean up memory after interaction search completes."""
        self.evaluated_candidates.clear()
        self._shap_interactions_df = None
        gc.collect()
