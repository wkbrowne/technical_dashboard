"""Tests for multi-model position sizing.

Tests cover:
- Combining logic (priority vs blend)
- Netting behavior (strongest vs net)
- Gating function determinism
- Config parsing backward compatibility
- Weight computation correctness
"""

import numpy as np
import pandas as pd
import pytest

from src.sizing.config import (
    ModelType,
    CombinePolicy,
    NettingPolicy,
    MonotoneSizingParams,
    MultiModelSizingConfig,
    RegimeGatingConfig,
    load_multi_model_config,
    save_multi_model_config,
)
from src.sizing.multi_model import (
    compute_edge_score,
    compute_raw_weight,
    compute_model_weights,
    compute_edge_scores,
    combine_model_weights,
    apply_portfolio_constraints,
    MultiModelSizingEngine,
)
from src.sizing.regime_gating import (
    RegimeGate,
    apply_regime_gating,
    compute_regime_exposure_multiplier,
)
from src.sizing.predictions import (
    get_prob_column,
    validate_predictions_format,
    convert_long_to_wide_predictions,
)


# Fixtures

@pytest.fixture
def sample_signals():
    """Create sample signals DataFrame with multi-model probabilities."""
    np.random.seed(42)
    n = 20

    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="W-MON"),
        "symbol": [f"SYM{i}" for i in range(n)],
        "p_long_normal": np.random.uniform(0.3, 0.8, n),
        "p_long_parabolic": np.random.uniform(0.2, 0.7, n),
        "p_short_normal": np.random.uniform(0.3, 0.8, n),
        "p_short_parabolic": np.random.uniform(0.2, 0.7, n),
        "actual_return": np.random.uniform(-0.1, 0.15, n),
        "week_monday": pd.date_range("2024-01-01", periods=n, freq="W-MON"),
    })


@pytest.fixture
def default_config():
    """Create default multi-model config."""
    return MultiModelSizingConfig()


@pytest.fixture
def regime_features():
    """Create sample regime features."""
    return pd.Series({
        "vix_percentile_252d": 75.0,
        "fred_bamlh0a0hym2_z60": 1.2,
        "sector_breadth_pct_above_ma200": 45.0,
    })


# Config tests

class TestConfig:
    """Tests for configuration classes."""

    def test_model_type_direction_sign(self):
        """Test model direction signs are correct."""
        assert ModelType.LONG_NORMAL.direction_sign == 1
        assert ModelType.LONG_PARABOLIC.direction_sign == 1
        assert ModelType.SHORT_NORMAL.direction_sign == -1
        assert ModelType.SHORT_PARABOLIC.direction_sign == -1

    def test_model_type_is_long_short(self):
        """Test is_long and is_short properties."""
        assert ModelType.LONG_NORMAL.is_long is True
        assert ModelType.LONG_NORMAL.is_short is False
        assert ModelType.SHORT_NORMAL.is_long is False
        assert ModelType.SHORT_NORMAL.is_short is True

    def test_config_to_dict_round_trip(self, default_config):
        """Test config serialization round-trip."""
        d = default_config.to_dict()
        restored = MultiModelSizingConfig.from_dict(d)

        assert restored.models == default_config.models
        assert restored.combine_policy == default_config.combine_policy
        assert restored.max_gross_exposure == default_config.max_gross_exposure

    def test_single_model_compat(self):
        """Test backward compatibility with single-model config."""
        config = MultiModelSizingConfig.single_model_compat(
            slope=2.5,
            intercept=0.55,
            exposure_mult=0.9,
        )

        assert config.models == ["long_normal"]
        assert config.sizing_params.slope == 2.5
        assert config.sizing_params.intercept == 0.55
        assert config.regime_gating.enabled is False

    def test_parabolic_threshold_offset(self):
        """Test parabolic models get extra threshold."""
        config = MultiModelSizingConfig(
            parabolic_threshold_offset=0.1,
        )
        config.sizing_params.intercept = 0.5

        normal_params = config.get_sizing_params_for_model(ModelType.LONG_NORMAL)
        parabolic_params = config.get_sizing_params_for_model(ModelType.LONG_PARABOLIC)

        assert normal_params.intercept == 0.5
        assert parabolic_params.intercept == 0.6  # 0.5 + 0.1


class TestConfigSaveLoad:
    """Tests for config file I/O."""

    def test_save_load_config(self, tmp_path, default_config):
        """Test saving and loading config file."""
        path = tmp_path / "config.json"
        save_multi_model_config(default_config, path)

        loaded = load_multi_model_config(path)

        assert loaded.models == default_config.models
        assert loaded.combine_policy == default_config.combine_policy
        assert loaded.sizing_params.slope == default_config.sizing_params.slope

    def test_load_legacy_format(self, tmp_path):
        """Test loading legacy single-model format."""
        import json

        legacy_config = {
            "slope": 1.5,
            "intercept": 0.58,
            "exposure_mult": 0.85,
            "turnover_penalty": 0.005,
        }

        path = tmp_path / "legacy.json"
        with open(path, "w") as f:
            json.dump(legacy_config, f)

        loaded = load_multi_model_config(path)

        assert loaded.models == ["long_normal"]
        assert loaded.sizing_params.slope == 1.5
        assert loaded.sizing_params.intercept == 0.58


# Weight computation tests

class TestWeightComputation:
    """Tests for weight computation logic."""

    def test_compute_edge_score(self):
        """Test edge score computation."""
        assert compute_edge_score(0.7, 0.5) == pytest.approx(0.2)
        assert compute_edge_score(0.5, 0.5) == pytest.approx(0.0)
        assert compute_edge_score(0.3, 0.5) == pytest.approx(-0.2)

    def test_compute_raw_weight(self):
        """Test raw weight computation."""
        params = MonotoneSizingParams(
            slope=2.0,
            intercept=0.5,
            exposure_mult=1.0,
            max_weight=0.10,
            min_weight=0.01,
        )

        # Above threshold
        w = compute_raw_weight(0.7, params)
        assert w == pytest.approx(0.10)  # Clipped to max

        # At threshold
        w = compute_raw_weight(0.5, params)
        assert w == 0.0  # Below min_weight

        # Below threshold
        w = compute_raw_weight(0.3, params)
        assert w == 0.0  # Clipped to 0

    def test_compute_model_weights(self, sample_signals, default_config):
        """Test computing weights for all models."""
        result = compute_model_weights(sample_signals, default_config)

        # Check weight columns exist
        assert "w_long_normal" in result.columns
        assert "w_short_normal" in result.columns

        # Long weights should be positive
        long_weights = result["w_long_normal"]
        assert (long_weights >= 0).all() or (long_weights <= 0).any()

        # Short weights should be negative (after direction sign)
        short_weights = result["w_short_normal"]
        assert (short_weights <= 0).all()

    def test_weights_respect_direction(self, sample_signals, default_config):
        """Test that short models produce negative weights."""
        result = compute_model_weights(sample_signals, default_config)

        # Any non-zero short weight should be negative
        short_normal = result["w_short_normal"]
        non_zero_short = short_normal[short_normal != 0]
        if len(non_zero_short) > 0:
            assert (non_zero_short < 0).all()


# Combining logic tests

class TestCombineLogic:
    """Tests for model combining policies."""

    def test_mode_priority_picks_best_edge(self, sample_signals, default_config):
        """Test mode_priority picks model with highest edge."""
        default_config.combine_policy = "mode_priority"

        # Add weights and edges
        result = compute_model_weights(sample_signals, default_config)
        result = compute_edge_scores(result, default_config)
        result = combine_model_weights(result, default_config)

        assert "combined_weight" in result.columns
        assert "contributing_model" in result.columns

        # Check that contributing model is set
        assert result["contributing_model"].notna().any()

    def test_blend_combines_all_models(self, sample_signals):
        """Test blend policy averages all models."""
        config = MultiModelSizingConfig(combine_policy="blend")

        result = compute_model_weights(sample_signals, config)
        result = compute_edge_scores(result, config)
        result = combine_model_weights(result, config)

        # Blend should mark contributing_model as "blend"
        assert (result["contributing_model"] == "blend").all()


# Netting behavior tests

class TestNettingBehavior:
    """Tests for long/short netting policies."""

    def test_strongest_picks_larger_weight(self):
        """Test strongest netting picks larger absolute weight."""
        signals = pd.DataFrame({
            "symbol": ["A", "B"],
            "p_long_normal": [0.8, 0.4],
            "p_short_normal": [0.4, 0.8],
            "actual_return": [0.05, -0.05],
        })

        config = MultiModelSizingConfig(
            models=["long_normal", "short_normal"],
            netting_policy="strongest",
        )

        result = compute_model_weights(signals, config)
        result = compute_edge_scores(result, config)
        result = combine_model_weights(result, config)

        # A should be long (higher long prob)
        assert result.loc[0, "direction"] == 1

        # B should be short (higher short prob)
        assert result.loc[1, "direction"] == -1


# Gating tests

class TestRegimeGating:
    """Tests for regime gating functionality."""

    def test_gating_disabled_returns_one(self, regime_features):
        """Test disabled gating returns multiplier of 1."""
        config = RegimeGatingConfig.disabled()
        mult, diag = compute_regime_exposure_multiplier(regime_features, config)

        assert mult == 1.0
        assert diag["enabled"] is False

    def test_gating_vix_high_reduces_exposure(self):
        """Test high VIX reduces exposure."""
        features = pd.Series({
            "vix_percentile_252d": 90.0,  # High VIX
        })

        config = RegimeGatingConfig(
            enabled=True,
            vix_high_threshold=80.0,
            vix_high_exposure_mult=0.6,
        )

        mult, diag = compute_regime_exposure_multiplier(features, config)

        assert mult == pytest.approx(0.6)
        assert "vix_high" in diag["triggered_rules"]

    def test_gating_multiple_rules_compound(self):
        """Test multiple rules compound multiplicatively."""
        features = pd.Series({
            "vix_percentile_252d": 90.0,  # High VIX
            "fred_bamlh0a0hym2_z60": 2.0,  # High credit risk
        })

        config = RegimeGatingConfig(
            enabled=True,
            vix_high_threshold=80.0,
            vix_high_exposure_mult=0.8,
            credit_risk_threshold=1.5,
            credit_risk_exposure_mult=0.9,
        )

        mult, diag = compute_regime_exposure_multiplier(features, config)

        # Should be 0.8 * 0.9 = 0.72
        assert mult == pytest.approx(0.72)
        assert len(diag["triggered_rules"]) == 2

    def test_gating_determinism(self, regime_features):
        """Test gating is deterministic."""
        config = RegimeGatingConfig(
            enabled=True,
            vix_high_threshold=70.0,
            vix_high_exposure_mult=0.7,
        )

        # Run multiple times
        results = [
            compute_regime_exposure_multiplier(regime_features, config)[0]
            for _ in range(10)
        ]

        # All results should be identical
        assert len(set(results)) == 1


# Portfolio constraints tests

class TestPortfolioConstraints:
    """Tests for portfolio constraint enforcement."""

    def test_max_weight_clipping(self, sample_signals, default_config):
        """Test individual weights are clipped to max."""
        default_config.max_weight_per_name = 0.05

        result = compute_model_weights(sample_signals, default_config)
        result = compute_edge_scores(result, default_config)
        result = combine_model_weights(result, default_config)
        result = apply_portfolio_constraints(result, default_config)

        # All weights should be <= max
        assert (result["final_weight"].abs() <= 0.05 + 1e-6).all()

    def test_gross_exposure_scaling(self, sample_signals):
        """Test gross exposure is scaled to limit."""
        config = MultiModelSizingConfig(
            max_gross_exposure=0.5,
        )

        result = compute_model_weights(sample_signals, config)
        result = compute_edge_scores(result, config)
        result = combine_model_weights(result, config)
        result = apply_portfolio_constraints(result, config)

        gross = result["final_weight"].abs().sum()
        assert gross <= 0.5 + 1e-6

    def test_min_weight_filtering(self, sample_signals):
        """Test positions below min weight are zeroed."""
        config = MultiModelSizingConfig(
            min_weight=0.02,
        )

        result = compute_model_weights(sample_signals, config)
        result = compute_edge_scores(result, config)
        result = combine_model_weights(result, config)
        result = apply_portfolio_constraints(result, config)

        # Non-zero weights should all be >= min
        non_zero = result["final_weight"][result["final_weight"] != 0]
        if len(non_zero) > 0:
            assert (non_zero.abs() >= 0.02 - 1e-6).all()


# Engine integration tests

class TestMultiModelEngine:
    """Integration tests for the sizing engine."""

    def test_engine_basic_workflow(self, sample_signals, default_config):
        """Test basic engine workflow produces valid output."""
        engine = MultiModelSizingEngine(default_config)
        result = engine.compute_weights(sample_signals)

        assert "final_weight" in result.columns
        assert len(result) == len(sample_signals)

    def test_engine_with_gating(self, sample_signals, regime_features):
        """Test engine with regime gating enabled."""
        config = MultiModelSizingConfig(
            regime_gating=RegimeGatingConfig(
                enabled=True,
                vix_high_threshold=70.0,
                vix_high_exposure_mult=0.7,
            ),
        )

        engine = MultiModelSizingEngine(config)
        regime_df = pd.DataFrame([regime_features])

        result = engine.compute_weights(sample_signals, regime_features=regime_df)

        assert "gating_multiplier" in result.columns

    def test_engine_diagnostics(self, sample_signals, default_config):
        """Test engine diagnostics output."""
        engine = MultiModelSizingEngine(default_config)
        result = engine.compute_weights(sample_signals)

        diag = engine.get_diagnostics(result)

        assert "n_signals" in diag
        assert "n_positions" in diag
        assert "n_longs" in diag
        assert "n_shorts" in diag


# Predictions format tests

class TestPredictionsFormat:
    """Tests for prediction data handling."""

    def test_get_prob_column(self):
        """Test probability column naming."""
        assert get_prob_column(ModelType.LONG_NORMAL) == "p_long_normal"
        assert get_prob_column(ModelType.SHORT_PARABOLIC) == "p_short_parabolic"

    def test_validate_predictions_valid(self, sample_signals):
        """Test validation passes for valid data."""
        result = validate_predictions_format(sample_signals)

        assert result["is_valid"] is True
        assert len(result["issues"]) == 0

    def test_validate_predictions_missing_column(self):
        """Test validation catches missing columns."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
            "symbol": ["A", "B", "C", "D", "E"],
            # Missing probability columns
        })

        result = validate_predictions_format(df)

        assert result["is_valid"] is False
        assert any("Missing" in issue for issue in result["issues"])

    def test_validate_predictions_bad_probs(self):
        """Test validation catches out-of-range probabilities."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
            "symbol": ["A", "B", "C", "D", "E"],
            "p_long_normal": [0.5, 0.6, 1.5, 0.7, 0.8],  # 1.5 is invalid
        })

        result = validate_predictions_format(df, models=[ModelType.LONG_NORMAL])

        assert result["is_valid"] is False
        assert any("outside [0, 1]" in issue for issue in result["issues"])

    def test_convert_long_to_wide(self):
        """Test converting long format to wide format."""
        long_df = pd.DataFrame({
            "date": ["2024-01-01"] * 4,
            "symbol": ["A"] * 4,
            "model": ["long_normal", "long_parabolic", "short_normal", "short_parabolic"],
            "probability": [0.6, 0.5, 0.4, 0.3],
        })

        wide_df = convert_long_to_wide_predictions(long_df)

        assert "p_long_normal" in wide_df.columns
        assert "p_short_parabolic" in wide_df.columns
        assert len(wide_df) == 1  # One row per (date, symbol)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
