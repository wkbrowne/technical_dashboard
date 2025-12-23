# Technical Dashboard Pipeline: An Honest Assessment

A candid evaluation of this trading system's methodology, strengths, weaknesses, and realistic expectations.

## What This System Does

This is a **momentum-based stock selection system** that:

1. **Computes technical features** across daily/weekly timeframes (RSI, MACD, relative strength, macro indicators)
2. **Labels historical data** using triple-barrier method (profit target, stop loss, time limit)
3. **Trains a classifier** (LightGBM) to predict which stocks will hit profit targets
4. **Optimizes position sizing** to translate predictions into portfolio weights
5. **Generates weekly signals** for a momentum-based trading strategy

---

## Core Technical Principles

### What's Done Well

#### 1. Proper Time-Series Cross-Validation
The system uses **purged walk-forward CV** - the gold standard for financial ML:
- Training data strictly before test data
- Purging removes samples whose label window extends into test
- Embargo gap prevents subtle leakage

This is correct and many practitioners get this wrong.

#### 2. Triple-Barrier Labeling
Labels are based on actual trade outcomes, not arbitrary future returns:
- Upper barrier (profit target): 3x ATR
- Lower barrier (stop loss): 1.5x ATR
- Time limit: 20 trading days

This respects how trades actually work.

#### 3. Sample Weighting for Overlapping Labels
Weekly signals with 20-day horizons overlap. The system weights samples inversely by overlap count, following Lopez de Prado's methodology. This prevents over-representation of certain regimes.

#### 4. Feature Engineering Discipline
- No lookahead bias in features (all lagged appropriately)
- Weekly features merged with `direction='backward'` to prevent leakage
- Lowercase column naming convention enforced

#### 5. Out-of-Sample Prediction Generation
The new `generate_cv_predictions.py` creates predictions where **every prediction is made on data the model never saw**. This is the only valid basis for backtesting.

---

## Critical Weaknesses & Limitations

### 1. Fundamental Problem: Is This Edge Real?

The current results show:
- **Out-of-sample AUC: ~0.60** (where 0.5 is random)
- **In-sample AUC: ~0.69** (showing the typical train-test gap)
- **Precision on top decile: ~40-45%** (vs ~35% base rate)

This is weak predictive power. To put it in perspective:
- You're slightly better than a coin flip
- Edge is maybe 10% above random
- Transaction costs and slippage can easily consume this

**Honest assessment**: This level of predictive power is typical for technical analysis ML systems. It's not obviously worthless, but it's not obviously profitable either.

**Recent backtest** (Aug 2024 - Nov 2025): Shows 89% raw return (67% CAGR), annualized Sharpe 0.51:
- Period included the April 2025 tariff correction - not purely a bull market
- Model AUC held up during volatile periods (0.65 during Feb-May 2025 correction)
- Annualized volatility is high (132%) - significant weekly swings
- Only 65 weeks of out-of-sample data - more history needed for confidence

### 2. No Transaction Cost Modeling in Training

The model optimizes for AUC/classification accuracy, but trading costs matter:
- Spread: ~5-10 bps each way
- Slippage: varies by liquidity
- Market impact: significant for larger positions

The sizing optimizer includes a `turnover_penalty`, but the model itself doesn't know about costs. It may prefer high-turnover features that don't survive after costs.

### 3. Regime Dependence

Technical momentum strategies are highly regime-dependent:
- **Bull markets**: Momentum works (buy winners, they keep winning)
- **Bear markets**: Momentum can accelerate losses
- **Choppy/ranging markets**: False signals dominate

The model has no explicit regime detection. Past performance may not persist.

### 4. Crowded Trade Risk

Momentum is one of the most studied and traded factors:
- Many quantitative funds trade similar signals
- Alpha gets arbitraged away over time
- Execution matters more than prediction

You're competing against Renaissance, Two Sigma, and thousands of retail quants using similar approaches.

### 5. Survivorship Bias Concerns

The system filters out:
- SPACs and ADRs (good)
- Delisted stocks (problematic - these are often the big losers)

True survivorship-free testing requires historical constituent lists, which may not be fully implemented.

### 6. Single Strategy, Single Asset Class

This is a long-only US equity momentum strategy. No diversification across:
- Short positions
- Other asset classes (bonds, commodities, crypto)
- Other strategies (value, mean-reversion, volatility)

A single strategy is high-risk for retirement.

---

## Realistic Expectations

### Best Case Scenario
If everything works and you execute well:
- **Annual return**: Maybe 5-10% above market
- **Sharpe ratio**: 0.5-0.8 (weak but positive)
- **Max drawdown**: 20-40% (yes, really)

This is not "retire on this program" territory for normal portfolio sizes.

### Likely Case Scenario
After transaction costs and real-world execution:
- **Annual return**: Close to market return (0-5% alpha)
- **Sharpe ratio**: 0.3-0.5
- **Significant drawdowns**: 30%+ at some point

You'd likely do similar to a market index fund with more volatility.

### Worst Case Scenario
The model's weak edge disappears:
- Regime change makes momentum unprofitable
- Competition arbitrages away the edge
- Execution costs exceed gross alpha

You underperform a simple buy-and-hold strategy.

---

## What Would Make This Viable for Retirement?

### 1. Much Higher Predictive Power
- Current: AUC ~0.68
- Needed: AUC ~0.75+ with consistent precision lift

This requires either:
- Alternative data (satellite imagery, credit card data, NLP on filings)
- Faster execution (intraday signals)
- More sophisticated modeling (deep learning on order flow)

### 2. Multiple Uncorrelated Strategies
- A momentum strategy (this one)
- A mean-reversion strategy
- A volatility strategy
- A macro strategy

Combined with proper portfolio allocation, this reduces drawdowns.

### 3. Realistic Capital Requirements
For a single strategy to support retirement income, you need:
- Consistent 10%+ annual alpha
- Low drawdowns (<15%)
- $2-5M+ capital base

Most retail traders don't have this.

### 4. Execution Infrastructure
- API access to brokers
- Real-time position management
- Automated risk controls

The current system generates signals but doesn't execute trades.

---

## Suggested Path Forward

### Short Term (1-3 months)
1. **Run the backtests properly** using `run_sizing_pipeline.py`
2. **Analyze results honestly**: What's the Sharpe after costs? Max drawdown?
3. **Paper trade**: Forward-test on live data without real money

### Medium Term (3-12 months)
4. **Add more strategies**: Mean-reversion, sector rotation, volatility
5. **Improve data**: Consider alternative data sources
6. **Build execution**: Connect to a broker API

### Long Term (1-3 years)
7. **Live trade with small capital**: 1-5% of portfolio
8. **Scale gradually**: Only if results are consistent
9. **Never bet the retirement**: This should be one piece of a diversified plan

---

## Honest Conclusion

This system is **technically competent** - it follows ML best practices for financial data. The methodology is sound.

However, **sound methodology doesn't guarantee profits**. Many well-built systems fail to produce alpha after costs.

### Is this worth your time?

**For learning**: Yes. This is a proper ML trading pipeline. You'll learn:
- Time-series cross-validation
- Feature engineering for finance
- Position sizing and risk management
- Walk-forward backtesting

**For retirement income**: Probably not as-is. The edge is too weak and the strategy too narrow. You'd need to:
- Significantly improve predictive power
- Add multiple uncorrelated strategies
- Build proper execution infrastructure
- Have substantial capital

**For supplemental income**: Possibly, if you:
- Paper trade for 6+ months first
- Use small position sizes (1-2% of portfolio per trade)
- Accept that some years will underperform

---

## Areas for Future Work

### High Priority
1. **Alternative data integration**: Fundamental data, sentiment, flow data
2. **Multi-strategy portfolio**: Combine with value, quality, low-vol
3. **Execution infrastructure**: Broker API, automated trading
4. **Live paper trading**: Forward-test before real capital

### Medium Priority
5. **Short selling**: Capture alpha on both sides
6. **Options overlay**: Enhance returns or reduce risk
7. **Intraday signals**: Faster alpha decay but more opportunities
8. **Risk parity allocation**: Volatility-weighted across strategies

### Lower Priority
9. **Deep learning models**: Potentially higher capacity
10. **Reinforcement learning**: End-to-end optimization
11. **Market making**: Different strategy entirely

---

## References

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Lopez de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge.
- Harvey, C. et al. (2016). "...and the Cross-Section of Expected Returns." Review of Financial Studies.
- Arnott, R. et al. (2019). "Reports of Value's Death May Be Greatly Exaggerated." Research Affiliates.

---

*Written with intellectual honesty. Trading is hard. Most strategies fail. Proceed with caution.*
