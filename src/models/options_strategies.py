import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional, Union
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class OptionsStrategyRecommender:
    """
    Options strategy recommender using Bollinger Bands, GARCH volatility,
    Markov chain models, and OHLC forecasts.
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 commission_per_contract: float = 1.0,
                 min_probability_threshold: float = 0.6):
        """
        Initialize the options strategy recommender.
        
        Parameters
        ----------
        risk_free_rate : float
            Risk-free interest rate (annual)
        commission_per_contract : float
            Commission cost per options contract
        min_probability_threshold : float
            Minimum probability threshold for recommendations
        """
        self.risk_free_rate = risk_free_rate
        self.commission = commission_per_contract
        self.min_prob_threshold = min_probability_threshold
        self.recommendations = []
        
    def generate_recommendations(self,
                               current_price: float,
                               ohlc_forecast: Dict,
                               bb_forecast: Dict,
                               vol_forecast: Dict,
                               markov_predictions: Dict,
                               days_to_expiry: int = 30) -> List[Dict]:
        """
        Generate options strategy recommendations based on forecasts.
        
        Parameters
        ----------
        current_price : float
            Current stock price
        ohlc_forecast : dict
            OHLC forecast results
        bb_forecast : dict
            Bollinger Bands forecast
        vol_forecast : dict
            Volatility forecast
        markov_predictions : dict
            Markov chain state predictions
        days_to_expiry : int
            Days until options expiration
            
        Returns
        -------
        list
            List of strategy recommendations
        """
        self.recommendations = []
        
        # Extract key forecast metrics
        forecast_days = min(days_to_expiry, len(ohlc_forecast['close']))
        
        # Price movement expectations
        expected_close = ohlc_forecast['close'][forecast_days-1]
        expected_high = max(ohlc_forecast['high'][:forecast_days])
        expected_low = min(ohlc_forecast['low'][:forecast_days])
        
        # Volatility expectations
        mean_vol = vol_forecast['mean_volatility']
        vol_trend = vol_forecast['volatility_trend']
        
        # BB state probabilities
        bb_state_probs = markov_predictions.get('state_probs', [0.2, 0.2, 0.2, 0.2, 0.2])
        
        # Calculate key metrics
        price_change_pct = (expected_close - current_price) / current_price
        max_upside = (expected_high - current_price) / current_price
        max_downside = (current_price - expected_low) / current_price
        
        # Generate strategy recommendations
        self._evaluate_directional_strategies(
            current_price, expected_close, expected_high, expected_low,
            price_change_pct, max_upside, max_downside, mean_vol, days_to_expiry
        )
        
        self._evaluate_volatility_strategies(
            current_price, mean_vol, vol_trend, bb_state_probs, days_to_expiry
        )
        
        self._evaluate_income_strategies(
            current_price, expected_high, expected_low, mean_vol, days_to_expiry
        )
        
        # Sort recommendations by expected return
        self.recommendations.sort(key=lambda x: x['expected_return'], reverse=True)
        
        return self.recommendations
    
    def _evaluate_directional_strategies(self, current_price: float, expected_close: float,
                                       expected_high: float, expected_low: float,
                                       price_change_pct: float, max_upside: float,
                                       max_downside: float, volatility: float, dte: int) -> None:
        """Evaluate directional options strategies."""
        
        # Long Call Strategy
        if price_change_pct > 0.05 and max_upside > 0.1:
            strike_price = current_price * 1.02  # Slightly OTM
            probability = self._calculate_success_probability(current_price, expected_high, strike_price, 'call')
            
            if probability > self.min_prob_threshold:
                self.recommendations.append({
                    'strategy': 'Long Call',
                    'direction': 'Bullish',
                    'strike_price': strike_price,
                    'probability_success': probability,
                    'max_profit': 'Unlimited',
                    'max_loss': self._estimate_option_premium(current_price, strike_price, volatility, dte, 'call'),
                    'expected_return': probability * max_upside * 10,  # Leverage factor
                    'risk_level': 'High',
                    'rationale': f'Expected {price_change_pct:.1%} price increase with {max_upside:.1%} upside potential'
                })
        
        # Long Put Strategy
        if price_change_pct < -0.05 and max_downside > 0.1:
            strike_price = current_price * 0.98  # Slightly OTM
            probability = self._calculate_success_probability(current_price, expected_low, strike_price, 'put')
            
            if probability > self.min_prob_threshold:
                self.recommendations.append({
                    'strategy': 'Long Put',
                    'direction': 'Bearish',
                    'strike_price': strike_price,
                    'probability_success': probability,
                    'max_profit': strike_price - 0,  # Assuming stock can go to 0
                    'max_loss': self._estimate_option_premium(current_price, strike_price, volatility, dte, 'put'),
                    'expected_return': probability * max_downside * 8,  # Leverage factor
                    'risk_level': 'High',
                    'rationale': f'Expected {abs(price_change_pct):.1%} price decrease with {max_downside:.1%} downside'
                })
        
        # Bull Call Spread
        if 0.02 < price_change_pct < 0.08:
            lower_strike = current_price * 1.01
            upper_strike = current_price * 1.05
            probability = self._calculate_success_probability(current_price, expected_close, lower_strike, 'call')
            
            if probability > self.min_prob_threshold:
                max_profit = upper_strike - lower_strike
                max_loss = self._estimate_spread_cost(current_price, lower_strike, upper_strike, volatility, dte, 'bull_call')
                
                self.recommendations.append({
                    'strategy': 'Bull Call Spread',
                    'direction': 'Moderately Bullish',
                    'strike_price': f'{lower_strike:.2f}/{upper_strike:.2f}',
                    'probability_success': probability,
                    'max_profit': max_profit,
                    'max_loss': max_loss,
                    'expected_return': probability * (max_profit / max_loss) * 100,
                    'risk_level': 'Medium',
                    'rationale': f'Moderate bullish outlook with limited risk/reward'
                })
    
    def _evaluate_volatility_strategies(self, current_price: float, volatility: float,
                                      vol_trend: float, bb_state_probs: List[float], dte: int) -> None:
        """Evaluate volatility-based strategies."""
        
        # Determine if we expect volatility expansion or contraction
        squeeze_prob = bb_state_probs[2] if len(bb_state_probs) > 2 else 0.2  # Middle state
        extreme_prob = bb_state_probs[0] + bb_state_probs[-1] if len(bb_state_probs) > 1 else 0.4
        
        # Long Straddle (volatility expansion expected)
        if vol_trend > 0 and extreme_prob > 0.4:
            strike_price = current_price
            probability = extreme_prob
            
            if probability > self.min_prob_threshold:
                premium_cost = self._estimate_straddle_cost(current_price, strike_price, volatility, dte)
                breakeven_up = strike_price + premium_cost
                breakeven_down = strike_price - premium_cost
                
                self.recommendations.append({
                    'strategy': 'Long Straddle',
                    'direction': 'High Volatility',
                    'strike_price': strike_price,
                    'probability_success': probability,
                    'max_profit': 'Unlimited',
                    'max_loss': premium_cost,
                    'expected_return': probability * volatility * 500,  # Volatility play multiplier
                    'risk_level': 'High',
                    'rationale': f'Expecting volatility breakout, need move beyond ${breakeven_down:.2f}-${breakeven_up:.2f}'
                })
        
        # Iron Condor (low volatility expected)
        if vol_trend < 0 and squeeze_prob > 0.5:
            otm_distance = current_price * 0.05
            put_strike = current_price - otm_distance
            call_strike = current_price + otm_distance
            
            probability = squeeze_prob
            if probability > self.min_prob_threshold:
                max_profit = self._estimate_condor_credit(current_price, put_strike, call_strike, volatility, dte)
                max_loss = otm_distance - max_profit
                
                self.recommendations.append({
                    'strategy': 'Iron Condor',
                    'direction': 'Low Volatility',
                    'strike_price': f'{put_strike:.2f}/{current_price:.2f}/{current_price:.2f}/{call_strike:.2f}',
                    'probability_success': probability,
                    'max_profit': max_profit,
                    'max_loss': max_loss,
                    'expected_return': probability * (max_profit / max_loss) * 100,
                    'risk_level': 'Medium',
                    'rationale': f'Expecting low volatility, profit if price stays between ${put_strike:.2f}-${call_strike:.2f}'
                })
    
    def _evaluate_income_strategies(self, current_price: float, expected_high: float,
                                  expected_low: float, volatility: float, dte: int) -> None:
        """Evaluate income-generating strategies."""
        
        # Covered Call (if expecting sideways to slightly bullish)
        resistance_level = expected_high
        if resistance_level > current_price * 1.02:
            strike_price = min(resistance_level, current_price * 1.05)
            probability = 0.7  # Conservative estimate for income strategy
            
            premium_income = self._estimate_option_premium(current_price, strike_price, volatility, dte, 'call')
            max_loss = current_price * 100  # Assuming 100 shares
            
            self.recommendations.append({
                'strategy': 'Covered Call',
                'direction': 'Neutral to Slightly Bullish',
                'strike_price': strike_price,
                'probability_success': probability,
                'max_profit': premium_income + (strike_price - current_price) * 100,
                'max_loss': max_loss,
                'expected_return': probability * (premium_income / current_price) * 100,
                'risk_level': 'Low',
                'rationale': f'Generate income while holding stock, profit if price stays below ${strike_price:.2f}'
            })
        
        # Cash-Secured Put (if expecting support to hold)
        support_level = expected_low
        if support_level < current_price * 0.98:
            strike_price = max(support_level, current_price * 0.95)
            probability = 0.65
            
            premium_income = self._estimate_option_premium(current_price, strike_price, volatility, dte, 'put')
            
            self.recommendations.append({
                'strategy': 'Cash-Secured Put',
                'direction': 'Neutral to Slightly Bearish',
                'strike_price': strike_price,
                'probability_success': probability,
                'max_profit': premium_income,
                'max_loss': strike_price * 100 - premium_income,
                'expected_return': probability * (premium_income / (strike_price * 100)) * 100,
                'risk_level': 'Medium',
                'rationale': f'Generate income with willingness to buy stock at ${strike_price:.2f}'
            })
    
    def _calculate_success_probability(self, current_price: float, target_price: float,
                                     strike_price: float, option_type: str) -> float:
        """Calculate probability of strategy success."""
        if option_type == 'call':
            if target_price > strike_price:
                # Use normal distribution to estimate probability
                std_dev = abs(target_price - current_price) / 2  # Rough estimate
                z_score = (strike_price - current_price) / std_dev if std_dev > 0 else 0
                return 1 - stats.norm.cdf(z_score)
            else:
                return 0.1
        else:  # put
            if target_price < strike_price:
                std_dev = abs(current_price - target_price) / 2
                z_score = (current_price - strike_price) / std_dev if std_dev > 0 else 0
                return 1 - stats.norm.cdf(z_score)
            else:
                return 0.1
    
    def _estimate_option_premium(self, spot_price: float, strike_price: float,
                               volatility: float, days_to_expiry: int, option_type: str) -> float:
        """Simplified Black-Scholes option pricing."""
        time_to_expiry = days_to_expiry / 365.0
        
        if time_to_expiry <= 0:
            # At expiration
            if option_type == 'call':
                return max(spot_price - strike_price, 0)
            else:
                return max(strike_price - spot_price, 0)
        
        # Simplified premium estimation
        moneyness = spot_price / strike_price
        time_value = volatility * np.sqrt(time_to_expiry) * spot_price * 0.4
        
        if option_type == 'call':
            intrinsic = max(spot_price - strike_price, 0)
            if moneyness > 1:  # ITM
                premium = intrinsic + time_value * 0.6
            else:  # OTM
                premium = time_value * (moneyness ** 0.5)
        else:  # put
            intrinsic = max(strike_price - spot_price, 0)
            if moneyness < 1:  # ITM
                premium = intrinsic + time_value * 0.6
            else:  # OTM
                premium = time_value * ((1/moneyness) ** 0.5)
        
        return max(premium, 0.01)  # Minimum premium
    
    def _estimate_spread_cost(self, spot_price: float, lower_strike: float, upper_strike: float,
                            volatility: float, days_to_expiry: int, spread_type: str) -> float:
        """Estimate cost of spread strategies."""
        if spread_type == 'bull_call':
            long_call = self._estimate_option_premium(spot_price, lower_strike, volatility, days_to_expiry, 'call')
            short_call = self._estimate_option_premium(spot_price, upper_strike, volatility, days_to_expiry, 'call')
            return long_call - short_call
        # Add other spread types as needed
        return 0
    
    def _estimate_straddle_cost(self, spot_price: float, strike_price: float,
                              volatility: float, days_to_expiry: int) -> float:
        """Estimate cost of straddle."""
        call_premium = self._estimate_option_premium(spot_price, strike_price, volatility, days_to_expiry, 'call')
        put_premium = self._estimate_option_premium(spot_price, strike_price, volatility, days_to_expiry, 'put')
        return call_premium + put_premium
    
    def _estimate_condor_credit(self, spot_price: float, put_strike: float, call_strike: float,
                              volatility: float, days_to_expiry: int) -> float:
        """Estimate credit received from iron condor."""
        # Simplified estimation - credit from selling closer strikes minus cost of buying further strikes
        wing_width = spot_price * 0.02  # Assume 2% wing width
        
        short_put = self._estimate_option_premium(spot_price, put_strike + wing_width, volatility, days_to_expiry, 'put')
        long_put = self._estimate_option_premium(spot_price, put_strike, volatility, days_to_expiry, 'put')
        short_call = self._estimate_option_premium(spot_price, call_strike - wing_width, volatility, days_to_expiry, 'call')
        long_call = self._estimate_option_premium(spot_price, call_strike, volatility, days_to_expiry, 'call')
        
        credit = (short_put + short_call) - (long_put + long_call)
        return max(credit, 0.1)
    
    def plot_strategy_comparison(self, top_n: int = 5, figsize: Tuple[int, int] = (15, 10)) -> None:
        """Plot comparison of top strategies."""
        if not self.recommendations:
            print("No recommendations to plot. Run generate_recommendations first.")
            return
        
        top_strategies = self.recommendations[:top_n]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Expected Return vs Risk
        strategies = [r['strategy'] for r in top_strategies]
        returns = [r['expected_return'] for r in top_strategies]
        risks = [r['risk_level'] for r in top_strategies]
        
        risk_map = {'Low': 1, 'Medium': 2, 'High': 3}
        risk_scores = [risk_map[r] for r in risks]
        
        scatter = ax1.scatter(risk_scores, returns, s=100, alpha=0.7, c=range(len(strategies)), cmap='viridis')
        ax1.set_xlabel('Risk Level')
        ax1.set_ylabel('Expected Return (%)')
        ax1.set_title('Expected Return vs Risk Level')
        ax1.set_xticks([1, 2, 3])
        ax1.set_xticklabels(['Low', 'Medium', 'High'])
        ax1.grid(True, alpha=0.3)
        
        # Add strategy labels
        for i, strategy in enumerate(strategies):
            ax1.annotate(strategy, (risk_scores[i], returns[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        # Success Probability
        probs = [r['probability_success'] for r in top_strategies]
        bars1 = ax2.bar(range(len(strategies)), probs, alpha=0.7, color='skyblue')
        ax2.set_xlabel('Strategy')
        ax2.set_ylabel('Success Probability')
        ax2.set_title('Strategy Success Probabilities')
        ax2.set_xticks(range(len(strategies)))
        ax2.set_xticklabels([s[:15] for s in strategies], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, prob in zip(bars1, probs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{prob:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Max Loss comparison (if available)
        max_losses = []
        for r in top_strategies:
            if isinstance(r['max_loss'], (int, float)):
                max_losses.append(r['max_loss'])
            else:
                max_losses.append(0)
        
        bars2 = ax3.bar(range(len(strategies)), max_losses, alpha=0.7, color='lightcoral')
        ax3.set_xlabel('Strategy')
        ax3.set_ylabel('Max Loss ($)')
        ax3.set_title('Maximum Loss Comparison')
        ax3.set_xticks(range(len(strategies)))
        ax3.set_xticklabels([s[:15] for s in strategies], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Strategy Direction Distribution
        directions = [r['direction'] for r in top_strategies]
        direction_counts = pd.Series(directions).value_counts()
        
        ax4.pie(direction_counts.values, labels=direction_counts.index, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Strategy Direction Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def get_strategy_summary(self) -> pd.DataFrame:
        """Get summary table of all recommendations."""
        if not self.recommendations:
            return pd.DataFrame()
        
        summary_data = []
        for rec in self.recommendations:
            summary_data.append({
                'Strategy': rec['strategy'],
                'Direction': rec['direction'],
                'Expected_Return_Pct': rec['expected_return'],
                'Success_Probability': rec['probability_success'],
                'Risk_Level': rec['risk_level'],
                'Strike_Price': rec['strike_price'],
                'Rationale': rec['rationale'][:50] + '...' if len(rec['rationale']) > 50 else rec['rationale']
            })
        
        return pd.DataFrame(summary_data)
    
    def filter_recommendations(self, 
                             min_return: float = None,
                             max_risk: str = None,
                             direction: str = None) -> List[Dict]:
        """Filter recommendations based on criteria."""
        filtered = self.recommendations.copy()
        
        if min_return is not None:
            filtered = [r for r in filtered if r['expected_return'] >= min_return]
        
        if max_risk is not None:
            risk_order = {'Low': 1, 'Medium': 2, 'High': 3}
            max_risk_score = risk_order.get(max_risk, 3)
            filtered = [r for r in filtered if risk_order.get(r['risk_level'], 3) <= max_risk_score]
        
        if direction is not None:
            filtered = [r for r in filtered if direction.lower() in r['direction'].lower()]
        
        return filtered