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
        
    def generate_enhanced_recommendations(self,
                                        current_price: float,
                                        ohlc_forecast: Dict,
                                        bb_forecast: Dict,
                                        vol_forecast: Dict,
                                        markov_predictions: Dict,
                                        max_dte: int = 45) -> Dict:
        """
        Generate enhanced options strategy recommendations with multi-DTE analysis.
        
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
        max_dte : int
            Maximum days to expiry for analysis
            
        Returns
        -------
        dict
            Enhanced recommendations with multi-DTE analysis
        """
        results = {
            'buying_strategies': [],
            'selling_strategies': {},
            'current_price': current_price,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M')
        }
        
        # Generate buying strategies (single DTE)
        results['buying_strategies'] = self._generate_buying_strategies(
            current_price, ohlc_forecast, bb_forecast, vol_forecast, markov_predictions, max_dte
        )
        
        # Generate selling strategies (multi-DTE: 0-10 DTE)
        results['selling_strategies'] = self._generate_multi_dte_selling_strategies(
            current_price, ohlc_forecast, bb_forecast, vol_forecast, markov_predictions
        )
        
        return results

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
        
        # Extend forecast if needed to match option expiry
        extended_forecast = self._extend_forecast_to_expiry(ohlc_forecast, days_to_expiry, current_price, vol_forecast)
        
        # Extract key forecast metrics using extended forecast
        forecast_days = days_to_expiry
        
        # Price movement expectations
        expected_close = extended_forecast['close'][forecast_days-1]
        expected_high = max(extended_forecast['high'][:forecast_days])
        expected_low = min(extended_forecast['low'][:forecast_days])
        
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
    
    def _extend_forecast_to_expiry(self, ohlc_forecast: Dict, days_to_expiry: int, 
                                  current_price: float, vol_forecast: Dict) -> Dict:
        """Extend OHLC forecast to match option expiry if needed."""
        current_forecast_length = len(ohlc_forecast['close'])
        
        if current_forecast_length >= days_to_expiry:
            # Forecast is long enough, return as-is
            return ohlc_forecast
        
        # Need to extend the forecast
        print(f"⚠️  Extending forecast from {current_forecast_length} to {days_to_expiry} days")
        
        extended_forecast = {
            'close': ohlc_forecast['close'].copy(),
            'high': ohlc_forecast['high'].copy(),
            'low': ohlc_forecast['low'].copy()
        }
        
        # Use the last forecasted values as a base
        last_close = extended_forecast['close'][-1]
        last_high = extended_forecast['high'][-1] 
        last_low = extended_forecast['low'][-1]
        
        # Get volatility for extension
        daily_vol = vol_forecast.get('mean_volatility', 0.02)
        
        # Extend using random walk with drift
        days_to_extend = days_to_expiry - current_forecast_length
        
        for i in range(days_to_extend):
            # Random daily return based on volatility
            daily_return = np.random.normal(0.0005, daily_vol)  # Small positive drift
            
            # Extend close prices
            next_close = last_close * (1 + daily_return)
            extended_forecast['close'].append(next_close)
            
            # Extend high/low with realistic ranges
            high_return = abs(daily_return) + np.random.exponential(daily_vol * 0.5)
            low_return = abs(daily_return) + np.random.exponential(daily_vol * 0.5)
            
            next_high = next_close * (1 + high_return)
            next_low = next_close * (1 - low_return)
            
            extended_forecast['high'].append(next_high)
            extended_forecast['low'].append(next_low)
            
            last_close = next_close
        
        return extended_forecast
    
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
                    'days_to_expiry': dte,
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
                    'days_to_expiry': dte,
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
                    'days_to_expiry': dte,
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
                    'days_to_expiry': dte,
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
                    'days_to_expiry': dte,
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
        
        # Covered Call (with proper strike positioning)
        resistance_level = expected_high
        
        # Strike should be ABOVE the expected high to be profitable
        # If forecast shows price going to $180, don't sell calls at $161!
        min_safe_strike = max(resistance_level * 1.02, current_price * 1.03)  # At least 2% above expected high
        
        if min_safe_strike > current_price:  # Only recommend if there's upside potential
            strike_price = min_safe_strike
            
            # Calculate probability that stock stays below strike
            # If expected high is $180 and strike is $184, this should be reasonable
            prob_below_strike = self._calculate_probability_below_strike(
                current_price, expected_high, strike_price, volatility, dte)
            
            if prob_below_strike > 0.5:  # Only recommend if >50% chance of success
                premium_income = self._estimate_option_premium(current_price, strike_price, volatility, dte, 'call')
                max_loss = current_price * 100  # Assuming 100 shares
                
                self.recommendations.append({
                    'strategy': 'Covered Call',
                    'direction': 'Neutral to Slightly Bullish',
                    'strike_price': strike_price,
                    'days_to_expiry': dte,
                    'probability_success': prob_below_strike,
                    'max_profit': premium_income + (strike_price - current_price) * 100,
                    'max_loss': max_loss,
                    'expected_return': prob_below_strike * (premium_income / current_price) * 100,
                    'risk_level': 'Low',
                    'rationale': f'Generate income while holding stock, strike ${strike_price:.2f} above expected high ${expected_high:.2f}'
                })
            else:
                print(f"⚠️  Covered call not recommended: expected high ${expected_high:.2f} too close to current price ${current_price:.2f}")
        else:
            print(f"⚠️  Covered call not recommended: insufficient upside potential (expected high: ${expected_high:.2f})")
        
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
                'days_to_expiry': dte,
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
    
    def _calculate_probability_below_strike(self, current_price: float, expected_high: float,
                                          strike_price: float, volatility: float, dte: int) -> float:
        """Calculate probability that stock stays below strike price."""
        # Use log-normal distribution to model stock price at expiry
        time_to_expiry = dte / 365.0
        
        if time_to_expiry <= 0:
            return 1.0 if current_price < strike_price else 0.0
        
        # Black-Scholes framework: S(T) ~ LogNormal(ln(S0) + (r-0.5*σ²)T, σ²T)
        risk_free_rate = self.risk_free_rate
        drift = (risk_free_rate - 0.5 * volatility**2) * time_to_expiry
        variance = volatility**2 * time_to_expiry
        
        # Expected log price at expiry
        log_current = np.log(current_price)
        log_strike = np.log(strike_price)
        
        # Standard normal variable: (ln(S(T)) - mean) / std
        mean_log_price = log_current + drift
        std_log_price = np.sqrt(variance)
        
        # P(S(T) < K) = P((ln(S(T)) - mean) / std < (ln(K) - mean) / std)
        z_score = (log_strike - mean_log_price) / std_log_price
        prob_below = stats.norm.cdf(z_score)
        
        return prob_below
    
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
                'Strike_Price': rec['strike_price'],
                'Days_To_Expiry': rec['days_to_expiry'],
                'Expected_Return_Pct': rec['expected_return'],
                'Success_Probability': rec['probability_success'],
                'Risk_Level': rec['risk_level'],
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
    
    def _generate_buying_strategies(self, current_price: float, ohlc_forecast: Dict,
                                  bb_forecast: Dict, vol_forecast: Dict,
                                  markov_predictions: Dict, max_dte: int) -> List[Dict]:
        """Generate enhanced buying strategies with detailed strike analysis."""
        buying_strategies = []
        
        # Extend forecast if needed to match option expiry
        extended_forecast = self._extend_forecast_to_expiry(ohlc_forecast, max_dte, current_price, vol_forecast)
        
        # Extract key forecast metrics using extended forecast
        forecast_days = max_dte
        expected_close = extended_forecast['close'][forecast_days-1]
        expected_high = max(extended_forecast['high'][:forecast_days])
        expected_low = min(extended_forecast['low'][:forecast_days])
        
        mean_vol = vol_forecast['mean_volatility']
        price_change_pct = (expected_close - current_price) / current_price
        max_upside = (expected_high - current_price) / current_price
        max_downside = (current_price - expected_low) / current_price
        
        # LONG CALL STRATEGIES
        if price_change_pct > 0.03 or max_upside > 0.08:
            call_strategies = []
            
            # ATM Call
            atm_strike = current_price
            atm_prob = self._calculate_success_probability(current_price, expected_high, atm_strike, 'call')
            call_strategies.append({
                'strategy': 'Long Call (ATM)',
                'direction': 'Bullish',
                'strike_price': atm_strike,
                'option_type': 'Call',
                'moneyness': 'ATM',
                'days_to_expiry': max_dte,
                'probability_success': atm_prob,
                'breakeven': atm_strike + self._estimate_option_premium(current_price, atm_strike, mean_vol, max_dte, 'call'),
                'max_profit': 'Unlimited',
                'max_loss': self._estimate_option_premium(current_price, atm_strike, mean_vol, max_dte, 'call'),
                'expected_return': atm_prob * max_upside * 1000 if atm_prob > 0.6 else 0,
                'risk_level': 'High'
            })
            
            # OTM Call (+2%)
            otm_strike = current_price * 1.02
            otm_prob = self._calculate_success_probability(current_price, expected_high, otm_strike, 'call')
            call_strategies.append({
                'strategy': 'Long Call (OTM +2%)',
                'direction': 'Bullish',
                'strike_price': otm_strike,
                'option_type': 'Call',
                'moneyness': 'OTM',
                'days_to_expiry': max_dte,
                'probability_success': otm_prob,
                'breakeven': otm_strike + self._estimate_option_premium(current_price, otm_strike, mean_vol, max_dte, 'call'),
                'max_profit': 'Unlimited',
                'max_loss': self._estimate_option_premium(current_price, otm_strike, mean_vol, max_dte, 'call'),
                'expected_return': otm_prob * max_upside * 1200 if otm_prob > 0.5 else 0,
                'risk_level': 'High'
            })
            
            # Deep OTM Call (+5%)
            dotm_strike = current_price * 1.05
            dotm_prob = self._calculate_success_probability(current_price, expected_high, dotm_strike, 'call')
            call_strategies.append({
                'strategy': 'Long Call (Deep OTM +5%)',
                'direction': 'Very Bullish',
                'strike_price': dotm_strike,
                'option_type': 'Call',
                'moneyness': 'Deep OTM',
                'days_to_expiry': max_dte,
                'probability_success': dotm_prob,
                'breakeven': dotm_strike + self._estimate_option_premium(current_price, dotm_strike, mean_vol, max_dte, 'call'),
                'max_profit': 'Unlimited',
                'max_loss': self._estimate_option_premium(current_price, dotm_strike, mean_vol, max_dte, 'call'),
                'expected_return': dotm_prob * max_upside * 1500 if dotm_prob > 0.3 else 0,
                'risk_level': 'Very High'
            })
            
            buying_strategies.extend(call_strategies)
        
        # LONG PUT STRATEGIES
        if price_change_pct < -0.03 or max_downside > 0.08:
            put_strategies = []
            
            # ATM Put
            atm_strike = current_price
            atm_prob = self._calculate_success_probability(current_price, expected_low, atm_strike, 'put')
            put_strategies.append({
                'strategy': 'Long Put (ATM)',
                'direction': 'Bearish',
                'strike_price': atm_strike,
                'option_type': 'Put',
                'moneyness': 'ATM',
                'days_to_expiry': max_dte,
                'probability_success': atm_prob,
                'breakeven': atm_strike - self._estimate_option_premium(current_price, atm_strike, mean_vol, max_dte, 'put'),
                'max_profit': atm_strike,
                'max_loss': self._estimate_option_premium(current_price, atm_strike, mean_vol, max_dte, 'put'),
                'expected_return': atm_prob * max_downside * 800 if atm_prob > 0.6 else 0,
                'risk_level': 'High'
            })
            
            # OTM Put (-2%)
            otm_strike = current_price * 0.98
            otm_prob = self._calculate_success_probability(current_price, expected_low, otm_strike, 'put')
            put_strategies.append({
                'strategy': 'Long Put (OTM -2%)',
                'direction': 'Bearish',
                'strike_price': otm_strike,
                'option_type': 'Put',
                'moneyness': 'OTM',
                'days_to_expiry': max_dte,
                'probability_success': otm_prob,
                'breakeven': otm_strike - self._estimate_option_premium(current_price, otm_strike, mean_vol, max_dte, 'put'),
                'max_profit': otm_strike,
                'max_loss': self._estimate_option_premium(current_price, otm_strike, mean_vol, max_dte, 'put'),
                'expected_return': otm_prob * max_downside * 1000 if otm_prob > 0.5 else 0,
                'risk_level': 'High'
            })
            
            # Deep OTM Put (-5%)
            dotm_strike = current_price * 0.95
            dotm_prob = self._calculate_success_probability(current_price, expected_low, dotm_strike, 'put')
            put_strategies.append({
                'strategy': 'Long Put (Deep OTM -5%)',
                'direction': 'Very Bearish',
                'strike_price': dotm_strike,
                'option_type': 'Put',
                'moneyness': 'Deep OTM',
                'days_to_expiry': max_dte,
                'probability_success': dotm_prob,
                'breakeven': dotm_strike - self._estimate_option_premium(current_price, dotm_strike, mean_vol, max_dte, 'put'),
                'max_profit': dotm_strike,
                'max_loss': self._estimate_option_premium(current_price, dotm_strike, mean_vol, max_dte, 'put'),
                'expected_return': dotm_prob * max_downside * 1200 if dotm_prob > 0.3 else 0,
                'risk_level': 'Very High'
            })
            
            buying_strategies.extend(put_strategies)
        
        # Sort by expected return
        buying_strategies.sort(key=lambda x: x['expected_return'], reverse=True)
        
        return buying_strategies
    
    def _generate_multi_dte_selling_strategies(self, current_price: float, ohlc_forecast: Dict,
                                             bb_forecast: Dict, vol_forecast: Dict,
                                             markov_predictions: Dict) -> Dict:
        """Generate selling strategies across multiple DTEs (0-10)."""
        dte_range = list(range(0, 11))  # 0DTE to 10DTE
        mean_vol = vol_forecast['mean_volatility']
        
        # Extend forecast to cover longest DTE
        max_dte = max(dte_range)
        extended_forecast = self._extend_forecast_to_expiry(ohlc_forecast, max_dte, current_price, vol_forecast)
        
        selling_strategies = {
            'covered_calls': {},
            'cash_secured_puts': {},
            'iron_condors': {},
            'summary_table': None,
            'forecast_info': {
                'extended_to_days': len(extended_forecast['close']),
                'expected_price_at_10d': extended_forecast['close'][min(9, len(extended_forecast['close'])-1)] if len(extended_forecast['close']) > 9 else current_price
            }
        }
        
        # COVERED CALLS - Use forecast-informed strikes
        if len(extended_forecast['close']) > 5:
            # Base strikes on forecast rather than arbitrary multipliers
            forecast_5d = extended_forecast['close'][min(4, len(extended_forecast['close'])-1)]
            forecast_10d = extended_forecast['close'][min(9, len(extended_forecast['close'])-1)]
            
            # Set strikes above forecasted levels to ensure profitability
            call_strikes = [
                max(current_price * 1.01, forecast_5d * 1.01),   # 1% above 5-day forecast
                max(current_price * 1.02, forecast_5d * 1.02),   # 2% above 5-day forecast  
                max(current_price * 1.03, forecast_10d * 1.01),  # 1% above 10-day forecast
                max(current_price * 1.05, forecast_10d * 1.03)   # 3% above 10-day forecast
            ]
        else:
            # Fallback to original multipliers if forecast is too short
            call_strikes = [current_price * mult for mult in [1.01, 1.02, 1.03, 1.05]]
        
        for dte in dte_range:
            selling_strategies['covered_calls'][f'{dte}DTE'] = []
            
            for strike in call_strikes:
                premium = self._estimate_option_premium(current_price, strike, mean_vol, max(dte, 1), 'call')
                prob_profit = 0.7 - (dte * 0.02)  # Decreases with time
                
                selling_strategies['covered_calls'][f'{dte}DTE'].append({
                    'strike_price': strike,
                    'premium_collected': premium,
                    'max_profit': premium + max(0, strike - current_price) * 100,
                    'probability_profit': max(0.3, prob_profit),
                    'breakeven': current_price - premium,
                    'return_if_assigned': ((strike - current_price + premium) / current_price) * 100,
                    'return_if_expires': (premium / current_price) * 100
                })
        
        # CASH SECURED PUTS
        put_strikes = [current_price * mult for mult in [0.99, 0.98, 0.97, 0.95]]
        for dte in dte_range:
            selling_strategies['cash_secured_puts'][f'{dte}DTE'] = []
            
            for strike in put_strikes:
                premium = self._estimate_option_premium(current_price, strike, mean_vol, max(dte, 1), 'put')
                prob_profit = 0.65 - (dte * 0.015)
                
                selling_strategies['cash_secured_puts'][f'{dte}DTE'].append({
                    'strike_price': strike,
                    'premium_collected': premium,
                    'max_profit': premium,
                    'probability_profit': max(0.3, prob_profit),
                    'breakeven': strike - premium,
                    'return_if_assigned': (premium / strike) * 100,
                    'return_if_expires': (premium / strike) * 100
                })
        
        # IRON CONDORS
        for dte in dte_range:
            wing_width = current_price * 0.03
            put_strike = current_price - wing_width
            call_strike = current_price + wing_width
            
            credit = self._estimate_condor_credit(current_price, put_strike, call_strike, mean_vol, max(dte, 1))
            prob_profit = 0.6 - (dte * 0.02)
            
            selling_strategies['iron_condors'][f'{dte}DTE'] = {
                'put_strike': put_strike,
                'call_strike': call_strike,
                'credit_collected': credit,
                'max_profit': credit,
                'max_loss': wing_width - credit,
                'probability_profit': max(0.3, prob_profit),
                'profit_range': [put_strike + credit, call_strike - credit],
                'return_on_risk': (credit / (wing_width - credit)) * 100 if wing_width > credit else 0
            }
        
        # Create summary table
        selling_strategies['summary_table'] = self._create_selling_summary_table(selling_strategies)
        
        return selling_strategies
    
    def _create_selling_summary_table(self, selling_strategies: Dict) -> pd.DataFrame:
        """Create an attractive summary table for selling strategies."""
        summary_data = []
        
        # Add covered calls summary
        for dte, strategies in selling_strategies['covered_calls'].items():
            best_cc = max(strategies, key=lambda x: x['return_if_expires']) if strategies else None
            if best_cc:
                summary_data.append({
                    'Strategy': 'Covered Call',
                    'DTE': dte,
                    'Strike': f"${best_cc['strike_price']:.2f}",
                    'Premium': f"${best_cc['premium_collected']:.2f}",
                    'Return%': f"{best_cc['return_if_expires']:.2f}%",
                    'Prob_Profit': f"{best_cc['probability_profit']:.1%}",
                    'Max_Profit': f"${best_cc['max_profit']:.2f}"
                })
        
        # Add cash secured puts summary
        for dte, strategies in selling_strategies['cash_secured_puts'].items():
            best_csp = max(strategies, key=lambda x: x['return_if_expires']) if strategies else None
            if best_csp:
                summary_data.append({
                    'Strategy': 'Cash Secured Put',
                    'DTE': dte,
                    'Strike': f"${best_csp['strike_price']:.2f}",
                    'Premium': f"${best_csp['premium_collected']:.2f}",
                    'Return%': f"{best_csp['return_if_expires']:.2f}%",
                    'Prob_Profit': f"{best_csp['probability_profit']:.1%}",
                    'Max_Profit': f"${best_csp['max_profit']:.2f}"
                })
        
        # Add iron condors summary
        for dte, strategy in selling_strategies['iron_condors'].items():
            summary_data.append({
                'Strategy': 'Iron Condor',
                'DTE': dte,
                'Strike': f"${strategy['put_strike']:.2f}/{strategy['call_strike']:.2f}",
                'Premium': f"${strategy['credit_collected']:.2f}",
                'Return%': f"{strategy['return_on_risk']:.2f}%",
                'Prob_Profit': f"{strategy['probability_profit']:.1%}",
                'Max_Profit': f"${strategy['max_profit']:.2f}"
            })
        
        return pd.DataFrame(summary_data)
    
    def display_enhanced_recommendations(self, recommendations: Dict, symbol: str = "STOCK") -> None:
        """Display recommendations in an attractive format."""
        print(f"🎯 ENHANCED OPTIONS ANALYSIS FOR {symbol}")
        print("=" * 80)
        print(f"📊 Analysis Date: {recommendations['analysis_date']}")
        print(f"💰 Current Price: ${recommendations['current_price']:.2f}")
        print()
        
        # BUYING STRATEGIES
        if recommendations['buying_strategies']:
            print("🚀 OPTION BUYING STRATEGIES")
            print("=" * 50)
            
            buying_df = pd.DataFrame(recommendations['buying_strategies'])
            display_cols = ['strategy', 'strike_price', 'days_to_expiry', 'probability_success', 
                          'expected_return', 'max_loss', 'risk_level']
            
            if not buying_df.empty:
                buying_display = buying_df[display_cols].copy()
                buying_display['strike_price'] = buying_display['strike_price'].apply(lambda x: f"${x:.2f}")
                buying_display['probability_success'] = buying_display['probability_success'].apply(lambda x: f"{x:.1%}")
                buying_display['expected_return'] = buying_display['expected_return'].apply(lambda x: f"{x:.1f}%")
                buying_display['max_loss'] = buying_display['max_loss'].apply(lambda x: f"${x:.2f}")
                
                print(buying_display.to_string(index=False))
            print()
        
        # SELLING STRATEGIES
        print("💼 OPTION SELLING STRATEGIES (Multi-DTE Analysis)")
        print("=" * 60)
        
        summary_table = recommendations['selling_strategies']['summary_table']
        if summary_table is not None and not summary_table.empty:
            print(summary_table.to_string(index=False))
        
        print(f"\n✅ Enhanced options analysis complete!")
        print(f"🔄 Strategies range from 0DTE to 10DTE for maximum flexibility")
        print(f"📊 All premiums and probabilities are model-estimated")

def analyze_etf_options(etf_symbol: str, max_dte: int = 45, display_results: bool = True) -> Dict:
    """
    Easy-to-use function for complete ETF multi-DTE options analysis.
    
    Parameters
    ----------
    etf_symbol : str
        ETF symbol (e.g., 'SPY', 'QQQ', 'IWM')
    max_dte : int
        Maximum days to expiry for analysis
    display_results : bool
        Whether to display formatted results
        
    Returns
    -------
    dict
        Complete enhanced options analysis results
    """
    import numpy as np
    from data.loader import get_etf_data
    from indicators.bollinger_bands import calculate_bollinger_bands
    
    print(f"🏦 ANALYZING {etf_symbol} OPTIONS (0-{max_dte} DTE)")
    print("=" * 60)
    
    try:
        # Load ETF data
        etf_data_dict = get_etf_data([etf_symbol], update=False, rate_limit=2.0)
        
        if not etf_data_dict or etf_symbol not in etf_data_dict:
            raise Exception(f"Failed to load {etf_symbol} data")
        
        selected_etf_raw_data = etf_data_dict[etf_symbol]
        
        # Calculate technical indicators
        bb_data_etf = calculate_bollinger_bands(selected_etf_raw_data['Close'], window=20, num_std=2.0)
        
        # Current ETF status
        etf_current_price = selected_etf_raw_data['Close'].iloc[-1]
        etf_current_bb_position = bb_data_etf['BB_Position'].iloc[-1]
        
        print(f"✅ Current {etf_symbol} Price: ${etf_current_price:.2f}")
        print(f"📊 BB Position: {etf_current_bb_position:.3f}")
        
        # Generate ETF-optimized forecasts
        etf_volatility = {
            'SPY': 0.015, 'QQQ': 0.022, 'IWM': 0.025, 
            'ARKK': 0.035, 'XLK': 0.020
        }.get(etf_symbol, 0.020)
        
        np.random.seed(42)  # Reproducible results
        etf_ohlc_forecast = {
            'close': [etf_current_price * (1 + np.random.normal(0, etf_volatility * 0.5)) for _ in range(max_dte)],
            'high': [etf_current_price * (1 + np.random.normal(0.005, etf_volatility)) for _ in range(max_dte)],
            'low': [etf_current_price * (1 + np.random.normal(-0.005, etf_volatility)) for _ in range(max_dte)]
        }
        
        etf_forecasts = {
            'vol_forecast': {
                'mean_volatility': etf_volatility,
                'volatility_trend': 0.0001
            },
            'markov_predictions': {
                'state_probs': [0.15, 0.2, 0.3, 0.2, 0.15],
                'trend_predictions': {'stable': 0.8}
            },
            'bb_forecast': {
                'bb_position_forecast': np.random.choice([1, 2, 3], size=max_dte, p=[0.3, 0.4, 0.3]),
                'bb_width_forecast': np.full(max_dte, 0.03)
            }
        }
        
        # Run enhanced options analysis
        etf_recommender = OptionsStrategyRecommender(
            risk_free_rate=0.05,
            commission_per_contract=1.0,
            min_probability_threshold=0.25
        )
        
        results = etf_recommender.generate_enhanced_recommendations(
            current_price=etf_current_price,
            ohlc_forecast=etf_ohlc_forecast,
            bb_forecast=etf_forecasts['bb_forecast'],
            vol_forecast=etf_forecasts['vol_forecast'],
            markov_predictions=etf_forecasts['markov_predictions'],
            max_dte=max_dte
        )
        
        if display_results:
            etf_recommender.display_enhanced_recommendations(results, etf_symbol)
            
            # ETF-specific summary
            print(f"\n🏦 {etf_symbol} SUMMARY:")
            print(f"  💰 Current Price: ${etf_current_price:.2f}")
            print(f"  📊 Est. Volatility: {etf_volatility:.1%}")
            print(f"  🎯 Buying Strategies: {len(results['buying_strategies'])}")
            print(f"  💼 Selling DTEs: 0-10 DTE coverage")
            print(f"  ⚖️ ETF Advantages: Lower vol, high liquidity, diversification")
        
        return results
        
    except Exception as e:
        print(f"❌ Error analyzing {etf_symbol}: {e}")
        return {}