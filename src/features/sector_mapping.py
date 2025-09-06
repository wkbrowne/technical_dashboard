"""
Enhanced sector and subsector ETF mapping with correlation validation.

This module automatically maps stock symbols to appropriate sector and subsector ETFs
using a combination of sector information from the universe CSV and correlation analysis
for validation and improvement.
"""
import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Subsector ETF keywords for matching (user-specified 22 ETFs)
SUBSECTOR_ETF_KEYWORDS = {
    # Technology subsectors
    'SMH': ['semiconductor', 'chip', 'memory', 'processor', 'intel', 'nvidia', 'amd', 'micron', 'broadcom'],
    'SKYY': ['cloud', 'saas', 'software', 'platform', 'computing', 'salesforce', 'microsoft', 'amazon web'],
    'HACK': ['cyber', 'security', 'firewall', 'antivirus', 'palo alto', 'symantec', 'mcafee'],
    'IGV': ['software', 'application', 'enterprise', 'database', 'oracle', 'sap', 'adobe'],
    
    # Financial subsectors  
    'KBE': ['bank', 'banking', 'jpmorgan', 'wells fargo', 'bank of america', 'citigroup'],
    'KRE': ['regional bank', 'community bank', 'zions', 'regions', 'fifth third', 'huntington'],
    
    # Healthcare/Biotech subsectors
    'IBB': ['biotech', 'biotechnology', 'biogen', 'gilead', 'amgen', 'vertex', 'regeneron'],
    'IHE': ['pharma', 'pharmaceutical', 'drug', 'pfizer', 'merck', 'johnson', 'abbott'],
    'XBI': ['biotech', 'small biotech', 'emerging biotech', 'clinical', 'therapeutics'],
    'PJP': ['pharma', 'pharmaceutical', 'medicine', 'healthcare'],
    
    # Industrial subsectors
    'ITA': ['aerospace', 'defense', 'aviation', 'boeing', 'lockheed', 'raytheon', 'northrop'],
    'XAR': ['aerospace', 'defense', 'military', 'contractor'],
    'ITB': ['homebuilder', 'construction', 'building', 'home depot', 'lowes', 'residential'],
    'XHB': ['homebuilder', 'housing', 'construction', 'building materials'],
    
    # Energy subsectors
    'XOP': ['oil', 'exploration', 'production', 'drilling', 'petroleum', 'exxon', 'chevron'],
    'XTN': ['transportation', 'pipeline', 'midstream', 'energy infrastructure'],
    
    # Consumer subsectors
    'XRT': ['retail', 'store', 'shopping', 'consumer', 'walmart', 'target', 'costco'],
    
    # Clean energy/materials
    'TAN': ['solar', 'renewable', 'clean energy', 'photovoltaic', 'green energy'],
    'ICLN': ['clean energy', 'renewable', 'wind', 'solar', 'green', 'sustainable'],
    'URA': ['uranium', 'nuclear', 'mining', 'energy'],
    'LIT': ['lithium', 'battery', 'electric vehicle', 'energy storage'],
    'COPX': ['copper', 'mining', 'metals', 'commodities']
}

# Equal-weighted ETF mapping (cap-weighted -> equal-weighted)
EQUAL_WEIGHT_ETF_MAP = {
    'SPY': 'RSP',    # S&P 500 -> S&P 500 Equal Weight
    'XLK': 'RYT',    # Technology -> Technology Equal Weight  
    'XLF': 'RYF',    # Financial -> Financial Equal Weight
    'XLE': 'RYE',    # Energy -> Energy Equal Weight
    'XLI': 'RYH',    # Industrial -> Industrial Equal Weight
    'XLU': 'RYU',    # Utilities -> Utilities Equal Weight
    'XLV': 'RHS',    # Healthcare -> Healthcare Equal Weight
    'XLB': 'RTM',    # Materials -> Materials Equal Weight
    'XLRE': 'EWRE'   # Real Estate -> Real Estate Equal Weight
}

# Standard sector ETF mapping (enhanced from orchestrator.py)
SECTOR_ETF_MAP = {
    "technology services": "XLK",
    "electronic technology": "XLK", 
    "finance": "XLF",
    "retail trade": "XLY",
    "health technology": "XLV",
    "consumer non-durables": "XLP",
    "producer manufacturing": "XLI",
    "energy minerals": "XLE",
    "consumer services": "XLY",
    "consumer durables": "XLY",
    "utilities": "XLU",
    "non-energy minerals": "XLB",
    "industrial services": "XLI",
    "transportation": "IYT",
    "commercial services": "XLC",
    "process industries": "XLB", 
    "communications": "XLC",
    "health services": "XLV",
    "distribution services": "XLI",
    "miscellaneous": "SPY",
}


def build_enhanced_sector_mappings(universe_csv: str, stock_data: Dict[str, pd.DataFrame], 
                                 etf_data: Dict[str, pd.DataFrame], 
                                 base_sectors: Dict[str, str]) -> Dict[str, Dict]:
    """
    Create comprehensive sector/subsector mappings with keyword matching and correlation validation.
    
    Args:
        universe_csv: Path to universe CSV with sector and industry information
        stock_data: Dictionary of stock price DataFrames {symbol: df}
        etf_data: Dictionary of ETF price DataFrames {etf: df}
        base_sectors: Base sector mapping from universe CSV {symbol: sector}
        
    Returns:
        Enhanced mapping dictionary with equal-weighted and subsector ETF assignments
    """
    logger.info("Building enhanced sector/subsector mappings with keyword matching...")
    
    # Load universe data with simplified field extraction
    universe_df = pd.read_csv(universe_csv)
    universe_df['Symbol'] = universe_df['Symbol'].astype(str)
    
    # Extract fields with exact matching (log errors if not found)
    required_fields = ['Description', 'Industry', 'Market capitalization', 'Sector']
    missing_fields = [f for f in required_fields if f not in universe_df.columns]
    if missing_fields:
        logger.error(f"Missing required fields in universe CSV: {missing_fields}")
    
    # Create symbol info lookup with simplified field access
    symbol_info = {}
    for _, row in universe_df.iterrows():
        info = {}
        try:
            info['description'] = row.get('Description', '').lower() if 'Description' in universe_df.columns else ''
            info['industry'] = row.get('Industry', '').lower() if 'Industry' in universe_df.columns else ''
            info['market_cap'] = pd.to_numeric(row.get('Market capitalization', 0), errors='coerce') or 0 if 'Market capitalization' in universe_df.columns else 0
            info['sector'] = row.get('Sector', '').lower() if 'Sector' in universe_df.columns else ''
        except Exception as e:
            logger.error(f"Error processing row for {row.get('Symbol', 'unknown')}: {e}")
            continue
        symbol_info[row['Symbol']] = info
    
    enhanced_mappings = {}
    symbols_processed = 0
    
    for symbol in stock_data.keys():
        if symbol not in base_sectors:
            continue
            
        sector_name = base_sectors[symbol].lower()
        sector_etf = SECTOR_ETF_MAP.get(sector_name, "SPY")
        
        # Get equal-weighted equivalent
        equal_weight_etf = EQUAL_WEIGHT_ETF_MAP.get(sector_etf)
        
        # Find best subsector ETF using keyword matching
        subsector_etf = _find_best_subsector_etf_by_keywords(
            symbol, symbol_info.get(symbol, {}), stock_data, etf_data
        )
        
        # Calculate correlations for validation
        correlations = _calculate_correlations(symbol, sector_etf, subsector_etf, 
                                            stock_data, etf_data, equal_weight_etf)
        
        # Determine confidence level
        confidence = _assess_mapping_confidence(correlations, sector_etf, subsector_etf)
        
        enhanced_mappings[symbol] = {
            'csv_sector': base_sectors[symbol],
            'sector_etf': sector_etf,
            'equal_weight_etf': equal_weight_etf,
            'subsector_etf': subsector_etf,
            'correlations': correlations,
            'confidence': confidence,
            'market_cap': symbol_info.get(symbol, {}).get('market_cap', 0)
        }
        
        symbols_processed += 1
    
    logger.info(f"Enhanced mapping completed for {symbols_processed} symbols")
    logger.info(f"Equal-weight coverage: {sum(1 for m in enhanced_mappings.values() if m['equal_weight_etf']) / len(enhanced_mappings):.1%}")
    logger.info(f"Subsector coverage: {sum(1 for m in enhanced_mappings.values() if m['subsector_etf']) / len(enhanced_mappings):.1%}")
    return enhanced_mappings


def _find_best_subsector_etf_by_keywords(symbol: str, symbol_info: Dict, 
                                        stock_data: Dict, etf_data: Dict) -> Optional[str]:
    """
    Find the best subsector ETF for a symbol using keyword matching across description and industry.
    
    Args:
        symbol: Stock symbol
        symbol_info: Symbol metadata (description, industry, etc.)
        stock_data: Stock price data (for correlation tie-breaking)
        etf_data: ETF price data (for correlation tie-breaking)
        
    Returns:
        Best matching subsector ETF symbol or None
    """
    description = symbol_info.get('description', '').lower()
    industry = symbol_info.get('industry', '').lower()
    combined_text = f"{description} {industry}".strip()
    
    if not combined_text:
        return None
    
    # Score all subsector ETFs by keyword matching
    candidates = []
    for subsector_etf, keywords in SUBSECTOR_ETF_KEYWORDS.items():
        keyword_score = sum(1 for keyword in keywords if keyword.lower() in combined_text)
        if keyword_score > 0:
            candidates.append((subsector_etf, keyword_score))
    
    if not candidates:
        return None
    
    # Sort by score (highest first)
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # If single best match or clear winner, return it
    if len(candidates) == 1 or candidates[0][1] > candidates[1][1]:
        return candidates[0][0]
    
    # Multiple candidates with same score - use correlation for tie-breaking
    best_candidates = [c for c in candidates if c[1] == candidates[0][1]]
    
    if symbol not in stock_data:
        return best_candidates[0][0]  # Just return first if no price data
    
    stock_returns = stock_data[symbol].get('adjclose')
    if stock_returns is None:
        return best_candidates[0][0]
    
    stock_returns = stock_returns.pct_change().dropna()
    best_subsector = None
    best_correlation = -1
    
    for subsector_etf, _ in best_candidates:
        if subsector_etf in etf_data and 'adjclose' in etf_data[subsector_etf].columns:
            etf_returns = etf_data[subsector_etf]['adjclose'].pct_change().dropna()
            
            # Align time series
            common_index = stock_returns.index.intersection(etf_returns.index)
            if len(common_index) > 100:  # Need sufficient data
                try:
                    correlation = stock_returns.loc[common_index].corr(
                        etf_returns.loc[common_index]
                    )
                    if correlation > best_correlation:
                        best_correlation = correlation
                        best_subsector = subsector_etf
                except:
                    continue
    
    return best_subsector or best_candidates[0][0]


def _calculate_correlations(symbol: str, sector_etf: str, subsector_etf: Optional[str],
                          stock_data: Dict, etf_data: Dict, equal_weight_etf: Optional[str] = None) -> Dict[str, float]:
    """
    Calculate correlation between symbol and its assigned ETFs (including equal-weighted).
    
    Returns:
        Dictionary with correlation values for different benchmarks
    """
    correlations = {}
    
    if symbol not in stock_data or 'adjclose' not in stock_data[symbol].columns:
        return correlations
    
    stock_returns = stock_data[symbol]['adjclose'].pct_change().dropna()
    
    # Market correlation (SPY)
    if 'SPY' in etf_data and 'adjclose' in etf_data['SPY'].columns:
        spy_returns = etf_data['SPY']['adjclose'].pct_change().dropna()
        common_index = stock_returns.index.intersection(spy_returns.index)
        if len(common_index) > 100:
            try:
                correlations['market'] = stock_returns.loc[common_index].corr(
                    spy_returns.loc[common_index]
                )
            except:
                pass
    
    # Equal-weighted market correlation (RSP)
    if 'RSP' in etf_data and 'adjclose' in etf_data['RSP'].columns:
        rsp_returns = etf_data['RSP']['adjclose'].pct_change().dropna()
        common_index = stock_returns.index.intersection(rsp_returns.index)
        if len(common_index) > 100:
            try:
                correlations['market_ew'] = stock_returns.loc[common_index].corr(
                    rsp_returns.loc[common_index]
                )
            except:
                pass
    
    # Sector correlation (cap-weighted)
    if sector_etf in etf_data and 'adjclose' in etf_data[sector_etf].columns:
        sector_returns = etf_data[sector_etf]['adjclose'].pct_change().dropna()
        common_index = stock_returns.index.intersection(sector_returns.index)
        if len(common_index) > 100:
            try:
                correlations['sector'] = stock_returns.loc[common_index].corr(
                    sector_returns.loc[common_index]
                )
            except:
                pass
    
    # Equal-weighted sector correlation
    if equal_weight_etf and equal_weight_etf in etf_data and 'adjclose' in etf_data[equal_weight_etf].columns:
        ew_sector_returns = etf_data[equal_weight_etf]['adjclose'].pct_change().dropna()
        common_index = stock_returns.index.intersection(ew_sector_returns.index)
        if len(common_index) > 100:
            try:
                correlations['sector_ew'] = stock_returns.loc[common_index].corr(
                    ew_sector_returns.loc[common_index]
                )
            except:
                pass
    
    # Subsector correlation
    if subsector_etf and subsector_etf in etf_data and 'adjclose' in etf_data[subsector_etf].columns:
        subsector_returns = etf_data[subsector_etf]['adjclose'].pct_change().dropna()
        common_index = stock_returns.index.intersection(subsector_returns.index)
        if len(common_index) > 100:
            try:
                correlations['subsector'] = stock_returns.loc[common_index].corr(
                    subsector_returns.loc[common_index]
                )
            except:
                pass
    
    return correlations


def _assess_mapping_confidence(correlations: Dict[str, float], sector_etf: str, 
                             subsector_etf: Optional[str]) -> str:
    """
    Assess confidence level of sector/subsector mapping based on correlations.
    
    Args:
        correlations: Dictionary of correlation values
        sector_etf: Assigned sector ETF
        subsector_etf: Assigned subsector ETF (if any)
        
    Returns:
        Confidence level: 'high', 'medium', 'low'
    """
    sector_corr = correlations.get('sector', 0)
    subsector_corr = correlations.get('subsector', 0)
    market_corr = correlations.get('market', 0)
    
    # High confidence: good sector correlation, subsector improves on sector
    if sector_corr > 0.6:
        if subsector_etf and subsector_corr > sector_corr + 0.05:
            return 'high'
        elif not subsector_etf:
            return 'high'
        else:
            return 'medium'
    
    # Medium confidence: reasonable sector correlation
    elif sector_corr > 0.4:
        return 'medium'
    
    # Low confidence: poor sector correlation
    else:
        return 'low'


def validate_sector_assignments(enhanced_mappings: Dict[str, Dict]) -> Dict[str, any]:
    """
    Validate sector assignments and identify potential improvements.
    
    Args:
        enhanced_mappings: Output from build_enhanced_sector_mappings()
        
    Returns:
        Validation report with statistics and recommendations
    """
    total_symbols = len(enhanced_mappings)
    confidence_counts = {'high': 0, 'medium': 0, 'low': 0}
    sector_improvements = []
    subsector_improvements = []
    
    for symbol, mapping in enhanced_mappings.items():
        confidence_counts[mapping['confidence']] += 1
        
        correlations = mapping['correlations']
        sector_corr = correlations.get('sector', 0)
        subsector_corr = correlations.get('subsector', 0)
        market_corr = correlations.get('market', 0)
        
        # Check if subsector provides improvement
        if subsector_corr > sector_corr + 0.05:
            subsector_improvements.append({
                'symbol': symbol,
                'sector_corr': sector_corr,
                'subsector_corr': subsector_corr,
                'improvement': subsector_corr - sector_corr,
                'subsector_etf': mapping['subsector_etf']
            })
        
        # Flag potential sector mis-assignments
        if sector_corr < market_corr - 0.1:
            sector_improvements.append({
                'symbol': symbol,
                'sector_etf': mapping['sector_etf'],
                'sector_corr': sector_corr,
                'market_corr': market_corr,
                'needs_review': True
            })
    
    return {
        'total_symbols': total_symbols,
        'confidence_distribution': confidence_counts,
        'avg_sector_correlation': np.mean([
            m['correlations'].get('sector', 0) for m in enhanced_mappings.values()
        ]),
        'subsector_improvements': subsector_improvements[:10],  # Top 10
        'sector_review_needed': sector_improvements[:10],  # Top 10
        'subsector_coverage': sum(1 for m in enhanced_mappings.values() 
                                if m['subsector_etf'] is not None) / total_symbols
    }


def get_required_etfs(enhanced_mappings: Dict[str, Dict]) -> List[str]:
    """
    Get list of all ETFs required for the enhanced mappings.
    
    Args:
        enhanced_mappings: Enhanced mapping dictionary
        
    Returns:
        List of unique ETF symbols needed
    """
    etfs = set(['SPY'])  # Always include market ETF
    
    for mapping in enhanced_mappings.values():
        etfs.add(mapping['sector_etf'])
        if mapping['subsector_etf']:
            etfs.add(mapping['subsector_etf'])
    
    return sorted(list(etfs))


def cache_enhanced_mappings(enhanced_mappings: Dict[str, Dict], 
                          cache_path: str) -> None:
    """
    Cache enhanced mappings to avoid recomputation.
    
    Args:
        enhanced_mappings: Mappings to cache
        cache_path: Path to save cache file
    """
    import pickle
    
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(cache_file, 'wb') as f:
        pickle.dump(enhanced_mappings, f)
    
    logger.debug(f"Cached enhanced mappings to {cache_path}")


def load_cached_mappings(cache_path: str) -> Optional[Dict[str, Dict]]:
    """
    Load cached enhanced mappings.
    
    Args:
        cache_path: Path to cache file
        
    Returns:
        Cached mappings or None if not available
    """
    import pickle
    
    cache_file = Path(cache_path)
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, 'rb') as f:
            mappings = pickle.load(f)
        logger.debug(f"Loaded cached mappings from {cache_path}")
        return mappings
    except Exception as e:
        logger.warning(f"Failed to load cached mappings: {e}")
        return None