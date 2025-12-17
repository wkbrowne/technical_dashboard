"""
Enhanced sector and subsector ETF mapping with correlation validation.

This module automatically maps stock symbols to appropriate sector and subsector ETFs
using a combination of sector information from the universe CSV and correlation analysis
for validation and improvement.

Uses parallel processing (200 stocks per worker) for correlation-based subsector discovery.
"""
import logging
import warnings
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

# Import parallel config for stocks_per_worker based parallelism
try:
    from ..config.parallel import calculate_workers_from_items, DEFAULT_STOCKS_PER_WORKER
except ImportError:
    from src.config.parallel import calculate_workers_from_items, DEFAULT_STOCKS_PER_WORKER

logger = logging.getLogger(__name__)

# Direct ticker-to-subsector ETF mapping for well-known stocks
# This is more reliable than keyword matching for major holdings
TICKER_TO_SUBSECTOR = {
    # SMH - Semiconductors (VanEck Semiconductor ETF major holdings + peers)
    'NVDA': 'SMH', 'AMD': 'SMH', 'INTC': 'SMH', 'AVGO': 'SMH', 'QCOM': 'SMH',
    'TXN': 'SMH', 'MU': 'SMH', 'AMAT': 'SMH', 'LRCX': 'SMH', 'KLAC': 'SMH',
    'ADI': 'SMH', 'MRVL': 'SMH', 'NXPI': 'SMH', 'ON': 'SMH', 'MCHP': 'SMH',
    'ASML': 'SMH', 'TSM': 'SMH', 'ARM': 'SMH', 'SNPS': 'SMH', 'CDNS': 'SMH',

    # IGV - Software (iShares Expanded Tech-Software ETF)
    # Also includes major tech companies with significant software/services
    'MSFT': 'IGV', 'ORCL': 'IGV', 'CRM': 'IGV', 'ADBE': 'IGV', 'NOW': 'IGV',
    'INTU': 'IGV', 'PANW': 'IGV', 'CRWD': 'IGV', 'WDAY': 'IGV', 'ADSK': 'IGV',
    'FTNT': 'IGV', 'TEAM': 'IGV', 'DDOG': 'IGV', 'ZS': 'IGV', 'SNOW': 'IGV',
    'PLTR': 'IGV', 'MDB': 'IGV', 'NET': 'IGV', 'HUBS': 'IGV', 'DOCU': 'IGV',
    # Consumer tech with significant services/software revenue
    'AAPL': 'IGV',  # Apple - major services business (App Store, Apple Music, iCloud)
    'META': 'IGV',  # Meta - software/platforms company
    'NFLX': 'IGV',  # Netflix - streaming software platform
    'SPOT': 'IGV',  # Spotify - streaming software
    'UBER': 'IGV',  # Uber - platform/software
    'LYFT': 'IGV',  # Lyft - platform/software
    'ABNB': 'IGV',  # Airbnb - platform/software
    'SNAP': 'IGV',  # Snap - social software

    # SKYY - Cloud Computing
    'AMZN': 'SKYY', 'GOOGL': 'SKYY', 'GOOG': 'SKYY', 'ZM': 'SKYY', 'TWLO': 'SKYY',
    'OKTA': 'SKYY', 'DBX': 'SKYY', 'BOX': 'SKYY', 'SPLK': 'SKYY', 'ESTC': 'SKYY',

    # HACK - Cybersecurity
    'CYBR': 'HACK', 'CHKP': 'HACK', 'QLYS': 'HACK', 'TENB': 'HACK', 'RPD': 'HACK',
    'S': 'HACK', 'VRNS': 'HACK', 'SAIL': 'HACK',

    # KBE - Banks (SPDR S&P Bank ETF)
    'JPM': 'KBE', 'BAC': 'KBE', 'WFC': 'KBE', 'C': 'KBE', 'GS': 'KBE',
    'MS': 'KBE', 'USB': 'KBE', 'PNC': 'KBE', 'TFC': 'KBE', 'COF': 'KBE',
    'BK': 'KBE', 'STT': 'KBE', 'SCHW': 'KBE', 'FITB': 'KBE', 'MTB': 'KBE',

    # KRE - Regional Banks
    'ZION': 'KRE', 'RF': 'KRE', 'HBAN': 'KRE', 'CFG': 'KRE', 'KEY': 'KRE',
    'CMA': 'KRE', 'FHN': 'KRE', 'ALLY': 'KRE', 'WAL': 'KRE', 'EWBC': 'KRE',

    # IBB - Biotech (iShares Biotechnology ETF)
    'AMGN': 'IBB', 'GILD': 'IBB', 'VRTX': 'IBB', 'REGN': 'IBB', 'BIIB': 'IBB',
    'MRNA': 'IBB', 'SGEN': 'IBB', 'ILMN': 'IBB', 'ALNY': 'IBB', 'BMRN': 'IBB',
    'INCY': 'IBB', 'EXEL': 'IBB', 'UTHR': 'IBB', 'SRPT': 'IBB', 'IONS': 'IBB',

    # XBI - Small/Mid Biotech (SPDR S&P Biotech ETF) - more speculative names
    'ACAD': 'XBI', 'RARE': 'XBI', 'NBIX': 'XBI', 'INSM': 'XBI', 'PCVX': 'XBI',
    'RYTM': 'XBI', 'APLS': 'XBI', 'KRYS': 'XBI', 'DVAX': 'XBI', 'TGTX': 'XBI',

    # IHE - Pharma (iShares U.S. Pharmaceuticals ETF)
    'JNJ': 'IHE', 'PFE': 'IHE', 'MRK': 'IHE', 'LLY': 'IHE', 'ABBV': 'IHE',
    'BMY': 'IHE', 'AZN': 'IHE', 'NVO': 'IHE', 'SNY': 'IHE', 'GSK': 'IHE',
    'ZTS': 'IHE', 'VTRS': 'IHE', 'TAK': 'IHE', 'TEVA': 'IHE',

    # ITA - Aerospace & Defense (iShares U.S. Aerospace & Defense ETF)
    'BA': 'ITA', 'LMT': 'ITA', 'RTX': 'ITA', 'NOC': 'ITA', 'GD': 'ITA',
    'GE': 'ITA', 'LHX': 'ITA', 'TDG': 'ITA', 'HII': 'ITA', 'TXT': 'ITA',
    'HWM': 'ITA', 'AXON': 'ITA', 'LDOS': 'ITA', 'KTOS': 'ITA',

    # XOP - Oil & Gas Exploration (SPDR S&P Oil & Gas Exploration ETF)
    'XOM': 'XOP', 'CVX': 'XOP', 'COP': 'XOP', 'EOG': 'XOP', 'PXD': 'XOP',
    'DVN': 'XOP', 'FANG': 'XOP', 'MRO': 'XOP', 'OXY': 'XOP', 'APA': 'XOP',
    'HAL': 'XOP', 'SLB': 'XOP', 'BKR': 'XOP', 'HES': 'XOP',

    # XRT - Retail (SPDR S&P Retail ETF)
    'WMT': 'XRT', 'COST': 'XRT', 'TGT': 'XRT', 'HD': 'XRT', 'LOW': 'XRT',
    'TJX': 'XRT', 'ROST': 'XRT', 'DG': 'XRT', 'DLTR': 'XRT', 'BBY': 'XRT',
    'ORLY': 'XRT', 'AZO': 'XRT', 'TSCO': 'XRT', 'ULTA': 'XRT',

    # ITB/XHB - Homebuilders
    'DHI': 'ITB', 'LEN': 'ITB', 'PHM': 'ITB', 'NVR': 'ITB', 'TOL': 'ITB',
    'KBH': 'ITB', 'MDC': 'ITB', 'TMHC': 'ITB', 'MTH': 'ITB', 'MHO': 'ITB',

    # TAN - Solar
    'ENPH': 'TAN', 'FSLR': 'TAN', 'SEDG': 'TAN', 'RUN': 'TAN', 'NOVA': 'TAN',
    'ARRY': 'TAN', 'MAXN': 'TAN', 'SPWR': 'TAN',

    # LIT - Lithium/Battery
    'ALB': 'LIT', 'SQM': 'LIT', 'LTHM': 'LIT', 'LAC': 'LIT', 'PLL': 'LIT',
    'RIVN': 'LIT', 'LCID': 'LIT',  # EV plays in lithium ETF

    # URA - Uranium
    'CCJ': 'URA', 'UEC': 'URA', 'DNN': 'URA', 'NXE': 'URA', 'UUUU': 'URA',
}

# Subsector ETF keywords for matching (user-specified 22 ETFs)
# Used as fallback when ticker not in TICKER_TO_SUBSECTOR
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


def _process_symbol_mapping_batch(
    work_items: List[Tuple[str, Dict, str]],
    batch_stock_data: Dict[str, pd.DataFrame],
    etf_data: Dict[str, pd.DataFrame]
) -> List[Tuple[str, Dict]]:
    """
    Process a batch of symbols for sector/subsector mapping.

    Args:
        work_items: List of (symbol, symbol_info, base_sector) tuples
        batch_stock_data: Stock price data dict (pre-filtered to batch symbols only)
        etf_data: ETF price data dict (full - ETFs are few and needed for all correlation tests)

    Returns:
        List of (symbol, mapping_dict) tuples
    """
    results = []
    for symbol, symbol_info, base_sector in work_items:
        sector_name = base_sector.lower()
        sector_etf = SECTOR_ETF_MAP.get(sector_name, "SPY")

        # Get equal-weighted equivalent
        equal_weight_etf = EQUAL_WEIGHT_ETF_MAP.get(sector_etf)

        # Find best subsector ETF using multi-stage approach
        subsector_etf = _find_best_subsector_etf(
            symbol, symbol_info, sector_etf, batch_stock_data, etf_data
        )

        # Calculate correlations for validation
        correlations = _calculate_correlations(
            symbol, sector_etf, subsector_etf,
            batch_stock_data, etf_data, equal_weight_etf
        )

        # Determine confidence level
        confidence = _assess_mapping_confidence(correlations, sector_etf, subsector_etf)

        mapping = {
            'csv_sector': base_sector,
            'sector_etf': sector_etf,
            'equal_weight_etf': equal_weight_etf,
            'subsector_etf': subsector_etf,
            'correlations': correlations,
            'confidence': confidence,
            'market_cap': symbol_info.get('market_cap', 0)
        }

        results.append((symbol, mapping))

    return results


def build_enhanced_sector_mappings(universe_csv: str, stock_data: Dict[str, pd.DataFrame],
                                   etf_data: Dict[str, pd.DataFrame],
                                   base_sectors: Dict[str, str],
                                   n_jobs: int = -1) -> Dict[str, Dict]:
    """
    Create comprehensive sector/subsector mappings with keyword matching and correlation validation.

    Uses parallel processing (200 stocks per worker) for correlation-based subsector discovery.

    Args:
        universe_csv: Path to universe CSV with sector and industry information
        stock_data: Dictionary of stock price DataFrames {symbol: df}
        etf_data: Dictionary of ETF price DataFrames {etf: df}
        base_sectors: Base sector mapping from universe CSV {symbol: sector}
        n_jobs: Number of parallel jobs (-1 for all cores)

    Returns:
        Enhanced mapping dictionary with equal-weighted and subsector ETF assignments
    """
    from joblib import Parallel, delayed

    logger.info("Building enhanced sector/subsector mappings (ticker lookup + keyword + correlation)...")

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

    # Build work items: (symbol, symbol_info, base_sector) tuples
    work_items = []
    for symbol in stock_data.keys():
        if symbol not in base_sectors:
            continue
        work_items.append((symbol, symbol_info.get(symbol, {}), base_sectors[symbol]))

    n_symbols = len(work_items)
    logger.info(f"Processing {n_symbols} symbols for sector/subsector mapping")

    # Calculate workers based on stocks_per_worker (200 stocks/worker default)
    chunk_size = DEFAULT_STOCKS_PER_WORKER
    n_workers = calculate_workers_from_items(n_symbols, items_per_worker=chunk_size)

    # Sequential for small datasets
    if n_symbols < 20 or n_jobs == 1 or n_workers == 1:
        all_results = _process_symbol_mapping_batch(work_items, stock_data, etf_data)
    else:
        # Parallel with batching
        chunks = [work_items[i:i + chunk_size] for i in range(0, n_symbols, chunk_size)]

        # Pre-subset stock_data per batch to avoid sending entire dataset to each worker
        # ETF data is small (few dozen ETFs) and needed for correlation tests, so pass full dict
        # See Section 4 of FEATURE_PIPELINE_ARCHITECTURE.md for parallelism best practices
        chunk_stock_data = []
        for chunk in chunks:
            batch_symbols = {item[0] for item in chunk}  # Extract symbols from work_items
            batch_data = {sym: stock_data[sym] for sym in batch_symbols if sym in stock_data}
            chunk_stock_data.append(batch_data)

        total_stock_symbols = len(stock_data)
        avg_symbols_per_batch = sum(len(d) for d in chunk_stock_data) / len(chunk_stock_data) if chunk_stock_data else 0
        logger.info(f"Parallel sector mapping: {len(chunks)} batches, {n_workers} workers")
        logger.info(f"Stock data subsetting: {total_stock_symbols} total -> {avg_symbols_per_batch:.0f} avg symbols/batch")

        try:
            batch_results = Parallel(
                n_jobs=n_workers,
                backend='loky',
                verbose=0
            )(
                delayed(_process_symbol_mapping_batch)(chunk, chunk_stock_data[i], etf_data)
                for i, chunk in enumerate(chunks)
            )

            # Flatten results
            all_results = []
            for batch in batch_results:
                all_results.extend(batch)

        except Exception as e:
            logger.warning(f"Parallel sector mapping failed ({e}), falling back to sequential")
            all_results = _process_symbol_mapping_batch(work_items, stock_data, etf_data)

    # Aggregate results into dict
    enhanced_mappings = {}
    for symbol, mapping in all_results:
        enhanced_mappings[symbol] = mapping

    symbols_processed = len(enhanced_mappings)
    logger.info(f"Enhanced mapping completed for {symbols_processed} symbols")
    if symbols_processed > 0:
        logger.info(f"Equal-weight coverage: {sum(1 for m in enhanced_mappings.values() if m['equal_weight_etf']) / symbols_processed:.1%}")
        logger.info(f"Subsector coverage: {sum(1 for m in enhanced_mappings.values() if m['subsector_etf']) / symbols_processed:.1%}")

    return enhanced_mappings


def _find_best_subsector_by_correlation(
    symbol: str,
    sector_etf: str,
    stock_data: Dict,
    etf_data: Dict,
    min_improvement: float = 0.03,
    min_data_points: int = 100
) -> Optional[str]:
    """
    Find the best subsector ETF by testing correlation improvement over sector ETF.

    This function tests ALL available subsector ETFs and returns the one with the
    highest correlation that also improves over the sector ETF correlation.

    Args:
        symbol: Stock symbol
        sector_etf: The stock's assigned sector ETF
        stock_data: Stock price data dict
        etf_data: ETF price data dict
        min_improvement: Minimum correlation improvement over sector required (default 0.03)
        min_data_points: Minimum overlapping data points required (default 100)

    Returns:
        Best subsector ETF symbol if it improves over sector, else None
    """
    if symbol not in stock_data:
        return None

    stock_df = stock_data[symbol]
    if 'adjclose' not in stock_df.columns:
        return None

    stock_returns = stock_df['adjclose'].pct_change(fill_method=None).dropna()
    if len(stock_returns) < min_data_points:
        return None

    # Calculate sector correlation as baseline
    sector_corr = 0.0
    if sector_etf in etf_data and 'adjclose' in etf_data[sector_etf].columns:
        sector_returns = etf_data[sector_etf]['adjclose'].pct_change(fill_method=None).dropna()
        common_idx = stock_returns.index.intersection(sector_returns.index)
        if len(common_idx) >= min_data_points:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    sector_corr = stock_returns.loc[common_idx].corr(sector_returns.loc[common_idx])
            except:
                pass

    # Test all subsector ETFs
    best_subsector = None
    best_improvement = 0.0

    all_subsector_etfs = set(SUBSECTOR_ETF_KEYWORDS.keys())

    for subsector_etf in all_subsector_etfs:
        if subsector_etf not in etf_data:
            continue
        if 'adjclose' not in etf_data[subsector_etf].columns:
            continue

        subsector_returns = etf_data[subsector_etf]['adjclose'].pct_change(fill_method=None).dropna()
        common_idx = stock_returns.index.intersection(subsector_returns.index)

        if len(common_idx) < min_data_points:
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                subsector_corr = stock_returns.loc[common_idx].corr(subsector_returns.loc[common_idx])
            improvement = subsector_corr - sector_corr

            # Only consider if it improves over sector by at least min_improvement
            if improvement >= min_improvement and improvement > best_improvement:
                best_improvement = improvement
                best_subsector = subsector_etf
        except:
            continue

    return best_subsector


def _find_best_subsector_etf(
    symbol: str,
    symbol_info: Dict,
    sector_etf: str,
    stock_data: Dict,
    etf_data: Dict
) -> Optional[str]:
    """
    Find the best subsector ETF using a multi-stage approach:
    1. Direct ticker mapping (most reliable for well-known stocks)
    2. Keyword matching from description/industry
    3. Correlation-based discovery (tests all subsector ETFs for improvement over sector)

    Args:
        symbol: Stock symbol
        symbol_info: Symbol metadata (description, industry, etc.)
        sector_etf: The stock's assigned sector ETF
        stock_data: Stock price data (for correlation analysis)
        etf_data: ETF price data (for correlation analysis)

    Returns:
        Best matching subsector ETF symbol or None
    """
    # Stage 1: Direct ticker-to-subsector mapping (most reliable)
    if symbol in TICKER_TO_SUBSECTOR:
        return TICKER_TO_SUBSECTOR[symbol]

    # Stage 2: Keyword matching from description/industry
    description = symbol_info.get('description', '').lower()
    industry = symbol_info.get('industry', '').lower()
    combined_text = f"{description} {industry}".strip()

    if combined_text:
        # Score all subsector ETFs by keyword matching
        candidates = []
        for subsector_etf, keywords in SUBSECTOR_ETF_KEYWORDS.items():
            keyword_score = sum(1 for keyword in keywords if keyword.lower() in combined_text)
            if keyword_score > 0:
                candidates.append((subsector_etf, keyword_score))

        if candidates:
            # Sort by score (highest first)
            candidates.sort(key=lambda x: x[1], reverse=True)
            # Return top keyword match
            return candidates[0][0]

    # Stage 3: Correlation-based discovery
    # Test all subsector ETFs and find one that improves correlation over sector
    return _find_best_subsector_by_correlation(
        symbol, sector_etf, stock_data, etf_data,
        min_improvement=0.03,  # Require at least 3% correlation improvement
        min_data_points=100
    )


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
    
    stock_returns = stock_data[symbol]['adjclose'].pct_change(fill_method=None).dropna()
    
    # Market correlation (SPY)
    if 'SPY' in etf_data and 'adjclose' in etf_data['SPY'].columns:
        spy_returns = etf_data['SPY']['adjclose'].pct_change(fill_method=None).dropna()
        common_index = stock_returns.index.intersection(spy_returns.index)
        if len(common_index) > 100:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    correlations['market'] = stock_returns.loc[common_index].corr(
                        spy_returns.loc[common_index]
                    )
            except:
                pass

    # Equal-weighted market correlation (RSP)
    if 'RSP' in etf_data and 'adjclose' in etf_data['RSP'].columns:
        rsp_returns = etf_data['RSP']['adjclose'].pct_change(fill_method=None).dropna()
        common_index = stock_returns.index.intersection(rsp_returns.index)
        if len(common_index) > 100:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    correlations['market_ew'] = stock_returns.loc[common_index].corr(
                        rsp_returns.loc[common_index]
                    )
            except:
                pass

    # Sector correlation (cap-weighted)
    if sector_etf in etf_data and 'adjclose' in etf_data[sector_etf].columns:
        sector_returns = etf_data[sector_etf]['adjclose'].pct_change(fill_method=None).dropna()
        common_index = stock_returns.index.intersection(sector_returns.index)
        if len(common_index) > 100:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    correlations['sector'] = stock_returns.loc[common_index].corr(
                        sector_returns.loc[common_index]
                    )
            except:
                pass

    # Equal-weighted sector correlation
    if equal_weight_etf and equal_weight_etf in etf_data and 'adjclose' in etf_data[equal_weight_etf].columns:
        ew_sector_returns = etf_data[equal_weight_etf]['adjclose'].pct_change(fill_method=None).dropna()
        common_index = stock_returns.index.intersection(ew_sector_returns.index)
        if len(common_index) > 100:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    correlations['sector_ew'] = stock_returns.loc[common_index].corr(
                        ew_sector_returns.loc[common_index]
                    )
            except:
                pass

    # Subsector correlation
    if subsector_etf and subsector_etf in etf_data and 'adjclose' in etf_data[subsector_etf].columns:
        subsector_returns = etf_data[subsector_etf]['adjclose'].pct_change(fill_method=None).dropna()
        common_index = stock_returns.index.intersection(subsector_returns.index)
        if len(common_index) > 100:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
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