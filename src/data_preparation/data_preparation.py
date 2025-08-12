#!/usr/bin/env python3
"""
Data preparation pipeline - main entry point.

This is the main CLI entry point for the feature computation pipeline.
It provides a thin wrapper around the orchestrator module.

Usage:
    python data_preparation.py
"""
import logging
import os
import sys
import warnings
from pathlib import Path

# Set NumExpr to use all available cores instead of the conservative default
if 'NUMEXPR_MAX_THREADS' not in os.environ:
    import multiprocessing
    os.environ['NUMEXPR_MAX_THREADS'] = str(multiprocessing.cpu_count())

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Suppress specific warnings  
warnings.filterwarnings("ignore", message="RANSAC did not reach consensus, using numpy's polyfit")
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the data preparation pipeline."""
    logger.info("Starting data preparation pipeline")
    
    try:
        # Import and run the orchestrator
        from src.pipelines.orchestrator import main as run_orchestrator
        run_orchestrator(include_sectors=True)
        
    except ImportError as e:
        logger.error(f"Failed to import orchestrator: {e}")
        logger.error("Make sure the src package is properly installed")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)
    
    logger.info("Data preparation pipeline completed successfully")


if __name__ == "__main__":
    main()