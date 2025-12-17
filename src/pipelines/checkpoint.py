"""
Checkpoint Manager for Feature Pipeline

Enables staged processing with intermediate outputs after each thematic computation step.
This allows:
1. Memory clearing between stages (via del + gc.collect())
2. Resumption from any checkpoint if pipeline fails
3. Debugging of specific stages in isolation
4. Incremental development - modify one stage without re-running all

Checkpoint stages (in order):
    01_single_stock      - Single-stock features (RSI, MACD, ATR, MA slopes, volume ratios)
    02_cross_sectional   - Cross-sectional features (alpha, beta, sector features)
    03_spread_features   - Spread features (QQQ-SPY, RSP-SPY, etc.)
    04_weekly_features   - Higher timeframe features (w_* prefixed)
    05_weekly_cs         - Weekly cross-sectional features
    06_weekly_spread     - Weekly spread features
    07_interpolated      - After NaN interpolation
    08_targets           - After target generation
    09_final             - Final merged output

Usage:
    # Enable checkpoints
    python -m src.cli.compute --checkpoint-dir artifacts/checkpoints

    # Resume from a specific checkpoint
    python -m src.cli.compute --resume-from 03_spread_features

    # Clean up checkpoints after success
    python -m src.cli.compute --cleanup-checkpoints
"""

import gc
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# Checkpoint stage definitions (ordered)
CHECKPOINT_STAGES = [
    ("01_single_stock", "Single-stock features"),
    ("02_cross_sectional", "Cross-sectional features"),
    ("03_spread_features", "Daily spread features"),
    ("04_weekly_features", "Higher timeframe features"),
    ("05_weekly_cs", "Weekly cross-sectional features"),
    ("06_weekly_spread", "Weekly spread features"),
    ("07_interpolated", "NaN interpolation"),
    ("08_targets", "Target generation"),
    ("09_final", "Final output"),
]

STAGE_ORDER = {name: idx for idx, (name, _) in enumerate(CHECKPOINT_STAGES)}


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management."""

    enabled: bool = False
    checkpoint_dir: Optional[Path] = None
    resume_from: Optional[str] = None
    cleanup_on_success: bool = False
    save_metadata: bool = True

    # Memory management
    clear_memory_after_save: bool = True

    def __post_init__(self):
        if self.checkpoint_dir is not None:
            self.checkpoint_dir = Path(self.checkpoint_dir)
            self.enabled = True

    @classmethod
    def from_args(cls, checkpoint_dir: Optional[str] = None,
                  resume_from: Optional[str] = None,
                  cleanup_checkpoints: bool = False) -> 'CheckpointConfig':
        """Create config from CLI arguments."""
        return cls(
            enabled=checkpoint_dir is not None or resume_from is not None,
            checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
            resume_from=resume_from,
            cleanup_on_success=cleanup_checkpoints,
        )


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint file."""

    stage: str
    timestamp: str
    n_symbols: int
    n_columns: int
    date_min: Optional[str] = None
    date_max: Optional[str] = None
    stage_time_seconds: float = 0.0
    cumulative_time_seconds: float = 0.0
    memory_mb: float = 0.0

    def to_dict(self) -> dict:
        return {
            'stage': self.stage,
            'timestamp': self.timestamp,
            'n_symbols': self.n_symbols,
            'n_columns': self.n_columns,
            'date_min': self.date_min,
            'date_max': self.date_max,
            'stage_time_seconds': self.stage_time_seconds,
            'cumulative_time_seconds': self.cumulative_time_seconds,
            'memory_mb': self.memory_mb,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CheckpointMetadata':
        return cls(**d)


class CheckpointManager:
    """
    Manages checkpoint saves/loads for the feature pipeline.

    The checkpoint format stores each symbol's DataFrame as a separate parquet
    file within a stage directory. This enables:
    - Parallel loading/saving
    - Partial updates (re-save only changed symbols)
    - Memory-efficient loading (load symbols on demand)

    Directory structure:
        checkpoints/
        ├── 01_single_stock/
        │   ├── AAPL.parquet
        │   ├── MSFT.parquet
        │   ├── ...
        │   └── _metadata.json
        ├── 02_cross_sectional/
        │   └── ...
        └── _pipeline_state.json
    """

    def __init__(self, config: CheckpointConfig):
        self.config = config
        self._start_time = datetime.now()
        self._stage_times: Dict[str, float] = {}
        self._last_stage_start: Optional[datetime] = None

        if self.config.enabled and self.config.checkpoint_dir:
            self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def should_skip_stage(self, stage: str) -> bool:
        """Check if stage should be skipped (already completed in resume)."""
        if not self.config.resume_from:
            return False

        resume_idx = STAGE_ORDER.get(self.config.resume_from, -1)
        current_idx = STAGE_ORDER.get(stage, -1)

        # Skip stages before the resume point
        return current_idx < resume_idx

    def should_load_checkpoint(self, stage: str) -> bool:
        """Check if we should load from checkpoint instead of computing."""
        if not self.config.resume_from:
            return False

        resume_idx = STAGE_ORDER.get(self.config.resume_from, -1)
        current_idx = STAGE_ORDER.get(stage, -1)

        # Load checkpoint for the stage immediately before resume point
        return current_idx == resume_idx - 1

    def get_checkpoint_path(self, stage: str) -> Path:
        """Get the directory path for a checkpoint stage."""
        if not self.config.checkpoint_dir:
            raise ValueError("Checkpoint directory not configured")
        return self.config.checkpoint_dir / stage

    def save_checkpoint(
        self,
        stage: str,
        indicators_by_symbol: Dict[str, pd.DataFrame],
        stage_start_time: Optional[datetime] = None,
    ) -> None:
        """
        Save checkpoint for a pipeline stage.

        Args:
            stage: Stage name (e.g., '01_single_stock')
            indicators_by_symbol: Dict of symbol -> DataFrame
            stage_start_time: When this stage started (for timing metadata)
        """
        if not self.config.enabled:
            return

        checkpoint_dir = self.get_checkpoint_path(stage)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Calculate timing
        stage_time = 0.0
        if stage_start_time:
            stage_time = (datetime.now() - stage_start_time).total_seconds()
        cumulative_time = (datetime.now() - self._start_time).total_seconds()

        # Calculate stats
        n_symbols = len(indicators_by_symbol)
        n_columns = 0
        date_min, date_max = None, None
        memory_mb = 0.0

        # Save each symbol's DataFrame
        logger.info(f"Saving checkpoint '{stage}' with {n_symbols} symbols...")
        dup_warn_logged = False
        for symbol, df in indicators_by_symbol.items():
            if df is None or df.empty:
                continue

            # Handle duplicate columns (keep last occurrence - most recent computation)
            # This can happen when multiple feature modules compute the same feature name
            if df.columns.duplicated().any():
                if not dup_warn_logged:
                    dup_cols = df.columns[df.columns.duplicated(keep=False)].unique().tolist()
                    logger.warning(f"Duplicate columns detected: {dup_cols[:5]}... (keeping last)")
                    dup_warn_logged = True
                df = df.loc[:, ~df.columns.duplicated(keep='last')]

            symbol_path = checkpoint_dir / f"{symbol}.parquet"
            df.to_parquet(symbol_path, index=True)

            # Update stats
            n_columns = max(n_columns, len(df.columns))
            memory_mb += df.memory_usage(deep=True).sum() / 1e6

            if hasattr(df.index, 'min'):
                sym_min = df.index.min()
                sym_max = df.index.max()
                if date_min is None or sym_min < date_min:
                    date_min = sym_min
                if date_max is None or sym_max > date_max:
                    date_max = sym_max

        # Save metadata
        if self.config.save_metadata:
            metadata = CheckpointMetadata(
                stage=stage,
                timestamp=datetime.now().isoformat(),
                n_symbols=n_symbols,
                n_columns=n_columns,
                date_min=str(date_min) if date_min else None,
                date_max=str(date_max) if date_max else None,
                stage_time_seconds=stage_time,
                cumulative_time_seconds=cumulative_time,
                memory_mb=memory_mb,
            )

            metadata_path = checkpoint_dir / "_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)

        self._stage_times[stage] = stage_time

        logger.info(
            f"Checkpoint '{stage}' saved: {n_symbols} symbols, {n_columns} cols, "
            f"{memory_mb:.1f} MB, {stage_time:.1f}s"
        )

        # Clear memory if configured
        if self.config.clear_memory_after_save:
            gc.collect()

    def load_checkpoint(self, stage: str) -> Dict[str, pd.DataFrame]:
        """
        Load checkpoint for a pipeline stage.

        Args:
            stage: Stage name to load

        Returns:
            Dict of symbol -> DataFrame
        """
        checkpoint_dir = self.get_checkpoint_path(stage)

        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

        # Load metadata if available
        metadata_path = checkpoint_dir / "_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = CheckpointMetadata.from_dict(json.load(f))
            logger.info(
                f"Loading checkpoint '{stage}': {metadata.n_symbols} symbols, "
                f"{metadata.n_columns} cols (saved at {metadata.timestamp})"
            )

        # Load all symbol files
        indicators_by_symbol = {}
        parquet_files = list(checkpoint_dir.glob("*.parquet"))

        for parquet_file in parquet_files:
            symbol = parquet_file.stem  # Filename without extension
            df = pd.read_parquet(parquet_file)
            indicators_by_symbol[symbol] = df

        logger.info(f"Loaded {len(indicators_by_symbol)} symbols from checkpoint '{stage}'")

        return indicators_by_symbol

    def get_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent checkpoint stage that exists."""
        if not self.config.checkpoint_dir or not self.config.checkpoint_dir.exists():
            return None

        latest_stage = None
        latest_idx = -1

        for stage, _ in CHECKPOINT_STAGES:
            stage_dir = self.config.checkpoint_dir / stage
            if stage_dir.exists() and any(stage_dir.glob("*.parquet")):
                idx = STAGE_ORDER.get(stage, -1)
                if idx > latest_idx:
                    latest_idx = idx
                    latest_stage = stage

        return latest_stage

    def cleanup_checkpoints(self, keep_final: bool = True) -> None:
        """
        Remove checkpoint files to free disk space.

        Args:
            keep_final: If True, keep the final checkpoint
        """
        if not self.config.checkpoint_dir or not self.config.checkpoint_dir.exists():
            return

        import shutil

        for stage, _ in CHECKPOINT_STAGES:
            if keep_final and stage == "09_final":
                continue

            stage_dir = self.config.checkpoint_dir / stage
            if stage_dir.exists():
                shutil.rmtree(stage_dir)
                logger.info(f"Cleaned up checkpoint: {stage}")

        # Remove pipeline state file
        state_file = self.config.checkpoint_dir / "_pipeline_state.json"
        if state_file.exists():
            state_file.unlink()

    def save_pipeline_state(self, current_stage: str, status: str = "running") -> None:
        """Save current pipeline state for recovery."""
        if not self.config.checkpoint_dir:
            return

        state = {
            'current_stage': current_stage,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'stage_times': self._stage_times,
            'total_time_seconds': (datetime.now() - self._start_time).total_seconds(),
        }

        state_file = self.config.checkpoint_dir / "_pipeline_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def load_pipeline_state(self) -> Optional[dict]:
        """Load saved pipeline state."""
        if not self.config.checkpoint_dir:
            return None

        state_file = self.config.checkpoint_dir / "_pipeline_state.json"
        if not state_file.exists():
            return None

        with open(state_file, 'r') as f:
            return json.load(f)

    def get_resume_stage(self) -> Optional[str]:
        """
        Determine which stage to resume from.

        Returns:
            Stage name to resume from, or None if starting fresh
        """
        if self.config.resume_from:
            # Explicit resume point
            if self.config.resume_from not in STAGE_ORDER:
                raise ValueError(
                    f"Invalid resume stage: {self.config.resume_from}. "
                    f"Valid stages: {list(STAGE_ORDER.keys())}"
                )
            return self.config.resume_from

        # Auto-detect from pipeline state
        state = self.load_pipeline_state()
        if state and state.get('status') == 'running':
            # Pipeline crashed - resume from last completed stage
            latest = self.get_latest_checkpoint()
            if latest:
                # Resume from the stage AFTER the latest checkpoint
                idx = STAGE_ORDER.get(latest, -1)
                for stage, _ in CHECKPOINT_STAGES:
                    if STAGE_ORDER[stage] == idx + 1:
                        logger.info(f"Auto-resuming from crashed pipeline at stage: {stage}")
                        return stage

        return None

    def print_checkpoint_summary(self) -> None:
        """Print summary of existing checkpoints."""
        if not self.config.checkpoint_dir or not self.config.checkpoint_dir.exists():
            print("No checkpoint directory found")
            return

        print(f"\nCheckpoint Summary ({self.config.checkpoint_dir})")
        print("=" * 60)

        for stage, description in CHECKPOINT_STAGES:
            stage_dir = self.config.checkpoint_dir / stage
            if stage_dir.exists():
                n_files = len(list(stage_dir.glob("*.parquet")))
                metadata_path = stage_dir / "_metadata.json"

                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        meta = json.load(f)
                    print(
                        f"  {stage:20} [{n_files:4} symbols] "
                        f"{meta.get('n_columns', '?'):4} cols  "
                        f"{meta.get('stage_time_seconds', 0):.1f}s"
                    )
                else:
                    print(f"  {stage:20} [{n_files:4} symbols]")
            else:
                print(f"  {stage:20} [not saved]")

        print("=" * 60)


def get_stage_before(stage: str) -> Optional[str]:
    """Get the checkpoint stage immediately before the given stage."""
    idx = STAGE_ORDER.get(stage, -1)
    if idx <= 0:
        return None

    for name, _ in CHECKPOINT_STAGES:
        if STAGE_ORDER[name] == idx - 1:
            return name
    return None


def list_available_stages() -> List[Tuple[str, str]]:
    """Return list of (stage_name, description) tuples."""
    return CHECKPOINT_STAGES.copy()
