"""
HPO artifact generation and report writing.

Handles:
- Per-round artifacts (search space, trials, metrics)
- Final summary report in Markdown
- Fold AUC analysis and pruning statistics
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


@dataclass
class HPORoundResult:
    """Results from a single HPO round."""
    round_num: int
    n_trials: int
    n_completed: int
    n_pruned: int
    best_score: float
    best_params: Dict[str, Any]
    metrics: Dict[str, float]  # auc_mean, aupr_mean, etc.
    fold_aucs: Dict[int, float]  # fold_idx -> mean AUC across trials
    elapsed_seconds: float
    search_space: Optional[Dict[str, Any]] = None  # Serialized SearchSpace
    param_importance: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'round_num': self.round_num,
            'n_trials': self.n_trials,
            'n_completed': self.n_completed,
            'n_pruned': self.n_pruned,
            'best_score': self.best_score,
            'best_params': self.best_params,
            'metrics': self.metrics,
            'fold_aucs': {str(k): v for k, v in self.fold_aucs.items()},
            'elapsed_seconds': self.elapsed_seconds,
            'search_space': self.search_space,
            'param_importance': self.param_importance,
        }


@dataclass
class PruningStats:
    """Statistics about pruning behavior."""
    n_pruned_absolute: int = 0  # Pruned by absolute thresholds
    n_pruned_relative: int = 0  # Pruned by relative threshold
    n_pruned_variance: int = 0  # Pruned by CV variance
    n_pruned_optuna: int = 0  # Pruned by Optuna's pruner
    prune_reasons: Dict[str, int] = field(default_factory=dict)
    prune_metric_distribution: List[float] = field(default_factory=list)

    def record_prune(self, reason: str) -> None:
        """Record a pruning event."""
        self.prune_reasons[reason] = self.prune_reasons.get(reason, 0) + 1

        if 'low_auc' in reason or 'high_brier' in reason:
            self.n_pruned_absolute += 1
        elif 'relative' in reason:
            self.n_pruned_relative += 1
        elif 'variance' in reason:
            self.n_pruned_variance += 1
        elif 'optuna' in reason:
            self.n_pruned_optuna += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'n_pruned_absolute': self.n_pruned_absolute,
            'n_pruned_relative': self.n_pruned_relative,
            'n_pruned_variance': self.n_pruned_variance,
            'n_pruned_optuna': self.n_pruned_optuna,
            'prune_reasons': self.prune_reasons,
            'prune_metric_stats': {
                'mean': float(np.mean(self.prune_metric_distribution)) if self.prune_metric_distribution else 0.0,
                'std': float(np.std(self.prune_metric_distribution)) if self.prune_metric_distribution else 0.0,
                'min': float(np.min(self.prune_metric_distribution)) if self.prune_metric_distribution else 0.0,
                'max': float(np.max(self.prune_metric_distribution)) if self.prune_metric_distribution else 0.0,
            }
        }


class HPOArtifactWriter:
    """
    Writes HPO artifacts to disk.

    Directory structure:
        artifacts/hpo/<study_name>/<run_id>/
        ├── search_space_round_0.json
        ├── search_space_round_1.json
        ├── best_params_round_0.json
        ├── best_params_round_1.json
        ├── trials_round_0.csv
        ├── trials_round_1.csv
        ├── param_importance_round_0.json
        ├── param_importance_round_1.json
        ├── fold_auc_summary_round_0.csv
        ├── fold_auc_summary_round_1.csv
        ├── pruning_stats_round_0.json
        ├── pruning_stats_round_1.json
        ├── final_best_params.json
        └── report.md
    """

    def __init__(
        self,
        base_dir: str = "artifacts/hpo",
        study_name: str = "default",
        run_id: Optional[str] = None,
    ):
        """
        Initialize artifact writer.

        Args:
            base_dir: Base directory for HPO artifacts
            study_name: Name of the study (e.g., model key)
            run_id: Unique run identifier (default: timestamp)
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.output_dir = Path(base_dir) / study_name / run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.study_name = study_name
        self.run_id = run_id
        self.round_results: List[HPORoundResult] = []
        self.pruning_stats_per_round: List[PruningStats] = []

    def save_search_space(
        self,
        round_num: int,
        search_space_dict: Dict[str, Any],
    ) -> None:
        """Save search space for a round."""
        path = self.output_dir / f"search_space_round_{round_num}.json"
        with open(path, 'w') as f:
            json.dump(search_space_dict, f, indent=2)

    def save_best_params(
        self,
        round_num: int,
        best_params: Dict[str, Any],
        metrics: Dict[str, float],
    ) -> None:
        """Save best parameters for a round."""
        path = self.output_dir / f"best_params_round_{round_num}.json"
        data = {
            'round_num': round_num,
            'best_params': best_params,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def save_trials_history(
        self,
        round_num: int,
        trials_df: pd.DataFrame,
    ) -> None:
        """Save trials history for a round."""
        path = self.output_dir / f"trials_round_{round_num}.csv"
        trials_df.to_csv(path, index=False)

    def save_param_importance(
        self,
        round_num: int,
        importance: Dict[str, float],
    ) -> None:
        """Save parameter importance for a round."""
        path = self.output_dir / f"param_importance_round_{round_num}.json"
        with open(path, 'w') as f:
            json.dump(importance, f, indent=2)

    def save_fold_auc_summary(
        self,
        round_num: int,
        fold_data: List[Dict[str, Any]],
    ) -> None:
        """
        Save fold AUC summary for a round.

        Args:
            round_num: Round number
            fold_data: List of dicts with fold metrics
                      [{'fold': 0, 'auc_mean': 0.7, 'auc_std': 0.02, ...}, ...]
        """
        path = self.output_dir / f"fold_auc_summary_round_{round_num}.csv"
        df = pd.DataFrame(fold_data)
        df.to_csv(path, index=False)

    def save_pruning_stats(
        self,
        round_num: int,
        stats: PruningStats,
    ) -> None:
        """Save pruning statistics for a round."""
        path = self.output_dir / f"pruning_stats_round_{round_num}.json"
        with open(path, 'w') as f:
            json.dump(stats.to_dict(), f, indent=2)
        self.pruning_stats_per_round.append(stats)

    def add_round_result(self, result: HPORoundResult) -> None:
        """Add a round result for report generation."""
        self.round_results.append(result)

    def save_final_best_params(
        self,
        best_params: Dict[str, Any],
        metrics: Dict[str, float],
        metadata: Dict[str, Any],
    ) -> None:
        """Save final best parameters across all rounds."""
        path = self.output_dir / "final_best_params.json"
        data = {
            'best_params': best_params,
            'metrics': metrics,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat(),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def generate_report(self) -> str:
        """
        Generate Markdown report summarizing HPO run.

        Returns:
            Path to generated report
        """
        report_path = self.output_dir / "report.md"
        lines = []

        # Header
        lines.append(f"# HPO Report: {self.study_name}")
        lines.append(f"\n**Run ID:** {self.run_id}")
        lines.append(f"**Generated:** {datetime.now().isoformat()}")
        lines.append(f"**Total Rounds:** {len(self.round_results)}")
        lines.append("")

        # Summary table
        if self.round_results:
            lines.append("## Round Summary")
            lines.append("")
            lines.append("| Round | Completed | Pruned | Best Score | Best AUC | Time |")
            lines.append("|-------|-----------|--------|------------|----------|------|")

            for r in self.round_results:
                auc = r.metrics.get('auc_mean', 0)
                time_min = r.elapsed_seconds / 60
                lines.append(
                    f"| {r.round_num} | {r.n_completed} | {r.n_pruned} | "
                    f"{r.best_score:.4f} | {auc:.4f} | {time_min:.1f}m |"
                )
            lines.append("")

        # Best parameters progression
        if self.round_results:
            lines.append("## Best Parameters Progression")
            lines.append("")

            # Identify key params that changed
            all_params = set()
            for r in self.round_results:
                all_params.update(r.best_params.keys())

            for param in sorted(all_params):
                values = [r.best_params.get(param) for r in self.round_results]
                if len(set(str(v) for v in values)) > 1:  # Changed across rounds
                    lines.append(f"- **{param}**: {' → '.join(str(v) for v in values)}")

            lines.append("")

        # Search space narrowing
        if len(self.round_results) > 1:
            lines.append("## Search Space Evolution")
            lines.append("")

            first_space = self.round_results[0].search_space or {}
            last_space = self.round_results[-1].search_space or {}

            if first_space and last_space:
                first_params = first_space.get('params', {})
                last_params = last_space.get('params', {})

                for name in sorted(first_params.keys()):
                    fp = first_params.get(name, {})
                    lp = last_params.get(name, {})

                    if fp.get('low') is not None and lp.get('low') is not None:
                        orig_range = fp.get('high', 0) - fp.get('low', 0)
                        new_range = lp.get('high', 0) - lp.get('low', 0)
                        if orig_range > 0:
                            shrink_pct = (1 - new_range / orig_range) * 100
                            lines.append(
                                f"- **{name}**: [{fp.get('low'):.4g}, {fp.get('high'):.4g}] → "
                                f"[{lp.get('low'):.4g}, {lp.get('high'):.4g}] "
                                f"({shrink_pct:.0f}% narrowed)"
                            )

            lines.append("")

        # Per-fold AUC analysis
        if self.round_results:
            last_round = self.round_results[-1]
            if last_round.fold_aucs:
                lines.append("## Per-Fold AUC (Final Round)")
                lines.append("")
                lines.append("| Fold | Mean AUC |")
                lines.append("|------|----------|")
                for fold_idx in sorted(last_round.fold_aucs.keys()):
                    auc = last_round.fold_aucs[fold_idx]
                    lines.append(f"| {fold_idx} | {auc:.4f} |")
                lines.append("")

        # Pruning analysis
        if self.pruning_stats_per_round:
            lines.append("## Pruning Analysis")
            lines.append("")

            for i, stats in enumerate(self.pruning_stats_per_round):
                lines.append(f"### Round {i}")
                lines.append(f"- Absolute threshold pruning: {stats.n_pruned_absolute}")
                lines.append(f"- Relative threshold pruning: {stats.n_pruned_relative}")
                lines.append(f"- Variance pruning: {stats.n_pruned_variance}")
                lines.append(f"- Optuna pruner: {stats.n_pruned_optuna}")
                lines.append("")

        # Final metrics
        if self.round_results:
            last = self.round_results[-1]
            lines.append("## Final Metrics")
            lines.append("")
            for key, value in sorted(last.metrics.items()):
                if value is not None:
                    lines.append(f"- **{key}**: {value:.4f}")
            lines.append("")

        # Parameter importance (if available)
        if self.round_results and self.round_results[-1].param_importance:
            lines.append("## Parameter Importance (Final Round)")
            lines.append("")
            importance = self.round_results[-1].param_importance
            for param, imp in sorted(importance.items(), key=lambda x: -x[1])[:10]:
                bar = "█" * int(imp * 20)
                lines.append(f"- {param}: {imp:.3f} {bar}")
            lines.append("")

        # Write report
        content = "\n".join(lines)
        with open(report_path, 'w') as f:
            f.write(content)

        return str(report_path)

    def get_output_dir(self) -> Path:
        """Get the output directory path."""
        return self.output_dir


def compute_fold_auc_summary(
    trials_data: List[Dict[str, Any]],
    n_folds: int,
    prune_folds: List[int],
) -> List[Dict[str, Any]]:
    """
    Compute per-fold AUC statistics from trial data.

    Args:
        trials_data: List of trial result dicts (must have fold AUC attrs)
        n_folds: Number of CV folds
        prune_folds: Which folds are used for pruning

    Returns:
        List of dicts with per-fold statistics
    """
    fold_summary = []

    for fold_idx in range(n_folds):
        # Extract AUCs for this fold from all completed trials
        fold_aucs = []
        for trial in trials_data:
            auc_key = f'fold_{fold_idx}_auc'
            if auc_key in trial and trial[auc_key] is not None:
                fold_aucs.append(trial[auc_key])

        if fold_aucs:
            fold_summary.append({
                'fold': fold_idx,
                'auc_mean': float(np.mean(fold_aucs)),
                'auc_std': float(np.std(fold_aucs)),
                'auc_min': float(np.min(fold_aucs)),
                'auc_max': float(np.max(fold_aucs)),
                'n_samples': len(fold_aucs),
                'is_prune_fold': fold_idx in prune_folds,
            })

    return fold_summary
