# Canonical owner: closed-loop-dbs-bench
"""Logging and debugging utilities for closed-loop DBS simulations.

Provides structured step-by-step logging, metrics export, and CSV/JSON output.

Example:
    >>> from dbs_bench.simulation.logging_utils import SimulationLogger
    >>> logger = SimulationLogger(log_dir)
    >>> for k in range(n_steps):
    ...     logger.log_step(t, y, u, u_prev=0.0)
    >>> logger.log_metrics_summary(metrics)
    >>> logger.export_step_data('simulation_steps.csv')
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from dbs_bench.simulation.simulate import SimulationResult


@dataclass
class StepRecord:
    """Record of a single simulation step.

    Attributes:
        time: Simulation time in seconds.
        y: Output (log-space beta power).
        u: Control input (stimulation amplitude).
        u_prev: Previous control input.
        y_ref: Reference threshold.
        error: Tracking error (max(0, y - y_ref)).
        solve_time: Solver time in seconds (optional).
    """

    time: float
    y: float
    u: float
    u_prev: float
    y_ref: float
    error: float = 0.0
    solve_time: Optional[float] = None

    @classmethod
    def from_dict(cls, d: Dict) -> "StepRecord":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class SimulationLogger:
    """Structured logging for closed-loop DBS simulation.

    Tracks step-by-step state and control, logs performance metrics,
    and exports data to CSV and JSON.

    Attributes:
        log_dir: Directory for log files.
        session_id: Unique identifier for this logging session.

    Example:
        >>> logger = SimulationLogger(Path('results/'))
        >>> logger.log_step(t=0.0, y=2.5, u=0.01, u_prev=0.0)
        >>> logger.log_metrics_summary({'tracking_error': 0.1})
        >>> logger.export_step_data('simulation.csv')
    """

    def __init__(
        self,
        log_dir: Path,
        session_id: Optional[str] = None,
        level: str = "INFO",
        console_output: bool = True,
    ):
        """Initialise simulation logger.

        Args:
            log_dir: Directory for log files.
            session_id: Unique session identifier. If None, uses timestamp.
            level: Logging level ('DEBUG', 'INFO', 'WARNING').
            console_output: Whether to also output to console.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = session_id

        self._step_records: List[StepRecord] = []
        self._setup_logger(level, console_output)
        self._y_ref = 2.3  # Default threshold
        self._step_count = 0

        self.logger.info("SimulationLogger initialised: session=%s", session_id)

    def _setup_logger(self, level: str, console_output: bool) -> None:
        """Setup Python logging infrastructure."""
        self.logger = logging.getLogger(f"dbs_bench.simulation.{self.session_id}")
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self.logger.handlers.clear()

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        log_file = self.log_dir / f"simulation_{self.session_id}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)

        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
            self.logger.addHandler(console_handler)

        self.logger.propagate = False

    def set_reference(self, y_ref: float) -> None:
        """Set reference threshold for error computation."""
        self._y_ref = y_ref

    def log_step(
        self,
        t: float,
        y: float,
        u: float,
        u_prev: float = 0.0,
        info: Optional[Dict] = None,
    ) -> None:
        """Log a single simulation step.

        Args:
            t: Simulation time.
            y: Current output (beta).
            u: Control input applied.
            u_prev: Previous control input.
            info: Optional dict with 'solve_time' key.
        """
        info = info or {}
        error = max(0.0, y - self._y_ref)

        record = StepRecord(
            time=t,
            y=y,
            u=u,
            u_prev=u_prev,
            y_ref=self._y_ref,
            error=error,
            solve_time=info.get("solve_time"),
        )

        self._step_records.append(record)
        self._step_count += 1

        self.logger.debug(
            "Step %d: t=%.3f, y=%.4f, u=%.5f, error=%.4f",
            self._step_count, t, y, u, error,
        )

        if y > self._y_ref + 0.5:
            self.logger.warning(
                "Large threshold violation at t=%.3f: y=%.4f (ref=%.4f)",
                t, y, self._y_ref,
            )

    def log_metrics_summary(self, metrics: Dict[str, float]) -> None:
        """Log final performance metrics and save to JSON.

        Args:
            metrics: Dictionary of performance metrics.
        """
        self.logger.info("=" * 50)
        self.logger.info("SIMULATION METRICS SUMMARY")
        self.logger.info("=" * 50)
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info("  %s: %.6f", key, value)
            else:
                self.logger.info("  %s: %s", key, value)
        self.logger.info("=" * 50)

        metrics_file = self.log_dir / f"metrics_{self.session_id}.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

    def export_step_data(self, filename: str = "simulation_steps.csv") -> Path:
        """Export step-by-step data to CSV.

        Args:
            filename: Output filename (relative to log_dir).

        Returns:
            Path to exported file.
        """
        filepath = self.log_dir / filename
        if not self._step_records:
            self.logger.warning("No step data to export")
            return filepath

        fieldnames = list(StepRecord.__dataclass_fields__.keys())
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in self._step_records:
                writer.writerow(record.to_dict())

        self.logger.info("Exported %d steps to %s", len(self._step_records), filepath)
        return filepath

    def get_step_arrays(self) -> Dict[str, np.ndarray]:
        """Return step data as numpy arrays.

        Returns:
            Dictionary with 'time', 'y', 'u', 'error' arrays.
        """
        if not self._step_records:
            return {}

        return {
            "time": np.array([r.time for r in self._step_records]),
            "y": np.array([r.y for r in self._step_records]),
            "u": np.array([r.u for r in self._step_records]),
            "error": np.array([r.error for r in self._step_records]),
        }

    def log_from_result(self, result: "SimulationResult") -> None:
        """Log all data from a completed SimulationResult.

        Args:
            result: SimulationResult from SimulationRunner.run().
        """
        self.set_reference(result.y_ref)

        for i in range(len(result.time)):
            info: Dict[str, Any] = {}
            if result.solver_info:
                if i < len(result.solver_info.get("solve_times", [])):
                    info["solve_time"] = result.solver_info["solve_times"][i]

            u_prev = result.u[i - 1] if i > 0 else 0.0
            self.log_step(
                t=result.time[i],
                y=result.y[i],
                u=result.u[i],
                u_prev=u_prev,
                info=info,
            )

        self.log_metrics_summary(result.metrics)


def create_simulation_logger(
    results_dir: Optional[Path] = None,
    level: str = "INFO",
) -> SimulationLogger:
    """Create a simulation logger with default settings.

    Args:
        results_dir: Results directory. Defaults to './results'.
        level: Logging level.

    Returns:
        Configured SimulationLogger.
    """
    if results_dir is None:
        results_dir = Path("results")

    return SimulationLogger(results_dir, level=level)
