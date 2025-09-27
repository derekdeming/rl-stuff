"""Lightweight metrics logging helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List


class MetricLogger:
    """Aggregates scalar metrics over iterations for quick inspection."""

    def __init__(self) -> None:
        self._history: Dict[str, List[float]] = defaultdict(list)

    def update(self, **metrics: float) -> None:
        for key, value in metrics.items():
            self._history[key].append(float(value))

    def latest(self) -> Dict[str, float]:
        return {key: values[-1] for key, values in self._history.items() if values}

    def averages(self) -> Dict[str, float]:
        return {
            key: sum(values) / len(values)
            for key, values in self._history.items()
            if values
        }

    def history(self) -> Dict[str, List[float]]:
        return dict(self._history)


__all__ = ["MetricLogger"]
