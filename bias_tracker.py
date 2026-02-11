"""
Gas Price Predictor - Adaptive Bias Correction

Tracks historical bias/shrinkage values across training cycles and uses
them to produce smarter corrections that evolve with fresh data.

Components:
  1. Auto-calibration window search (optimal window size per cycle)
  2. Regime detection (rising / falling / flat market)
  3. EMA-blended bias (smooths noise across cycles)
  4. Regime-aware adjustment (different bias per market regime)
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import BIAS_HISTORY_FILE


def _calibrate_shrinkage(
    raw_pred_changes: np.ndarray,
    actual_abs: np.ndarray,
    base_prices: np.ndarray,
    target_cents: float = 0.02,
) -> float:
    """
    Find the shrinkage factor s in [0, 1] that maximizes
    the % of predictions within +/-target_cents of actual.
    """
    best_s = 0.0
    best_rate = 0.0

    for s in np.arange(0.0, 1.01, 0.02):
        dampened = raw_pred_changes * s
        pred_abs = base_prices + dampened
        rate = (np.abs(pred_abs - actual_abs) <= target_cents).mean()
        if rate > best_rate or (rate == best_rate and s > best_s):
            best_rate = rate
            best_s = s

    return round(best_s, 2)


class BiasTracker:
    """Tracks historical bias and shrinkage values across training cycles."""

    def __init__(self):
        self.history: List[dict] = []
        self._load()

    def _load(self):
        if BIAS_HISTORY_FILE.exists():
            try:
                with open(BIAS_HISTORY_FILE) as f:
                    self.history = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.history = []

    def _save(self):
        BIAS_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(BIAS_HISTORY_FILE, "w") as f:
            json.dump(self.history, f, indent=2, default=str)

    def record(
        self,
        shrinkage: float,
        bias: float,
        regime: str,
        cal_window: int,
        accuracy_2c: float,
    ):
        """Record bias/shrinkage from a training cycle."""
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "shrinkage": shrinkage,
            "bias": bias,
            "regime": regime,
            "cal_window": cal_window,
            "accuracy_2c": accuracy_2c,
        })
        # Keep last 52 entries
        self.history = self.history[-52:]
        self._save()

    @staticmethod
    def _ema(values: List[float], alpha: float = 0.3) -> float:
        """Exponentially-weighted moving average, most recent weighted highest."""
        if not values:
            return 0.0
        result = values[0]
        for v in values[1:]:
            result = alpha * v + (1 - alpha) * result
        return result

    @staticmethod
    def _detect_regime(base_prices: np.ndarray) -> str:
        """Detect market regime from recent base prices."""
        recent = base_prices[-8:]
        if len(recent) < 4:
            return "flat"
        slope = np.polyfit(range(len(recent)), recent, 1)[0]
        if slope > 0.005:
            return "rising"
        elif slope < -0.005:
            return "falling"
        return "flat"

    def _find_best_cal_window(
        self,
        raw_pred: np.ndarray,
        actual_abs: np.ndarray,
        base_prices: np.ndarray,
        min_w: int = 13,
        max_w: int = 78,
    ) -> int:
        """Search for optimal calibration window size."""
        n = len(raw_pred)
        best_w = min(52, n // 4) if n >= 52 else max(min_w, n // 4)
        best_rate = 0.0

        for w in range(min_w, min(max_w, n) + 1, 4):
            sl = slice(n - w, None)
            s = _calibrate_shrinkage(
                raw_pred[sl], actual_abs[sl], base_prices[sl]
            )
            dampened = raw_pred[sl] * s
            pred = base_prices[sl] + dampened
            bias = float(np.mean(pred - actual_abs[sl]))
            final_pred = pred - bias
            rate = (np.abs(final_pred - actual_abs[sl]) <= 0.02).mean()
            if rate > best_rate or (rate == best_rate and w > best_w):
                best_rate = rate
                best_w = w

        return best_w

    def compute_adaptive_bias(
        self,
        raw_ensemble_preds: np.ndarray,
        actual_abs: np.ndarray,
        base_prices: np.ndarray,
    ) -> Tuple[float, float, Dict]:
        """
        Full adaptive bias correction pipeline.

        Returns:
            (shrinkage, bias, metadata_dict)
        """
        # Step 1: Find optimal calibration window
        best_w = self._find_best_cal_window(
            raw_ensemble_preds, actual_abs, base_prices
        )
        sl = slice(len(raw_ensemble_preds) - best_w, None)

        # Step 2: Calibrate shrinkage on best window
        shrinkage = _calibrate_shrinkage(
            raw_ensemble_preds[sl], actual_abs[sl], base_prices[sl]
        )

        # Step 3: Compute raw bias
        dampened = raw_ensemble_preds[sl] * shrinkage
        pred = base_prices[sl] + dampened
        raw_bias = float(np.mean(pred - actual_abs[sl]))

        # Step 4: Detect regime
        regime = self._detect_regime(base_prices)

        # Step 5: Blend with historical (EMA + regime-aware)
        adaptive_bias = raw_bias
        if len(self.history) >= 3:
            recent_biases = [h["bias"] for h in self.history[-5:]]
            ema_bias = self._ema(recent_biases, alpha=0.3)
            adaptive_bias = 0.7 * raw_bias + 0.3 * ema_bias

        regime_entries = [h for h in self.history if h["regime"] == regime]
        if len(regime_entries) >= 2:
            regime_bias = float(
                np.mean([h["bias"] for h in regime_entries[-3:]])
            )
            adaptive_bias = 0.6 * adaptive_bias + 0.4 * regime_bias

        # Step 6: Compute accuracy on calibration window with adaptive bias
        final_pred = (
            base_prices[sl]
            + raw_ensemble_preds[sl] * shrinkage
            - adaptive_bias
        )
        accuracy_2c = float(
            (np.abs(final_pred - actual_abs[sl]) <= 0.02).mean()
        )

        # Step 7: Record this cycle
        self.record(shrinkage, adaptive_bias, regime, best_w, accuracy_2c)

        metadata = {
            "raw_bias": round(raw_bias, 6),
            "adaptive_bias": round(adaptive_bias, 6),
            "regime": regime,
            "cal_window": best_w,
            "cal_accuracy_2c": round(accuracy_2c * 100, 1),
            "history_entries": len(self.history),
        }
        return shrinkage, adaptive_bias, metadata
