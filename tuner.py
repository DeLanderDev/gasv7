"""
Gas Price Predictor - Optuna Hyperparameter Tuning

Automated Bayesian hyperparameter optimization using Optuna with
time-series-aware walk-forward validation as the objective.

Tunes: XGBoost params, Ridge alpha, ensemble weights, feature
selection count, and sample weighting.
"""

import json
import logging
from datetime import datetime
from typing import Callable, Dict, Optional

import numpy as np
import optuna
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import (
    BEST_PARAMS_FILE,
    DEFAULT_XGBOOST_PARAMS,
    MIN_TRAINING_WEEKS,
    TUNING_N_TRIALS,
    TUNING_TIMEOUT_PER_TRIAL,
    TUNING_MAX_TIMEOUT,
)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers (duplicated from model.py to avoid circular imports)
# ═══════════════════════════════════════════════════════════════════════════════

def _sample_weights(n: int, recent_frac: float = 0.4) -> np.ndarray:
    """Recent data gets 2x weight; older data ramps from 0.5 -> 1.0."""
    w = np.ones(n)
    cut = int(n * (1 - recent_frac))
    w[:cut] = np.linspace(0.5, 1.0, cut)
    w[cut:] = 2.0
    return w


def _top_features(X, y, names, top_n=35, xgb_params=None):
    """Quick XGBoost to rank features, return top N names."""
    p = xgb_params or DEFAULT_XGBOOST_PARAMS
    m = XGBRegressor(**{**p, "n_estimators": 100})
    m.fit(X[names], y, verbose=False)
    fi = dict(zip(names, m.feature_importances_))
    sorted_feats = sorted(fi, key=fi.get, reverse=True)
    return sorted_feats[:top_n]


def _calibrate_shrinkage(
    raw_pred_changes: np.ndarray,
    actual_abs: np.ndarray,
    base_prices: np.ndarray,
    target_cents: float = 0.02,
) -> float:
    """Find shrinkage s in [0,1] maximizing +/-target_cents accuracy."""
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Search Space
# ═══════════════════════════════════════════════════════════════════════════════

def _suggest_params(trial) -> dict:
    """Suggest hyperparameters for one Optuna trial."""
    xgb_params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
        "random_state": 42,
    }

    w1 = trial.suggest_float("ensemble_w1", 0.2, 0.6)
    w2 = trial.suggest_float("ensemble_w2", 0.15, 0.5)

    return {
        "xgb_params": xgb_params,
        "ridge_alpha": trial.suggest_float("ridge_alpha", 0.01, 10.0, log=True),
        "ensemble_w1": w1,
        "ensemble_w2": w2,
        "top_n": trial.suggest_int("top_n", 15, 55, step=5),
        "recent_frac": trial.suggest_float("recent_frac", 0.2, 0.6),
    }


def _normalize_weights(w1: float, w2: float):
    """Normalize ensemble weights so they sum to 1, with w3 >= 0.05."""
    w3 = max(1.0 - w1 - w2, 0.05)
    total = w1 + w2 + w3
    return [w1 / total, w2 / total, w3 / total]


# ═══════════════════════════════════════════════════════════════════════════════
#  Objective Function
# ═══════════════════════════════════════════════════════════════════════════════

def _objective(trial, X, y_chg, y_abs, bases, names) -> float:
    """
    Optuna objective: walk-forward validation with trial params.
    Returns within-2-cents rate (study direction is "maximize").
    """
    params = _suggest_params(trial)
    xgb_p = params["xgb_params"]
    ridge_alpha = params["ridge_alpha"]
    top_n = params["top_n"]
    recent_frac = params["recent_frac"]
    ew = _normalize_weights(params["ensemble_w1"], params["ensemble_w2"])

    n = len(X)
    test_n = max(13, min(26, n // 5))
    test_start = n - test_n

    if test_start < MIN_TRAINING_WEEKS:
        return 0.0

    errors = []

    for i in range(test_start, n):
        Xtr, ytr = X.iloc[:i], y_chg.iloc[:i]
        btr, atr = bases.iloc[:i], y_abs.iloc[:i]
        sw = _sample_weights(len(Xtr), recent_frac)

        # Feature selection
        sel = _top_features(Xtr, ytr, names, top_n=top_n, xgb_params=xgb_p)

        # Fit 3 models
        m1 = XGBRegressor(**xgb_p)
        m1.fit(Xtr, ytr, sample_weight=sw, verbose=False)

        m2 = XGBRegressor(**xgb_p)
        m2.fit(Xtr[sel], ytr, sample_weight=sw, verbose=False)

        sc = StandardScaler()
        Xsc = sc.fit_transform(Xtr[sel])
        m3 = Ridge(alpha=ridge_alpha)
        m3.fit(Xsc, ytr, sample_weight=sw)

        # Calibrate shrinkage on last portion of training data
        cal_n = min(26, len(Xtr) // 4)
        cal_sl = slice(len(Xtr) - cal_n, None)

        p1c = m1.predict(Xtr.iloc[cal_sl])
        p2c = m2.predict(Xtr.iloc[cal_sl][sel])
        p3c = m3.predict(sc.transform(Xtr.iloc[cal_sl][sel]))
        raw_cal = ew[0] * p1c + ew[1] * p2c + ew[2] * p3c

        shrink = _calibrate_shrinkage(
            raw_cal, atr.values[cal_sl], btr.values[cal_sl]
        )

        # Calibrate bias
        damp_cal = raw_cal * shrink
        bias = float(
            np.mean(btr.values[cal_sl] + damp_cal - atr.values[cal_sl])
        )

        # Predict test point
        X_test = X.iloc[i : i + 1]
        p1 = m1.predict(X_test)[0]
        p2 = m2.predict(X_test[sel])[0]
        p3 = m3.predict(sc.transform(X_test[sel]))[0]
        raw = ew[0] * p1 + ew[1] * p2 + ew[2] * p3

        final_chg = raw * shrink - bias
        pred = bases.iloc[i] + final_chg
        errors.append(abs(pred - y_abs.iloc[i]))

        # Intermediate pruning: report running accuracy every 5 steps
        if len(errors) % 5 == 0:
            running_rate = float(np.mean(np.array(errors) <= 0.02))
            trial.report(running_rate, step=len(errors))
            if trial.should_prune():
                raise optuna.TrialPruned()

    within_2_cents = float(np.mean(np.array(errors) <= 0.02))
    return within_2_cents


# ═══════════════════════════════════════════════════════════════════════════════
#  Study Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_tuning(
    df,
    n_trials: int = TUNING_N_TRIALS,
    timeout: int = min(TUNING_N_TRIALS * TUNING_TIMEOUT_PER_TRIAL, TUNING_MAX_TIMEOUT),
    callback: Optional[Callable] = None,
) -> dict:
    """
    Run Optuna hyperparameter optimization.

    Args:
        df: Input DataFrame with gas price data.
        n_trials: Max number of Optuna trials.
        timeout: Max seconds for the study.
        callback: Optional fn(trial_number, total_trials, best_value) for progress.

    Returns:
        Dict with best_params, best_value, n_trials, and study_summary.
    """
    # Lazy import to avoid circular dependency
    from model import GasPriceModel

    temp = GasPriceModel()
    X, y_chg, y_abs, bases, names, fdf = temp._prep(df)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    def objective_wrapper(trial):
        result = _objective(trial, X, y_chg, y_abs, bases, names)
        if callback:
            try:
                best_val = max(study.best_value, result)
            except ValueError:
                best_val = result
            callback(trial.number + 1, n_trials, best_val)
        return result

    study.optimize(
        objective_wrapper,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
    )

    # Extract best params into structured format
    best = study.best_trial.params
    best_params = {
        "xgb_params": {
            "n_estimators": best["n_estimators"],
            "max_depth": best["max_depth"],
            "learning_rate": best["learning_rate"],
            "subsample": best["subsample"],
            "colsample_bytree": best["colsample_bytree"],
            "min_child_weight": best["min_child_weight"],
            "reg_alpha": best["reg_alpha"],
            "reg_lambda": best["reg_lambda"],
            "random_state": 42,
        },
        "ridge_alpha": best["ridge_alpha"],
        "ensemble_w1": best["ensemble_w1"],
        "ensemble_w2": best["ensemble_w2"],
        "top_n": best["top_n"],
        "recent_frac": best["recent_frac"],
    }

    # Build summary
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    summary = {
        "n_completed": len(completed),
        "n_pruned": len(pruned),
        "n_total": len(study.trials),
        "best_value": study.best_value,
        "best_trial_number": study.best_trial.number,
    }

    return {
        "best_params": best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "study_summary": summary,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Persistence
# ═══════════════════════════════════════════════════════════════════════════════

def save_best_params(params: dict):
    """Persist best hyperparameters to JSON."""
    BEST_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BEST_PARAMS_FILE, "w") as f:
        json.dump(
            {**params, "tuned_at": datetime.now().isoformat()},
            f,
            indent=2,
        )


def load_best_params() -> Optional[dict]:
    """Load persisted best hyperparameters, or None if not found."""
    if BEST_PARAMS_FILE.exists():
        try:
            with open(BEST_PARAMS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None
