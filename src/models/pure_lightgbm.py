"""Pure LightGBM Native Implementation."""

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluate import safe_cross_sectional_corr


def _daily_ic_metric(preds: np.ndarray, dataset: lgb.Dataset, dates: np.ndarray):
    labels = dataset.get_label()
    frame = pd.DataFrame(
        {
            "pred": np.asarray(preds, dtype=np.float32),
            "label": np.asarray(labels, dtype=np.float32),
            "date": pd.to_datetime(np.asarray(dates)),
        }
    ).dropna()
    if frame.empty:
        return "daily_ic", 0.0, True

    daily_ic = frame.groupby("date", sort=True).apply(
        lambda x: safe_cross_sectional_corr(x["pred"], x["label"], method="pearson"),
        include_groups=False,
    ).dropna()
    if daily_ic.empty:
        return "daily_ic", 0.0, True
    return "daily_ic", float(daily_ic.mean()), True


class NativeLGBM:
    """Native LightGBM wrapper mimicking the interface expected by our pipeline."""
    
    def __init__(
        self,
        loss: str = "mse",
        colsample_bytree: float = 0.8879,
        learning_rate: float = 0.2,
        subsample: float = 0.8789,
        lambda_l1: float = 205.6999,
        lambda_l2: float = 580.9768,
        max_depth: int = 8,
        num_leaves: int = 210,
        num_threads: int = 20,
        early_stop: int = 50,
        **kwargs
    ):
        # Map generic loss names to lightgbm specific objectives
        objective = "regression"
        if loss == "mse":
            objective = "regression"
        elif loss == "mae":
            objective = "regression_l1"
            
        self.params = {
            "objective": objective,
            "metric": "None",
            "colsample_bytree": colsample_bytree,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "lambda_l1": lambda_l1,
            "lambda_l2": lambda_l2,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "num_threads": num_threads,
            "verbosity": -1,
            "seed": 42
        }
        self.early_stop = early_stop
        self.model = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame = None,
        y_valid: pd.Series = None,
        valid_dates: np.ndarray | None = None,
    ):
        """Fit the LightGBM model."""
        dtrain = lgb.Dataset(X_train.values, label=y_train.values)
        
        valid_sets = [dtrain]
        valid_names = ["train"]
        feval = None
        
        if X_valid is not None and y_valid is not None:
            dvalid = lgb.Dataset(X_valid.values, label=y_valid.values, reference=dtrain)
            valid_sets = [dvalid]
            valid_names = ["valid"]
            if valid_dates is not None:
                feval = lambda preds, data: _daily_ic_metric(preds, data, valid_dates)
            
        callbacks = []
        if self.early_stop > 0 and X_valid is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=self.early_stop, verbose=True))
            
        callbacks.append(lgb.log_evaluation(period=50))
            
        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=1000, # Large number, rely on early stopping
            valid_sets=valid_sets,
            valid_names=valid_names,
            feval=feval,
            callbacks=callbacks
        )
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict using the trained model."""
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
        return self.model.predict(X_test.values)
        
    def get_feature_importance(self, importance_type="split"):
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
        return self.model.feature_importance(importance_type=importance_type)
