"""LightGBM model for baseline comparison."""

from qlib.contrib.model.gbdt import LGBModel

def build_lgbm_model(
    loss: str = "mse",
    colsample_bytree: float = 0.8879,
    learning_rate: float = 0.2,
    subsample: float = 0.8789,
    lambda_l1: float = 205.6999,
    lambda_l2: float = 580.9768,
    max_depth: int = 8,
    num_leaves: int = 210,
    num_threads: int = 20,
    early_step: int = 50,
) -> LGBModel:
    """Build a LightGBM model for tabular stock prediction.

    LightGBM is highly efficient for cross-sectional data and often serves
    as a strong baseline in quantitative finance.
    """
    model = LGBModel(
        loss=loss,
        colsample_bytree=colsample_bytree,
        learning_rate=learning_rate,
        subsample=subsample,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        max_depth=max_depth,
        num_leaves=num_leaves,
        num_threads=num_threads,
        early_step=early_step,
    )
    print(f"LightGBM model built: max_depth={max_depth}, lr={learning_rate}")
    return model
