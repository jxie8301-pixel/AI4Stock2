"""Feature engineering using Qlib Alpha158 handler."""

from qlib.contrib.data.handler import Alpha158


def build_alpha158_handler(
    instruments: str = "csi300",
    start_time: str = "2008-01-01",
    end_time: str = "2023-12-31",
    fit_start_time: str = "2008-01-01",
    fit_end_time: str = "2018-12-31",
) -> Alpha158:
    """Build an Alpha158 data handler with standard normalization.

    The handler generates 158 technical factors (moving averages, momentum,
    volatility, volume, etc.) and applies cross-sectional z-score normalization.

    Parameters
    ----------
    instruments : str
        Stock universe, e.g. "csi300", "csi500".
    start_time / end_time : str
        Overall data time range.
    fit_start_time / fit_end_time : str
        Time range for fitting the normalization parameters (typically the training period).
    """
    handler = Alpha158(
        instruments=instruments,
        start_time=start_time,
        end_time=end_time,
        fit_start_time=fit_start_time,
        fit_end_time=fit_end_time,
        infer_processors=[
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        learn_processors=[
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
        ],
    )
    print(f"Alpha158 handler built: instruments={instruments}, "
          f"time=[{start_time}, {end_time}], fit=[{fit_start_time}, {fit_end_time}]")
    return handler
