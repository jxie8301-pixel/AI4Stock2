"""Feature engineering using Qlib Alpha158 handler with Valuation extensions."""

from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import RobustZScoreNorm, Fillna, DropnaLabel, CSRankNorm


class Alpha158WithValuation(Alpha158):
    """Extended Alpha158 handler that includes fundamental valuation factors."""

    def get_feature_config(self):
        # Get the base Alpha158 technical factors
        feature_config = super().get_feature_config()
        
        # Add our custom valuation factors from the binary database
        # These names must match the .bin filenames in data/qlib_data_cn/features/
        valuation_factors = [
            "pe_ttm", "pb", "ps", "pcf", "peg", 
            "total_mv", "circ_mv", "turnover"
        ]
        
        # Qlib syntax: ["$pe_ttm", "$pb", ...]
        # We append them to the existing feature list
        for factor in valuation_factors:
            expr = f"${factor}"
            if expr not in feature_config[0]:
                feature_config[0].append(expr)
                feature_config[1].append(factor.upper())
                
        return feature_config


def build_alpha158_handler(
    instruments: str = "csi300",
    start_time: str = "2008-01-01",
    end_time: str = "2023-12-31",
    fit_start_time: str = "2008-01-01",
    fit_end_time: str = "2018-12-31",
    use_valuation: bool = True,
) -> DataHandlerLP:
    """Build a data handler with optional valuation factors.

    Parameters
    ----------
    use_valuation : bool
        If True, use the extended Alpha158WithValuation handler.
    """
    handler_cls = Alpha158WithValuation if use_valuation else Alpha158
    
    handler = handler_cls(
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
    
    msg = "Alpha158" + ("+Valuation" if use_valuation else "")
    print(f"{msg} handler built: instruments={instruments}, "
          f"time=[{start_time}, {end_time}], fit=[{fit_start_time}, {fit_end_time}]")
    return handler
