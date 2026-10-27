import pandas as pd
import numpy as np

class FeatureEngineeringFactory:
    """Computes mathematical ratio and net financial indicators."""
    @staticmethod
    def construct_features(df: pd.DataFrame) -> pd.DataFrame:
        res = df.copy()
        res["capital_net"] = res["capital-gain"] - res["capital-loss"]
        # hours_per_age = hours-per-week / (age - 16)
        age_delta = np.maximum(res["age"] - 16.0, 1.0)
        res["hours_per_age"] = res["hours-per-week"] / age_delta
        return res
