import numpy as np
import pandas as pd
from typing import Dict, Tuple

class DomainDataSanitizer:
    """Encapsulates data cleaning rules and learns parameters strictly from training data."""
    def __init__(self, num_cols, cat_cols):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.impute_medians_: Dict[str, float] = {}
        self.impute_modes_: Dict[str, str] = {}
        self.clipping_intervals_: Dict[str, Tuple[float, float]] = {}
        self.is_ready_ = False

    def learn_training_priors(self, df: pd.DataFrame):
        for col in self.num_cols:
            vals = pd.to_numeric(df[col], errors="coerce")
            self.impute_medians_[col] = float(vals.median())
            self.clipping_intervals_[col] = (float(vals.quantile(0.01)), float(vals.quantile(0.99)))

        for col in self.cat_cols:
            clean_s = df[col].astype(str).str.strip().replace("?", np.nan)
            mode_s = clean_s.dropna().mode()
            self.impute_modes_[col] = mode_s.iloc[0] if not mode_s.empty else "Missing"

        self.is_ready_ = True
        return self

    def apply_sanitization(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_ready_:
            raise RuntimeError("Sanitizer must learn training priors before transforming data.")
        out = df.copy()
        for col in self.num_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(self.impute_medians_[col])
            lo, hi = self.clipping_intervals_[col]
            out[col] = out[col].clip(lower=lo, upper=hi)

        for col in self.cat_cols:
            out[col] = out[col].astype(str).str.strip().replace("?", np.nan).fillna(self.impute_modes_[col])
        return out
