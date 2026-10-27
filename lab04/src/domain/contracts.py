from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass(frozen=True)
class NumericalConstraint:
    min_value: float
    max_value: float

@dataclass(frozen=True)
class DatasetContract:
    expected_rows: int = 48842
    expected_cols: int = 15
    age_bounds: NumericalConstraint = NumericalConstraint(17, 90)
    hours_bounds: NumericalConstraint = NumericalConstraint(1, 99)
    capital_gain_bounds: NumericalConstraint = NumericalConstraint(0, 99999)
    capital_loss_bounds: NumericalConstraint = NumericalConstraint(0, 4356)
    missing_token: str = "?"
