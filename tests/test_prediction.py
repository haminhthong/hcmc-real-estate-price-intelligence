import numpy as np
import pandas as pd

from src.feature_engineering import make_features
from src.train import build_pipeline, conformal_quantile


def test_prediction_is_non_negative():
    rows = pd.DataFrame(
        [
            {
                "Area": 50,
                "Bedrooms": 2,
                "Bathrooms": 1,
                "Floors": 1,
                "Width": 4,
                "Length": 12,
                "Alley Width": 2,
                "Property Type": "Nhà riêng",
                "location_area": "Quận 1",
                "Direction": "Đông",
                "Position": "Trong hẻm",
            }
        ]
        * 8
    )
    model = build_pipeline().fit(
        make_features(rows),
        np.log1p(np.arange(4000, 4008)),
    )
    assert np.expm1(model.predict(make_features(rows))).min() >= 0


def test_conformal_quantile_is_conservative():
    residuals = np.arange(1, 11, dtype=float)
    assert conformal_quantile(residuals, coverage=0.8) == 9.0
