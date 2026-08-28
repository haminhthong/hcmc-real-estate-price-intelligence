import numpy as np
import pandas as pd

from src.data_processing import clean_data
from src.feature_engineering import make_features
from src.train import build_pipeline


def sample_df():
    return pd.DataFrame(
        {
            "Price": [5000.0, 6000.0],
            "Area": [50.0, 60.0],
            "Property Type": ["Nhà riêng"] * 2,
            "Location": ["Quận 1, TP.HCM"] * 2,
            "Listing ID": [1, 2],
            "Bedrooms": [2, 3],
            "Bathrooms": [1, 2],
            "Floors": [1, 2],
            "Width": [4.0, 5.0],
            "Length": [12.0, 12.0],
            "Alley Width": [2.0, 3.0],
            "Direction": ["Đông", "Tây"],
            "Position": ["Trong hẻm", "Đường chính"],
        }
    )


def test_preprocessing_has_no_nonfinite_values():
    frame = make_features(clean_data(sample_df()))
    transformed = (
        build_pipeline()
        .named_steps["preprocessor"]
        .fit_transform(frame)
    )
    assert np.isfinite(transformed).all()


def test_feature_count_is_stable_after_transform():
    frame = make_features(clean_data(sample_df()))
    prep = build_pipeline().named_steps["preprocessor"].fit(frame)
    assert prep.transform(frame).shape[1] == len(prep.get_feature_names_out())


def test_target_is_not_a_feature():
    assert "Price" not in make_features(clean_data(sample_df())).columns


def test_supplied_amenities_are_not_overwritten():
    frame = pd.DataFrame(
        [{"has_furniture": True, "car_alley": True}]
    )
    features = make_features(frame)
    assert features.loc[0, "has_furniture"] == 1
    assert features.loc[0, "car_alley"] == 1


def test_duplicate_rows_without_listing_id_share_one_group():
    frame = sample_df().drop(columns="Listing ID")
    duplicated = pd.concat([frame.iloc[[0]], frame.iloc[[0]]], ignore_index=True)
    assert len(clean_data(duplicated)) == 1


def test_spatial_and_quality_features_are_created():
    frame = pd.DataFrame(
        [{"Latitude": 10.7769, "Longitude": 106.7009, "Bedrooms": 2}]
    )
    features = make_features(frame)
    assert features.loc[0, "distance_to_cbd_km"] < 0.1
    assert 0 < features.loc[0, "data_quality_score"] < 100
