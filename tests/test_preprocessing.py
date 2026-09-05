import numpy as np
import pandas as pd

from src.data_processing import (
    _make_property_group_id,
    clean_data,
    make_property_signature,
)
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
            "Latitude": [10.7769, 10.7770],
            "Longitude": [106.7009, 106.7010],
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


def test_different_listing_id_same_property_shares_group():
    row1 = sample_df().iloc[[0]].copy()
    row2 = sample_df().iloc[[0]].copy()
    row1["Listing ID"] = 99991
    row2["Listing ID"] = 99992
    concat_df = pd.concat([row1, row2], ignore_index=True)
    signatures = make_property_signature(concat_df)
    assert signatures.iloc[0] == signatures.iloc[1]


def test_reposted_property_with_changed_price_stays_in_one_group():
    frame = sample_df().iloc[[0]].copy()
    repost = frame.copy()
    repost["Price"] = 5200.0
    repost["Listing ID"] = 88888
    concat_df = pd.concat([frame, repost], ignore_index=True)
    signatures = make_property_signature(concat_df)
    assert signatures.iloc[0] == signatures.iloc[1]


def test_small_coordinate_shift_shares_group():
    frame = sample_df().iloc[[0]].copy()
    shift = frame.copy()
    shift["Latitude"] = float(frame["Latitude"].iloc[0]) + 0.00001
    shift["Longitude"] = float(frame["Longitude"].iloc[0]) + 0.00001
    shift["Listing ID"] = 77777
    concat_df = pd.concat([frame, shift], ignore_index=True)
    signatures = make_property_signature(concat_df)
    assert signatures.iloc[0] == signatures.iloc[1]


def test_different_houses_same_district_not_merged():
    house1 = sample_df().iloc[[0]].copy()
    house2 = sample_df().iloc[[0]].copy()
    house2["Area"] = 120.0
    house2["Width"] = 8.0
    concat_df = pd.concat([house1, house2], ignore_index=True)
    signatures = make_property_signature(concat_df)
    assert signatures.iloc[0] != signatures.iloc[1]


def test_property_group_size_max_greater_than_one():
    frame = sample_df().iloc[[0]].copy()
    repost = frame.copy()
    repost["Listing ID"] = 12345
    concat_df = pd.concat([frame, repost], ignore_index=True)
    concat_df["property_group_id"] = _make_property_group_id(concat_df)
    assert concat_df.groupby("property_group_id").size().max() > 1


def test_spatial_and_quality_features_are_created():
    frame = pd.DataFrame(
        [{"Latitude": 10.7769, "Longitude": 106.7009, "Bedrooms": 2}]
    )
    features = make_features(frame)
    assert features.loc[0, "distance_to_cbd_km"] < 0.1
    assert 0 < features.loc[0, "input_completeness_score"] < 100


def test_reference_date_prevents_serving_skew_and_leakage():
    frame = pd.DataFrame(
        [
            {"listing_date": pd.Timestamp("2025-01-01")},
            {"listing_date": pd.Timestamp("2025-01-10")},
        ]
    )
    ref_date = pd.Timestamp("2025-01-15")
    features = make_features(frame, reference_date=ref_date)
    assert features.loc[0, "days_from_train_reference"] == -14
    assert features.loc[1, "days_from_train_reference"] == -5
