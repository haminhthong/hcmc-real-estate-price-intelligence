import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    FLAG_FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    MODEL_VERSION,
    NUMERIC_FEATURES,
    RANDOM_STATE,
)
from .data_processing import clean_data
from .feature_engineering import make_features


def build_pipeline() -> Pipeline:
    """Tạo quy trình tiền xử lý và mô hình dùng chung khi huấn luyện."""
    numeric_features = NUMERIC_FEATURES + FLAG_FEATURES
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                SimpleImputer(strategy="median", add_indicator=True),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    regressor = RandomForestRegressor(
        n_estimators=180,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return Pipeline(
        [("preprocessor", preprocessor), ("model", regressor)]
    )


def split_group_indices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chia ba tập huấn luyện, hiệu chỉnh và kiểm tra theo nhóm tin đăng."""
    outer_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    train_calibration_idx, test_idx = next(
        outer_split.split(df, groups=df["property_group_id"])
    )

    train_calibration = df.iloc[train_calibration_idx]
    inner_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=RANDOM_STATE + 1,
    )
    local_train_idx, local_calibration_idx = next(
        inner_split.split(
            train_calibration,
            groups=train_calibration["property_group_id"],
        )
    )
    train_idx = train_calibration_idx[local_train_idx]
    calibration_idx = train_calibration_idx[local_calibration_idx]

    group_sets = [
        set(df.iloc[index]["property_group_id"])
        for index in (train_idx, calibration_idx, test_idx)
    ]
    if (
        group_sets[0] & group_sets[1]
        or group_sets[0] & group_sets[2]
        or group_sets[1] & group_sets[2]
    ):
        raise RuntimeError("Phát hiện nhóm bất động sản bị trùng giữa các tập.")
    return train_idx, calibration_idx, test_idx


def train(data_path=DATA_PATH) -> dict[str, float | int | str]:
    """Huấn luyện mô hình, hiệu chỉnh khoảng giá và lưu kết quả đánh giá."""
    df = clean_data(pd.read_csv(data_path))
    if len(df) < 50:
        raise ValueError("Cần ít nhất 50 mẫu hợp lệ để chia ba tập dữ liệu.")

    train_idx, calibration_idx, test_idx = split_group_indices(df)
    features = make_features(df)
    log_target = np.log1p(df["Price"])
    pipeline = build_pipeline().fit(
        features.iloc[train_idx],
        log_target.iloc[train_idx],
    )

    calibration_prediction = pipeline.predict(features.iloc[calibration_idx])
    calibration_residual = np.abs(
        log_target.iloc[calibration_idx].to_numpy() - calibration_prediction
    )
    residual_log_quantile = float(np.quantile(calibration_residual, 0.8))

    prediction = np.maximum(
        np.expm1(pipeline.predict(features.iloc[test_idx])),
        0,
    )
    actual = df.iloc[test_idx]["Price"].to_numpy()
    percentage_error = np.abs(prediction - actual) / np.maximum(actual, 1)

    training_data = df.iloc[train_idx]
    segment_unit_prices = (
        training_data.assign(unit_price=training_data["Price"] / training_data["Area"])
        .groupby(["Property Type", "location_area"])["unit_price"]
        .median()
        .to_dict()
    )
    artifact = {
        "pipeline": pipeline,
        "version": MODEL_VERSION,
        "features": features.columns.tolist(),
        "supported_areas": sorted(training_data["location_area"].unique().tolist()),
        "supported_property_types": sorted(
            training_data["Property Type"].unique().tolist()
        ),
        "training_ranges": {
            column: [
                float(training_data[column].min()),
                float(training_data[column].max()),
            ]
            for column in NUMERIC_FEATURES
            if training_data[column].notna().any()
        },
        "residual_log_quantile": residual_log_quantile,
        "segment_unit_prices": segment_unit_prices,
    }
    metrics = {
        "model_version": MODEL_VERSION,
        "train_rows": int(len(train_idx)),
        "calibration_rows": int(len(calibration_idx)),
        "test_rows": int(len(test_idx)),
        "mae_million": float(mean_absolute_error(actual, prediction)),
        "rmse_million": float(mean_squared_error(actual, prediction) ** 0.5),
        "r2": float(r2_score(actual, prediction)),
        "mape_percent": float(percentage_error.mean() * 100),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    train()
