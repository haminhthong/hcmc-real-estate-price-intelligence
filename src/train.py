import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import (
    CATEGORICAL_FEATURES,
    DATA_CARD_PATH,
    DATA_PATH,
    ERROR_ANALYSIS_PATH,
    FLAG_FEATURES,
    METRICS_PATH,
    MODEL_COMPARISON_PATH,
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
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="most_frequent",
                                keep_empty_features=True,
                            ),
                        ),
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
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    return Pipeline(
        [("preprocessor", preprocessor), ("model", regressor)]
    )


def split_group_indices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chia theo thời gian; mỗi nhóm tin chỉ xuất hiện trong một tập."""
    ordered_groups = (
        df.groupby("property_group_id", as_index=False)["listing_date"]
        .max()
        .sort_values(["listing_date", "property_group_id"])
    )
    number_of_groups = len(ordered_groups)
    test_start = int(number_of_groups * 0.8)
    calibration_start = int(number_of_groups * 0.64)
    train_groups = set(ordered_groups.iloc[:calibration_start]["property_group_id"])
    calibration_groups = set(
        ordered_groups.iloc[calibration_start:test_start]["property_group_id"]
    )
    test_groups = set(ordered_groups.iloc[test_start:]["property_group_id"])
    train_idx = df.index[df["property_group_id"].isin(train_groups)].to_numpy()
    calibration_idx = df.index[
        df["property_group_id"].isin(calibration_groups)
    ].to_numpy()
    test_idx = df.index[df["property_group_id"].isin(test_groups)].to_numpy()

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


def conformal_quantile(residuals: np.ndarray, coverage: float = 0.8) -> float:
    """Tính phân vị conformal có hiệu chỉnh theo cỡ tập hiệu chỉnh."""
    if len(residuals) == 0:
        raise ValueError("Tập phần dư hiệu chỉnh không được rỗng.")
    if not 0 < coverage < 1:
        raise ValueError("Mức bao phủ phải nằm trong khoảng (0, 1).")
    rank = min(int(np.ceil((len(residuals) + 1) * coverage)), len(residuals))
    return float(np.sort(residuals)[rank - 1])


def regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict:
    """Tính các chỉ số đánh giá trên thang giá thật."""
    percentage_error = np.abs(prediction - actual) / np.maximum(actual, 1)
    return {
        "mae_million": float(mean_absolute_error(actual, prediction)),
        "rmse_million": float(mean_squared_error(actual, prediction) ** 0.5),
        "r2": float(r2_score(actual, prediction)),
        "mape_percent": float(percentage_error.mean() * 100),
    }


def save_model_atomically(artifact: dict, destination: Path) -> None:
    """Ghi mô hình qua tệp tạm để tránh làm hỏng tệp đang phục vụ."""
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    joblib.dump(artifact, temporary_path)
    os.replace(temporary_path, destination)


def train(data_path: Path = DATA_PATH) -> dict[str, float | int | str]:
    """Huấn luyện mô hình, hiệu chỉnh khoảng giá và lưu kết quả đánh giá."""
    raw_data = pd.read_csv(data_path)
    df = clean_data(raw_data)
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
    target_coverage = 0.8
    residual_log_quantile = conformal_quantile(
        calibration_residual,
        coverage=target_coverage,
    )

    test_log_prediction = pipeline.predict(features.iloc[test_idx])
    prediction = np.maximum(np.expm1(test_log_prediction), 0)
    actual = df.iloc[test_idx]["Price"].to_numpy()
    lower_bound = np.maximum(
        np.expm1(test_log_prediction - residual_log_quantile),
        0,
    )
    upper_bound = np.expm1(test_log_prediction + residual_log_quantile)
    interval_coverage = float(
        np.mean((actual >= lower_bound) & (actual <= upper_bound))
    )

    baseline = DummyRegressor(strategy="median").fit(
        features.iloc[train_idx],
        log_target.iloc[train_idx],
    )
    baseline_prediction = np.maximum(
        np.expm1(baseline.predict(features.iloc[test_idx])),
        0,
    )
    baseline_calibration_prediction = np.maximum(
        np.expm1(baseline.predict(features.iloc[calibration_idx])),
        0,
    )
    calibration_actual = df.iloc[calibration_idx]["Price"].to_numpy()
    random_forest_calibration_prediction = np.maximum(
        np.expm1(calibration_prediction),
        0,
    )
    baseline_calibration_metrics = regression_metrics(
        calibration_actual,
        baseline_calibration_prediction,
    )
    random_forest_calibration_metrics = regression_metrics(
        calibration_actual,
        random_forest_calibration_prediction,
    )
    champion = (
        "random_forest"
        if random_forest_calibration_metrics["mae_million"]
        < baseline_calibration_metrics["mae_million"]
        else "baseline_median"
    )
    comparison = {
        "selection_split": "calibration",
        "selection_metric": "mae_million",
        "champion": champion,
        "calibration": {
            "baseline_median": baseline_calibration_metrics,
            "random_forest": random_forest_calibration_metrics,
        },
        "test_report_only": {
            "baseline_median": regression_metrics(actual, baseline_prediction),
            "random_forest": regression_metrics(actual, prediction),
        },
    }

    training_data = df.iloc[train_idx]
    training_features = features.iloc[train_idx]
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
                float(training_features[column].min()),
                float(training_features[column].max()),
            ]
            for column in NUMERIC_FEATURES
            if training_features[column].notna().any()
        },
        "residual_log_quantile": residual_log_quantile,
        "target_coverage": target_coverage,
        "split_protocol": "grouped temporal 64/16/20",
        "segment_unit_prices": segment_unit_prices,
    }
    metrics = {
        "model_version": MODEL_VERSION,
        "train_rows": int(len(train_idx)),
        "calibration_rows": int(len(calibration_idx)),
        "test_rows": int(len(test_idx)),
        **regression_metrics(actual, prediction),
        "prediction_interval_target_coverage": target_coverage,
        "prediction_interval_test_coverage": interval_coverage,
    }

    test_result = df.iloc[test_idx][["Property Type", "location_area"]].copy()
    test_result["absolute_error_million"] = np.abs(prediction - actual)
    error_analysis = {
        "by_property_type": test_result.groupby("Property Type")[
            "absolute_error_million"
        ].agg(["count", "mean", "median"]).round(2).to_dict("index"),
        "by_location_area": test_result.groupby("location_area")[
            "absolute_error_million"
        ].agg(["count", "mean", "median"]).round(2).to_dict("index"),
    }
    data_card = {
        "source_file": Path(data_path).name,
        "rows_raw": int(len(raw_data)),
        "rows_after_cleaning": int(len(df)),
        "date_min": df["listing_date"].min().isoformat(),
        "date_max": df["listing_date"].max().isoformat(),
        "residential_property_types": sorted(df["Property Type"].unique().tolist()),
        "unknown_location_percent": float(
            (df["location_area"] == "Unknown").mean() * 100
        ),
        "coordinates_available_percent": float(
            df[["Latitude", "Longitude"]].notna().all(axis=1).mean() * 100
        ),
        "target": "Price (triệu VND, giá đăng)",
        "known_limitations": [
            "Dữ liệu là giá đăng, không phải giá giao dịch.",
            "Địa chỉ và tọa độ còn thiếu ở một phần đáng kể bản ghi.",
            "Dữ liệu mẫu không đại diện đầy đủ cho toàn bộ thị trường TP.HCM.",
        ],
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_model_atomically(artifact, MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    MODEL_COMPARISON_PATH.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    ERROR_ANALYSIS_PATH.write_text(
        json.dumps(error_analysis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    DATA_CARD_PATH.write_text(
        json.dumps(data_card, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    train()
