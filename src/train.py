"""Mô-đun huấn luyện, hiệu chỉnh khoảng tin cậy và lưu trữ mô hình định giá.

Tệp này đảm nhiệm quy trình MLOps huấn luyện end-to-end:
1. Xây dựng scikit-learn Pipeline (SimpleImputer + OneHotEncoder + RandomForestRegressor).
2. Phân chia tập dữ liệu theo nhóm thời gian (Grouped Temporal Split 64/16/20) chống rò rỉ dữ liệu (Data Leakage).
3. Sử dụng Conformal Prediction xác định khoảng tin cậy không phụ thuộc phân phối (Target Coverage 80%).
4. So sánh mô hình Random Forest với Median Baseline trên tập Calibration.
5. Ghi nhận thông số đánh giá (MAE, RMSE, R2, MAPE) và xuất các artifacts (metrics, model comparison, error analysis, data card).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Set, Tuple

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
    logger,
)
from .data_processing import clean_data
from .feature_engineering import make_features


def build_pipeline() -> Pipeline:
    """Xây dựng Pipeline xử lý đặc trưng và mô hình Random Forest cho định giá bất động sản.

    Pipeline bao gồm:
    - Xử lý đặc trưng số: Xử lý giá trị khuyết bằng trung vị (median imputer) + Indicator.
    - Xử lý đặc trưng danh mục: Điền yếu vị (most frequent imputer) + OneHotEncoder.
    - Regressor: RandomForestRegressor (180 trees, min_samples_leaf=2).

    Returns:
        Pipeline chưa được fit.
    """
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


def split_group_indices(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chia tập dữ liệu theo nhóm bất động sản và mốc thời gian (Grouped Temporal Split).

    NGUYÊN LÝ CHỐNG LEAKAGE:
    1. Nhóm toàn bộ tin đăng theo `property_group_id`.
    2. Lấy ngày đăng bài muộn nhất của từng nhóm và sắp xếp tăng dần theo thời gian.
    3. Phân chia danh sách nhóm theo tỷ lệ:
       - Train: 64% nhóm đầu tiên (dữ liệu cũ nhất trong quá khứ).
       - Calibration: 16% nhóm tiếp theo (dùng hiệu chỉnh khoảng tin cậy conformal).
       - Test: 20% nhóm muộn nhất (dùng để kiểm thử giả lập dữ liệu thực tế tương lai).
    4. Kiểm tra đảm bảo tập Train, Calibration và Test hoàn toàn giao rỗng (Disjoint sets).

    Args:
        df: DataFrame dữ liệu đã được làm sạch và bổ sung `property_group_id`.

    Returns:
        Tuple chứa các mảng chỉ số (indices): (train_idx, calibration_idx, test_idx).

    Raises:
        RuntimeError: Nếu phát hiện bất kỳ nhóm bất động sản nào bị trùng giữa các tập.
    """
    ordered_groups = (
        df.groupby("property_group_id", as_index=False)["listing_date"]
        .max()
        .sort_values(["listing_date", "property_group_id"])
    )
    number_of_groups = len(ordered_groups)
    test_start = int(number_of_groups * 0.8)
    calibration_start = int(number_of_groups * 0.64)

    train_groups: Set[str] = set(
        ordered_groups.iloc[:calibration_start]["property_group_id"]
    )
    calibration_groups: Set[str] = set(
        ordered_groups.iloc[calibration_start:test_start]["property_group_id"]
    )
    test_groups: Set[str] = set(
        ordered_groups.iloc[test_start:]["property_group_id"]
    )

    train_idx = df.index[df["property_group_id"].isin(train_groups)].to_numpy()
    calibration_idx = df.index[
        df["property_group_id"].isin(calibration_groups)
    ].to_numpy()
    test_idx = df.index[df["property_group_id"].isin(test_groups)].to_numpy()

    # Kiểm tra tính toàn vẹn (Disjointness test)
    group_sets = [
        set(df.iloc[index]["property_group_id"])
        for index in (train_idx, calibration_idx, test_idx)
    ]
    if (
        group_sets[0] & group_sets[1]
        or group_sets[0] & group_sets[2]
        or group_sets[1] & group_sets[2]
    ):
        raise RuntimeError("CẢNH BÁO LEAKAGE: Phát hiện nhóm bất động sản bị trùng giữa các tập!")

    return train_idx, calibration_idx, test_idx


def conformal_quantile(residuals: np.ndarray, coverage: float = 0.8) -> float:
    """Tính toán phân vị (Quantile) phần dư phục vụ Split Conformal Prediction.

    Công thức chọn rank phân vị có điều chỉnh kích thước mẫu calibration n:
        rank = ceil((n + 1) * coverage)

    Args:
        residuals: Mảng chứa giá trị phần dư tuyệt đối trên tập Calibration: |y_true - y_pred|.
        coverage: Tỷ lệ bao phủ mục tiêu (Mặc định 80% tức 0.8).

    Returns:
        Ngưỡng giá trị phần dư tương ứng với mức bao phủ mục tiêu.
    """
    if len(residuals) == 0:
        raise ValueError("Tập phần dư hiệu chỉnh (calibration residuals) không được rỗng.")
    if not 0 < coverage < 1:
        raise ValueError("Mức bao phủ (coverage) phải nằm trong khoảng (0, 1).")
    rank = min(int(np.ceil((len(residuals) + 1) * coverage)), len(residuals))
    return float(np.sort(residuals)[rank - 1])


def regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    """Tính các chỉ số đo lường hiệu năng mô hình hồi quy (đơn vị: triệu VND).

    Args:
        actual: Mảng giá trị thực tế (triệu VND).
        prediction: Mảng giá trị dự báo từ mô hình (triệu VND).

    Returns:
        Dict chứa các chỉ số: mae_million, rmse_million, r2, mape_percent.
    """
    percentage_error = np.abs(prediction - actual) / np.maximum(actual, 1)
    return {
        "mae_million": float(mean_absolute_error(actual, prediction)),
        "rmse_million": float(mean_squared_error(actual, prediction) ** 0.5),
        "r2": float(r2_score(actual, prediction)),
        "mape_percent": float(percentage_error.mean() * 100),
    }


def save_model_atomically(artifact: Dict[str, Any], destination: Path) -> None:
    """Ghi file mô hình nguyên tử (Atomic Write) qua tệp tạm `.tmp`.

    Đảm bảo nếu quá trình ghi file bị gián đoạn, file mô hình gốc phục vụ API
    không bị hỏng hóc hoặc thiếu hụt dữ liệu.

    Args:
        artifact: Dictionary chứa pipeline và các thông số mô hình.
        destination: Đường dẫn đích lưu file `.joblib`.
    """
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    joblib.dump(artifact, temporary_path)
    os.replace(temporary_path, destination)
    logger.info("Đã lưu file mô hình nguyên tử thành công tại: %s", destination)


def train(data_path: Path = DATA_PATH) -> Dict[str, Any]:
    """Huấn luyện mô hình, hiệu chỉnh khoảng tin cậy conformal và lưu toàn bộ kết quả.

    Args:
        data_path: Đường dẫn tới file dữ liệu thô (.csv).

    Returns:
        Dict chứa thông số đánh giá cuối cùng trên tập Test.
    """
    logger.info("Bắt đầu quy trình huấn luyện từ file: %s", data_path)
    raw_data = pd.read_csv(data_path)
    df = clean_data(raw_data)
    if len(df) < 50:
        raise ValueError("Số lượng mẫu hợp lệ quá nhỏ (< 50 mẫu) để tiến hành phân chia 3 tập.")

    # 1. Phân chia Grouped Temporal Split
    train_idx, calibration_idx, test_idx = split_group_indices(df)
    logger.info(
        "Đã chia tập dữ liệu: Train=%d mẫu, Calibration=%d mẫu, Test=%d mẫu.",
        len(train_idx),
        len(calibration_idx),
        len(test_idx),
    )

    # 2. Chuẩn bị đặc trưng và log-transform biến mục tiêu Price
    features = make_features(df)
    log_target = np.log1p(df["Price"])

    # 3. Fit Pipeline trên tập Train
    pipeline = build_pipeline().fit(
        features.iloc[train_idx],
        log_target.iloc[train_idx],
    )

    # 4. Hiệu chỉnh Conformal Prediction trên tập Calibration
    calibration_prediction = pipeline.predict(features.iloc[calibration_idx])
    calibration_residual = np.abs(
        log_target.iloc[calibration_idx].to_numpy() - calibration_prediction
    )
    target_coverage = 0.8
    residual_log_quantile = conformal_quantile(
        calibration_residual,
        coverage=target_coverage,
    )

    # 5. Đánh giá kiểm thử độc lập trên tập Test
    test_log_prediction = pipeline.predict(features.iloc[test_idx])
    prediction = np.maximum(np.expm1(test_log_prediction), 0)
    actual = df.iloc[test_idx]["Price"].to_numpy()

    # Tính khoảng dự báo conformal trên tập Test
    lower_bound = np.maximum(
        np.expm1(test_log_prediction - residual_log_quantile),
        0,
    )
    upper_bound = np.expm1(test_log_prediction + residual_log_quantile)
    interval_coverage = float(
        np.mean((actual >= lower_bound) & (actual <= upper_bound))
    )

    # 6. So sánh với mô hình Baseline (Median Regressor)
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

    # 7. Tính đơn giá trung vị phân khúc theo từng Loại hình x Quận/huyện
    training_data = df.iloc[train_idx]
    training_features = features.iloc[train_idx]
    segment_unit_prices = (
        training_data.assign(unit_price=training_data["Price"] / training_data["Area"])
        .groupby(["Property Type", "location_area"])["unit_price"]
        .median()
        .to_dict()
    )

    # Đóng gói Model Artifact
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

    # Đóng gói Metrics
    metrics = {
        "model_version": MODEL_VERSION,
        "train_rows": int(len(train_idx)),
        "calibration_rows": int(len(calibration_idx)),
        "test_rows": int(len(test_idx)),
        **regression_metrics(actual, prediction),
        "prediction_interval_target_coverage": target_coverage,
        "prediction_interval_test_coverage": interval_coverage,
    }

    # Phân tích sai số theo nhóm
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

    # Thẻ thông tin dữ liệu (Data Card)
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
            "Dữ liệu là giá đăng tham khảo, không phải giá giao dịch thực tế.",
            "Địa chỉ và tọa độ còn thiếu ở một số bản ghi.",
            "Dữ liệu mẫu chưa bao phủ đầy đủ toàn bộ phân khúc thị trường TP.HCM.",
        ],
    }

    # 8. Lưu trữ các Artifacts
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

    logger.info("Hoàn tất huấn luyện! Kết quả metrics: %s", json.dumps(metrics, ensure_ascii=False))
    return metrics


if __name__ == "__main__":
    train()

