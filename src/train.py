"""Mô-đun huấn luyện, hiệu chỉnh khoảng tin cậy và lưu trữ mô hình định giá.

Tệp này đảm nhiệm quy trình MLOps huấn luyện end-to-end:
1. Xây dựng pipeline scikit-learn gồm xử lý thiếu, mã hóa danh mục và ExtraTreesRegressor.
2. Chia nhóm theo thời gian thành Train/Validation/Calibration/Test theo tỷ lệ 60/15/10/15.
3. Sử dụng Conformal Prediction xác định khoảng tin cậy không phụ thuộc phân phối (Target Coverage 80%).
4. So sánh Extra Trees với baseline trung vị trên Validation; chỉ dùng Calibration để hiệu chỉnh khoảng dự báo.
5. Ghi nhận thông số đánh giá (MAE, RMSE, R2, MAPE) và xuất các artifacts (metrics, model comparison, error analysis, data card).
"""

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
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
    """Xây dựng pipeline xử lý đặc trưng và mô hình Extra Trees.

    Pipeline bao gồm:
    - Xử lý đặc trưng số: Xử lý giá trị khuyết bằng trung vị (median imputer) + Indicator.
    - Xử lý đặc trưng danh mục: Điền yếu vị (most frequent imputer) + OneHotEncoder.
    - Mô hình: ExtraTreesRegressor với 300 cây và min_samples_leaf=1.

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
    regressor = ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=1,
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    return Pipeline(
        [("preprocessor", preprocessor), ("model", regressor)]
    )


def split_group_indices(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chia tập dữ liệu theo nhóm bất động sản và mốc thời gian (Grouped Temporal Split 60/15/10/15).

    NGUYÊN LÝ CHỐNG LEAKAGE & PHÂN TÁCH ĐỘC LẬP:
    1. Nhóm toàn bộ tin đăng theo `property_group_id`.
    2. Lấy ngày đăng bài muộn nhất của từng nhóm và sắp xếp tăng dần theo thời gian.
    3. Phân chia danh sách nhóm theo tỷ lệ:
       - Train: 60% nhóm đầu tiên (Fit preprocessing & model).
       - Validation: 15% nhóm tiếp theo (Chọn mô hình, hyperparameter và so sánh baseline).
       - Calibration: 10% nhóm tiếp theo (Chỉ tính conformal residuals quantile).
       - Test: 15% nhóm muộn nhất (Đánh giá kiểm thử độc lập duy nhất 1 lần).
    4. Kiểm tra đảm bảo cả 4 tập hoàn toàn giao rỗng (Disjoint sets).

    Args:
        df: DataFrame dữ liệu đã được làm sạch và bổ sung `property_group_id`.

    Returns:
        Tuple chứa các mảng chỉ số: (train_idx, validation_idx, calibration_idx, test_idx).

    Raises:
        RuntimeError: Nếu phát hiện bất kỳ nhóm bất động sản nào bị trùng giữa các tập.
    """
    ordered_groups = (
        df.groupby("property_group_id", as_index=False)["listing_date"]
        .max()
        .sort_values(["listing_date", "property_group_id"])
    )
    number_of_groups = len(ordered_groups)
    train_end = int(number_of_groups * 0.60)
    validation_end = int(number_of_groups * 0.75)
    calibration_end = int(number_of_groups * 0.85)

    train_groups: set[str] = set(
        ordered_groups.iloc[:train_end]["property_group_id"]
    )
    validation_groups: set[str] = set(
        ordered_groups.iloc[train_end:validation_end]["property_group_id"]
    )
    calibration_groups: set[str] = set(
        ordered_groups.iloc[validation_end:calibration_end]["property_group_id"]
    )
    test_groups: set[str] = set(
        ordered_groups.iloc[calibration_end:]["property_group_id"]
    )

    train_idx = df.index[df["property_group_id"].isin(train_groups)].to_numpy()
    validation_idx = df.index[
        df["property_group_id"].isin(validation_groups)
    ].to_numpy()
    calibration_idx = df.index[
        df["property_group_id"].isin(calibration_groups)
    ].to_numpy()
    test_idx = df.index[df["property_group_id"].isin(test_groups)].to_numpy()

    # Kiểm tra tính toàn vẹn (Disjointness test giữa 4 tập)
    group_sets = [
        set(df.iloc[index]["property_group_id"])
        for index in (train_idx, validation_idx, calibration_idx, test_idx)
    ]
    if (
        group_sets[0] & group_sets[1]
        or group_sets[0] & group_sets[2]
        or group_sets[0] & group_sets[3]
        or group_sets[1] & group_sets[2]
        or group_sets[1] & group_sets[3]
        or group_sets[2] & group_sets[3]
    ):
        raise RuntimeError("CẢNH BÁO LEAKAGE: Phát hiện nhóm bất động sản bị trùng giữa các tập!")

    return train_idx, validation_idx, calibration_idx, test_idx


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


def regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Tính các chỉ số đo lường hiệu năng mô hình hồi quy (đơn vị: triệu VND).

    Args:
        actual: Mảng giá trị thực tế (triệu VND).
        prediction: Mảng giá trị dự báo từ mô hình (triệu VND).

    Returns:
        Dict chứa các chỉ số: mae_million, median_ae_million, rmse_million, r2, mape_percent.
    """
    percentage_error = np.abs(prediction - actual) / np.maximum(actual, 1)
    return {
        "mae_million": float(mean_absolute_error(actual, prediction)),
        "median_ae_million": float(median_absolute_error(actual, prediction)),
        "rmse_million": float(mean_squared_error(actual, prediction) ** 0.5),
        "r2": float(r2_score(actual, prediction)),
        "mape_percent": float(percentage_error.mean() * 100),
    }


def save_model_atomically(artifact: dict[str, Any], destination: Path) -> None:
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


def _write_json(data: dict[str, Any], destination: Path) -> None:
    """Ghi dữ liệu JSON theo cùng một định dạng UTF-8."""
    destination.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def train(data_path: Path = DATA_PATH) -> dict[str, Any]:
    """Huấn luyện mô hình, hiệu chỉnh khoảng tin cậy conformal và lưu toàn bộ kết quả.

    Args:
        data_path: Đường dẫn tới file dữ liệu thô (.csv).

    Returns:
        Dict chứa thông số đánh giá cuối cùng trên tập Test.
    """
    logger.info("Bắt đầu quy trình huấn luyện từ file: %s", data_path)
    raw_data = pd.read_csv(data_path)
    df = clean_data(raw_data)
    audit_stats = getattr(df, "attrs", {}).get("data_audit", {})

    if len(df) < 50:
        raise ValueError("Số lượng mẫu hợp lệ quá nhỏ (< 50 mẫu) để tiến hành phân chia 4 tập.")

    # Chia dữ liệu theo nhóm và thời gian trước khi tạo mô hình.
    train_idx, validation_idx, calibration_idx, test_idx = split_group_indices(df)
    logger.info(
        "Đã chia tập dữ liệu: Train=%d mẫu, Validation=%d mẫu, Calibration=%d mẫu, Test=%d mẫu.",
        len(train_idx),
        len(validation_idx),
        len(calibration_idx),
        len(test_idx),
    )

    # Tạo đặc trưng và biến đổi log cho giá mục tiêu.
    features = make_features(df)
    log_target = np.log1p(df["Price"])

    # Chỉ khớp pipeline trên tập huấn luyện.
    pipeline = build_pipeline().fit(
        features.iloc[train_idx],
        log_target.iloc[train_idx],
    )

    # Chọn mô hình bằng tập validation, không sử dụng calibration hoặc test.
    validation_log_prediction = pipeline.predict(features.iloc[validation_idx])
    validation_prediction = np.maximum(np.expm1(validation_log_prediction), 0)
    validation_actual = df.iloc[validation_idx]["Price"].to_numpy()
    extra_trees_val_metrics = regression_metrics(validation_actual, validation_prediction)

    baseline = DummyRegressor(strategy="median").fit(
        features.iloc[train_idx],
        log_target.iloc[train_idx],
    )
    baseline_val_prediction = np.maximum(
        np.expm1(baseline.predict(features.iloc[validation_idx])),
        0,
    )
    baseline_val_metrics = regression_metrics(validation_actual, baseline_val_prediction)

    # Chặn triển khai nếu Extra Trees không vượt baseline theo MAE.
    deployment_approved = bool(
        extra_trees_val_metrics["mae_million"] < baseline_val_metrics["mae_million"]
    )
    deployment_reason = (
        "Extra Trees đạt MAE tốt hơn baseline trên tập Validation."
        if deployment_approved
        else "Extra Trees không cải thiện MAE so với baseline trên tập Validation."
    )
    if not deployment_approved:
        raise RuntimeError(deployment_reason)

    recommended_model = "extra_trees" if deployment_approved else "baseline_median"

    # Chỉ dùng calibration để tính phần dư conformal.
    calibration_prediction = pipeline.predict(features.iloc[calibration_idx])
    calibration_residual = np.abs(
        log_target.iloc[calibration_idx].to_numpy() - calibration_prediction
    )
    target_coverage = 0.8
    residual_log_quantile = conformal_quantile(
        calibration_residual,
        coverage=target_coverage,
    )

    # Chỉ dùng test để báo cáo kết quả cuối cùng.
    test_log_prediction = pipeline.predict(features.iloc[test_idx])
    prediction = np.maximum(np.expm1(test_log_prediction), 0)
    actual = df.iloc[test_idx]["Price"].to_numpy()

    # Tính khoảng dự báo conformal trên tập test.
    lower_bound = np.maximum(
        np.expm1(test_log_prediction - residual_log_quantile),
        0,
    )
    upper_bound = np.expm1(test_log_prediction + residual_log_quantile)
    interval_coverage = float(
        np.mean((actual >= lower_bound) & (actual <= upper_bound))
    )
    mean_interval_width = float(np.mean(upper_bound - lower_bound))
    coverage_gap = float(interval_coverage - target_coverage)
    relative_interval_width = float(mean_interval_width / max(float(np.mean(actual)), 1.0))

    baseline_test_prediction = np.maximum(
        np.expm1(baseline.predict(features.iloc[test_idx])),
        0,
    )

    comparison = {
        "selection_split": "validation",
        "selection_metric": "mae_million",
        "recommended_model": recommended_model,
        "deployed_model": "extra_trees",
        "deployment_approved": deployment_approved,
        "deployment_reason": deployment_reason,
        "validation": {
            "baseline_median": baseline_val_metrics,
            "extra_trees": extra_trees_val_metrics,
        },
        "test_report_only": {
            "baseline_median": regression_metrics(actual, baseline_test_prediction),
            "extra_trees": regression_metrics(actual, prediction),
        },
    }

    # Tính trung vị đơn giá từ tập huấn luyện cho từng phân khúc.
    training_data = df.iloc[train_idx]
    training_features = features.iloc[train_idx]
    segment_unit_prices = (
        training_data.assign(unit_price=training_data["Price"] / training_data["Area"])
        .groupby(["Property Type", "location_area"])["unit_price"]
        .median()
        .to_dict()
    )

    # Đóng gói pipeline và metadata phục vụ suy luận.
    artifact = {
        "pipeline": pipeline,
        "model_type": "ExtraTreesRegressor",
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
        "split_protocol": "grouped temporal 60/15/10/15",
        "segment_unit_prices": segment_unit_prices,
    }

    # Tổng hợp metric trên tập test.
    metrics = {
        "model_version": MODEL_VERSION,
        "train_rows": len(train_idx),
        "validation_rows": len(validation_idx),
        "calibration_rows": len(calibration_idx),
        "test_rows": len(test_idx),
        **regression_metrics(actual, prediction),
        "prediction_interval_target_coverage": target_coverage,
        "prediction_interval_test_coverage": interval_coverage,
        "coverage_gap": coverage_gap,
        "mean_interval_width_million": mean_interval_width,
        "relative_interval_width": relative_interval_width,
        "deployment_approved": deployment_approved,
        "deployment_reason": deployment_reason,
    }

    # Phân tích sai số và độ bao phủ theo phân khúc.
    test_result = df.iloc[test_idx][["Property Type", "location_area", "Price"]].copy()
    test_result["absolute_error_million"] = np.abs(prediction - actual)
    test_result["in_interval"] = (actual >= lower_bound) & (actual <= upper_bound)

    # Chia nhóm theo khoảng giá để đọc sai số dễ hơn.
    price_bins = [0, 3000, 7000, 15000, np.inf]
    price_labels = ["< 3 tỷ", "3 - 7 tỷ", "7 - 15 tỷ", "> 15 tỷ"]
    test_result["price_range"] = pd.cut(test_result["Price"], bins=price_bins, labels=price_labels)

    min_segment_size = 20

    def summarize_segment(df_group: pd.DataFrame) -> dict[str, Any]:
        count = len(df_group)
        if count < min_segment_size:
            return {"count": count, "note": f"Mẫu quá ít (< {min_segment_size})"}
        return {
            "count": count,
            "mean_mae_million": round(float(df_group["absolute_error_million"].mean()), 2),
            "median_mae_million": round(float(df_group["absolute_error_million"].median()), 2),
            "interval_coverage": round(float(df_group["in_interval"].mean()), 4),
        }

    error_analysis = {
        "by_property_type": {
            str(name): summarize_segment(group)
            for name, group in test_result.groupby("Property Type", observed=False)
        },
        "by_location_area": {
            str(name): summarize_segment(group)
            for name, group in test_result.groupby("location_area", observed=False)
        },
        "by_price_range": {
            str(name): summarize_segment(group)
            for name, group in test_result.groupby("price_range", observed=False)
        },
    }

    # Tạo thẻ mô tả chất lượng và phạm vi dữ liệu.
    data_card = {
        "source_file": Path(data_path).name,
        "rows_raw": audit_stats.get("rows_raw", len(raw_data)),
        "rows_after_cleaning": audit_stats.get("rows_after_cleaning", len(df)),
        "duplicate_group_percent": audit_stats.get("duplicate_group_percent", 0.0),
        "rows_removed_by_reason": audit_stats.get("rows_removed_by_reason", {}),
        "missing_rate_by_column": {
            col: round(float(df[col].isna().mean() * 100), 2) for col in df.columns
        },
        "target_percentiles": {
            f"p{p}": round(float(df["Price"].quantile(p / 100)), 1)
            for p in [10, 25, 50, 75, 90]
        },
        "area_percentiles": {
            f"p{p}": round(float(df["Area"].quantile(p / 100)), 1)
            for p in [10, 25, 50, 75, 90]
        },
        "rows_per_property_type": df["Property Type"].value_counts().to_dict(),
        "rows_per_area": df["location_area"].value_counts().to_dict(),
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
            f"Tập dữ liệu sau làm sạch nhỏ ({len(df)} mẫu), một số phân khúc có số lượng mẫu < 20.",
        ],
    }

    # Lưu mô hình và các báo cáo bằng định dạng thống nhất.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_model_atomically(artifact, MODEL_PATH)

    for destination, report in (
        (METRICS_PATH, metrics),
        (MODEL_COMPARISON_PATH, comparison),
        (ERROR_ANALYSIS_PATH, error_analysis),
        (DATA_CARD_PATH, data_card),
    ):
        _write_json(report, destination)

    logger.info("Hoàn tất huấn luyện! Kết quả metrics: %s", json.dumps(metrics, ensure_ascii=False))
    return metrics


if __name__ == "__main__":
    train()
