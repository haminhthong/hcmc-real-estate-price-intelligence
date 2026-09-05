"""Mô-đun huấn luyện, so sánh mô hình, hiệu chỉnh khoảng tin cậy và lưu trữ artifacts.

Tệp này đảm nhiệm quy trình MLOps huấn luyện end-to-end:
1. Xây dựng pipeline scikit-learn với tiền xử lý chuẩn hóa và các thuật toán hồi quy.
2. Chia nhóm theo thời gian (Grouped Temporal Split) 60/15/10/15 dựa trên ngày đăng muộn nhất của mỗi nhóm bất động sản.
3. Sử dụng mốc ngày tham chiếu (`reference_date`) tính từ tập Train để chống Data Leakage & Training-Serving Skew.
4. Thử nghiệm hai cách đặt bài toán Target: Total Price (`log1p(Price)`) vs Price/m² (`log1p(Price/Area)`).
5. So sánh đa mô hình (Naive Median, District x Property-Type Median, Ridge Linear, Random Forest, HistGradientBoosting, ExtraTrees) trên tập Validation.
6. Sử dụng Conformal Prediction xác định khoảng tin cậy không phụ thuộc phân phối (Target Coverage 80%) trên tập Calibration.
7. Đánh giá kiểm thử độc lập và phân tích sâu độ bao phủ/độ rộng khoảng tin cậy theo phân khúc loại hình, quận/huyện, tầm giá trên tập Test.
"""

json_import = True
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


def build_pipeline(model_name: str = "extra_trees") -> Pipeline:
    """Xây dựng pipeline xử lý đặc trưng và thuật toán dự báo theo cấu hình.

    Args:
        model_name: Tên thuật toán ('naive_median', 'ridge_linear', 'random_forest',
            'hist_gradient_boosting', 'extra_trees').

    Returns:
        Pipeline scikit-learn chưa được fit.
    """
    numeric_features = NUMERIC_FEATURES + FLAG_FEATURES
    cat_pipeline = Pipeline(
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
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    if model_name == "ridge_linear":
        num_pipeline = Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                        keep_empty_features=True,
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        num_pipeline = SimpleImputer(
            strategy="median",
            add_indicator=True,
            keep_empty_features=True,
        )

    preprocessor = ColumnTransformer(
        [
            ("num", num_pipeline, numeric_features),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    regressors = {
        "naive_median": DummyRegressor(strategy="median"),
        "ridge_linear": Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=1,
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=200,
            random_state=RANDOM_STATE,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=1,
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
    }

    if model_name not in regressors:
        raise ValueError(f"Tên mô hình không hợp lệ: {model_name}. Chọn từ {list(regressors.keys())}")

    return Pipeline([("preprocessor", preprocessor), ("model", regressors[model_name])])


def split_group_indices(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chia tập dữ liệu theo nhóm bất động sản và mốc thời gian (Grouped Temporal Split 60/15/10/15).

    NGUYÊN LÝ CHỐNG LEAKAGE & PHÂN TÁCH ĐỘC LẬP:
    1. Nhóm toàn bộ tin đăng theo `property_group_id`.
    2. Lấy ngày đăng bài muộn nhất của từng nhóm và sắp xếp tăng dần theo thời gian.
    3. Phân chia danh sách nhóm theo tỷ lệ 60/15/10/15:
       - Train (60%): Fit đặc trưng và huấn luyện mô hình.
       - Validation (15%): So sánh đa mô hình, chọn target formulation.
       - Calibration (10%): Tính phần dư conformal prediction.
       - Test (15%): Báo cáo đánh giá cuối cùng.
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

    # Kiểm tra tính toàn vẹn (Disjointness test)
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
    """Tính toán phân vị (Quantile) phần dư tuyệt đối phục vụ Split Conformal Prediction."""
    if len(residuals) == 0:
        raise ValueError("Tập phần dư hiệu chỉnh (calibration residuals) không được rỗng.")
    if not 0 < coverage < 1:
        raise ValueError("Mức bao phủ (coverage) phải nằm trong khoảng (0, 1).")
    rank = min(int(np.ceil((len(residuals) + 1) * coverage)), len(residuals))
    return float(np.sort(residuals)[rank - 1])


def regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Tính các chỉ số đo lường hiệu năng hồi quy (đơn vị: triệu VND)."""
    percentage_error = np.abs(prediction - actual) / np.maximum(actual, 1)
    return {
        "mae_million": float(mean_absolute_error(actual, prediction)),
        "median_ae_million": float(median_absolute_error(actual, prediction)),
        "rmse_million": float(mean_squared_error(actual, prediction) ** 0.5),
        "r2": float(r2_score(actual, prediction)),
        "mape_percent": float(percentage_error.mean() * 100),
    }


def save_model_atomically(artifact: dict[str, Any], destination: Path) -> None:
    """Ghi file mô hình nguyên tử (Atomic Write) qua tệp tạm `.tmp`."""
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    joblib.dump(artifact, temporary_path)
    os.replace(temporary_path, destination)
    logger.info("Đã lưu file mô hình nguyên tử thành công tại: %s", destination)


def _write_json(data: dict[str, Any], destination: Path) -> None:
    """Ghi dữ liệu JSON định dạng UTF-8."""
    destination.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def train(data_path: Path = DATA_PATH) -> dict[str, Any]:
    """Huấn luyện mô hình, thực nghiệm target, so sánh đa thuật toán và lưu artifacts."""
    logger.info("Bắt đầu quy trình huấn luyện từ file: %s", data_path)
    raw_data = pd.read_csv(data_path)
    df = clean_data(raw_data)
    audit_stats = getattr(df, "attrs", {}).get("data_audit", {})

    if len(df) < 50:
        raise ValueError("Số lượng mẫu hợp lệ quá nhỏ (< 50 mẫu) để tiến hành phân chia 4 tập.")

    train_idx, validation_idx, calibration_idx, test_idx = split_group_indices(df)
    logger.info(
        "Đã chia tập dữ liệu (Grouped Temporal Split): Train=%d, Validation=%d, Calibration=%d, Test=%d.",
        len(train_idx),
        len(validation_idx),
        len(calibration_idx),
        len(test_idx),
    )

    # 1. Xác định mốc thời gian tham chiếu CHỈ từ tập Train (chống leakage)
    reference_date = df.iloc[train_idx]["listing_date"].max()
    features = make_features(df, reference_date=reference_date)

    df_train = df.iloc[train_idx]
    df_val = df.iloc[validation_idx]
    df_calib = df.iloc[calibration_idx]
    df_test = df.iloc[test_idx]

    features_train = features.iloc[train_idx]
    features_val = features.iloc[validation_idx]
    features_calib = features.iloc[calibration_idx]
    features_test = features.iloc[test_idx]

    # 2. Xây dựng baseline segment trung vị theo (Property Type x location_area)
    train_segment_medians = (
        df_train.groupby(["Property Type", "location_area"])["Price"]
        .median()
        .to_dict()
    )
    overall_train_median = float(df_train["Price"].median())

    def predict_segment_median(df_sub: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, r in df_sub.iterrows():
            pt, loc = r["Property Type"], r["location_area"]
            preds.append(train_segment_medians.get((pt, loc), overall_train_median))
        return np.array(preds)

    val_actual = df_val["Price"].to_numpy()
    test_actual = df_test["Price"].to_numpy()

    # Baseline segment metrics
    segment_val_preds = predict_segment_median(df_val)
    segment_val_metrics = regression_metrics(val_actual, segment_val_preds)

    segment_test_preds = predict_segment_median(df_test)
    segment_test_metrics = regression_metrics(test_actual, segment_test_preds)

    # 3. Thực nghiệm 2 Target Formulations x 5 ML Models trên tập Validation
    target_formulations = ["total_price", "price_per_m2"]
    candidate_models = [
        "naive_median",
        "ridge_linear",
        "random_forest",
        "hist_gradient_boosting",
        "extra_trees",
    ]

    validation_benchmark_results: dict[str, dict[str, dict[str, float]]] = {
        fmt: {} for fmt in target_formulations
    }
    fitted_pipelines: dict[tuple[str, str], Pipeline] = {}

    for fmt in target_formulations:
        if fmt == "total_price":
            y_train = np.log1p(df_train["Price"])
        else:
            y_train = np.log1p(df_train["Price"] / df_train["Area"])

        for m_name in candidate_models:
            pipe = build_pipeline(m_name).fit(features_train, y_train)
            fitted_pipelines[(fmt, m_name)] = pipe

            val_raw_pred = pipe.predict(features_val)
            if fmt == "price_per_m2":
                val_pred_price = np.maximum(
                    np.expm1(val_raw_pred) * df_val["Area"].to_numpy(),
                    0,
                )
            else:
                val_pred_price = np.maximum(np.expm1(val_raw_pred), 0)

            validation_benchmark_results[fmt][m_name] = regression_metrics(
                val_actual, val_pred_price
            )

    # Bổ sung segment median vào kết quả validation benchmark
    validation_benchmark_results["total_price"]["district_property_segment_median"] = segment_val_metrics
    validation_benchmark_results["price_per_m2"]["district_property_segment_median"] = segment_val_metrics

    # 4. Lựa chọn mô hình & target formulation tốt nhất dựa vào MAE trên tập Validation
    best_combo = None
    best_val_mae = float("inf")

    for fmt in target_formulations:
        for m_name in candidate_models:
            mae = validation_benchmark_results[fmt][m_name]["mae_million"]
            if mae < best_val_mae:
                best_val_mae = mae
                best_combo = (fmt, m_name)

    selected_target_fmt, selected_model_name = best_combo
    winning_pipeline = fitted_pipelines[best_combo]

    naive_val_mae = validation_benchmark_results["total_price"]["naive_median"]["mae_million"]
    deployment_approved = bool(best_val_mae < naive_val_mae)
    deployment_reason = (
        f"Mô hình {selected_model_name} (target={selected_target_fmt}) đạt MAE tốt nhất trên Validation "
        f"({best_val_mae:.1f} triệu vs Naive Median {naive_val_mae:.1f} triệu)."
    )

    logger.info("Mô hình được chọn: %s với target formulation=%s", selected_model_name, selected_target_fmt)

    # 5. Conformal Calibration trên tập Calibration
    calib_raw_pred = winning_pipeline.predict(features_calib)
    if selected_target_fmt == "price_per_m2":
        calib_y_true = np.log1p(df_calib["Price"] / df_calib["Area"]).to_numpy()
    else:
        calib_y_true = np.log1p(df_calib["Price"]).to_numpy()

    calibration_residuals = np.abs(calib_y_true - calib_raw_pred)
    target_coverage = 0.8
    residual_log_quantile = conformal_quantile(
        calibration_residuals,
        coverage=target_coverage,
    )

    # 6. Đánh giá độc lập trên tập Test (Report Only)
    test_report_only: dict[str, dict[str, float]] = {
        "district_property_segment_median": segment_test_metrics
    }
    for m_name in candidate_models:
        pipe = fitted_pipelines[(selected_target_fmt, m_name)]
        test_raw_pred = pipe.predict(features_test)
        if selected_target_fmt == "price_per_m2":
            t_pred_price = np.maximum(
                np.expm1(test_raw_pred) * df_test["Area"].to_numpy(),
                0,
            )
        else:
            t_pred_price = np.maximum(np.expm1(test_raw_pred), 0)
        test_report_only[m_name] = regression_metrics(test_actual, t_pred_price)

    # Tính toán khoảng tin cậy conformal trên tập Test cho mô hình được chọn
    test_raw_pred_winning = winning_pipeline.predict(features_test)
    if selected_target_fmt == "price_per_m2":
        test_pred_price = np.maximum(
            np.expm1(test_raw_pred_winning) * df_test["Area"].to_numpy(),
            0,
        )
        lower_bound = np.maximum(
            np.expm1(test_raw_pred_winning - residual_log_quantile) * df_test["Area"].to_numpy(),
            0,
        )
        upper_bound = np.expm1(test_raw_pred_winning + residual_log_quantile) * df_test["Area"].to_numpy()
    else:
        test_pred_price = np.maximum(np.expm1(test_raw_pred_winning), 0)
        lower_bound = np.maximum(
            np.expm1(test_raw_pred_winning - residual_log_quantile),
            0,
        )
        upper_bound = np.expm1(test_raw_pred_winning + residual_log_quantile)

    interval_coverage = float(np.mean((test_actual >= lower_bound) & (test_actual <= upper_bound)))
    interval_widths = upper_bound - lower_bound
    mean_interval_width = float(np.mean(interval_widths))
    median_interval_width = float(np.median(interval_widths))
    coverage_gap = float(interval_coverage - target_coverage)
    relative_interval_width = float(mean_interval_width / max(float(np.mean(test_actual)), 1.0))

    # 7. Lưu trữ Comparison & Metrics Artifacts
    comparison = {
        "selection_split": "validation",
        "selection_metric": "mae_million",
        "recommended_model": selected_model_name,
        "deployed_model": selected_model_name,
        "selected_target_formulation": selected_target_fmt,
        "deployment_approved": deployment_approved,
        "deployment_reason": deployment_reason,
        "validation_candidate_benchmarks": validation_benchmark_results,
        "test_report_only": test_report_only,
    }

    # Bảng giá đơn giá theo phân khúc (cho tra cứu API)
    segment_unit_prices = (
        df_train.assign(unit_price=df_train["Price"] / df_train["Area"])
        .groupby(["Property Type", "location_area"])["unit_price"]
        .median()
        .to_dict()
    )

    artifact = {
        "pipeline": winning_pipeline,
        "model_type": selected_model_name,
        "version": MODEL_VERSION,
        "features": features.columns.tolist(),
        "supported_areas": sorted(df_train["location_area"].unique().tolist()),
        "supported_property_types": sorted(df_train["Property Type"].unique().tolist()),
        "training_ranges": {
            col: [
                float(features_train[col].min()),
                float(features_train[col].max()),
            ]
            for col in NUMERIC_FEATURES
            if features_train[col].notna().any()
        },
        "residual_log_quantile": residual_log_quantile,
        "target_coverage": target_coverage,
        "target_formulation": selected_target_fmt,
        "reference_date": reference_date.isoformat(),
        "split_protocol": "grouped temporal split based on latest listing date per property group (60/15/10/15)",
        "segment_unit_prices": segment_unit_prices,
    }

    winning_test_metrics = regression_metrics(test_actual, test_pred_price)
    metrics = {
        "model_version": MODEL_VERSION,
        "deployed_model": selected_model_name,
        "target_formulation": selected_target_fmt,
        "train_rows": len(train_idx),
        "validation_rows": len(validation_idx),
        "calibration_rows": len(calibration_idx),
        "test_rows": len(test_idx),
        **winning_test_metrics,
        "prediction_interval_target_coverage": target_coverage,
        "prediction_interval_test_coverage": interval_coverage,
        "coverage_gap": coverage_gap,
        "mean_interval_width_million": mean_interval_width,
        "median_interval_width_million": median_interval_width,
        "relative_interval_width": relative_interval_width,
        "deployment_approved": deployment_approved,
        "deployment_reason": deployment_reason,
    }

    # 8. Phân tích sâu Conformal Coverage & Error theo lát cắt dữ liệu (Slices)
    test_result = df_test[["Property Type", "location_area", "Price"]].copy()
    test_result["predicted_price_million"] = test_pred_price
    test_result["absolute_error_million"] = np.abs(test_pred_price - test_actual)
    test_result["in_interval"] = (test_actual >= lower_bound) & (test_actual <= upper_bound)
    test_result["interval_width_million"] = interval_widths

    price_bins = [0, 3000, 7000, 15000, np.inf]
    price_labels = ["< 3 tỷ", "3 - 7 tỷ", "7 - 15 tỷ", "> 15 tỷ"]
    test_result["price_range"] = pd.cut(test_result["Price"], bins=price_bins, labels=price_labels)

    def summarize_slice(df_group: pd.DataFrame) -> dict[str, Any]:
        count = len(df_group)
        if count == 0:
            return {"count": 0, "note": "Không có mẫu"}
        return {
            "count": count,
            "mean_mae_million": round(float(df_group["absolute_error_million"].mean()), 2),
            "median_mae_million": round(float(df_group["absolute_error_million"].median()), 2),
            "interval_coverage": round(float(df_group["in_interval"].mean()), 4),
            "median_interval_width_million": round(float(df_group["interval_width_million"].median()), 2),
        }

    error_analysis = {
        "by_property_type": {
            str(name): summarize_slice(group)
            for name, group in test_result.groupby("Property Type", observed=False)
        },
        "by_location_area": {
            str(name): summarize_slice(group)
            for name, group in test_result.groupby("location_area", observed=False)
        },
        "by_price_range": {
            str(name): summarize_slice(group)
            for name, group in test_result.groupby("price_range", observed=False)
        },
    }

    # 9. Đóng gói Thẻ Dữ Liệu (Data Card)
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
        "reference_date": reference_date.isoformat(),
        "residential_property_types": sorted(df["Property Type"].unique().tolist()),
        "unknown_location_percent": float((df["location_area"] == "Unknown").mean() * 100),
        "coordinates_available_percent": float(
            df[["Latitude", "Longitude"]].notna().all(axis=1).mean() * 100
        ),
        "target": "Price (triệu VND, giá đăng)",
        "known_limitations": [
            "Dữ liệu là giá đăng tham khảo từ tin rao, không phải giá giao dịch thực tế.",
            "Phân phối giá nhà lệch mạnh (long tail) dẫn tới khoảng conformal rộng ở phân khúc cao cấp.",
            "Thông tin địa chỉ và tọa độ GPS còn khuyết ở một số bản ghi.",
            f"Tập dữ liệu sau làm sạch quy mô nhỏ ({len(df)} mẫu), một số phân khúc ít mẫu.",
        ],
    }

    # 10. Xuất các file Artifacts
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

    logger.info("Hoàn tất huấn luyện! Metrics: %s", json.dumps(metrics, ensure_ascii=False))
    return metrics


if __name__ == "__main__":
    train()
