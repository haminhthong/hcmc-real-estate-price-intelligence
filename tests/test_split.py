import pandas as pd

from src.train import split_group_indices


def test_train_validation_calibration_test_property_groups_do_not_overlap():
    frame = pd.DataFrame(
        {
            "property_group_id": [f"group-{index}" for index in range(100)],
            "listing_date": pd.date_range("2025-01-01", periods=100),
        }
    )
    train, validation, calibration, test = split_group_indices(frame)
    train_groups = set(frame.iloc[train].property_group_id)
    val_groups = set(frame.iloc[validation].property_group_id)
    calib_groups = set(frame.iloc[calibration].property_group_id)
    test_groups = set(frame.iloc[test].property_group_id)

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(calib_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(calib_groups)
    assert val_groups.isdisjoint(test_groups)
    assert calib_groups.isdisjoint(test_groups)

    assert frame.iloc[train].listing_date.max() <= frame.iloc[validation].listing_date.min()
    assert frame.iloc[validation].listing_date.max() <= frame.iloc[calibration].listing_date.min()
    assert frame.iloc[calibration].listing_date.max() <= frame.iloc[test].listing_date.min()


def test_grouped_temporal_split_prioritizes_group_isolation_over_strict_row_dates():
    """Kiểm tra: Bất động sản có nhiều tin đăng rải rác (Jan, Mar, Jul) sẽ cùng đi vào

    split tương ứng với ngày muộn nhất (Jul), cô lập toàn bộ các tin đăng của căn nhà
    vào cùng một tập để chống Data Leakage.
    """
    # 100 nhóm cơ sở trải dài từ tháng 1 tới tháng 10
    dates = pd.date_range("2025-01-01", periods=100)
    records = [{"property_group_id": f"group-{i}", "listing_date": dates[i]} for i in range(100)]

    # Nhóm đặc biệt: Có tin đăng rất sớm (Jan 02) và tin đăng muộn (Oct 10)
    records.append({"property_group_id": "group-multi-listing", "listing_date": pd.Timestamp("2025-01-02")})
    records.append({"property_group_id": "group-multi-listing", "listing_date": pd.Timestamp("2025-10-15")})

    frame = pd.DataFrame(records)
    train, val, calib, test = split_group_indices(frame)

    # Đảm bảo cả hai tin đăng của group-multi-listing đều nằm trọn vẹn trong tập Test (tập mới nhất)
    multi_indices = frame.index[frame["property_group_id"] == "group-multi-listing"].to_numpy()
    assert set(multi_indices).issubset(set(test)), "Mọi tin đăng của nhóm phải cùng nằm trong tập mới nhất (Test)"
    assert set(multi_indices).isdisjoint(set(train)), "Không được rò rỉ tin đăng cũ của nhóm sang tập Train"
