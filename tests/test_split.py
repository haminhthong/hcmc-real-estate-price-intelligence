import pandas as pd

from src.train import split_group_indices


def test_train_test_property_groups_do_not_overlap():
    frame = pd.DataFrame(
        {"property_group_id": [f"group-{index}" for index in range(100)]}
    )
    train, calibration, test = split_group_indices(frame)
    train_groups = set(frame.iloc[train].property_group_id)
    calibration_groups = set(frame.iloc[calibration].property_group_id)
    test_groups = set(frame.iloc[test].property_group_id)
    assert train_groups.isdisjoint(calibration_groups)
    assert train_groups.isdisjoint(test_groups)
    assert calibration_groups.isdisjoint(test_groups)
