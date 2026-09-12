import pandas as pd
from geoai_copilot.data import qc_summary, prepare_features


def test_qc_summary():
    df = pd.DataFrame({"GR": [10, 20], "Phi": [0.1, 0.2]})
    result = qc_summary(df)
    assert result["rows"] == 2
    assert result["duplicate_rows"] == 0


def test_prepare_features():
    df = pd.DataFrame({"GR": [10, None], "Phi": [0.1, 0.2]})
    X, y = prepare_features(df, ["GR"], "Phi")
    assert len(X) == 1
    assert len(y) == 1
