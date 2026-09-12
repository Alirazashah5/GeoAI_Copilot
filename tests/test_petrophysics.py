import pandas as pd
from geoai_copilot.petrophysics import density_porosity, shale_volume_linear


def test_density_porosity():
    s = pd.Series([2.65, 1.0])
    out = density_porosity(s)
    assert out.iloc[0] == 0
    assert out.iloc[1] == 1


def test_shale_volume_is_clipped():
    s = pd.Series([0, 50, 100])
    out = shale_volume_linear(s, 0, 100)
    assert out.tolist() == [0.0, 0.5, 1.0]
