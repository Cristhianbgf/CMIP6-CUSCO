"""Pruebas de la corrección de sesgo: el PSS y la propiedad clave del QDM
(corrige el sesgo y conserva la señal de cambio del modelo)."""

import numpy as np
import pandas as pd
import xarray as xr
from xsdba import QuantileDeltaMapping

from cmip6cusco.ajuste_sesgo.qdm import pss


def test_pss_identicas_y_disjuntas():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    bins = np.linspace(-10, 30, 81)
    assert np.isclose(pss(a, a, bins), 1.0)
    assert pss(a, a + 15, bins) == 0.0


def _temperatura(media, n_anios, rng, inicio):
    t = pd.date_range(f"{inicio}-01-01", periods=365 * n_anios, freq="D")
    ciclo = 3 * np.sin(2 * np.pi * t.dayofyear / 365)
    da = xr.DataArray(media + ciclo + rng.normal(0, 1.5, len(t)),
                      coords={"time": t}, dims="time")
    da.attrs["units"] = "degC"
    return da


def test_qdm_corrige_el_sesgo_y_conserva_el_calentamiento():
    """Mismo esquema que el proyecto para temperatura (aditivo, 50 cuantiles,
    por mes): un modelo 4 °C más frío que lo observado, que en el futuro se
    calienta 2,5 °C, debe quedar en lo observado + 2,5 °C."""
    rng = np.random.default_rng(42)
    obs = _temperatura(20.0, 20, rng, 1981)
    hist = _temperatura(16.0, 20, rng, 1981)
    fut = _temperatura(18.5, 20, rng, 2071)
    qdm = QuantileDeltaMapping.train(obs, hist, nquantiles=50,
                                     group="time.month", kind="+")
    corr = qdm.adjust(fut, interp="linear")
    assert abs(float(corr.mean()) - (float(obs.mean()) + 2.5)) < 0.3
