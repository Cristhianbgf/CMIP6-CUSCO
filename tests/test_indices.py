"""Pruebas de los índices ETCCDI con series sintéticas de resultado conocido."""

import numpy as np
import pandas as pd
import xarray as xr

from cmip6cusco.indices import etccdi

TIEMPO = pd.date_range("2001-01-01", "2001-12-31", freq="D")


def _serie(valores):
    return xr.DataArray(np.asarray(valores, dtype="float64"), coords={"time": TIEMPO},
                        dims="time")


def test_dias_con_heladas_y_extremos_de_temperatura():
    tmin = np.full(len(TIEMPO), 5.0)
    tmin[10:20] = -2.0          # 10 días bajo 0 °C
    tmin[100] = -6.5            # la noche más fría
    tmax = np.full(len(TIEMPO), 18.0)
    tmax[200] = 29.3            # la tarde más calurosa
    ds = etccdi.indices_temp(_serie(tmax), _serie(tmin))
    assert int(ds.fd.item()) == 11
    assert ds.tnn.item() == -6.5
    assert ds.txx.item() == 29.3


def test_sin_tmax_solo_indices_de_tmin():
    ds = etccdi.indices_temp(None, _serie(np.full(len(TIEMPO), 3.0)))
    assert set(ds.data_vars) == {"tnn", "fd"}


def test_indices_de_precipitacion():
    pr = np.zeros(len(TIEMPO))
    pr[50:53] = [12.0, 30.0, 8.0]   # 3 días húmedos seguidos, máximo 30
    pr[300] = 0.5                   # llovizna: no cuenta como día húmedo
    ds = etccdi.indices_precip(_serie(pr))
    assert ds.rx1day.item() == 30.0
    assert ds.rx5day.item() == 50.0
    assert ds.prcptot.item() == 50.0   # la llovizna (<1 mm) queda fuera
    assert int(ds.cwd.item()) == 3
    # racha seca más larga: del día 53 al 364 (la llovizna también es día seco)
    assert int(ds.cdd.item()) == len(TIEMPO) - 53


def test_r95p_usa_el_umbral_del_periodo_base():
    pr = np.zeros(len(TIEMPO))
    pr[:100] = 2.0
    pr[100:105] = [20.0, 25.0, 40.0, 3.0, 2.0]
    serie = _serie(pr)
    per95, per99 = etccdi.percentiles_precip_base(serie)
    ds = etccdi.indices_precip(serie, per95, per99)
    # solo los días sobre el p95 de los días húmedos entran a la suma
    esperado = pr[pr > float(per95)].sum()
    assert np.isclose(ds.r95p.item(), esperado)
    assert ds.r99p.item() <= ds.r95p.item()
