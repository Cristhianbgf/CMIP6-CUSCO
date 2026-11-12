"""Índices de extremos ETCCDI con xclim.

Envoltorios delgados sobre xclim 0.61. La lista de índices está en
configs/dominio.yml:
  precipitación: rx1day, rx5day, r95p, r99p, cdd, cwd, prcptot, sdii
  temperatura:   tnn, txx, tn10p, tx90p, fd, dtr

Cuatro índices necesitan periodo base (percentil calculado en la línea de
referencia y aplicado al periodo objetivo):
  - r95p/r99p: umbral = percentil 95/99 de la precip de días húmedos (≥1 mm)
    del periodo base, un valor por celda (ETCCDI clásico). El índice es la
    suma anual de la precip que supera ese umbral.
  - tn10p/tx90p: percentil por día-del-año (ventana 5 días, percentile_doy)
    del periodo base; el índice es el % de días bajo/sobre ese percentil.

Todos se agregan por año hidrológico/civil (freq="YS" por defecto).
"""

import numpy as np
import xarray as xr
from xclim import indices as xi
from xclim.core.calendar import percentile_doy


def _unidades(da: xr.DataArray, var: str) -> xr.DataArray:
    """Fija unidades que xclim sabe parsear ('mm/d' no lo reconoce; se usa 'mm/day')."""
    da = da.copy()
    da.attrs["units"] = "mm/day" if var == "pr" else "degC"
    return da


# --------------------------- periodo base ---------------------------

def percentiles_precip_base(pr_base: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Percentil 95 y 99 de la precip de días húmedos (≥1 mm), por celda."""
    pr_base = _unidades(pr_base, "pr")
    humedos = pr_base.where(pr_base >= 1)
    q = humedos.quantile([0.95, 0.99], dim="time")
    return (q.sel(quantile=0.95, drop=True).rename("per95"),
            q.sel(quantile=0.99, drop=True).rename("per99"))


def percentiles_temp_base(tasmax_base: xr.DataArray,
                          tasmin_base: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Percentil 90 de Tmax y 10 de Tmin por día-del-año (ventana 5 días)."""
    tx90 = percentile_doy(_unidades(tasmax_base, "tasmax"), per=90, window=5)
    tn10 = percentile_doy(_unidades(tasmin_base, "tasmin"), per=10, window=5)
    return tx90, tn10


# --------------------------- índices ---------------------------

def indices_precip(pr: xr.DataArray, per95: xr.DataArray | None = None,
                   per99: xr.DataArray | None = None, freq: str = "YS") -> xr.Dataset:
    """Los 8 índices de precipitación (r95p/r99p solo si se pasa el periodo base)."""
    pr = _unidades(pr, "pr")
    out = {
        "rx1day": xi.max_n_day_precipitation_amount(pr, window=1, freq=freq),
        "rx5day": xi.max_n_day_precipitation_amount(pr, window=5, freq=freq),
        "cdd": xi.maximum_consecutive_dry_days(pr, thresh="1 mm/day", freq=freq),
        "cwd": xi.maximum_consecutive_wet_days(pr, thresh="1 mm/day", freq=freq),
        # PRCPTOT ETCCDI = total anual de días húmedos (≥1 mm)
        "prcptot": pr.where(pr >= 1).resample(time=freq).sum().assign_attrs(
            units="mm", long_name="Precipitación total de días húmedos"),
        "sdii": xi.daily_pr_intensity(pr, thresh="1 mm/day", freq=freq),
    }
    if per95 is not None:
        out["r95p"] = pr.where(pr > per95).resample(time=freq).sum().assign_attrs(
            units="mm", long_name="Precip. anual sobre el p95 del periodo base")
        out["r99p"] = pr.where(pr > per99).resample(time=freq).sum().assign_attrs(
            units="mm", long_name="Precip. anual sobre el p99 del periodo base")
    return xr.Dataset(out)


def indices_temp(tasmax: xr.DataArray | None, tasmin: xr.DataArray,
                 tx90_per: xr.DataArray | None = None,
                 tn10_per: xr.DataArray | None = None, freq: str = "YS") -> xr.Dataset:
    """Índices de temperatura. Si tasmax es None (p.ej. UKESM sin tasmax
    diaria), se omiten los que la requieren (txx, dtr, tx90p) y se calculan
    solo los de tasmin (tnn, fd, tn10p)."""
    tasmin = _unidades(tasmin, "tasmin")
    out = {
        "tnn": xi.tn_min(tasmin, freq=freq),
        "fd": xi.frost_days(tasmin, thresh="0 degC", freq=freq),
    }
    if tn10_per is not None:
        out["tn10p"] = xi.tn10p(tasmin, tasmin_per=tn10_per, freq=freq)
    if tasmax is not None:
        tasmax = _unidades(tasmax, "tasmax")
        out["txx"] = xi.tx_max(tasmax, freq=freq)
        out["dtr"] = xi.daily_temperature_range(tasmin, tasmax, freq=freq)
        if tx90_per is not None:
            out["tx90p"] = xi.tx90p(tasmax, tasmax_per=tx90_per, freq=freq)
    return xr.Dataset(out)
