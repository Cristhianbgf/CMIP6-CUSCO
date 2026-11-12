"""Grilla canónica del proyecto y alineación de productos a ella.

Motivo: los productos 0.1° no comparten centros
de celda — PISCOp v3 y RAIN4PE usan centros en x.x5 (con ruido de punto
flotante entre sí) y PISCOt v1.2 viene desfasada hasta 0.025° (~2.8 km).
Con xarray, cualquier operación entre ellos alinea por igualdad exacta de
coordenadas -> intersección vacía y resultados degenerados silenciosos.

Por eso toda serie del proyecto se ajusta a la grilla canónica (la de
PISCOp/RAIN4PE, derivada del dominio de configs/dominio.yml) con
`alinear_a_grilla`. Para desfases pequeños (≤ tolerancia) es un simple
re-etiquetado por vecino más cercano: los valores no se alteran.
"""

import numpy as np
import xarray as xr

from cmip6cusco.config import bbox, cargar_config


def grilla_canonica(config: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Centros de celda canónicos (lat, lon) del dominio a la resolución objetivo.

    Ej.: dominio 11-16°S / 70-75°W a 0.1° -> lat -15.95..-11.05, lon -74.95..-70.05.
    """
    config = config or cargar_config()
    res = config["dominio"]["resolucion_objetivo"]
    lon_min, lat_min, lon_max, lat_max = bbox(config)
    lats = np.round(np.arange(lat_min + res / 2, lat_max, res), 3)
    lons = np.round(np.arange(lon_min + res / 2, lon_max, res), 3)
    return lats, lons


def alinear_a_grilla(ds: xr.Dataset | xr.DataArray,
                     config: dict | None = None,
                     tolerancia: float = 0.03) -> xr.Dataset | xr.DataArray:
    """Ajusta un dataset 0.1° a la grilla canónica por vecino más cercano.

    - Desfase < ruido numérico (PISCOp, RAIN4PE): re-etiquetado exacto.
    - Desfase real ≤ tolerancia (PISCOt, 0.025°): re-etiquetado nearest;
      los valores no se interpolan (se documenta el corrimiento ≤ 2.8 km).
    - Desfase > tolerancia: celdas quedan NaN -> señal de que el producto
      no está en la grilla 0.1° y necesita regrillado de verdad (xarray-regrid).
    """
    lats, lons = grilla_canonica(config)
    return ds.reindex(lat=lats, lon=lons, method="nearest", tolerance=tolerancia)
