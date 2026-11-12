"""Máscara de cobertura común de las observaciones en el dominio.

Problema: el dominio (11-16°S, 70-75°W) incluye una franja boliviana donde
RAIN4PE y PISCO no tienen datos (NaN). Toda evaluación se hace solo
sobre las celdas donde todas las referencias observacionales tienen datos;
si no, las sumas con NaN producen ceros falsos (visto en la verificación de
RAIN4PE: celdas con "0 mm/año").

Produce: data/interim/mascara_comun.nc (1 = celda válida, 0 = sin cobertura).

Uso:  python -m cmip6cusco.datos.mascara
"""

import numpy as np
import xarray as xr

from cmip6cusco.config import RAIZ_REPO
from cmip6cusco.utils.grilla import alinear_a_grilla

RUTA_INTERIM = RAIZ_REPO / "data" / "interim"


def crear_mascara_comun() -> xr.DataArray:
    """Intersección de la cobertura de PISCOp v3, RAIN4PE y PISCOt.

    Los tres productos se ajustan primero a la grilla canónica (sus centros
    de celda no coinciden entre sí; ver utils/grilla.py).
    """
    piscop = alinear_a_grilla(xr.open_dataset(RUTA_INTERIM / "piscop_v3_prec_diaria_cusco.nc"))
    rain = alinear_a_grilla(xr.open_dataset(RUTA_INTERIM / "rain4pe_prec_diaria_cusco.nc"))
    tmax = alinear_a_grilla(xr.open_dataset(RUTA_INTERIM / "piscot_v12_tmax_diaria_cusco.nc"))

    # Una celda es válida si tiene datos finitos el primer día Y una
    # climatología no degenerada (P anual > 0 descarta el relleno-cero).
    p3_anual = piscop["precipitation"].sel(time=slice("1981", "2010")).sum("time")
    ra_anual = rain["pcp"].sum("time")
    valida = (
        np.isfinite(piscop["precipitation"].isel(time=0))
        & np.isfinite(rain["pcp"].isel(time=0))
        & np.isfinite(tmax[list(tmax.data_vars)[0]].isel(time=0))
        & (p3_anual > 0)
        & (ra_anual > 0)
    )
    mascara = valida.astype("uint8").rename("mascara")
    mascara.attrs["descripcion"] = (
        "1 = celda con cobertura en PISCOp v3, RAIN4PE y PISCOt v1.2; "
        "0 = sin datos (p. ej. franja boliviana)")
    mascara = mascara.drop_vars([c for c in mascara.coords if c not in ("lat", "lon")])
    # limpiar codificaciones heredadas de los archivos fuente (rompen el guardado)
    mascara.encoding = {}
    for coord in mascara.coords.values():
        coord.encoding = {}
    return mascara


if __name__ == "__main__":
    mascara = crear_mascara_comun()
    ruta = RUTA_INTERIM / "mascara_comun.nc"
    mascara.to_netcdf(ruta)
    n_validas = int(mascara.sum())
    n_total = mascara.size
    print(f"Máscara guardada en {ruta}")
    print(f"Celdas válidas: {n_validas}/{n_total} ({100*n_validas/n_total:.0f} %)")
