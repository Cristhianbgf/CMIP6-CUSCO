"""Comparación del QDM de este repositorio con BASD-CMIP6-PE.

Evalúa el producto publicado (BASD-CMIP6-PE, recortes de datos/basd_cmip6_pe.py)
con las mismas métricas y periodo de validación (2005-2014) que usamos para
nuestro QDM, y las pone lado a lado leyendo los f2_qdm_validacion_{GCM}.csv.

Dos diferencias que favorecen a BASD:
1. BASD-CMIP6-PE fue calibrado por sus autores con 1981-2015 -> nuestro
   periodo de validación 2005-2014 está dentro de su calibración (in-sample);
   nuestro QDM se evalúa out-of-sample (calibración 1981-2004).
2. Su referencia de P es RAIN4PE; la nuestra es PISCOp v3. Parte de las
   diferencias refleja desacuerdo observacional (~6 % en el dominio), no
   calidad del método.

Uso:  python -m cmip6cusco.evaluacion.comparacion_basd
"""

import numpy as np
import pandas as pd
import xarray as xr

from cmip6cusco.ajuste_sesgo.qdm import (
    PERIODO_VAL,
    RUTA_INTERIM,
    RUTA_RESULTADOS,
    _cargar_obs,
    _metricas,
)
from cmip6cusco.config import cargar_config
from cmip6cusco.utils.grilla import alinear_a_grilla

RUTA_BASD = RUTA_INTERIM / "basd_cmip6_pe"


def _cargar_basd(gcm: str, var: str, config, mascara) -> xr.DataArray:
    ruta = RUTA_BASD / f"{gcm}_{var}_historical_cusco.nc"
    if not ruta.exists():
        raise FileNotFoundError(ruta.name)
    ds = xr.open_dataset(ruta)
    nombres = [v for v in ds.data_vars if v not in ("crs",) and "bnds" not in v]
    da = ds[nombres[0]]

    # unidades canónicas con verificación (no asumir a ciegas)
    unidades = str(da.attrs.get("units", "")).lower()
    if var == "pr" and ("kg" in unidades or float(da.isel(time=0).max()) < 0.01):
        da = da * 86400
    if var != "pr" and ("k" == unidades or float(da.isel(time=0).mean()) > 150):
        da = da - 273.15
    da = alinear_a_grilla(da, config).where(mascara == 1)
    return da.sel(time=PERIODO_VAL).astype("float32")


def comparar() -> pd.DataFrame:
    config = cargar_config()
    mascara = xr.open_dataarray(RUTA_INTERIM / "mascara_comun.nc")
    filas = []

    for gcm in config["gcms"]:
        ruta_qdm = RUTA_RESULTADOS / f"f2_qdm_validacion_{gcm}.csv"
        if not ruta_qdm.exists():
            print(f"OMITIDO {gcm}: sin validación QDM")
            continue
        t_qdm = pd.read_csv(ruta_qdm).set_index(["variable", "métrica"])

        for var in ["pr", "tasmax", "tasmin"]:
            try:
                basd = _cargar_basd(gcm, var, config, mascara)
            except FileNotFoundError as e:
                print(f"OMITIDO {gcm}/{var}: falta {e}")
                continue
            obs_val = _cargar_obs(var, config, mascara).sel(time=PERIODO_VAL)
            # calendario del BASD puede diferir de las obs: métricas agrupadas
            # (pooled) no exigen alineación día a día
            m_basd = _metricas(var, obs_val.values.ravel(), basd.values.ravel())
            for metrica, (v_obs, v_basd) in m_basd.items():
                try:
                    fila_qdm = t_qdm.loc[(var, metrica)]
                except KeyError:
                    continue
                filas.append({
                    "GCM": gcm, "variable": var, "métrica": metrica,
                    "obs": v_obs,
                    "GCM crudo": float(fila_qdm["GCM crudo"]),
                    "QDM (nuestro)": float(fila_qdm["GCM+QDM"]),
                    "BASD (publicado)": v_basd,
                })
            print(f"ok {gcm}/{var}")

    tabla = pd.DataFrame(filas)
    salida = RUTA_RESULTADOS / "f2_comparacion_qdm_vs_basd.csv"
    tabla.round(3).to_csv(salida, index=False)
    print(f"\nGuardado en {salida}\n")

    # resumen por métrica (media entre GCMs) para lectura rápida
    resumen = (tabla.groupby(["variable", "métrica"])
               [["obs", "GCM crudo", "QDM (nuestro)", "BASD (publicado)"]]
               .mean())
    print(resumen.round(2).to_string())
    return tabla


if __name__ == "__main__":
    comparar()
