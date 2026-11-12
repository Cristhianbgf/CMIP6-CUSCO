"""Archivos de resultados para usuarios.

1. NetCDF por celda (10 km) con el cambio de los 14 índices ETCCDI, por
   escenario y horizonte: percentil 10, mediana y percentil 90 de los 10 GCM,
   más el valor observado 1985-2014. Van en resultados/datos/ (no se versionan:
   se publican como archivos de un release).
2. Tabla del departamento del Cusco: cambio promedio sobre las celdas del
   departamento, por GCM y resumido (mediana, P10, P90).

Requiere los Δ por celda en caché (python -m figuras.f5_mapas_cambio deltas 2050|2100).

Uso:  python scripts/exportar_productos.py
"""
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from shapely.ops import unary_union

from cmip6cusco.config import RAIZ_REPO, cargar_config
from figuras.f5_mapas_cambio import _contorno_cusco, deltas

RES = RAIZ_REPO / "resultados"
DATOS = RES / "datos"

# índice: (nombre ETCCDI en español, unidad, ¿se resume también en %?)
INDICES = {
    "txx": ("Temperatura máxima extrema", "°C", False),
    "tnn": ("Temperatura mínima extrema", "°C", False),
    "tx90p": ("Días cálidos (Tmax sobre el percentil 90 de 1985-2014)", "días/año", False),
    "tn10p": ("Noches frías (Tmin bajo el percentil 10 de 1985-2014)", "días/año", False),
    "fd": ("Días con heladas", "días/año", False),
    "dtr": ("Rango diurno de temperatura", "°C", False),
    "prcptot": ("Precipitación total anual en días húmedos", "mm/año", True),
    "rx1day": ("Precipitación máxima en 1 día", "mm", True),
    "rx5day": ("Precipitación máxima en 5 días consecutivos", "mm", True),
    "r95p": ("Precipitación en días muy húmedos", "mm/año", True),
    "r99p": ("Precipitación en días extremadamente húmedos", "mm/año", True),
    "sdii": ("Índice simple de intensidad diaria", "mm/día", True),
    "cdd": ("Días secos consecutivos", "días", False),
    "cwd": ("Días húmedos consecutivos", "días", False),
}
PERIODOS = {"2050": "2035-2064", "2100": "2071-2100"}


def _plano(da):
    """tx90p/tn10p arrastran la dimensión 'percentiles' (un valor válido)."""
    return da.mean("percentiles") if "percentiles" in da.dims else da


def _indice(d, idx, m):
    """Índice de un GCM; NaN si falta (UKESM1-0-LL no tiene tasmax)."""
    if idx in d:
        da = _plano(d[idx])
    else:
        da = xr.full_like(m, np.nan, dtype="float64")
    # coordenadas sobrantes (p. ej. 'height' de tasmax) estorban al apilar
    return da.drop_vars([c for c in da.coords if c not in ("lat", "lon")])


def mascara_departamento(m):
    cusco = unary_union(_contorno_cusco())
    lon2, lat2 = np.meshgrid(m.lon.values, m.lat.values)
    dentro = shapely.contains_xy(cusco, lon2, lat2)
    return xr.DataArray(dentro, coords={"lat": m.lat, "lon": m.lon},
                        dims=("lat", "lon")) & (m == 1)


def exportar(ventana, cfg):
    m, base_idx, acum = deltas(ventana)
    dep = mascara_departamento(m)
    print(f"{ventana}: {int(dep.sum())} celdas en el departamento del Cusco")

    # --- NetCDF por celda ---
    vars_ = {}
    for idx, (nombre, uni, _) in INDICES.items():
        por_ssp = []
        for ssp in cfg["escenarios"]:
            pila = xr.concat([_indice(d, idx, m) for _, d in acum[ssp]], dim="gcm")
            q = pila.quantile([0.1, 0.5, 0.9], dim="gcm", skipna=True)
            por_ssp.append(q.assign_coords(quantile=["p10", "mediana", "p90"]))
        da = xr.concat(por_ssp, dim=pd.Index(cfg["escenarios"], name="escenario"))
        da = da.rename(quantile="estadistico").where(m == 1).astype("float32")
        da.attrs = {"long_name": f"Cambio de {nombre} ({idx.upper()}), "
                    f"{PERIODOS[ventana]} frente a 1985-2014", "units": uni}
        vars_[idx] = da
        ob = _plano(base_idx[idx]).where(m == 1).astype("float32")
        ob.attrs = {"long_name": f"{nombre}, observado 1985-2014 (PISCO)", "units": uni}
        vars_[f"{idx}_obs"] = ob
    ds = xr.Dataset(vars_)
    ds.attrs = {
        "title": f"CMIP6-CUSCO: cambio de índices ETCCDI, {PERIODOS[ventana]}",
        "source": "10 GCM CMIP6 corregidos con QDM a 0,1° contra PISCOp v3 y PISCOt v1.2 (SENAMHI)",
        "estadistico": "p10, mediana y p90 entre los 10 GCM (9 en índices de Tmax)",
        "references": "https://github.com/Cristhianbgf/CMIP6-CUSCO",
        "license": "CC BY 4.0 (citar también PISCO, SENAMHI)",
    }
    DATOS.mkdir(parents=True, exist_ok=True)
    salida = DATOS / f"cmip6_cusco_cambio_etccdi_{ventana}.nc"
    ds.to_netcdf(salida, encoding={v: {"zlib": True, "complevel": 4} for v in ds})
    print(f"  -> {salida.relative_to(RAIZ_REPO)}")

    # --- departamento: promedio de las celdas por GCM ---
    filas = []
    for idx, (_, _, en_pct) in INDICES.items():
        base = float(_plano(base_idx[idx]).where(dep).mean())
        for ssp in cfg["escenarios"]:
            for gcm, d in acum[ssp]:
                cambio = float(_indice(d, idx, m).where(dep).mean())
                filas.append({"ventana": ventana, "ssp": ssp, "gcm": gcm, "indice": idx,
                              "obs_1985_2014": base, "cambio": cambio,
                              "cambio_pct": 100 * cambio / base if en_pct else np.nan})
    return filas


def main():
    cfg = cargar_config()
    filas = []
    for v in ("2050", "2100"):
        filas += exportar(v, cfg)
    por_gcm = pd.DataFrame(filas)
    por_gcm.round(3).to_csv(RES / "f5_departamento_cusco_por_gcm.csv", index=False)

    def resumen(g):
        out = {"obs_1985_2014": g.obs_1985_2014.iloc[0],
               "n_gcm": int(g.cambio.notna().sum())}
        for col in ("cambio", "cambio_pct"):
            x = g[col].dropna()
            out[f"{col}_mediana"] = x.median() if len(x) else np.nan
            out[f"{col}_p10"] = x.quantile(0.1) if len(x) else np.nan
            out[f"{col}_p90"] = x.quantile(0.9) if len(x) else np.nan
        return pd.Series(out)

    res = por_gcm.groupby(["ventana", "ssp", "indice"]).apply(resumen).reset_index()
    res.round(3).to_csv(RES / "f5_departamento_cusco.csv", index=False)
    print("Guardado resultados/f5_departamento_cusco.csv")


if __name__ == "__main__":
    main()
