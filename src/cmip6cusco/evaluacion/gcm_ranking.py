"""Evaluación histórica de los 10 GCM en el dominio Cusco.

Compara la climatología mensual 1981-2014 de cada GCM (mensuales Amon del
catálogo Pangeo) contra las observaciones (PISCOp v3 y RAIN4PE para P;
PISCOt v1.2 para Tmax/Tmin), como promedio espacial del dominio sobre la
máscara de cobertura común.

Métricas por GCM:
- P: correlación del ciclo anual (media de las 2 referencias) y sesgo anual %.
- T: sesgo medio de Tmax y Tmin (°C) y correlación del ciclo anual de Tmax.
- Puntaje compuesto: promedio de los rankings por métrica (menor = mejor).

Nota: a escala de GCM (~1-2°) el dominio son ~7×7 celdas; aquí
no se evalúa patrón espacial fino (eso es tarea del downscaling), solo ciclo
anual y sesgos de conjunto. La comparación interanual directa no procede
porque la variabilidad interna del GCM no está en fase con la realidad.

Produce:
- data/interim/cmip6_mensual/{gcm}_{var}_historical.nc (subconjuntos cacheados)
- resultados/f1_ranking_gcms.csv (tabla versionada en git)

Uso:  python -m cmip6cusco.evaluacion.gcm_ranking
"""

import numpy as np
import pandas as pd
import xarray as xr

from cmip6cusco.config import RAIZ_REPO, bbox, cargar_config
from cmip6cusco.datos.cmip6_nube import abrir_catalogo, buscar
from cmip6cusco.utils.grilla import alinear_a_grilla, grilla_canonica

RUTA_INTERIM = RAIZ_REPO / "data" / "interim"
RUTA_CACHE = RUTA_INTERIM / "cmip6_mensual"
RUTA_RESULTADOS = RAIZ_REPO / "resultados"

PERIODO = slice("1981", "2014")  # solape obs / historical CMIP6
MARGEN = 2.0  # grados extra al recortar el GCM (para interpolar sin bordes NaN)


# ----------------------------- observaciones -----------------------------

def climatologias_observadas(config) -> dict[str, xr.DataArray]:
    """Ciclo anual observado (12 valores, promedio del dominio enmascarado).

    P en mm/día; T en °C.
    """
    mascara = xr.open_dataarray(RUTA_INTERIM / "mascara_comun.nc")

    def clim(archivo: str, variable: str) -> xr.DataArray:
        da = xr.open_dataset(RUTA_INTERIM / archivo)[variable]
        da = alinear_a_grilla(da, config).where(mascara == 1)
        c = da.sel(time=PERIODO).groupby("time.month").mean("time")
        return c.mean(["lat", "lon"]).compute()

    return {
        "pr_piscop": clim("piscop_v3_prec_diaria_cusco.nc", "precipitation"),
        "pr_rain4pe": clim("rain4pe_prec_diaria_cusco.nc", "pcp"),
        "tasmax": clim("piscot_v12_tmax_diaria_cusco.nc", "tmax"),
        "tasmin": clim("piscot_v12_tmin_diaria_cusco.nc", "tmin"),
    }


# --------------------------------- GCMs ----------------------------------

def _subconjunto_gcm(busqueda, config, gcm: str, var: str) -> xr.DataArray:
    """Serie mensual 1981-2014 del GCM recortada al dominio (con margen).

    Se cachea en data/interim/cmip6_mensual/ — la primera corrida descarga
    de la nube; las siguientes leen el disco.
    """
    RUTA_CACHE.mkdir(parents=True, exist_ok=True)
    ruta = RUTA_CACHE / f"{gcm}_{var}_historical.nc"
    if ruta.exists():
        return xr.open_dataarray(ruta)

    df = busqueda.df
    df = df[(df.source_id == gcm) & (df.experiment_id == "historical")
            & (df.variable_id == var)].sort_values("member_id")
    if df.empty:
        raise ValueError(f"No hay {var} mensual de {gcm} en el catálogo")
    preferido = df[df.member_id == "r1i1p1f1"]
    fila = (preferido if not preferido.empty else df).iloc[0]

    ds = xr.open_zarr(fila["zstore"], storage_options={"token": "anon"},
                      consolidated=True)
    lon_min, lat_min, lon_max, lat_max = bbox(config)
    lon_min, lon_max = lon_min - MARGEN, lon_max + MARGEN
    lat_min, lat_max = lat_min - MARGEN, lat_max + MARGEN
    if float(ds.lon.max()) > 180:
        lon_min, lon_max = lon_min % 360, lon_max % 360
    ds = ds.sortby("lat").sel(lon=slice(lon_min, lon_max),
                              lat=slice(lat_min, lat_max))
    da = ds[var].sel(time=PERIODO).compute()

    # unidades canónicas: pr kg m-2 s-1 -> mm/día; tas* K -> °C
    da = da * 86400 if var == "pr" else da - 273.15
    # longitudes de vuelta a -180..180 para que casen con la grilla canónica
    da = da.assign_coords(lon=(da.lon + 180) % 360 - 180).sortby("lon")
    da.attrs["member_id"] = fila["member_id"]

    da.name = var
    comp = {da.name: {"zlib": True, "complevel": 4}}
    da.to_netcdf(ruta, encoding=comp)
    return da


def _clim_dominio(da: xr.DataArray, config, mascara: xr.DataArray) -> xr.DataArray:
    """Climatología mensual del GCM interpolada a la grilla canónica y
    promediada sobre la máscara común (mismo muestreo que las obs)."""
    clim = da.groupby("time.month").mean("time")
    lats, lons = grilla_canonica(config)
    clim = clim.interp(lat=lats, lon=lons, method="linear")
    return clim.where(mascara == 1).mean(["lat", "lon"])


# -------------------------------- ranking --------------------------------

def evaluar_gcms() -> pd.DataFrame:
    config = cargar_config()
    mascara = xr.open_dataarray(RUTA_INTERIM / "mascara_comun.nc")

    print("Climatologías observadas (1981-2014, dominio enmascarado)...")
    obs = climatologias_observadas(config)
    obs_p = (obs["pr_piscop"] + obs["pr_rain4pe"]) / 2  # ensamble observacional

    print("Abriendo catálogo Pangeo (mensuales Amon)...")
    catalogo = abrir_catalogo()
    busqueda = buscar(catalogo, config, table_id="Amon",
                      experimentos=["historical"])

    filas = []
    for gcm in config["gcms"]:
        print(f"{gcm}...")
        try:
            clim = {v: _clim_dominio(_subconjunto_gcm(busqueda, config, gcm, v),
                                     config, mascara)
                    for v in ["pr", "tasmax", "tasmin"]}
        except ValueError as e:
            print(f"   OMITIDO: {e}")
            continue
        r_p = float(np.corrcoef(clim["pr"], obs_p)[0, 1])
        sesgo_p = float((clim["pr"].mean() - obs_p.mean()) / obs_p.mean() * 100)
        sesgo_tmax = float(clim["tasmax"].mean() - obs["tasmax"].mean())
        sesgo_tmin = float(clim["tasmin"].mean() - obs["tasmin"].mean())
        r_t = float(np.corrcoef(clim["tasmax"], obs["tasmax"])[0, 1])
        filas.append({"GCM": gcm, "r_ciclo_P": r_p, "sesgo_P_pct": sesgo_p,
                      "sesgo_Tmax_C": sesgo_tmax, "sesgo_Tmin_C": sesgo_tmin,
                      "r_ciclo_Tmax": r_t})

    tabla = pd.DataFrame(filas).set_index("GCM")
    # puntaje = promedio de rankings por métrica (1 = mejor en esa métrica)
    rangos = pd.DataFrame({
        "r_ciclo_P": (-tabla.r_ciclo_P).rank(),
        "sesgo_P": tabla.sesgo_P_pct.abs().rank(),
        "sesgo_Tmax": tabla.sesgo_Tmax_C.abs().rank(),
        "sesgo_Tmin": tabla.sesgo_Tmin_C.abs().rank(),
        "r_ciclo_Tmax": (-tabla.r_ciclo_Tmax).rank(),
    })
    tabla["puntaje"] = rangos.mean(axis=1)
    tabla = tabla.sort_values("puntaje")

    RUTA_RESULTADOS.mkdir(exist_ok=True)
    salida = RUTA_RESULTADOS / "f1_ranking_gcms.csv"
    tabla.round(3).to_csv(salida)
    print(f"\nRanking guardado en {salida}\n")
    print(tabla.round(2).to_string())
    return tabla


if __name__ == "__main__":
    evaluar_gcms()
