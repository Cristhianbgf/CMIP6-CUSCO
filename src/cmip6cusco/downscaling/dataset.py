"""Dataset de entrenamiento para la U-Net y la difusión.

Marco MOS-SR: la red aprende la relación entre la escala gruesa y la fina
del clima observado. Para eso se
construyen pares perfectamente alineados en el tiempo:

    entrada  = obs agregada a ~1° (coarsen 10×) y re-interpolada a 0.1°
               (el "GCM idealizado": misma información que una celda gruesa)
    objetivo = obs original a 0.1°

más covariables estáticas (elevación media y rugosidad del DEM a 0.1°).
En aplicación, la entrada es el GCM corregido con QDM.

Salidas:
- data/interim/covariables_estaticas.nc  (dem_media, dem_std a 0.1°)
- data/processed/f3_pares_{var}.nc       (entrada, objetivo; time,lat,lon)

Los cortes temporales (entrenamiento/validación/prueba) no se guardan
en archivos separados: viven en configs/dominio.yml y se aplican al cargar.

Uso:  python -m cmip6cusco.downscaling.dataset
"""

import numpy as np
import xarray as xr

from cmip6cusco.config import RAIZ_REPO, cargar_config
from cmip6cusco.utils.grilla import alinear_a_grilla

RUTA_INTERIM = RAIZ_REPO / "data" / "interim"
RUTA_PROCESSED = RAIZ_REPO / "data" / "processed"

FACTOR = 10  # 0.1° × 10 = 1.0° (escala "GCM idealizado")

ARCHIVOS_OBS = {
    "pr": ("piscop_v3_prec_diaria_cusco.nc", "precipitation"),
    "tasmax": ("piscot_v12_tmax_diaria_cusco.nc", "tmax"),
    "tasmin": ("piscot_v12_tmin_diaria_cusco.nc", "tmin"),
}


def covariables_estaticas(config) -> xr.Dataset:
    """Elevación media y rugosidad (std) a 0.1° desde el DEM de 0.01°."""
    ruta = RUTA_INTERIM / "covariables_estaticas.nc"
    if ruta.exists():
        return xr.open_dataset(ruta)

    dem = xr.open_dataset(RUTA_INTERIM / "dem_piscot_cusco.nc")["DEM"]
    # 501 puntos a 0.01° -> recortar el punto sobrante para agrupar exacto 10×10
    dem = dem.isel(lat=slice(0, 500), lon=slice(0, 500))
    grueso = dem.coarsen(lat=FACTOR, lon=FACTOR)
    est = xr.Dataset({
        "dem_media": grueso.mean().astype("float32"),
        "dem_std": grueso.std().astype("float32"),
    })
    est = alinear_a_grilla(est, config)
    est.dem_media.attrs.update(units="m", descripcion="elevación media 0.1°")
    est.dem_std.attrs.update(units="m", descripcion="rugosidad (std intra-celda)")
    est.to_netcdf(ruta)
    print(f"Covariables estáticas guardadas en {ruta.name}")
    return est


def a_escala_gruesa(fino: xr.DataArray) -> xr.DataArray:
    """Agrega a ~1° (media de bloques FACTOR×FACTOR) y re-interpola a 0.1°.

    Es la entrada con la que se entrenó la U-Net: la misma malla fina, pero
    solo con la información de la escala gruesa. En aplicación, el QDM a 0.1°
    tiene que pasar por aquí antes de la red; si no, la red añade el detalle
    fino a un campo que ya lo tiene.
    Deja sin dato un borde de 0.5°: la interpolación no extrapola más allá del
    centro del bloque exterior.
    """
    grueso = fino.coarsen(lat=FACTOR, lon=FACTOR).mean()
    # interp devuelve float64; se vuelve al tipo de entrada para no duplicar RAM
    return grueso.interp(lat=fino.lat, lon=fino.lon, method="linear").astype(fino.dtype)


def construir_pares(var: str, config, mascara) -> None:
    """Pares (entrada gruesa re-interpolada, objetivo fino) para una variable."""
    RUTA_PROCESSED.mkdir(parents=True, exist_ok=True)
    ruta = RUTA_PROCESSED / f"f3_pares_{var}.nc"
    if ruta.exists():
        print(f"Ya existe {ruta.name}; nada que hacer.")
        return

    archivo, nombre = ARCHIVOS_OBS[var]
    fino = xr.open_dataset(RUTA_INTERIM / archivo)[nombre]
    fino = alinear_a_grilla(fino, config).where(mascara == 1)
    fino = fino.sel(time=slice("1981", "2014")).astype("float32")

    # agregar a ~1° (como lo vería un GCM) y volver a 0.1° por bilineal:
    # la entrada tiene la misma malla que el objetivo pero solo la
    # información de la escala gruesa — eso es lo que la red debe refinar
    entrada = a_escala_gruesa(fino)

    pares = xr.Dataset({"entrada": entrada.astype("float32"),
                        "objetivo": fino})
    pares.attrs.update(
        variable=var,
        factor_downscaling=FACTOR,
        entrenamiento=str(config["periodos"]["entrenamiento"]),
        validacion=str(config["periodos"]["validacion"]),
        prueba=str(config["periodos"]["prueba"]),
    )
    comp = {v: {"zlib": True, "complevel": 4} for v in pares.data_vars}
    pares.to_netcdf(ruta, encoding=comp)
    print(f"Guardado {ruta.name} {dict(pares.sizes)}")


if __name__ == "__main__":
    config = cargar_config()
    mascara = xr.open_dataarray(RUTA_INTERIM / "mascara_comun.nc")
    covariables_estaticas(config)
    for var in ARCHIVOS_OBS:
        construir_pares(var, config, mascara)
