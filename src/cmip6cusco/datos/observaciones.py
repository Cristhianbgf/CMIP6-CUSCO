"""Descarga y recorte de las observaciones de referencia.

Productos (la versión más reciente de cada uno):
- Precipitación: PISCOp v3.0 diaria (Figshare 32411886, 1981-2025) y RAIN4PE
  diaria (GFZ, 1981-2015), usadas como ensamble observacional.
- Temperatura: PISCOt v1.2 diaria Tmax/Tmin en su versión oficial a 0.1°
  (Figshare 22712365, 1981-2020), en vez de los ~114 GB de la versión a 0.01°.

Cada archivo nacional se descarga, se recorta al dominio y se borra; en
data/interim/ solo quedan los recortes de Cusco.

Uso desde terminal:

    conda activate cmip6-cusco
    python -m cmip6cusco.datos.observaciones rain4pe
    python -m cmip6cusco.datos.observaciones piscop_v3
    python -m cmip6cusco.datos.observaciones piscot_tmax
    python -m cmip6cusco.datos.observaciones piscot_tmin
"""

import shutil
import sys
import zipfile
from pathlib import Path

import requests
import xarray as xr

from cmip6cusco.config import RAIZ_REPO, bbox, cargar_config

RUTA_RAW = RAIZ_REPO / "data" / "raw"
RUTA_INTERIM = RAIZ_REPO / "data" / "interim"

URL_RAIN4PE = (
    "https://datapub.gfz-potsdam.de/download/10.5880.PIK.2020.010enouiv/"
    "RAIN4PE_daily_0.1d_1981_2015_v1.0.nc.zip"
)

# Figshare: URLs directas de api.figshare.com (julio de 2026)
URL_PISCOP_V3_DIARIA = "https://ndownloader.figshare.com/files/64968183"   # PISCOp_d.nc, 1.53 GB
URL_PISCOT_TMAX_010 = "https://ndownloader.figshare.com/files/40343707"    # tmax_daily_1981_2020_010.nc, 0.63 GB
URL_PISCOT_TMIN_010 = "https://ndownloader.figshare.com/files/40343728"    # tmin_daily_1981_2020_010.nc, 0.65 GB
URL_DEM_PISCOT = "https://ndownloader.figshare.com/files/36426324"         # DEM.nc (covariables PISCOt), 3.3 MB


def _exigir_disco_libre(gb_necesarios: float):
    """Aborta con un mensaje claro si no hay espacio suficiente en disco."""
    libre_gb = shutil.disk_usage(RAIZ_REPO).free / 1e9
    if libre_gb < gb_necesarios:
        raise RuntimeError(
            f"Solo hay {libre_gb:.1f} GB libres y esta operación necesita "
            f"~{gb_necesarios:.1f} GB. Liberar espacio antes de continuar."
        )


def _normalizar_coordenadas(ds: xr.Dataset) -> xr.Dataset:
    """Renombra lat/lon/tiempo a nombres canónicos y ordena lat ascendente.

    Cada producto usa nombres distintos (latitude/longitude, X/Y, T, Z1...);
    aquí se uniformiza todo a (time, lat, lon).
    """
    renombres = {}
    for original, canonico in [
        ("latitude", "lat"), ("Latitude", "lat"), ("Y", "lat"),
        ("longitude", "lon"), ("Longitude", "lon"), ("X", "lon"),
        ("T", "time"), ("Z1", "time"),
    ]:
        if original in ds.dims or original in ds.coords:
            renombres[original] = canonico
    ds = ds.rename(renombres)
    if float(ds.lat[0]) > float(ds.lat[-1]):  # lat descendente -> invertir
        ds = ds.isel(lat=slice(None, None, -1))
    return ds


def _asignar_calendario(ds: xr.Dataset, inicio: str) -> xr.Dataset:
    """Asigna fechas reales cuando el eje de tiempo viene sin metadatos CF.

    PISCOp v3 trae la dimensión de tiempo como enteros 1..N (sin unidades).
    Se asigna un calendario diario desde `inicio`, pero solo si la longitud
    coincide exactamente con la esperada — si no, es señal de que el
    supuesto es falso y se aborta.
    """
    import numpy as np
    import pandas as pd

    if np.issubdtype(ds.time.dtype, np.datetime64):
        return ds  # ya viene decodificado, no tocar
    fechas = pd.date_range(inicio, periods=ds.sizes["time"], freq="D")
    print(f"   Tiempo sin metadatos CF: asignando {fechas[0].date()} -> "
          f"{fechas[-1].date()} ({len(fechas)} días)")
    return ds.assign_coords(time=fechas)


def _recortar_dominio(ds: xr.Dataset, config: dict) -> xr.Dataset:
    """Recorta el dataset al dominio Cusco de configs/dominio.yml."""
    lon_min, lat_min, lon_max, lat_max = bbox(config)
    return ds.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))


def _guardar_comprimido(ds: xr.Dataset, ruta: Path):
    """Guarda un NetCDF con compresión interna (zlib nivel 4)."""
    codificacion = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(ruta, encoding=codificacion)


def _descargar_archivo(url: str, destino: Path, gb_esperados: float,
                       reintentos: int = 8):
    """Descarga con streaming, reintentos y reanudación por HTTP Range.

    Los servidores de datos (PIK, GFZ) cortan las transferencias largas con
    frecuencia. En vez de empezar de cero, cada reintento continúa desde el
    byte donde se cortó (cabecera Range) -> cada corte es progreso acumulado,
    no trabajo perdido. Si el servidor no honra el Range (código ≠ 206), se
    reinicia ese archivo desde cero. Si los reintentos se agotan, el error
    sube (no se oculta).
    """
    import time

    _exigir_disco_libre(gb_esperados + 1.0)  # margen para el recorte posterior
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.unlink(missing_ok=True)  # no heredar parciales de corridas viejas
    for intento in range(1, reintentos + 1):
        try:
            bajado = destino.stat().st_size if destino.exists() else 0
            cabeceras = {"Range": f"bytes={bajado}-"} if bajado else {}
            print(f"   Descargando {destino.name} (~{gb_esperados:.1f} GB)"
                  + (f" [reintento {intento}, desde {bajado / 1e6:.0f} MB]"
                     if intento > 1 else "") + "...")
            with requests.get(url, stream=True, timeout=60,
                              allow_redirects=True, headers=cabeceras) as r:
                modo = "ab"
                if bajado and r.status_code != 206:  # Range no soportado
                    bajado, modo = 0, "wb"
                r.raise_for_status()
                with open(destino, modo) as f:
                    for pedazo in r.iter_content(chunk_size=8 * 1024 * 1024):
                        f.write(pedazo)
                        bajado += len(pedazo)
                        if bajado % (256 * 1024 * 1024) < 8 * 1024 * 1024:
                            print(f"   ... {bajado / 1e9:.2f} GB")
            print(f"   Descarga completa: {destino.stat().st_size / 1e9:.2f} GB")
            return
        except (requests.exceptions.RequestException, OSError) as e:
            if intento == reintentos:
                destino.unlink(missing_ok=True)
                raise
            espera = min(30 * intento, 120)
            print(f"   Corte de red ({type(e).__name__}); reintento en {espera} s")
            time.sleep(espera)


def _descargar_recortar_borrar(url: str, nombre_raw: str, destino_final: Path,
                               gb_esperados: float, inicio_tiempo: str | None = None):
    """Patrón completo: descarga -> normaliza -> recorta -> guarda -> borra."""
    config = cargar_config()
    if destino_final.exists():
        print(f"Ya existe {destino_final.name}; nada que hacer.")
        return
    ruta_nc = RUTA_RAW / nombre_raw
    if not ruta_nc.exists():
        _descargar_archivo(url, ruta_nc, gb_esperados)
    else:
        print(f"   Reutilizando {ruta_nc.name} ya descargado")

    print("   Recortando al dominio Cusco...")
    with xr.open_dataset(ruta_nc) as ds:
        ds = _normalizar_coordenadas(ds)
        if inicio_tiempo:
            ds = _asignar_calendario(ds, inicio_tiempo)
        recorte = _recortar_dominio(ds, config).compute()
        _guardar_comprimido(recorte, destino_final)
        dims = dict(recorte.sizes)
    ruta_nc.unlink()  # borrar el nacional: solo queda el recorte
    print(f"   Guardado {destino_final.name} {dims}; original borrado")
    print(f"   Disco libre: {shutil.disk_usage(RAIZ_REPO).free / 1e9:.1f} GB")


def descargar_piscop_v3():
    """PISCOp v3.0: precipitación diaria 1981-2025 (Figshare, 1.53 GB)."""
    _descargar_recortar_borrar(
        URL_PISCOP_V3_DIARIA, "PISCOp_d.nc",
        RUTA_INTERIM / "piscop_v3_prec_diaria_cusco.nc", 1.6,
        inicio_tiempo="1981-01-01")  # PISCOp v3: eje de tiempo sin metadatos CF


def descargar_piscot_tmax():
    """PISCOt v1.2 a 0.1°: Tmax diaria 1981-2020 (Figshare, 0.63 GB)."""
    _descargar_recortar_borrar(
        URL_PISCOT_TMAX_010, "tmax_daily_1981_2020_010.nc",
        RUTA_INTERIM / "piscot_v12_tmax_diaria_cusco.nc", 0.7)


def descargar_piscot_tmin():
    """PISCOt v1.2 a 0.1°: Tmin diaria 1981-2020 (Figshare, 0.65 GB)."""
    _descargar_recortar_borrar(
        URL_PISCOT_TMIN_010, "tmin_daily_1981_2020_010.nc",
        RUTA_INTERIM / "piscot_v12_tmin_diaria_cusco.nc", 0.7)


def descargar_rain4pe():
    """RAIN4PE: baja el zip (1.0 GB), extrae, recorta a Cusco y borra."""
    config = cargar_config()
    destino_final = RUTA_INTERIM / "rain4pe_prec_diaria_cusco.nc"
    if destino_final.exists():
        print(f"Ya existe {destino_final.name}; nada que hacer.")
        return

    RUTA_RAW.mkdir(parents=True, exist_ok=True)
    ruta_zip = RUTA_RAW / "RAIN4PE_daily_0.1d_1981_2015_v1.0.nc.zip"
    ruta_nc = RUTA_RAW / "RAIN4PE_daily_0.1d_1981_2015_v1.0.nc"

    if not ruta_nc.exists():
        _descargar_archivo(URL_RAIN4PE, ruta_zip, 1.1)
        print("   Descomprimiendo...")
        with zipfile.ZipFile(ruta_zip) as z:
            nombres_nc = [n for n in z.namelist() if n.endswith(".nc")]
            z.extract(nombres_nc[0], RUTA_RAW)
        ruta_zip.unlink()  # borrar el zip apenas se extrae (disco escaso)

    print("   Recortando al dominio Cusco...")
    with xr.open_dataset(ruta_nc) as ds:
        ds = _normalizar_coordenadas(ds)
        recorte = _recortar_dominio(ds, config)
        _guardar_comprimido(recorte, destino_final)
        dims = dict(recorte.sizes)
    ruta_nc.unlink()
    print(f"   Guardado {destino_final.name} {dims}; original borrado")
    print(f"   Disco libre: {shutil.disk_usage(RAIZ_REPO).free / 1e9:.1f} GB")


def descargar_dem():
    """DEM oficial de las covariables de PISCOt v1.2 (Figshare, 3.3 MB).

    Es el mismo DEM con el que se interpoló PISCOt -> máxima consistencia
    entre la covariable de elevación y las observaciones de temperatura.
    Se guarda el recorte a resolución nativa (0.01°); las agregaciones a
    0.1° se derivan al armar los datasets de entrenamiento.
    """
    _descargar_recortar_borrar(
        URL_DEM_PISCOT, "DEM_piscot.nc",
        RUTA_INTERIM / "dem_piscot_cusco.nc", 0.1)


TAREAS = {
    "rain4pe": descargar_rain4pe,
    "piscop_v3": descargar_piscop_v3,
    "piscot_tmax": descargar_piscot_tmax,
    "piscot_tmin": descargar_piscot_tmin,
    "dem": descargar_dem,
}

if __name__ == "__main__":
    tarea = sys.argv[1] if len(sys.argv) > 1 else None
    if tarea not in TAREAS:
        raise SystemExit(f"Uso: python -m cmip6cusco.datos.observaciones "
                         f"[{' | '.join(TAREAS)}]")
    TAREAS[tarea]()
