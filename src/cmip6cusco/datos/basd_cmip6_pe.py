"""Descarga y recorte de BASD-CMIP6-PE, el producto con el que se compara el QDM.

BASD-CMIP6-PE (Fernandez-Palomino et al. 2024, DOI 10.5880/pik.2023.001) es
el CMIP6 corregido y downscaleado a 0.1° para Perú/Ecuador con ISIMIP3BASD.

Estructura del servidor (julio de 2026):
  https://www.pik-potsdam.de/data/doi/10.5880/PIK.2023.001/BASD-CMIP6-PE/
      daily/{experimento}/{GCM}/
          {gcm_min}_{miembro}_{exp}_{var}_daily_{y1}_{y2}.nc   (~240 MB / 10 años)

Como los archivos vienen por décadas, se procesa uno a la vez (descargar ~240 MB -> recortar a Cusco -> borrar) y al final se concatenan
los recortes. Pico de disco ~ 300 MB.

Uso:
    python -m cmip6cusco.datos.basd_cmip6_pe MPI-ESM1-2-HR historical
    python -m cmip6cusco.datos.basd_cmip6_pe todos historical
"""

import re
import sys

import requests
import xarray as xr

from cmip6cusco.config import RAIZ_REPO, cargar_config
from cmip6cusco.datos.observaciones import (
    _descargar_archivo,
    _exigir_disco_libre,
    _guardar_comprimido,
    _normalizar_coordenadas,
    _recortar_dominio,
)

URL_BASE = ("https://www.pik-potsdam.de/data/doi/10.5880/PIK.2023.001/"
            "BASD-CMIP6-PE/daily")
RUTA_RAW = RAIZ_REPO / "data" / "raw"
RUTA_SALIDA = RAIZ_REPO / "data" / "interim" / "basd_cmip6_pe"

# años necesarios por experimento (mismos periodos que cmip6_diario)
PERIODOS = {"historical": (1981, 2014), "ssp126": (2015, 2100),
            "ssp370": (2015, 2100), "ssp585": (2015, 2100)}


def _listar_archivos(gcm: str, experimento: str) -> list[str]:
    """Nombres .nc del directorio del GCM/experimento en el servidor del PIK."""
    url = f"{URL_BASE}/{experimento}/{gcm}/"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return sorted(set(re.findall(r'href="([^"?]+\.nc)"', r.text)))


def _tramo_util(nombre: str, y_ini: int, y_fin: int) -> bool:
    """¿El tramo decadal del archivo (..._y1_y2.nc) solapa el periodo útil?"""
    m = re.search(r"_(\d{4})_(\d{4})\.nc$", nombre)
    if not m:
        return False
    y1, y2 = int(m.group(1)), int(m.group(2))
    return y2 >= y_ini and y1 <= y_fin


def descargar_basd(gcm: str, experimento: str, variables=None):
    config = cargar_config()
    variables = variables or config["variables"]
    y_ini, y_fin = PERIODOS[experimento]
    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

    archivos = _listar_archivos(gcm, experimento)
    for var in variables:
        destino = RUTA_SALIDA / f"{gcm}_{var}_{experimento}_cusco.nc"
        if destino.exists():
            print(f"Ya existe {destino.name}; nada que hacer.")
            continue

        tramos = [a for a in archivos
                  if f"_{var}_daily_" in a and _tramo_util(a, y_ini, y_fin)]
        if not tramos:
            print(f"OMITIDO {gcm}/{experimento}/{var}: sin archivos en el índice")
            continue
        print(f"{gcm} {var} {experimento}: {len(tramos)} tramos decadales")

        recortes = []
        for nombre in tramos:
            _exigir_disco_libre(1.0)
            ruta_raw = RUTA_RAW / nombre
            _descargar_archivo(f"{URL_BASE}/{experimento}/{gcm}/{nombre}",
                               ruta_raw, 0.25)
            with xr.open_dataset(ruta_raw) as ds:
                ds = _normalizar_coordenadas(ds)
                recortes.append(_recortar_dominio(ds, config)
                                .sel(time=slice(str(y_ini), str(y_fin)))
                                .compute())
            ruta_raw.unlink()  # borrar el nacional apenas recortado

        junto = xr.concat(recortes, dim="time").sortby("time")
        _guardar_comprimido(junto, destino)
        print(f"   Guardado {destino.name} {dict(junto.sizes)}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("Uso: python -m cmip6cusco.datos.basd_cmip6_pe "
                         "<GCM|todos> <experimento>")
    objetivo, experimento = sys.argv[1], sys.argv[2]
    gcms = cargar_config()["gcms"] if objetivo == "todos" else [objetivo]
    for gcm in gcms:
        descargar_basd(gcm, experimento)
