"""Carga de la configuración central del proyecto (configs/dominio.yml)."""

from pathlib import Path

import yaml

# Raíz del repositorio: dos niveles arriba de este archivo (src/cmip6cusco/)
RAIZ_REPO = Path(__file__).resolve().parents[2]
RUTA_CONFIG = RAIZ_REPO / "configs" / "dominio.yml"


def cargar_config(ruta: Path | str = RUTA_CONFIG) -> dict:
    """Lee el YAML de configuración y lo devuelve como diccionario.

    Todos los scripts del proyecto obtienen dominio, periodos, GCMs y
    escenarios de aquí — nunca hardcodeados.
    """
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def bbox(config: dict) -> tuple[float, float, float, float]:
    """Devuelve el dominio como (lon_min, lat_min, lon_max, lat_max)."""
    d = config["dominio"]
    return (d["lon_min"], d["lat_min"], d["lon_max"], d["lat_max"])
