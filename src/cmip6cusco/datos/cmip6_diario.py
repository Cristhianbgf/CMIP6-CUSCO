"""Series diarias de CMIP6 recortadas al dominio.

Igual que los mensuales del ranking, pero en paso diario: se descarga de la
nube el recorte (con margen) y se cachea en data/interim/cmip6_diario/.
Cada archivo (GCM × variable × experimento) queda de unos pocos MB; el costo
es solo la transferencia de la primera vez.

En Windows conviene exportar GRPC_DNS_RESOLVER=native antes de correr (ver docs/data.md).

Uso:
    python -m cmip6cusco.datos.cmip6_diario MPI-ESM1-2-HR historical
    python -m cmip6cusco.datos.cmip6_diario EC-Earth3 historical ssp370
"""

import sys

import xarray as xr

from cmip6cusco.config import RAIZ_REPO, bbox, cargar_config
from cmip6cusco.datos.cmip6_nube import abrir_catalogo, buscar

RUTA_CACHE = RAIZ_REPO / "data" / "interim" / "cmip6_diario"
# Margen (grados) alrededor del dominio para que la interpolación bilineal a
# 0.1° cubra los bordes sin NaN. Debe superar el espaciado nativo del GCM más
# grueso (CanESM5, 2.8°): con 2.0 los puntos nativos que "envuelven" el dominio
# quedaban fuera del recorte y el sur/este salían NaN (459 celdas). 4.0 cubre
# hasta CanESM5. Los GCM finos no se afectan (solo bajan unas celdas nativas más).
MARGEN = 4.0

# periodo útil por experimento (no bajar 1850-1980 que no se usa)
PERIODOS = {
    "historical": slice("1981", "2014"),
    "ssp126": slice("2015", "2100"),
    "ssp370": slice("2015", "2100"),
    "ssp585": slice("2015", "2100"),
}


def subconjunto_diario(busqueda, config, gcm: str, var: str,
                       experimento: str) -> xr.DataArray:
    """Serie diaria del GCM recortada al dominio (cacheada en disco)."""
    RUTA_CACHE.mkdir(parents=True, exist_ok=True)
    ruta = RUTA_CACHE / f"{gcm}_{var}_{experimento}.nc"
    if ruta.exists():
        return xr.open_dataarray(ruta)

    df = busqueda.df
    df = df[(df.source_id == gcm) & (df.experiment_id == experimento)
            & (df.variable_id == var)].sort_values("member_id")
    if df.empty:
        raise ValueError(f"No hay {var} diaria de {gcm}/{experimento} en el catálogo")
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
    da = ds[var].sel(time=PERIODOS[experimento]).compute()

    # unidades canónicas: pr kg m-2 s-1 -> mm/día; tas* K -> °C
    if var == "pr":
        da = da * 86400
        da.attrs["units"] = "mm/d"
    else:
        da = da - 273.15
        da.attrs["units"] = "degC"
    da = da.assign_coords(lon=(da.lon + 180) % 360 - 180).sortby("lon")
    da.attrs["member_id"] = fila["member_id"]
    da.attrs["calendario"] = str(getattr(da.time.dt, "calendar", "desconocido"))

    da.name = var
    da.to_netcdf(ruta, encoding={da.name: {"zlib": True, "complevel": 4}})
    return da


def descargar_gcm(gcm: str, experimentos: list[str]):
    config = cargar_config()
    catalogo = abrir_catalogo()
    busqueda = buscar(catalogo, config, table_id="day",
                      experimentos=experimentos)
    for exp in experimentos:
        for var in config["variables"]:
            print(f"{gcm} {var} {exp}...")
            try:
                da = subconjunto_diario(busqueda, config, gcm, var, exp)
                print(f"   OK: {da.sizes} [{da.attrs.get('member_id', '?')}]")
            except ValueError as e:
                print(f"   OMITIDO: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Uso: python -m cmip6cusco.datos.cmip6_diario "
                         "<GCM|todos> [experimentos...]")
    objetivo = sys.argv[1]
    experimentos = sys.argv[2:] or ["historical"]
    gcms = cargar_config()["gcms"] if objetivo == "todos" else [objetivo]
    for gcm in gcms:
        descargar_gcm(gcm, experimentos)
