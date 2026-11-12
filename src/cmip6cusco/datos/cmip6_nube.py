"""Acceso a CMIP6 en la nube (catálogo Pangeo en Google Cloud, formato zarr).

Los GCM no se descargan como archivos. El catálogo es una tabla
(un CSV gigante) donde cada fila es un dataset disponible (modelo × experimento
× variable × miembro) con la URL de su almacén zarr en Google Cloud. Con esa
URL, xarray abre el dataset "en streaming": solo baja los pedacitos (chunks)
que uno realmente usa — por ejemplo, el recorte del dominio Cusco.

Uso desde terminal (prueba de acceso):

    conda activate cmip6-cusco
    python -m cmip6cusco.datos.cmip6_nube
"""

import intake
import pandas as pd
import xarray as xr

from cmip6cusco.config import bbox, cargar_config

# Catálogo oficial Pangeo/ESGF en Google Cloud (ver docs/03_datos.md)
URL_CATALOGO = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"


def abrir_catalogo():
    """Abre el catálogo CMIP6 de Pangeo (descarga solo la tabla índice)."""
    return intake.open_esm_datastore(URL_CATALOGO)


def buscar(catalogo, config, table_id="day", experimentos=None):
    """Filtra el catálogo a los GCMs/variables del proyecto.

    table_id: "day" (diario) o "Amon" (mensual atmosférico).
    Por defecto busca 'historical' + los escenarios SSP de configs/dominio.yml.
    """
    experimentos = experimentos or (["historical"] + list(config["escenarios"]))
    return catalogo.search(
        source_id=config["gcms"],
        experiment_id=experimentos,
        variable_id=config["variables"],
        table_id=table_id,
    )


def buscar_diarios(catalogo, config):
    """Atajo: variables diarias del proyecto (ver `buscar`)."""
    return buscar(catalogo, config, table_id="day")


def tabla_disponibilidad(busqueda, config):
    """Matriz GCM × experimento: qué variables diarias existen en la nube.

    Cada celda dice cuántas de las 3 variables del proyecto (pr, tasmax,
    tasmin) están disponibles. Sirve para detectar de entrada qué GCM/SSP
    habría que buscar en ESGF como respaldo.
    """
    df = busqueda.df
    filas = []
    experimentos = ["historical"] + list(config["escenarios"])
    for gcm in config["gcms"]:
        fila = {"GCM": gcm}
        for exp in experimentos:
            variables = sorted(
                df[(df.source_id == gcm) & (df.experiment_id == exp)]
                .variable_id.unique()
            )
            n = len(variables)
            total = len(config["variables"])
            fila[exp] = f"{n}/{total}" if n < total else "3/3 ✓"
        filas.append(fila)
    return pd.DataFrame(filas).set_index("GCM")


def abrir_recorte(busqueda, config, gcm, experimento, variable):
    """Abre un dataset zarr en streaming y lo recorta al dominio Cusco.

    Prefiere el miembro r1i1p1f1 (o el primero alfabético si no existe,
    p. ej. UKESM usa f2). Devuelve el xarray.Dataset ya recortado.
    """
    df = busqueda.df
    df = df[
        (df.source_id == gcm)
        & (df.experiment_id == experimento)
        & (df.variable_id == variable)
    ].sort_values("member_id")
    if df.empty:
        raise ValueError(f"No hay {variable} de {gcm}/{experimento} en el catálogo")
    preferido = df[df.member_id == "r1i1p1f1"]
    zstore = (preferido if not preferido.empty else df).iloc[0]["zstore"]

    ds = xr.open_zarr(zstore, storage_options={"token": "anon"}, consolidated=True)

    # Recorte al dominio. Los GCM usan longitud 0–360, nuestro dominio
    # está en -180–180 -> se convierte (p. ej. -75° = 285°).
    lon_min, lat_min, lon_max, lat_max = bbox(config)
    if float(ds.lon.max()) > 180:
        lon_min, lon_max = lon_min % 360, lon_max % 360
    ds = ds.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
    ds.attrs["zstore_origen"] = zstore
    return ds


def _prueba_de_humo():
    """Prueba de acceso al catálogo y recorte del dominio Cusco."""
    config = cargar_config()

    print("1) Abriendo catálogo Pangeo/GCS...")
    catalogo = abrir_catalogo()
    busqueda = buscar_diarios(catalogo, config)
    print(f"   Datasets diarios encontrados: {len(busqueda.df)}\n")

    print("2) Disponibilidad (variables diarias pr/tasmax/tasmin por GCM):")
    tabla = tabla_disponibilidad(busqueda, config)
    print(tabla.to_string(), "\n")

    gcm, exp, var = "EC-Earth3", "historical", "pr"
    print(f"3) Abriendo {var} de {gcm}/{exp} en streaming y recortando el dominio...")
    ds = abrir_recorte(busqueda, config, gcm, exp, var)
    n_lat, n_lon = ds.sizes["lat"], ds.sizes["lon"]
    print(f"   Grilla del GCM en el dominio Cusco: {n_lat} lat × {n_lon} lon "
          f"({n_lat * n_lon} celdas)")

    # Un día concreto, promedio del dominio, convertido de kg m-2 s-1 a mm/día
    dia = ds.pr.sel(time="1990-01-15", method="nearest").mean().compute()
    print(f"   P media del dominio el 1990-01-15: {float(dia) * 86400:.2f} mm/día")
    print("\nAcceso correcto: catálogo, streaming y recorte.")


if __name__ == "__main__":
    _prueba_de_humo()
