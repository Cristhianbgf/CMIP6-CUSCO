"""Mapas del cambio de los índices ETCCDI por celda.

Calcula el cambio por celda de índices ETCCDI clave: mediana inter-GCM de
Δ = índice(QDM-futuro 2071-2100) − índice(obs 1985-2014), y lo dibuja sobre
los Andes de Cusco con cartopy + cmcrameri (Scientific Colour Maps).

Baseline = obs (el QDM-histórico ~ obs por construcción, enfoque QDM clásico).
El cómputo por celda se cachea en NetCDF; la figura se re-dibuja al instante.

Uso:
    python -m figuras.f5_mapas_cambio calcular   # ~15 min, escribe el NetCDF
    python -m figuras.f5_mapas_cambio dibujar     # rápido, desde el NetCDF
"""
import gc
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from cmip6cusco.config import RAIZ_REPO, cargar_config
from cmip6cusco.indices import etccdi
from cmip6cusco.utils.grilla import alinear_a_grilla

INT = RAIZ_REPO / "data" / "interim"
QDM = INT / "cmip6_qdm"
SAL = RAIZ_REPO / "resultados" / "figuras"
NC = RAIZ_REPO / "resultados" / "f5_mapas_cambio_2100.nc"
CACHE = INT / "f5_mapas_cache"

REF = ("1985", "2014")
VENTANA = ("2071", "2100")
# índices a mapear: (clave, familia). Elegidos para que los lea un no especialista.
CLAVES = [("txx", "T"), ("fd", "T"), ("prcptot", "P"), ("rx1day", "P")]
# índices de lluvia que se muestran en % de su valor observado
EN_PCT = ("prcptot", "rx1day")
OBS = {"pr": ("piscop_v3_prec_diaria_cusco.nc", "precipitation"),
       "tasmax": ("piscot_v12_tmax_diaria_cusco.nc", "tmax"),
       "tasmin": ("piscot_v12_tmin_diaria_cusco.nc", "tmin")}


def _obs(var, cfg, m, anios):
    a, n = OBS[var]
    da = alinear_a_grilla(xr.open_dataset(INT / a)[n], cfg).where(m == 1)
    return da.sel(time=slice(*anios)).astype("float32")


def _indices_celda(pr, tx, tn, base) -> xr.Dataset:
    """Índices ETCCDI por celda (climatología = media sobre años)."""
    ds = xr.merge([etccdi.indices_precip(pr, base["per95"], base["per99"]),
                   etccdi.indices_temp(tx, tn, base["tx90"], base["tn10"])])
    return ds.mean("time")


def _contorno_cusco():
    """Geometría del límite del departamento de Cusco (Natural Earth admin-1,
    10m). La descarga se cachea en la caché de cartopy (~/.local/share/cartopy)
    tras la primera vez, así que las re-corridas son offline."""
    from cartopy.io import shapereader
    shp = shapereader.natural_earth(resolution="10m", category="cultural",
                                    name="admin_1_states_provinces")
    return [r.geometry for r in shapereader.Reader(shp).records()
            if r.attributes.get("name") == "Cusco"]


VENTANAS = {"2050": ("2035", "2064"), "2100": ("2071", "2100")}
BASE_OBS = INT / "f5_base_obs.nc"


def _dir_cache(ventana):
    return CACHE if ventana == "2100" else INT / f"f5_mapas_cache_{ventana}"


def deltas(ventana="2100"):
    """Δ por celda de los 14 índices (futuro QDM − obs 1985-2014) para cada GCM
    y escenario. Cada combinación se guarda en disco; si el proceso se corta,
    retoma donde quedó. Devuelve (máscara, índices obs por celda, {ssp: [(gcm, Δ)]})."""
    cfg = cargar_config()
    m = xr.open_dataarray(INT / "mascara_comun.nc")
    cache_dir = _dir_cache(ventana)
    cache_dir.mkdir(parents=True, exist_ok=True)
    faltan = [(g, s) for g in cfg["gcms"] for s in cfg["escenarios"]
              if not (cache_dir / f"{g}_{s}.nc").exists()]

    base = base_idx = None
    if faltan or not BASE_OBS.exists():
        pr_r, tx_r, tn_r = (_obs(v, cfg, m, REF) for v in ["pr", "tasmax", "tasmin"])
        per95, per99 = etccdi.percentiles_precip_base(pr_r)
        tx90, tn10 = etccdi.percentiles_temp_base(tx_r, tn_r)
        base = {"per95": per95, "per99": per99, "tx90": tx90, "tn10": tn10}
        print("baseline obs (por celda)...", flush=True)
        base_idx = _indices_celda(pr_r, tx_r, tn_r, base).load()
        base_idx.to_netcdf(BASE_OBS)
    else:
        with xr.open_dataset(BASE_OBS) as ds:
            base_idx = ds.load()

    acum = {ssp: [] for ssp in cfg["escenarios"]}
    for gcm in cfg["gcms"]:
        for ssp in cfg["escenarios"]:
            cache = cache_dir / f"{gcm}_{ssp}.nc"
            if cache.exists():
                with xr.open_dataset(cache) as ds:
                    acum[ssp].append((gcm, ds.load()))
                continue
            fut = {}
            for v in ["pr", "tasmax", "tasmin"]:
                f = QDM / f"{gcm}_{v}_{ssp}.nc"
                if not f.exists():
                    # UKESM sin tasmax -> índices de T de tasmax quedan NaN
                    fut[v] = None
                    continue
                with xr.open_dataarray(f) as da:
                    fut[v] = da.sel(time=slice(*VENTANAS[ventana])).transpose(
                        "time", "lat", "lon").load()
            fut_idx = _indices_celda(fut["pr"], fut["tasmax"], fut["tasmin"], base)
            delta = (fut_idx - base_idx).load()
            delta.to_netcdf(cache)
            acum[ssp].append((gcm, delta))
            del fut, fut_idx
            gc.collect()
            print(f"  {ventana} {gcm} {ssp} listo", flush=True)
    return m, base_idx, acum


def calcular():
    cfg = cargar_config()
    m, base_idx, acum = deltas("2100")
    # mediana inter-GCM por celda, por SSP
    salida = {}
    for ssp in cfg["escenarios"]:
        med = xr.concat([d for _, d in acum[ssp]], dim="gcm").median("gcm")
        for clave, _ in CLAVES:
            # re-enmascarar: los índices de conteo/suma devuelven 0 en celdas sin
            # dato en vez de NaN -> se fuerza NaN con la máscara común
            salida[f"{clave}_{ssp}"] = med[clave].where(m == 1)
    for clave in EN_PCT:
        salida[f"base_{clave}"] = base_idx[clave].where(m == 1)
    xr.Dataset(salida).to_netcdf(NC)
    print(f"Guardado {NC.name}")


def _limites_admin():
    """Límites departamentales (Natural Earth admin-1, 10m) de Perú."""
    from cartopy.io import shapereader
    shp = shapereader.natural_earth(resolution="10m", category="cultural",
                                    name="admin_1_states_provinces")
    return [r.geometry for r in shapereader.Reader(shp).records()
            if r.attributes.get("admin") == "Peru"]


# ciudades de referencia (lon, lat)
CIUDADES = {"Cusco": (-71.97, -13.53), "Quillabamba": (-72.69, -12.86),
            "Sicuani": (-71.23, -14.27), "Yauri": (-71.41, -14.79)}


def dibujar():
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from matplotlib import colors

    from figuras import estilo

    estilo.aplicar()
    plt.rcParams["axes.grid"] = False
    ds = xr.open_dataset(NC)
    mask = xr.open_dataarray(INT / "mascara_comun.nc").reindex(
        lat=ds.lat, lon=ds.lon, method="nearest")

    def campo(clave, ssp):
        da = ds[f"{clave}_{ssp}"]
        da = da.mean("percentiles") if "percentiles" in da.dims else da
        if clave in EN_PCT:
            base = ds[f"base_{clave}"]
            # celdas casi sin lluvia (desierto costero): el % no tiene sentido
            minimo = {"prcptot": 300, "rx1day": 10}[clave]
            da = 100 * da / base.where(base > minimo)
        return da.where(mask == 1)

    cfg = cargar_config()
    ssps = cfg["escenarios"]
    lon0, lat0, lon1, lat1 = (cfg["dominio"]["lon_min"], cfg["dominio"]["lat_min"],
                              cfg["dominio"]["lon_max"], cfg["dominio"]["lat_max"])
    # (nombre ETCCDI en español, qué mide, unidad, colormap, ¿divergente?)
    filas = {
        "txx": ("Maximum extreme\ntemperature (TXx)", "hottest afternoon\nof the year",
                "°C increase", "OrRd", False),
        "fd": ("Frost days\n(FD)", "days per year with\nminimum below 0 °C",
               "change in days", "Blues_r", False),
        "prcptot": ("Annual total\nprecipitation (PRCPTOT)", "rain over the whole year",
                    "% change", "BrBG", True),
        "rx1day": ("Maximum 1-day\nprecipitation (Rx1day)",
                   "heaviest downpour\nof the year", "% change", "BrBG", True),
    }
    import matplotlib.patheffects as pe
    from shapely.geometry import box
    from shapely.ops import unary_union

    proj = ccrs.PlateCarree()
    admin = _limites_admin()
    cusco = _contorno_cusco()
    # encuadre en el departamento, con un margen; fuera de él, velo blanco
    cu = unary_union(cusco)
    bx0, by0, bx1, by1 = cu.bounds
    ext = [bx0 - 0.25, bx1 + 0.25, by0 - 0.2, by1 + 0.2]
    velo = box(ext[0] - 1, ext[2] - 1, ext[1] + 1, ext[3] + 1).difference(cu)
    halo = [pe.withStroke(linewidth=2.2, foreground="white")]

    fig = plt.figure(figsize=(11.5, 15))
    gs = fig.add_gridspec(len(CLAVES), len(ssps), left=0.19, right=0.87,
                          top=0.89, bottom=0.045, wspace=0.04, hspace=0.06)
    axes = np.array([[fig.add_subplot(gs[i, j], projection=proj)
                      for j in range(len(ssps))] for i in range(len(CLAVES))])
    escalas = {}
    for i, (clave, _) in enumerate(CLAVES):
        nombre, _, uni, cmap, diverg = filas[clave]
        todos = np.concatenate([campo(clave, s).values.ravel() for s in ssps])
        todos = todos[np.isfinite(todos)]
        if diverg:
            lim = float(np.nanpercentile(np.abs(todos), 98))
            norm = colors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
        elif clave == "fd":
            norm = colors.Normalize(vmin=float(np.nanpercentile(todos, 2)), vmax=0)
        else:
            norm = colors.Normalize(vmin=0, vmax=float(np.nanpercentile(todos, 99)))
        for j, ssp in enumerate(ssps):
            ax = axes[i, j]
            ax.set_extent(ext, crs=proj)
            ax.set_facecolor("#F2F2F2")
            im = campo(clave, ssp).plot.pcolormesh(
                ax=ax, transform=proj, cmap=cmap, norm=norm,
                add_colorbar=False, rasterized=True)
            ax.add_geometries([velo], proj, facecolor="white", edgecolor="none",
                              alpha=0.72, zorder=3)
            ax.add_geometries(admin, proj, facecolor="none", edgecolor="#B5B5B5",
                              lw=0.35, zorder=3)
            ax.add_geometries(cusco, proj, facecolor="none", edgecolor="#1A1A1A",
                              lw=1.2, zorder=4)
            for nom, (x, y) in CIUDADES.items():
                ax.plot(x, y, "o", ms=3.2, color="#1A1A1A", mec="white", mew=0.6,
                        zorder=6, transform=proj)
                ax.text(x + 0.07, y + 0.05, nom, fontsize=7.2, zorder=6,
                        transform=proj, color="#1A1A1A", path_effects=halo)
            ax.set_title("")
            for s in ax.spines.values():
                s.set_edgecolor("#BBBBBB")
                s.set_linewidth(0.5)
        escalas[i] = im

    fig.canvas.draw()
    for i, (clave, _) in enumerate(CLAVES):
        nombre, que, uni, _, _ = filas[clave]
        pr = axes[i, -1].get_position()
        pl = axes[i, 0].get_position()
        cax = fig.add_axes([pr.x1 + 0.012, pr.y0 + 0.01, 0.011, pr.height - 0.02])
        cb = fig.colorbar(escalas[i], cax=cax)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7.5, length=0)
        cb.set_label(uni, fontsize=8.5)
        cb.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:+.0f}".replace("-", "−") if v else "0"))
        yc = (pl.y0 + pl.y1) / 2
        fig.text(pl.x0 - 0.012, yc + 0.006, nombre, ha="right", va="bottom",
                 fontsize=10.5, weight="bold", linespacing=1.25)
        fig.text(pl.x0 - 0.012, yc - 0.006, que, ha="right", va="top",
                 fontsize=8.5, color=estilo.TINTA_SUAVE, linespacing=1.25)
    for j, ssp in enumerate(ssps):
        p = axes[0, j].get_position()
        xc = (p.x0 + p.x1) / 2
        fig.text(xc, p.y1 + 0.022, estilo.NOMBRE_SSP[ssp], ha="center", fontsize=11,
                 weight="bold", color=estilo.COLOR_SSP[ssp])
        fig.text(xc, p.y1 + 0.007, estilo.LECTURA_SSP[ssp], ha="center",
                 fontsize=8.5, color=estilo.TINTA_SUAVE)

    estilo.titulo(fig, "Where will the climate of Cusco change most by the end of the century?",
                  "Change in 2071–2100 relative to 1985–2014, cell by cell (10 km). "
                  "Median of 10 CMIP6 models bias-adjusted with QDM.", y=0.985)
    estilo.fuente(fig, "Black line: Cusco department (region); colours are faded outside it. Near-rainless "
                  "cells have no data. Index names follow the ETCCDI. Source: CMIP6-CUSCO, PISCO (SENAMHI).", y=0.012)

    SAL.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(SAL / f"f5_mapas_cambio_2100.{ext}")
    print(f"Guardado f5_mapas_cambio_2100.png/.pdf en {SAL}")


if __name__ == "__main__":
    accion = sys.argv[1] if len(sys.argv) > 1 else "calcular"
    if accion == "calcular":
        calcular()
    elif accion == "deltas":
        deltas(sys.argv[2] if len(sys.argv) > 2 else "2100")
    elif accion == "dibujar":
        dibujar()
    else:
        calcular(); dibujar()
