"""Cambio de los índices ETCCDI en las proyecciones.

Para un GCM×SSP calcula, por ventana de horizonte (2035-64 y 2071-2100), los
índices ETCCDI de:
  - QDM directo (producto clásico), y
  - QDM + U-Net (producto DL),
y el cambio de cada producto contra su propio baseline 1985-2014.

Cada Δ se calcula por el mismo camino en baseline y futuro: QDM(futuro) −
QDM(histórico) y U-Net(QDM futuro) − U-Net(QDM histórico). Los percentiles
de los índices de umbral (tx90p, tn10p, r95p, r99p) salen de la obs.
La fila «baseline (obs)» queda en la tabla como referencia del sesgo
histórico de cada producto.

La U-Net se entrenó con obs agregada a 1° y re-interpolada (campo suave).
Una versión anterior le pasaba aquí el QDM a 0.1°, que ya trae el detalle
fino, y la red lo añadía por segunda vez: en el histórico la U-Net se
alejaba de la obs en los índices de umbral (mediana de los 10 GCM: FD +20
d/año, TX90p +70 d/año, TN10p +54 d/año, R95p +98 mm, PRCPTOT −136 mm). Ahora
el QDM pasa por `a_escala_gruesa`, la misma agregación del entrenamiento.

Todas las filas (obs, QDM y U-Net) se promedian sobre la misma región: las
celdas donde existe el producto DL, que son las interiores (el borde de 0.5°
del dominio no tiene entrada gruesa y es el margen de contexto). Así los
productos se comparan sobre las mismas celdas.

El resultado es una tabla por índice: valor baseline y Δ de cada ventana,
promediado sobre esa región. Los campos diarios no se guardan;
solo la tabla de índices.

Uso:  python -m cmip6cusco.indices.proyecciones MPI-ESM1-2-HR ssp585
"""

import argparse
import faulthandler
import gc

import numpy as np
import pandas as pd
import torch
import xarray as xr

from cmip6cusco.ajuste_sesgo.qdm import corregir_periodo
from cmip6cusco.config import RAIZ_REPO, cargar_config
from cmip6cusco.downscaling.aplicar import (RUTA_QDM, _canales_estaticos,
                                            _cargar_unet)
from cmip6cusco.downscaling.dataset import a_escala_gruesa
from cmip6cusco.indices import etccdi
from cmip6cusco.utils.grilla import alinear_a_grilla

RUTA_INTERIM = RAIZ_REPO / "data" / "interim"
RUTA_RESULTADOS = RAIZ_REPO / "resultados"

VENTANAS = {"2050": (2035, 2064), "2100": (2071, 2100)}
REF = (1985, 2014)

ARCHIVOS_OBS = {
    "pr": ("piscop_v3_prec_diaria_cusco.nc", "precipitation"),
    "tasmax": ("piscot_v12_tmax_diaria_cusco.nc", "tmax"),
    "tasmin": ("piscot_v12_tmin_diaria_cusco.nc", "tmin"),
}


def _obs(var, config, mascara, anios=None) -> xr.DataArray:
    archivo, nombre = ARCHIVOS_OBS[var]
    da = alinear_a_grilla(xr.open_dataset(RUTA_INTERIM / archivo)[nombre],
                          config).where(mascara == 1)
    if anios:
        da = da.sel(time=slice(str(anios[0]), str(anios[1])))
    return da.astype("float32")


def _unet_sobre(fino_da, var, unet, stats, est) -> xr.DataArray:
    """Aplica la U-Net a un campo QDM a 0.1°, llevado antes a la escala gruesa
    del entrenamiento (`a_escala_gruesa`)."""
    # el orden de dims puede variar (p.ej. CanESM5 noleap -> (lat,time,lon));
    # forzar (time,lat,lon) para el stack posicional de canales
    fino_da = fino_da.transpose("time", "lat", "lon")
    entrada_da = a_escala_gruesa(fino_da).transpose("time", "lat", "lon")
    entrada = entrada_da.values.astype("float32")
    dem_n, rug_n = _canales_estaticos(stats, est)
    x_var = ((entrada - stats["ent_media"]) / stats["ent_std"]).astype("float32")
    # lotes de 64 días armados al vuelo: evita el arreglo (T, 3, lat, lon)
    # completo, que con 30 años no entra en los ~6 GB de RAM de la PC
    res = np.empty_like(entrada)
    with torch.no_grad():
        for i in range(0, len(x_var), 64):
            xv = x_var[i:i + 64]
            xb = np.stack([xv, np.broadcast_to(dem_n, xv.shape),
                           np.broadcast_to(rug_n, xv.shape)], axis=1)
            xb = torch.from_numpy(np.nan_to_num(xb).astype("float32"))
            res[i:i + 64] = unet(xb).numpy()[:, 0]
    # todo en float32: con 30 años, una copia en float64 de cada campo pesa
    # ~220 MB y en la ventana 2071-2100 la RAM no alcanzaba
    recon = entrada + res * np.float32(stats["res_std"])
    del res, x_var
    if var == "pr":
        np.clip(recon, 0, None, out=recon)
    recon[~(np.isfinite(entrada) & np.isfinite(fino_da.values))] = np.nan
    return xr.DataArray(recon.astype("float32", copy=False),
                        coords=entrada_da.coords, dims=entrada_da.dims, name=var)


def _region_dl(fino_da) -> np.ndarray:
    """Celdas (lat, lon) donde existe el producto DL: dato fino y entrada gruesa."""
    dia = fino_da.transpose("time", "lat", "lon").isel(time=slice(0, 1))
    grueso = a_escala_gruesa(dia).transpose("time", "lat", "lon")
    return (np.isfinite(dia.values[0]) & np.isfinite(grueso.values[0]))


def _indices_dominio(pr, tx, tn, base, region=None) -> dict:
    """Índices ETCCDI -> climatología (media anual) -> media de dominio.

    Los índices de conteo/suma (tx90p, tn10p, fd, prcptot, r95p, r99p)
    devuelven 0 en celdas sin dato (entrada NaN) en vez de NaN, lo que diluiría
    la media de dominio (~0.8% con 20/2500 celdas enmascaradas). Se toma la
    validez de rx1day (índice de máximo, que sí deja NaN donde no hay dato) como
    máscara común y se aplica antes del promedio, para que todos los índices se
    promedien solo sobre las celdas con dato. Con `region` (lat, lon) se
    promedia además solo dentro de esa región, común a todos los productos.
    """
    ip = etccdi.indices_precip(pr, base["per95"], base["per99"])
    it = etccdi.indices_temp(tx, tn, base["tx90"], base["tn10"])
    ds = xr.merge([ip, it])
    # media temporal (sobre años) -> (lat,lon). tx90p/tn10p arrastran una
    # dim 'percentiles' (size 2); se colapsa promediándola para poder enmascarar
    # con la validez espacial (si no, np.where rompe por broadcast (50,50) vs
    # (50,50,2)). El promedio de esa dim es lineal -> no altera la media de dominio.
    clim = {}
    for v in ds.data_vars:
        a = ds[v].mean("time")
        extra = [d for d in a.dims if d not in ("lat", "lon")]
        clim[v] = (a.mean(extra) if extra else a).transpose("lat", "lon").values
    valido = np.isfinite(clim["rx1day"])  # (lat,lon): 2480 con dato, 20 sin dato
    if region is not None:
        valido &= region
    return {v: float(np.nanmean(np.where(valido, a, np.nan)))
            for v, a in clim.items()}


def procesar(gcm: str, ssp: str, sobrescribir: bool = False) -> pd.DataFrame | None:
    salida = RUTA_RESULTADOS / f"f5_etccdi_{gcm}_{ssp}.csv"
    if salida.exists() and not sobrescribir:
        print(f"[{gcm} {ssp}] ya existe {salida.name}; omito")
        return None
    config = cargar_config()
    mascara = xr.open_dataarray(RUTA_INTERIM / "mascara_comun.nc")

    # --- periodo base: percentiles desde la obs de referencia ---
    pr_ref = _obs("pr", config, mascara, REF)
    tx_ref = _obs("tasmax", config, mascara, REF)
    tn_ref = _obs("tasmin", config, mascara, REF)
    per95, per99 = etccdi.percentiles_precip_base(pr_ref)
    tx90, tn10 = etccdi.percentiles_temp_base(tx_ref, tn_ref)
    base = {"per95": per95, "per99": per99, "tx90": tx90, "tn10": tn10}
    print(f"[{gcm} {ssp}] periodo base {REF} listo.")
    # región común a todas las filas: donde existe el producto DL
    region = _region_dl(pr_ref)
    print(f"   región común (producto DL): {int(region.sum())} celdas")

    est = xr.open_dataset(RUTA_INTERIM / "covariables_estaticas.nc")
    # UKESM no tiene tasmax diaria -> se procesa con pr + tasmin (índices de
    # tasmax se omiten limpiamente, ver indices_temp)
    tiene_tasmax = (RUTA_QDM / f"{gcm}_tasmax_{ssp}.nc").exists()
    variables = ["pr", "tasmin"] + (["tasmax"] if tiene_tasmax else [])
    if not tiene_tasmax:
        print(f"   [aviso] {gcm} sin tasmax -> índices txx/dtr/tx90p omitidos")
    unets = {v: _cargar_unet(v) for v in variables}

    filas = {}
    filas["baseline (obs)"] = _indices_dominio(
        pr_ref, tx_ref if tiene_tasmax else None, tn_ref, base, region)

    # baseline calculado igual que el futuro: QDM-histórico (~obs) para el
    # producto QDM, y U-Net(QDM-histórico) para la U-Net — así el offset
    # estable del downscaling se cancela en el Δ (ver aplicar.py).
    print(f"   QDM del histórico {REF} (baseline consistente)...")
    qdm_ref = {v: corregir_periodo(gcm, v, "historical", config, mascara, REF)
               for v in variables}
    filas["baseline QDM"] = _indices_dominio(
        qdm_ref["pr"], qdm_ref.get("tasmax"), qdm_ref["tasmin"], base, region)
    print("   baseline U-Net (sobre el QDM-histórico)...")
    bb_ref = {v: _unet_sobre(qdm_ref[v], v, *unets[v], est) for v in variables}
    filas["baseline UNet"] = _indices_dominio(
        bb_ref["pr"], bb_ref.get("tasmax"), bb_ref["tasmin"], base, region)
    del qdm_ref, bb_ref  # liberar la RAM antes de las ventanas de futuro
    gc.collect()

    # --- ventanas de futuro: QDM directo y U-Net ---
    for etiqueta, (a0, a1) in VENTANAS.items():
        qdm = {v: xr.open_dataarray(RUTA_QDM / f"{gcm}_{v}_{ssp}.nc").sel(
            time=slice(str(a0), str(a1))) for v in variables}
        print(f"   ventana {etiqueta} ({a0}-{a1}): índices QDM...")
        filas[f"QDM {etiqueta}"] = _indices_dominio(
            qdm["pr"], qdm.get("tasmax"), qdm["tasmin"], base, region)
        print(f"   ventana {etiqueta}: downscaling U-Net + índices...")
        bb = {v: _unet_sobre(qdm[v], v, *unets[v], est) for v in variables}
        filas[f"UNet {etiqueta}"] = _indices_dominio(
            bb["pr"], bb.get("tasmax"), bb["tasmin"], base, region)
        del qdm, bb  # cada ventana libera lo suyo antes de la siguiente
        gc.collect()

    tabla = pd.DataFrame(filas).T
    # cambio: cada producto contra su baseline del mismo pipeline
    for et in VENTANAS:
        tabla.loc[f"Δ QDM {et}"] = tabla.loc[f"QDM {et}"] - tabla.loc["baseline QDM"]
        tabla.loc[f"Δ UNet {et}"] = tabla.loc[f"UNet {et}"] - tabla.loc["baseline UNet"]

    RUTA_RESULTADOS.mkdir(exist_ok=True)
    tabla.round(2).to_csv(salida)
    print(f"\nGuardado en {salida.name}\n")
    print(tabla.round(2).to_string())
    return tabla


if __name__ == "__main__":
    # si el proceso muere en código nativo (p. ej. sin memoria), que deje la pila
    faulthandler.enable()
    p = argparse.ArgumentParser()
    p.add_argument("gcm", help="nombre del GCM o 'todos'")
    p.add_argument("ssp", nargs="?", help="ssp o vacío = todos los del config")
    p.add_argument("--sobrescribir", action="store_true",
                   help="recalcular aunque ya exista la tabla del GCM×SSP")
    args = p.parse_args()
    cfg = cargar_config()
    gcms = cfg["gcms"] if args.gcm == "todos" else [args.gcm]
    ssps = [args.ssp] if args.ssp else cfg["escenarios"]
    for g in gcms:
        for s in ssps:
            print(f"===== {g} {s} =====")
            try:
                procesar(g, s, sobrescribir=args.sobrescribir)
            except FileNotFoundError as e:
                print(f"   OMITIDO: {e}")
