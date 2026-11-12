"""Aplica la U-Net a las proyecciones corregidas con QDM.

A diferencia de evaluar.py (que mide la U-Net en la prueba 2011-2014 con
los pares obs), aquí la ENTRADA es una proyección corregida por QDM
(`data/interim/cmip6_qdm/{gcm}_{var}_{ssp}.nc`, 0.1°, 2015-2100). La red
aprendió obs_gruesa->obs_fina, así que el QDM se lleva primero a esa misma
escala gruesa (`a_escala_gruesa`: media a 1° y re-interpolación a 0.1°) y
recién entonces se refina. Sin ese paso la red recibe un campo que ya trae el
detalle fino y lo añade dos veces. El producto queda en las celdas
interiores: el borde de 0.5° del dominio no tiene entrada gruesa, igual que
en la evaluación.

Devuelve un DataArray con coordenadas (time, lat, lon) y la máscara del
dominio, listo para que xclim calcule los índices ETCCDI encima. Se procesa
por ventana de años (no los 86 de golpe) para no reventar la RAM (6 GB).

Uso:
    python -m cmip6cusco.downscaling.aplicar MPI-ESM1-2-HR pr ssp585 --anios 2071 2100
"""

import argparse

import numpy as np
import torch
import xarray as xr

from cmip6cusco.config import RAIZ_REPO, cargar_config
from cmip6cusco.downscaling.dataset import a_escala_gruesa
from cmip6cusco.downscaling.entrenar import RUTA_MODELOS
from cmip6cusco.downscaling.unet import UNet

RUTA_INTERIM = RAIZ_REPO / "data" / "interim"
RUTA_QDM = RUTA_INTERIM / "cmip6_qdm"


def _cargar_unet(var: str):
    """Carga la U-Net entrenada y sus estadísticos de normalización."""
    punto = torch.load(RUTA_MODELOS / f"unet_{var}.pt", weights_only=False)
    unet = UNet(3)
    unet.load_state_dict(punto["estado"])
    unet.eval()
    return unet, punto["stats"]


def _canales_estaticos(stats, est) -> tuple[np.ndarray, np.ndarray]:
    """DEM medio y rugosidad normalizados con los estadísticos del entrenamiento."""
    dem_n = ((est.dem_media.values - stats["dem_media"]) / stats["dem_stdv"]).astype("float32")
    rug_n = ((est.dem_std.values - stats["rug_media"]) / stats["rug_std"]).astype("float32")
    return dem_n, rug_n


def aplicar_unet(gcm: str, var: str, ssp: str,
                    anios: tuple[int, int] | None = None,
                    lote: int = 64) -> xr.DataArray:
    """Downscalea una proyección QDM con la U-Net.

    reconstrucción = gruesa(QDM) + residuo_predicho·res_std  (pr recortada a ≥0),
    con gruesa() = la misma agregación a 1° del entrenamiento.
    Conserva coords y máscara -> apto para xclim.
    """
    ruta = RUTA_QDM / f"{gcm}_{var}_{ssp}.nc"
    if not ruta.exists():
        raise FileNotFoundError(f"Falta la proyección QDM {ruta.name} (correr antes: python -m cmip6cusco.ajuste_sesgo.qdm proyectar)")

    qdm_da = xr.open_dataarray(ruta)
    if anios is not None:
        qdm_da = qdm_da.sel(time=slice(str(anios[0]), str(anios[1])))
    # el orden de dims puede variar (p.ej. calendario noleap -> (lat,time,lon));
    # forzar (time,lat,lon) para el stack posicional de canales
    qdm_da = qdm_da.transpose("time", "lat", "lon")
    entrada_da = a_escala_gruesa(qdm_da).transpose("time", "lat", "lon")
    entrada = entrada_da.values.astype("float32")  # (T, lat, lon)

    est = xr.open_dataset(RUTA_INTERIM / "covariables_estaticas.nc")
    unet, stats = _cargar_unet(var)
    dem_n, rug_n = _canales_estaticos(stats, est)

    x_var = (entrada - stats["ent_media"]) / stats["ent_std"]
    x = np.stack([x_var,
                  np.broadcast_to(dem_n, x_var.shape).copy(),
                  np.broadcast_to(rug_n, x_var.shape).copy()], axis=1)
    x = torch.from_numpy(np.nan_to_num(x).astype("float32"))

    with torch.no_grad():
        residuo = np.concatenate(
            [unet(x[i:i + lote]).numpy()[:, 0] for i in range(0, len(x), lote)])
    recon = entrada + residuo * stats["res_std"]
    if var == "pr":
        recon = np.clip(recon, 0, None)

    # devolver como DataArray con las coords de la entrada + re-enmascarar
    # (sin entrada gruesa en el borde, ni QDM en las celdas fuera de la
    #  máscara; el nan_to_num las volvió 0 -> restaurar NaN para que xclim no
    #  cuente celdas de relleno)
    salida = xr.DataArray(recon.astype("float32"), coords=entrada_da.coords,
                          dims=entrada_da.dims, name=var)
    salida = salida.where(np.isfinite(entrada_da) & np.isfinite(qdm_da))
    salida.attrs.update(qdm_da.attrs)
    salida.attrs.update(downscaling="U-Net", gcm=gcm, escenario=ssp)
    return salida


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("gcm")
    p.add_argument("variable", choices=["pr", "tasmax", "tasmin"])
    p.add_argument("ssp")
    p.add_argument("--anios", type=int, nargs=2, default=None, metavar=("A0", "A1"))
    args = p.parse_args()

    da = aplicar_unet(args.gcm, args.variable, args.ssp, anios=args.anios)
    qdm = xr.open_dataarray(RUTA_QDM / f"{args.gcm}_{args.variable}_{args.ssp}.nc")
    if args.anios:
        qdm = qdm.sel(time=slice(str(args.anios[0]), str(args.anios[1])))
    # comparar sobre las mismas celdas: la U-Net no cubre el borde de 0.5°
    qdm = qdm.transpose("time", "lat", "lon").where(da.notnull())
    u = "mm/d" if args.variable == "pr" else "°C"
    # comparar QDM (entrada) con la U-Net (salida): np.nanmean para float32 fiel
    m_in = float(np.nanmean(qdm.values))
    m_out = float(np.nanmean(da.values))
    p99_in = float(np.nanpercentile(qdm.values, 99))
    p99_out = float(np.nanpercentile(da.values, 99))
    print(f"[{args.gcm} {args.variable} {args.ssp} {args.anios or 'todo'}]")
    print(f"  celdas válidas (día0): {int(np.isfinite(da.isel(time=0)).sum())}")
    print(f"  media  QDM {m_in:.2f} -> U-Net {m_out:.2f} {u}")
    print(f"  p99    QDM {p99_in:.2f} -> U-Net {p99_out:.2f} {u}")
