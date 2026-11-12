"""Ensamble de difusión aplicado a una proyección.

Aplica la difusión (residuo crudo, censura de días secos de 0.5 mm y centrado,
la variante con mejor resultado en el periodo de prueba) a una ventana de
proyección (MPI-ESM1-2-HR, ssp585, fin de siglo) y compara la banda de los N
miembros en el p99 de la precipitación diaria con QDM, con la U-Net y con la
observación.

La ventana es corta porque el muestreo DDIM en CPU es lento. La evaluación de
la difusión está en downscaling/evaluar_difusion.py.

Uso:  python -m figuras.f5_demo_difusion              # MPI ssp585, 2098-2100
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from cmcrameri import cm as cmc

from cmip6cusco.config import RAIZ_REPO, cargar_config
from cmip6cusco.downscaling.dataset import a_escala_gruesa
from cmip6cusco.downscaling.difusion import (DenoiserUNet, ProgramaRuido,
                                             muestrear_ddim)
from cmip6cusco.downscaling.entrenar import RUTA_MODELOS
from cmip6cusco.downscaling.unet import UNet
from cmip6cusco.utils.grilla import alinear_a_grilla

INT = RAIZ_REPO / "data" / "interim"
QDM = INT / "cmip6_qdm"
SAL = RAIZ_REPO / "resultados" / "figuras"


def _p99(campo, mascara):
    v = campo[:, mascara]
    return float(np.nanpercentile(v[np.isfinite(v)], 99))


def demo(gcm="MPI-ESM1-2-HR", ssp="ssp585", a0=2098, a1=2100,
         miembros=8, pasos=25, censura=0.5):
    cfg = cargar_config()
    torch.manual_seed(cfg["semilla"])
    mascara = xr.open_dataarray(INT / "mascara_comun.nc").values.astype(bool)
    est = xr.open_dataset(INT / "covariables_estaticas.nc")

    qdm_da = xr.open_dataarray(QDM / f"{gcm}_pr_{ssp}.nc").sel(
        time=slice(str(a0), str(a1))).transpose("time", "lat", "lon")
    qdm_fino = qdm_da.values.astype("float32")
    # la red (y la difusión, condicionada en ella) se entrenó con entrada
    # gruesa: el QDM se agrega a 1° y se re-interpola antes de refinarlo
    entrada = a_escala_gruesa(qdm_da).transpose(
        "time", "lat", "lon").values.astype("float32")
    n = entrada.shape[0]
    print(f"[demo difusión] {gcm} {ssp} {a0}-{a1}: {n} días × {miembros} miembros")

    # --- U-Net (media determinística) ---
    pb = torch.load(RUTA_MODELOS / "unet_pr.pt", weights_only=False)
    stats = pb["stats"]
    unet = UNet(3); unet.load_state_dict(pb["estado"]); unet.eval()
    dem_n = ((est.dem_media.values - stats["dem_media"]) / stats["dem_stdv"]).astype("float32")
    rug_n = ((est.dem_std.values - stats["rug_media"]) / stats["rug_std"]).astype("float32")
    x_var = ((entrada - stats["ent_media"]) / stats["ent_std"]).astype("float32")
    cond3 = np.nan_to_num(np.stack(
        [x_var, np.broadcast_to(dem_n, x_var.shape).copy(),
         np.broadcast_to(rug_n, x_var.shape).copy()], axis=1))
    with torch.no_grad():
        xt = torch.from_numpy(cond3)
        res_b = np.concatenate([unet(xt[i:i + 64]).numpy()[:, 0]
                                for i in range(0, len(xt), 64)])
    pred_b = np.clip(entrada + res_b * stats["res_std"], 0, None)

    # --- difusión: ensamble con censura y centrado ---
    pc = torch.load(RUTA_MODELOS / "difusion_pr.pt", weights_only=False)
    std_c = pc["stats"]["res_c_std"]
    den = DenoiserUNet(canales_condicion=4); den.load_state_dict(pc["estado"]); den.eval()
    prog = ProgramaRuido(pc["config"]["pasos"])
    pred_b_n = ((pred_b - stats["ent_media"]) / stats["ent_std"]).astype("float32")
    cond = torch.from_numpy(np.nan_to_num(
        np.concatenate([cond3, pred_b_n[:, None]], axis=1)))

    residuos = np.empty((miembros, *entrada.shape), dtype="float32")
    for m in range(miembros):
        print(f"   miembro {m + 1}/{miembros}...")
        partes = []
        for i in range(0, n, 32):
            r = muestrear_ddim(den, prog, cond[i:i + 32], n_pasos=pasos,
                               semilla=cfg["semilla"] * 1000 + m * 100 + i)
            partes.append(r.numpy()[:, 0])
        residuos[m] = np.concatenate(partes) * std_c
    residuos -= residuos.mean(axis=0, keepdims=True)  # centrado

    ensamble = np.empty_like(residuos)
    for m in range(miembros):
        campo = pred_b + residuos[m]
        campo = np.where(pred_b >= censura, campo, pred_b)  # censura de secos
        ensamble[m] = np.clip(campo, 0, None)

    # --- p99 por método, todos sobre las mismas celdas (donde hay producto DL;
    #     el borde de 0.5° no tiene entrada gruesa) ---
    region = mascara & np.isfinite(pred_b[0]) & np.isfinite(qdm_fino[0])
    obs = alinear_a_grilla(xr.open_dataset(INT / "piscop_v3_prec_diaria_cusco.nc")
                           ["precipitation"], cfg).where(
        xr.open_dataarray(INT / "mascara_comun.nc") == 1).sel(
        time=slice("1985", "2014")).transpose(
        "time", "lat", "lon").values.astype("float32")
    p99 = {
        "obs 1985-2014": _p99(obs, region),
        "QDM": _p99(qdm_fino, region),
        "unet": _p99(pred_b, region),
        "diffusion (members)": [_p99(ensamble[m], region) for m in range(miembros)],
    }
    for k, v in p99.items():
        print(f"   p99 {k}: {np.round(v, 2)}")
    return p99, gcm, ssp, a0, a1


def figura(p99, gcm, ssp, a0, a1):
    from figuras import estilo
    estilo.aplicar()
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    fig.subplots_adjust(left=0.27, right=0.96, top=0.74, bottom=0.17)
    miembros = np.array(p99["diffusion (members)"])
    filas = [
        ("Observed (1985–2014)", [p99["obs 1985-2014"]], estilo.COLOR_OBS, "s"),
        ("QDM (classical bias adjustment)", [p99["QDM"]], estilo.COLOR_QDM, "o"),
        ("QDM + U-Net", [p99["unet"]], estilo.COLOR_DL, "D"),
        ("Diffusion, 8 members", miembros, "#7B5EA7", "o"),
    ]
    y = np.arange(len(filas))[::-1]
    for yi, (nom, vals, col, mk) in zip(y, filas):
        if len(vals) > 1:
            ax.plot([vals.min(), vals.max()], [yi, yi], color=col, lw=6, alpha=0.18,
                    solid_capstyle="round")
            ax.scatter(vals, np.full(len(vals), yi), s=28, color=col, alpha=0.8, zorder=3)
            v = float(np.median(vals))
        else:
            v = float(vals[0])
            ax.scatter([v], [yi], s=70, color=col, marker=mk, zorder=3)
        ax.annotate(f"{v:.1f} mm", (v, yi), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8.5)
    ax.axvline(p99["obs 1985-2014"], color=estilo.COLOR_OBS, lw=0.8, ls=(0, (4, 2)))
    ax.set_yticks(y)
    ax.set_yticklabels([f[0] for f in filas], fontsize=9)
    ax.set_ylim(-0.6, len(filas) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Precipitation of the wettest 1 % of days (p99, mm/day)", fontsize=9)
    estilo.titulo(fig, "End-of-century extreme precipitation by method",
                  f"{gcm}, {estilo.NOMBRE_SSP[ssp]}, {a0}–{a1}. "
                  "Diffusion gives a band of possible values.")
    SAL.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(SAL / f"f5_demo_difusion_{ssp}.{ext}")
    print(f"Guardado f5_demo_difusion_{ssp}.png/.pdf en {SAL}")


if __name__ == "__main__":
    # el cálculo (demo) necesita el archivo QDM y muestrea la difusión en CPU;
    # se cachea el resultado (p99 por método + miembros) para poder redibujar
    # sin recomputar. Usar --recompute para forzar el recálculo.
    ssp = "ssp585"
    recompute = "--recompute" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    a0 = int(args[0]) if len(args) > 0 else 2098
    a1 = int(args[1]) if len(args) > 1 else 2100
    cache = RAIZ_REPO / "resultados" / f"f5_demo_difusion_{ssp}.json"

    if recompute or not cache.exists():
        p99, gcm, ssp, a0, a1 = demo(ssp=ssp, a0=a0, a1=a1)
        cache.write_text(json.dumps(
            {"p99": p99, "gcm": gcm, "ssp": ssp, "a0": a0, "a1": a1}, indent=2))
        print(f"cache -> {cache.name}")
    else:
        d = json.loads(cache.read_text())
        p99, gcm, ssp, a0, a1 = d["p99"], d["gcm"], d["ssp"], d["a0"], d["a1"]
        print(f"(desde cache {cache.name}; --recompute para recalcular)")
    figura(p99, gcm, ssp, a0, a1)
