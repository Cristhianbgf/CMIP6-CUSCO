"""Validación del ajuste de sesgo (QDM) por GCM.

Cuatro paneles (crudo y QDM por GCM, y comparación con BASD-CMIP6-PE):
  (a) sesgo de P (%)          crudo -> QDM      [f2_qdm_resumen.csv]
  (b) error del p99 de P (%)  crudo -> QDM      [f2_qdm_resumen.csv]
  (c) PSS de P                crudo / QDM / BASD  [f2_comparacion_qdm_vs_basd.csv]
  (d) sesgo de Tmax (°C)      crudo -> QDM      [f2_qdm_resumen.csv]

Textos en inglés; colores de cmcrameri.

Uso:  python -m figuras.f2_validacion_qdm
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "resultados"
SAL = RES / "figuras"

# etiquetas cortas de GCM (para el eje y)
ETIQ = {
    "CanESM5": "CanESM5", "CNRM-CM6-1": "CNRM-CM6", "CNRM-ESM2-1": "CNRM-ESM2",
    "EC-Earth3": "EC-Earth3", "GFDL-ESM4": "GFDL-ESM4", "IPSL-CM6A-LR": "IPSL-CM6A",
    "MIROC6": "MIROC6", "MPI-ESM1-2-HR": "MPI-ESM1-2", "MRI-ESM2-0": "MRI-ESM2",
    "UKESM1-0-LL": "UKESM1",
}
# colores que se distinguen también con daltonismo
C_CRUDO = "#9A9A9A"   # gris: modelo crudo
C_QDM = "#1F4E79"     # azul: QDM
C_BASD = "#C77D2E"    # ámbar: BASD-CMIP6-PE


def figura():
    res = pd.read_csv(RES / "f2_qdm_resumen.csv")
    res = res[res.GCM != "MEDIA ENSAMBLE"].reset_index(drop=True)
    comp = pd.read_csv(RES / "f2_comparacion_qdm_vs_basd.csv")

    gcms = list(res.GCM)
    y = np.arange(len(gcms))[::-1]  # primer GCM arriba
    etiquetas = [ETIQ.get(g, g) for g in gcms]

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "figure.dpi": 140,
    })
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4))
    (axa, axb), (axc, axd) = axes

    def dumbbell(ax, col_crudo, col_qdm, ref, titulo, unidad):
        xc = res[col_crudo].to_numpy(dtype=float)
        xq = res[col_qdm].to_numpy(dtype=float)
        for yi, c, q in zip(y, xc, xq):
            if np.isnan(c) or np.isnan(q):
                continue
            ax.plot([c, q], [yi, yi], color="0.7", lw=1.4, zorder=1)
        ax.scatter(xc, y, s=42, facecolor="white", edgecolor=C_CRUDO,
                   linewidth=1.6, zorder=3, label="Raw global model")
        ax.scatter(xq, y, s=44, facecolor=C_QDM, edgecolor=C_QDM,
                   zorder=3, label="This repository (QDM)")
        ax.axvline(ref, color="0.35", lw=1.0, ls="--", zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels(etiquetas, fontsize=8)
        ax.set_title(titulo, fontsize=10, weight="bold", loc="left", pad=6)
        ax.set_xlabel(unidad, fontsize=8.5)
        ax.tick_params(length=2.5, labelsize=8)
        ax.grid(axis="x", alpha=0.25, lw=0.5)

    # (a) sesgo de P (%)
    dumbbell(axa, "sesgo P % (crudo)", "sesgo P % (QDM)", 0.0,
             "Mean precipitation bias", "%  (0 = observed)")
    # (b) error del p99 de P (%)
    dumbbell(axb, "err p99 % (crudo)", "err p99 % (QDM)", 0.0,
             "Extreme precipitation error (p99)", "%  (0 = observed)")
    # (d) sesgo de Tmax (°C)
    dumbbell(axd, "sesgo Tmax (crudo)", "sesgo Tmax (QDM)", 0.0,
             "Mean maximum temperature bias", "°C  (0 = observed)")

    # (c) PSS de P: crudo / QDM / BASD
    pss = comp[(comp.variable == "pr") & (comp["métrica"] == "PSS")]
    pss = pss.set_index("GCM")
    for yi, g in zip(y, gcms):
        if g not in pss.index:
            continue
        fila = pss.loc[g]
        cr, qd, ba = (float(fila["GCM crudo"]), float(fila["QDM (nuestro)"]),
                      float(fila["BASD (publicado)"]))
        axc.plot([cr, max(qd, ba)], [yi, yi], color="0.85", lw=1.2, zorder=1)
        axc.scatter(cr, yi, s=42, facecolor="white", edgecolor=C_CRUDO,
                    linewidth=1.6, zorder=3)
        axc.scatter(qd, yi, s=44, facecolor=C_QDM, edgecolor=C_QDM, zorder=3)
        axc.scatter(ba, yi, s=50, marker="D", facecolor="none",
                    edgecolor=C_BASD, linewidth=1.7, zorder=4)
    axc.axvline(1.0, color="0.35", lw=1.0, ls="--", zorder=0)
    axc.set_yticks(y)
    axc.set_yticklabels(etiquetas, fontsize=8)
    axc.set_title("Precipitation distribution (PSS)", fontsize=10,
                  weight="bold", loc="left", pad=6)
    axc.set_xlabel("PSS  (1 = identical to observed)", fontsize=8.5)
    axc.set_xlim(0.55, 1.02)
    axc.tick_params(length=2.5, labelsize=8)
    axc.grid(axis="x", alpha=0.25, lw=0.5)

    # leyenda global
    handles = [
        Line2D([0], [0], marker="o", ls="none", mfc="white", mec=C_CRUDO,
               mew=1.6, ms=8, label="Raw global model"),
        Line2D([0], [0], marker="o", ls="none", color=C_QDM, ms=8,
               label="This repository (QDM)"),
        Line2D([0], [0], marker="D", ls="none", mfc="none", mec=C_BASD,
               mew=1.7, ms=8, label="BASD-CMIP6-PE (published)"),
    ]
    fig.legend(handles=handles, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=9.5,
               handletextpad=0.4, columnspacing=2.0)

    fig.suptitle("Error of each model before and after bias adjustment · "
                 "Cusco, 1981–2014", fontsize=12.5, weight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.04, 1, 0.965])

    SAL.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(SAL / f"f2_validacion_qdm.{ext}",
                    bbox_inches="tight", dpi=300 if ext == "png" else None)
    print(f"Guardado f2_validacion_qdm.png/.pdf en {SAL}")


if __name__ == "__main__":
    figura()
