"""Cambio de los 14 índices ETCCDI en el ensamble de modelos.

Un panel por índice: mediana del cambio entre los GCM por escenario, con la
banda P10-P90, para QDM y para QDM + U-Net. Colores de cmcrameri.

Uso:  python -m figuras.f5_cambios_ensamble          # ventana 2100 (defecto)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmcrameri import cm as cmc
from matplotlib.lines import Line2D

RAIZ = Path(__file__).resolve().parents[1]
CSV = RAIZ / "resultados" / "f5_etccdi_ensamble.csv"
SALIDA = RAIZ / "resultados" / "figuras"

from figuras import estilo

# orden e info de cada índice: (clave, título, unidad, familia)
INDICES = [
    ("txx", "TXx · hottest day", "°C", "T"),
    ("tnn", "TNn · coldest night", "°C", "T"),
    ("tx90p", "TX90p · warm days", "days/yr", "T"),
    ("tn10p", "TN10p · cold nights", "days/yr", "T"),
    ("fd", "FD · frost days", "days/yr", "T"),
    ("dtr", "DTR · diurnal range", "°C", "T"),
    ("prcptot", "PRCPTOT · annual total", "mm/yr", "P"),
    ("rx1day", "Rx1day · max 1-day", "mm", "P"),
    ("rx5day", "Rx5day · max 5-day", "mm", "P"),
    ("r95p", "R95p · very wet days", "mm/yr", "P"),
    ("r99p", "R99p · extremely wet days", "mm/yr", "P"),
    ("sdii", "SDII · daily intensity", "mm/day", "P"),
    ("cdd", "CDD · dry spell", "days", "P"),
    ("cwd", "CWD · wet spell", "days", "P"),
]
SSPS = estilo.SSPS
ETIQ_SSP = estilo.NOMBRE_SSP
COL_SSP = estilo.COLOR_SSP


def figura(ventana: str = "2100"):
    df = pd.read_csv(CSV)
    # ventana puede leerse como int (2100) -> comparar como texto
    df = df[df.ventana.astype(str) == str(ventana)]

    estilo.aplicar()
    fig, axes = plt.subplots(2, 7, figsize=(15.5, 5.6))
    x = np.arange(3)

    for ax, (clave, titulo, unidad, fam) in zip(axes.ravel(), INDICES):
        for k, (prod, dx, mk, relleno) in enumerate(
                [("QDM", -0.13, "o", False), ("UNet", 0.13, "D", True)]):
            for j, ssp in enumerate(SSPS):
                r = df[(df.producto == prod) & (df.ssp == ssp) & (df.indice == clave)]
                if r.empty:
                    continue
                med, p10, p90 = (float(r.mediana.iloc[0]), float(r.p10.iloc[0]),
                                 float(r.p90.iloc[0]))
                col = COL_SSP[ssp]
                ax.errorbar(x[j] + dx, med, yerr=[[med - p10], [p90 - med]],
                            fmt=mk, ms=5.5, color=col,
                            mfc=col if relleno else "white",
                            mec=col, mew=1.3, elinewidth=1.3, capsize=2.5,
                            ecolor=col, alpha=0.95, zorder=3)
        ax.axhline(0, color="#888888", lw=0.8, zorder=1)
        ax.set_title(titulo, fontsize=8.6, pad=5)
        ax.set_ylabel(unidad, fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["1-2.6", "3-7.0", "5-8.5"], fontsize=7)
        ax.tick_params(length=2.5, labelsize=7)

    # dos viñetas sobran (14 índices en 14 ejes -> grid 2×7 exacto)
    # leyenda global
    handles = [
        Line2D([0], [0], marker="o", ls="none", color="0.3", mfc="white",
               mec="0.3", ms=6, label="QDM (classical)"),
        Line2D([0], [0], marker="D", ls="none", color="0.3", mfc="0.3",
               ms=6, label="QDM + U-Net"),
    ]
    handles += [Line2D([0], [0], marker="s", ls="none", color=COL_SSP[s],
                       ms=7, label=ETIQ_SSP[s]) for s in SSPS]
    fig.legend(handles=handles, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=8.5,
               handletextpad=0.4, columnspacing=1.6)

    per = '2071–2100' if str(ventana) == '2100' else '2035–2064'
    fig.suptitle(f"Change in the 14 extreme indices · domain 11–16 °S, 70–75 °W · {per} "
                 "vs 1985–2014", fontsize=12.5, weight="bold", x=0.01,
                 ha="left", y=1.0)
    fig.text(0.01, 0.94, "Point: median of 10 CMIP6 models · bar: range of 80 % "
             "of the models (P10–P90)", ha="left", fontsize=9, color=estilo.TINTA_SUAVE)
    fig.text(0.01, 0.905, "TX90p and TN10p count the days per year above (or below) the 1985–2014 "
             "threshold; today there are about 36. In the tropics temperature varies "
             "little from day to day, so with +4–5 °C almost the whole year exceeds "
             "that threshold.", ha="left", fontsize=8.5, color=estilo.TINTA_SUAVE)
    fig.tight_layout(rect=[0, 0.04, 1, 0.87])

    SALIDA.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(SALIDA / f"f5_cambios_ensamble_{ventana}.{ext}",
                    bbox_inches="tight", dpi=300 if ext == "png" else None)
    print(f"Guardado f5_cambios_ensamble_{ventana}.png/.pdf en {SALIDA}")


if __name__ == "__main__":
    figura(sys.argv[1] if len(sys.argv) > 1 else "2100")
