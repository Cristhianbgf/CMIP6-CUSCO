"""Modelo global crudo, QDM de este repositorio y BASD-CMIP6-PE frente a la
observación, promediados sobre los 10 GCM.

Seis paneles con un punto por producto y una línea para la observación.
Textos en español, con el estilo de figuras/estilo.py.

Uso:  python -m figuras.f2_comparacion_productos
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "resultados"
SAL = RES / "figuras"

from figuras import estilo

# producto: (etiqueta, columna en el CSV, color, marcador)
PRODUCTOS = [
    ("Raw global model", "GCM crudo", estilo.COLOR_CRUDO, "o"),
    ("This repository (QDM, 10 km)", "QDM (nuestro)", estilo.COLOR_QDM, "o"),
    ("BASD-CMIP6-PE (published)", "BASD (publicado)", estilo.COLOR_BASD, "D"),
]


def figura():
    df = pd.read_csv(RES / "f2_comparacion_qdm_vs_basd.csv")

    def ens(var, metrica, col):
        """Promedio inter-GCM de una columna para (variable, métrica)."""
        sub = df[(df.variable == var) & (df["métrica"] == metrica)]
        return float(sub[col].mean())

    # paneles: (título, unidad, obs_objetivo, valores por producto, fmt, es_pss)
    def panel(var, metrica, titulo, uni, es_pss=False, fmt="{:.1f}"):
        obs = ens(var, metrica, "obs")
        vals = [ens(var, metrica, c) for _, c, *_ in PRODUCTOS]
        return dict(titulo=titulo, uni=uni, obs=obs, vals=vals, fmt=fmt,
                    es_pss=es_pss)

    # PSS de temperatura = promedio de tasmax y tasmin por producto
    def panel_tpss():
        obs = 1.0
        vals = []
        for _, c, *_ in PRODUCTOS:
            vals.append(np.mean([ens("tasmax", "PSS", c), ens("tasmin", "PSS", c)]))
        return dict(titulo="(f)  Temperature — distribution skill (PSS)",
                    uni="PSS  (1 = perfect)", obs=obs, vals=vals,
                    fmt="{:.2f}", es_pss=True)

    paneles = [
        panel("pr", "media (mm/d)", "Mean daily precipitation", "mm/day", fmt="{:.1f}"),
        panel("pr", "p99 (mm/d)", "Extreme-day precipitation (p99)", "mm/day"),
        panel("pr", "PSS", "Precipitation distribution (PSS)",
              "PSS (1 = identical)", es_pss=True, fmt="{:.2f}"),
        panel("tasmax", "media (°C)", "Mean maximum temperature", "°C"),
        panel("tasmin", "p05 (°C)", "Coldest nights (Tmin p05)", "°C"),
        panel_tpss(),
    ]
    paneles[-1]["titulo"] = "Temperature distribution (PSS)"
    paneles[-1]["uni"] = "PSS (1 = identical)"

    estilo.aplicar()
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.4))
    y = np.arange(len(PRODUCTOS))[::-1]
    for ax, p in zip(axes.ravel(), paneles):
        ax.axvline(p["obs"], color=estilo.COLOR_OBS, lw=1.2, ls=(0, (4, 2)), zorder=2)
        for yi, v, (_, _, col, mk) in zip(y, p["vals"], PRODUCTOS):
            ax.plot([p["obs"], v], [yi, yi], color=col, lw=2, alpha=0.45, zorder=2)
            ax.plot(v, yi, mk, ms=9, color=col, mec="white", mew=0.8, zorder=3)
            ax.annotate(p["fmt"].format(v).replace("-", "−"), (v, yi),
                        textcoords="offset points", xytext=(0, 9), ha="center",
                        fontsize=8.5, color=estilo.TINTA)
        ax.set_title(p["titulo"])
        ax.set_xlabel(p["uni"], fontsize=8.5, color=estilo.TINTA_SUAVE)
        ax.set_yticks(y)
        ax.set_yticklabels([e for e, _, _, _ in PRODUCTOS] if ax in axes[:, 0] else [],
                           fontsize=8.5)
        ax.set_ylim(-0.6, len(PRODUCTOS) - 0.3)
        ax.grid(axis="y", visible=False)
        lo, hi = min(p["vals"] + [p["obs"]]), max(p["vals"] + [p["obs"]])
        pad = 0.15 * (hi - lo) + 1e-6
        ax.set_xlim(lo - pad, hi + pad)
        ax.text(p["obs"], len(PRODUCTOS) - 0.35, " observed" if not p["es_pss"] else " ideal",
                fontsize=7.5, color=estilo.TINTA_SUAVE, va="top")
    estilo.titulo(fig, "Does the model reproduce the climate we already know? (1981–2014)",
                  "Mean of 10 CMIP6 models over the Cusco domain, against PISCO observations "
                  "(dashed line). Closer to the line is better.")
    estilo.fuente(fig, "Source: CMIP6-CUSCO; BASD-CMIP6-PE (Fernandez-Palomino et al., "
                  "2024); PISCO (SENAMHI).")
    fig.subplots_adjust(left=0.20, right=0.97, top=0.80, bottom=0.14, hspace=0.75, wspace=0.16)

    SAL.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(SAL / f"f2_comparacion_productos.{ext}",
                    )
    print(f"Guardado f2_comparacion_productos.png/.pdf en {SAL}")


if __name__ == "__main__":
    figura()
