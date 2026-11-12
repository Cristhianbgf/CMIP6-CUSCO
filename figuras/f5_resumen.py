"""Resumen para tomadores de decisión: cuánto cambian siete indicadores del
clima del departamento del Cusco a mediados y a fines de siglo, en tres
escenarios. Promedio sobre las celdas del departamento
(resultados/f5_departamento_cusco.csv, de scripts/exportar_productos.py).

Cada celda: mediana de los 10 GCM (producto QDM) y, debajo, el rango en que
cae el 80 % de los modelos (P10–P90). Si ese rango cruza el cero, los modelos
no coinciden en el signo y la celda se marca como «incierto».

Uso:  python -m figuras.f5_resumen
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from figuras import estilo

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "resultados"
SAL = RES / "figuras"

# (índice, nombre ETCCDI en español, unidad del cambio, ¿en % de la línea base?, decimales)
FILAS = [
    ("txx", "Maximum extreme temperature", "°C", False, 1),
    ("tnn", "Minimum extreme temperature", "°C", False, 1),
    ("fd", "Frost days", "days", False, 0),
    ("prcptot", "Annual total precipitation", "%", True, 0),
    ("rx1day", "Maximum 1-day precipitation", "%", True, 0),
    ("r95p", "Very wet day precipitation", "%", True, 0),
    ("cdd", "Consecutive dry days", "days", False, 0),
]
# qué mide cada índice, en palabras simples
QUE = {"txx": "hottest afternoon of the year", "tnn": "coldest night of the year",
       "fd": "days per year below 0 °C", "prcptot": "rain over the whole year",
       "rx1day": "heaviest downpour of the year",
       "r95p": "rain on the wettest 5 % of days",
       "cdd": "longest dry spell of the year"}
UNID_BASE = {"txx": "°C", "tnn": "°C", "fd": "days", "prcptot": "mm",
             "rx1day": "mm", "r95p": "mm", "cdd": "days"}
VENTANAS = [("2050", "2035–2064"), ("2100", "2071–2100")]


ETCCDI = {"txx": "TXx", "tnn": "TNn", "fd": "FD", "prcptot": "PRCPTOT",
          "rx1day": "Rx1day", "r95p": "R95p", "cdd": "CDD"}


def _num(v, dec, signo=True):
    v = round(float(v), dec)
    if v == 0:
        return "0"
    s = f"{v:+.{dec}f}" if signo else f"{v:.{dec}f}"
    return s.replace("-", "−")


def figura():
    estilo.aplicar()
    dep = pd.read_csv(RES / "f5_departamento_cusco.csv")

    cols = [(v, s) for v, _ in VENTANAS for s in estilo.SSPS]
    n_f, n_c = len(FILAS), len(cols)

    fig = plt.figure(figsize=(12.5, 7.4))
    ax = fig.add_axes([0.30, 0.10, 0.69, 0.68])
    ax.set_xlim(0, n_c)
    ax.set_ylim(n_f, 0)
    ax.axis("off")

    for i, (idx, nombre, uni, en_pct, dec) in enumerate(FILAS):
        vals = []
        for v, s in cols:
            r = dep[(dep.ventana.astype(str) == v) & (dep.ssp == s) & (dep.indice == idx)].iloc[0]
            col = "cambio_pct" if en_pct else "cambio"
            vals.append((r[f"{col}_mediana"], r[f"{col}_p10"], r[f"{col}_p90"]))
            b = float(r.obs_1985_2014)
        tope = max(abs(m) for m, _, _ in vals) or 1

        for j, (med, p10, p90) in enumerate(vals):
            incierto = p10 < 0 < p90
            if incierto:
                color, alfa = "#EDEDED", 1
            else:
                color = "#C0392B" if med > 0 else "#2E6FA7"
                alfa = 0.12 + 0.6 * abs(med) / tope
            ax.add_patch(Rectangle((j + 0.04, i + 0.06), 0.92, 0.88,
                                   facecolor=color, alpha=alfa, lw=0))
            txt = f"{_num(med, dec)} {uni}"
            ax.text(j + 0.5, i + 0.40, txt, ha="center", va="center",
                    fontsize=11.5, weight="bold",
                    color=estilo.TINTA_SUAVE if incierto else estilo.TINTA)
            rango = f"{_num(p10, dec, False)} to {_num(p90, dec, False)}"
            ax.text(j + 0.5, i + 0.70, "uncertain · " + rango if incierto else rango,
                    ha="center", va="center", fontsize=7.8, color=estilo.TINTA_SUAVE)

        # nombre del indicador y valor actual, a la izquierda
        y = 0.78 - (i + 0.5) / n_f * 0.68
        fig.text(0.012, y + 0.014, f"{nombre} ({ETCCDI[idx]})", ha="left", va="center", fontsize=10.5,
                 weight="bold")
        fig.text(0.012, y - 0.018, f"{QUE[idx]} · today: "
                 f"{_num(b, 1 if UNID_BASE[idx] == '°C' else 0, False)} {UNID_BASE[idx]}", ha="left", va="center",
                 fontsize=8, color=estilo.TINTA_SUAVE)

    # encabezados: ventana arriba, escenario debajo
    for k, (v, rotulo) in enumerate(VENTANAS):
        x0 = 0.30 + 0.69 * (3 * k) / n_c
        fig.text(x0 + 0.69 * 1.5 / n_c, 0.865, f"{'Mid-century' if v == '2050' else 'End of century'} "
                 f"({rotulo})", ha="center", fontsize=10.5, weight="bold")
    for j, (v, s) in enumerate(cols):
        xc = 0.30 + 0.69 * (j + 0.5) / n_c
        fig.text(xc, 0.828, estilo.NOMBRE_SSP[s], ha="center", fontsize=9.5,
                 weight="bold", color=estilo.COLOR_SSP[s])
        fig.text(xc, 0.800, estilo.LECTURA_SSP[s].replace("intermediate-high", "intermediate"),
                 ha="center", fontsize=7.8, color=estilo.TINTA_SUAVE)
    ax.axvline(3, color="white", lw=6)

    estilo.titulo(fig, "How will the climate of the Cusco region change?",
                  "Change relative to 1985–2014. Median of 10 climate models (CMIP6) "
                  "bias-adjusted with QDM at 10 km; below, the range of 80 % of the models.",
                  y=0.985)
    estilo.fuente(fig, "Red: increases · blue: decreases · grey: models disagree on the sign of the change. "
                  "Average of the 10 km cells inside the Cusco department (region). "
                  "Source: CMIP6-CUSCO, PISCO (SENAMHI).", y=0.03)

    SAL.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(SAL / f"f5_resumen.{ext}")
    print(f"Guardado f5_resumen.png/.pdf en {SAL}")


if __name__ == "__main__":
    figura()
