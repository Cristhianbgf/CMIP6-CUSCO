"""Esquema del flujo de trabajo (figura del README).

GCM crudo -> ajuste de sesgo con QDM -> downscaling con U-Net y con difusión
(entrenados con pares de observaciones gruesa/fina) -> índices ETCCDI ->
proyecciones. No lee datos; los textos van en inglés.

Uso:  python -m figuras.f0_pipeline
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

RAIZ = Path(__file__).resolve().parents[1]
SAL = RAIZ / "resultados" / "figuras"

# paleta por etapa (relleno, borde)
GRIS = ("#e8e8ea", "#7a7a7a")     # entrada
AZUL = ("#d6e4f0", "#2f6f9f")     # ajuste de sesgo (QDM)
VERDE = ("#dcebe1", "#2f6b4f")    # deep learning
AMBAR = ("#f6e6c8", "#c98a2e")    # salidas


def caja(ax, cx, cy, w, h, texto, col, fs=10, negrita=True):
    relleno, borde = col
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        linewidth=1.6, facecolor=relleno, edgecolor=borde, zorder=2))
    ax.text(cx, cy, texto, ha="center", va="center", fontsize=fs,
            weight="bold" if negrita else "normal", zorder=3, color="#1a1a1a")


def flecha(ax, xy0, xy1, col="#444", lw=1.8, estilo="-|>"):
    ax.annotate("", xy=xy1, xytext=xy0,
                arrowprops=dict(arrowstyle=estilo, color=col, lw=lw,
                                shrinkA=2, shrinkB=2), zorder=1)


def figura():
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(9.5, 11))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 103)
    ax.axis("off")

    # --- cajas ---
    # Las tres ramas van en una sola fila (QDM a 0.1°, U-Net y difusión), con los pares
    # de entrenamiento entre las dos redes, para que ninguna flecha cruce una caja.
    caja(ax, 45, 91, 46, 9,
         "CMIP6 global models (daily)\n10 models · 3 scenarios", GRIS, 10)
    caja(ax, 45, 76, 46, 9,
         "QDM bias adjustment\n(Cannon 2015)", AZUL, 10)

    caja(ax, 11, 57, 17, 10,
         "QDM (0.1°)\nclassical\nmethod", AZUL, 9)
    caja(ax, 36, 57, 17, 10, "U-Net", VERDE, 9.5)
    caja(ax, 61, 57, 18, 13,
         "Training pairs\nobs 1° → obs 0.1°\n(PISCO + RAIN4PE,\nPISCOt) + DEM",
         GRIS, 8, negrita=False)
    caja(ax, 86, 57, 17, 10, "Residual\ndiffusion", VERDE, 9.5)

    caja(ax, 61, 39, 30, 8,
         "Downscaled\n0.1° fields", VERDE, 9.5)
    caja(ax, 45, 24, 50, 8.5,
         "ETCCDI extreme indices (xclim)\n14 indices · median and P10–P90", AMBAR, 10)
    caja(ax, 45, 9, 50, 8.5,
         "Projections 2035–2064 and 2071–2100\nSSP1-2.6 · SSP3-7.0 · SSP5-8.5", AMBAR, 10)

    # --- flechas ---
    flecha(ax, (45, 86.1), (45, 81.1))            # GCM -> QDM
    flecha(ax, (30, 71.1), (14, 62.6))            # QDM -> producto QDM
    flecha(ax, (40, 71.1), (37, 62.6))            # QDM -> U-Net
    flecha(ax, (62, 71.1), (83, 62.6))            # QDM -> difusión
    flecha(ax, (51.6, 57), (44.9, 57), col="#7a7a7a", lw=1.4)   # entrenamiento -> U-Net
    flecha(ax, (70.4, 57), (77.1, 57), col="#7a7a7a", lw=1.4)   # entrenamiento -> difusión
    flecha(ax, (38, 51.6), (52, 43.6), lw=1.6)    # U-Net -> campos finos
    flecha(ax, (84, 51.6), (70, 43.6), lw=1.6)    # difusión -> campos finos
    flecha(ax, (58, 34.6), (52, 28.9))            # downscaled -> ETCCDI
    flecha(ax, (12, 51.6), (30, 28.9))            # producto QDM -> ETCCDI
    flecha(ax, (45, 19.4), (45, 13.9))            # ETCCDI -> Projections

    # nota del marco MOS-SR
    ax.text(50, 100.5,
            "CMIP6-CUSCO workflow", ha="center", va="top",
            fontsize=13.5, weight="bold", color="#1a1a1a")
    ax.text(50, 1.5,
            "The climate-change signal is carried by QDM; "
            "the network only learns the 1° → 0.1° mapping from observations.",
            ha="center", va="bottom", fontsize=8.5, style="italic",
            color="#555")

    SAL.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(SAL / f"f0_pipeline.{ext}",
                    bbox_inches="tight", dpi=300 if ext == "png" else None)
    print(f"Guardado f0_pipeline.png/.pdf en {SAL}")


if __name__ == "__main__":
    figura()
