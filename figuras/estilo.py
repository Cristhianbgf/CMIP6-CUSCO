"""Estilo común de las figuras: tipografía, colores y nombres en español.

Todas las figuras importan de aquí para verse como una sola familia.
"""
import matplotlib.pyplot as plt

# Escenarios: nombre técnico y lectura para no especialistas
SSPS = ["ssp126", "ssp370", "ssp585"]
NOMBRE_SSP = {"ssp126": "SSP1-2.6", "ssp370": "SSP3-7.0", "ssp585": "SSP5-8.5"}
LECTURA_SSP = {"ssp126": "low emissions", "ssp370": "intermediate-high emissions",
               "ssp585": "very high emissions"}
COLOR_SSP = {"ssp126": "#3A7CA5", "ssp370": "#E09F3E", "ssp585": "#9E2A2B"}

# Productos
COLOR_OBS = "#222222"
COLOR_CRUDO = "#B0B0B0"
COLOR_QDM = "#1F4E79"
COLOR_DL = "#2A9D8F"
COLOR_BASD = "#C77D2E"

TINTA = "#222222"
TINTA_SUAVE = "#5A5A5A"
REJILLA = "#E6E6E6"


def aplicar():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "text.color": TINTA,
        "axes.labelcolor": TINTA,
        "axes.edgecolor": "#888888",
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.axisbelow": True,
        "axes.grid": True,
        "grid.color": REJILLA,
        "grid.linewidth": 0.6,
        "xtick.color": TINTA_SUAVE,
        "ytick.color": TINTA_SUAVE,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
    })


def titulo(fig, texto, subtitulo=None, y=0.985):
    """Título a la izquierda, al estilo de un informe."""
    fig.text(0.012, y, texto, ha="left", va="top", fontsize=14,
             weight="bold", color=TINTA)
    if subtitulo:
        fig.text(0.012, y - 0.36 / fig.get_figheight(), subtitulo, ha="left", va="top",
                 fontsize=9.5, color=TINTA_SUAVE)


def fuente(fig, texto, y=0.01):
    fig.text(0.012, y, texto, ha="left", va="bottom", fontsize=7.5,
             color=TINTA_SUAVE)
