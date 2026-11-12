"""Agregación del ensamble de índices ETCCDI.

Lee todas las tablas por GCM (resultados/f5_etccdi_{gcm}_{ssp}.csv) y consolida
las señales de cambio entre GCM: mediana + banda P10–P90 (spread del ensamble)
por SSP, ventana, producto (QDM / U-Net) e índice.

La mediana es la mejor estimación central; el P10–P90 es la incertidumbre
estructural entre modelos. CanESM5 y UKESM
(ECS alta) inflan el extremo cálido — el spread lo captura sin descartarlos.

Uso:  python -m cmip6cusco.indices.ensamble
"""

import numpy as np
import pandas as pd

from cmip6cusco.config import RAIZ_REPO, cargar_config

RUTA_RESULTADOS = RAIZ_REPO / "resultados"


def _cargar_tablas() -> dict[tuple[str, str], pd.DataFrame]:
    """{(gcm, ssp): tabla} de todos los f5_etccdi_*.csv presentes."""
    tablas = {}
    for ruta in sorted(RUTA_RESULTADOS.glob("f5_etccdi_*.csv")):
        resto = ruta.stem.replace("f5_etccdi_", "")
        gcm, ssp = resto.rsplit("_", 1)  # el ssp es el último segmento
        tablas[(gcm, ssp)] = pd.read_csv(ruta, index_col=0)
    return tablas


def consolidar() -> pd.DataFrame:
    config = cargar_config()
    tablas = _cargar_tablas()
    if not tablas:
        raise FileNotFoundError("No hay f5_etccdi_*.csv (correr proyecciones primero)")
    gcms_presentes = sorted({g for g, _ in tablas})
    print(f"GCMs presentes: {len(gcms_presentes)} — {', '.join(gcms_presentes)}")

    filas = []
    for ssp in config["escenarios"]:
        for producto in ["QDM", "UNet"]:
            for ventana in ["2050", "2100"]:
                fila_delta = f"Δ {producto} {ventana}"
                # apilar el vector de índices de cada GCM para este ssp
                por_gcm = {g: t.loc[fila_delta] for (g, s), t in tablas.items()
                           if s == ssp and fila_delta in t.index}
                if not por_gcm:
                    continue
                m = pd.DataFrame(por_gcm).T  # filas=GCM, columnas=índices
                for indice in m.columns:
                    vals = m[indice].dropna().values
                    if len(vals) == 0:
                        continue
                    filas.append({
                        "ssp": ssp, "producto": producto, "ventana": ventana,
                        "indice": indice, "n_gcm": len(vals),
                        "mediana": np.median(vals),
                        "p10": np.percentile(vals, 10),
                        "p90": np.percentile(vals, 90),
                    })

    resumen = pd.DataFrame(filas)
    salida = RUTA_RESULTADOS / "f5_etccdi_ensamble.csv"
    resumen.round(2).to_csv(salida, index=False)
    print(f"\nGuardado en {salida.name} ({len(resumen)} filas)\n")

    # vista compacta: Δ mediana [p10, p90] por índice, U-Net, 2100, por SSP
    vista = resumen[(resumen.producto == "UNet") & (resumen.ventana == "2100")]
    if not vista.empty:
        print("=== Δ U-Net 2071-2100 (mediana [P10, P90] inter-GCM) ===")
        for ssp in config["escenarios"]:
            sub = vista[vista.ssp == ssp]
            if sub.empty:
                continue
            print(f"\n[{ssp}]  n_gcm~{int(sub.n_gcm.max())}")
            for _, r in sub.iterrows():
                print(f"  {r.indice:8s}: {r.mediana:8.2f}  [{r.p10:7.2f}, {r.p90:7.2f}]")
    return resumen


if __name__ == "__main__":
    consolidar()
