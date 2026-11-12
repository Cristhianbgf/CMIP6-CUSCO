"""Evaluación de la U-Net en el periodo de prueba (2011-2014).

La validación (2006-2010) se usó para elegir el checkpoint, así que no
puede dar el número final. La prueba (2011-2014) no fue vista ni por
el entrenamiento ni por la selección de modelo; lo que salga de aquí es lo
que se reporta.

Compara, sobre los mismos días y celdas válidas:
- listón: la entrada gruesa re-interpolada (lo que habría sin red)
- U-Net: entrada + residuo predicho (pr recortada a ≥ 0)
con las mismas métricas de la validación del QDM: media, extremos
(p95/p99 o p05/p95) y Perkins Skill Score, más RMSE y correlación.

Uso:  python -m cmip6cusco.downscaling.evaluar pr
"""

import sys

import numpy as np
import pandas as pd
import torch
import xarray as xr

from cmip6cusco.ajuste_sesgo.qdm import _metricas
from cmip6cusco.config import RAIZ_REPO, cargar_config
from cmip6cusco.downscaling.entrenar import RUTA_MODELOS, _cortar
from cmip6cusco.downscaling.unet import UNet

RUTA_PROCESSED = RAIZ_REPO / "data" / "processed"
RUTA_INTERIM = RAIZ_REPO / "data" / "interim"
RUTA_RESULTADOS = RAIZ_REPO / "resultados"


def evaluar(var: str) -> pd.DataFrame:
    config = cargar_config()
    pares = xr.open_dataset(RUTA_PROCESSED / f"f3_pares_{var}.nc")
    est = xr.open_dataset(RUTA_INTERIM / "covariables_estaticas.nc")
    mascara = xr.open_dataarray(RUTA_INTERIM / "mascara_comun.nc").values.astype(bool)

    anios = config["periodos"]["prueba"]
    entrada = _cortar(pares.entrada, anios).values
    objetivo = _cortar(pares.objetivo, anios).values
    print(f"[{var}] prueba {anios}: {entrada.shape[0]} días")

    punto = torch.load(RUTA_MODELOS / f"unet_{var}.pt", weights_only=False)
    stats = punto["stats"]
    modelo = UNet(3)
    modelo.load_state_dict(punto["estado"])
    modelo.eval()

    dem_n = ((est.dem_media.values - stats["dem_media"]) / stats["dem_stdv"]).astype("float32")
    rug_n = ((est.dem_std.values - stats["rug_media"]) / stats["rug_std"]).astype("float32")
    x_var = (entrada - stats["ent_media"]) / stats["ent_std"]
    x = np.stack([x_var,
                  np.broadcast_to(dem_n, x_var.shape).copy(),
                  np.broadcast_to(rug_n, x_var.shape).copy()], axis=1)
    x = torch.from_numpy(np.nan_to_num(x).astype("float32"))

    with torch.no_grad():
        residuo = np.concatenate(
            [modelo(x[i:i + 64]).numpy()[:, 0] for i in range(0, len(x), 64)])
    reconstruccion = entrada + residuo * stats["res_std"]
    if var == "pr":
        reconstruccion = np.clip(reconstruccion, 0, None)

    obs = objetivo[:, mascara].ravel()
    campos = [("listón (interp.)", entrada[:, mascara].ravel()),
              ("U-Net", reconstruccion[:, mascara].ravel())]
    # Muestra común: celdas-día con dato en obs y en todos los métodos, también
    # para la fila obs. La entrada gruesa no cubre un borde de 0.5° del dominio
    # (1600 de las 2480 celdas); con obs sobre las 2480 se compararían regiones
    # distintas (p. ej. un sesgo de media de −0.27 °C que en realidad es +0.02).
    ok = np.isfinite(obs)
    for _, vals in campos:
        ok &= np.isfinite(vals)
    print(f"   muestra común: {ok.sum()} de {np.isfinite(obs).sum()} "
          f"valores obs ({100 * ok.sum() / np.isfinite(obs).sum():.1f} %)")

    m_obs = _metricas(var, obs[ok], obs[ok])
    filas = [{"método": "obs", "RMSE": 0.0, "r": 1.0,
              **{met: v_obs for met, (v_obs, _) in m_obs.items()}}]
    for nombre, vals in campos:
        m = _metricas(var, obs[ok], vals[ok])
        fila = {"método": nombre,
                "RMSE": float(np.sqrt(np.mean((vals[ok] - obs[ok]) ** 2))),
                "r": float(np.corrcoef(vals[ok], obs[ok])[0, 1])}
        fila.update({met: mod for met, (_, mod) in m.items()})
        filas.append(fila)

    tabla = pd.DataFrame(filas).set_index("método")
    salida = RUTA_RESULTADOS / f"f3_unet_prueba_{var}.csv"
    tabla.round(3).to_csv(salida)
    print(f"Guardado en {salida}\n")
    print(tabla.round(3).to_string())
    return tabla


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Uso: python -m cmip6cusco.downscaling.evaluar <variable>")
    evaluar(sys.argv[1])
