"""Evaluación de la difusión en el periodo de prueba (2011-2014): ensambles, colas y CRPS.

La pregunta que responde: ¿el ensamble generativo recupera la cola extrema
(p99) que la U-Net determinística suaviza (−7.6 % medido), sin romper lo que
la U-Net ya hace bien (media, PSS)?

Métricas:
- Media del ensamble: RMSE y r (debe parecerse a la U-Net).
- Distribución agrupada del ensamble (todas las muestras): p95, p99 y PSS.
- CRPS (properscoring): calidad probabilística; menor es mejor. Se compara
  contra el "CRPS determinístico" de la U-Net (= su MAE), que es la cota
  que un ensamble útil debe batir.

Censura de días secos (--censura-secos UMBRAL): el residuo estocástico se
aplica solo donde la U-Net predice celda húmeda (pred_b ≥ umbral); en celdas
secas se conserva la predicción determinística. Motivo: la ocurrencia ya la
resuelve la U-Net (PSS 0.99); el ruido de la difusión en celdas secas se
rectifica con el clip ≥0 en llovizna espuria (medido: media +21 %, frec 65 %
vs 51 %). Es la estructura en dos partes (ocurrencia/intensidad) clásica de
los modelos estocásticos de precipitación.

Centrado del ensamble (--centrar-residuo): a cada miembro se le resta la
media de los M miembros por celda-día antes de componer el campo. En la
descomposición residual tipo CorrDiff la media condicional la pone la U-Net
y la difusión debe aportar variabilidad de media cero; el DDPM entrenado no
lo garantiza (medido: deriva ~+0.9 mm/d en celdas húmedas, media +9.8 % y
p95 corrido +1.8 mm aun con censura). El centrado impone esa propiedad sin
parámetros nuevos; costo: encoge la dispersión en el factor √(1−1/M) (~4 %
con M=8).

Uso:  python -m cmip6cusco.downscaling.evaluar_difusion pr --miembros 8 --pasos 25
      python -m cmip6cusco.downscaling.evaluar_difusion pr --miembros 8 --pasos 25 \
          --checkpoint difusion_pr_crudo80.pt --censura-secos 1.0
"""

import argparse

import numpy as np
import pandas as pd
import properscoring as ps
import torch
import xarray as xr

from cmip6cusco.ajuste_sesgo.qdm import _metricas
from cmip6cusco.config import RAIZ_REPO, cargar_config
from cmip6cusco.downscaling.difusion import (DenoiserUNet, ProgramaRuido,
                                             muestrear_ddim)
from cmip6cusco.downscaling.entrenar import RUTA_MODELOS, _cortar
from cmip6cusco.downscaling.unet import UNet

RUTA_PROCESSED = RAIZ_REPO / "data" / "processed"
RUTA_INTERIM = RAIZ_REPO / "data" / "interim"
RUTA_RESULTADOS = RAIZ_REPO / "resultados"


def evaluar(var: str, miembros: int, pasos_ddim: int, lote: int = 32,
            checkpoint: str | None = None,
            censura_secos: float | None = None,
            centrar_residuo: bool = False):
    config = cargar_config()
    torch.manual_seed(config["semilla"])

    pares = xr.open_dataset(RUTA_PROCESSED / f"f3_pares_{var}.nc")
    est = xr.open_dataset(RUTA_INTERIM / "covariables_estaticas.nc")
    mascara = xr.open_dataarray(RUTA_INTERIM / "mascara_comun.nc").values.astype(bool)

    anios = config["periodos"]["prueba"]
    entrada = _cortar(pares.entrada, anios).values
    objetivo = _cortar(pares.objetivo, anios).values
    n_dias = entrada.shape[0]
    print(f"[{var}] prueba {anios}: {n_dias} días × {miembros} miembros × "
          f"{pasos_ddim} pasos DDIM")

    # --- U-Net congelada (media determinística) ---
    punto_b = torch.load(RUTA_MODELOS / f"unet_{var}.pt", weights_only=False)
    stats = punto_b["stats"]
    unet = UNet(3)
    unet.load_state_dict(punto_b["estado"])
    unet.eval()

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
    pred_b = entrada + res_b * stats["res_std"]
    if var == "pr":
        pred_b = np.clip(pred_b, 0, None)

    # --- difusión: muestrear el residuo restante ---
    nombre_pt = checkpoint or f"difusion_{var}.pt"
    punto_c = torch.load(RUTA_MODELOS / nombre_pt, weights_only=False)
    std_c = punto_c["stats"]["res_c_std"]
    espacio = punto_c["config"].get("espacio", "crudo")
    print(f"   checkpoint: {nombre_pt} | espacio del residuo: {espacio}"
          + (f" | censura de secos: pred_b < {censura_secos} mm/d"
             if censura_secos is not None else "")
          + (" | residuo centrado (media 0 entre miembros)"
             if centrar_residuo else ""))
    denoiser = DenoiserUNet(canales_condicion=4)
    denoiser.load_state_dict(punto_c["estado"])
    denoiser.eval()
    programa = ProgramaRuido(punto_c["config"]["pasos"])

    pred_b_n = ((pred_b - stats["ent_media"]) / stats["ent_std"]).astype("float32")
    cond = torch.from_numpy(np.nan_to_num(
        np.concatenate([cond3, pred_b_n[:, None]], axis=1)))

    # 1ª pasada: muestrear los residuos de todos los miembros
    residuos = np.empty((miembros, n_dias, *entrada.shape[1:]), dtype="float32")
    for m in range(miembros):
        print(f"   miembro {m + 1}/{miembros}...")
        partes = []
        for i in range(0, n_dias, lote):
            res = muestrear_ddim(denoiser, programa, cond[i:i + lote],
                                 n_pasos=pasos_ddim,
                                 semilla=config["semilla"] * 1000 + m * 100 + i)
            partes.append(res.numpy()[:, 0])
        residuos[m] = np.concatenate(partes) * std_c

    if centrar_residuo:
        residuos -= residuos.mean(axis=0, keepdims=True)

    # 2ª pasada: componer los campos (censura y clip por miembro)
    ensamble = np.empty_like(residuos)
    for m in range(miembros):
        if espacio == "log1p":
            campo = np.expm1(np.log1p(np.clip(pred_b, 0, None)) + residuos[m])
        else:
            campo = pred_b + residuos[m]
        if censura_secos is not None:
            # dos partes: en celdas secas según la U-Net manda la determinística
            campo = np.where(pred_b >= censura_secos, campo, pred_b)
        if var == "pr":
            campo = np.clip(campo, 0, None)
        ensamble[m] = campo

    # --- métricas ---
    obs = objetivo[:, mascara].ravel()
    media_ens = ensamble.mean(axis=0)[:, mascara].ravel()
    agrupado = ensamble[:, :, mascara].reshape(miembros, -1)
    campos = [("listón (interp.)", entrada[:, mascara].ravel()),
              ("U-Net", pred_b[:, mascara].ravel())]
    # Muestra común a todas las filas, obs incluida: la entrada gruesa no cubre
    # un borde de 0.5° (1600 de 2480 celdas). Antes la fila obs usaba las 2480
    # y comparaba regiones distintas (un p99 «exacto» que en realidad es +1.5 %).
    ok = (np.isfinite(obs) & np.isfinite(media_ens)
          & np.isfinite(agrupado).all(axis=0))
    for _, campo in campos:
        ok &= np.isfinite(campo)
    print(f"   muestra común: {ok.sum()} de {np.isfinite(obs).sum()} "
          f"valores obs ({100 * ok.sum() / np.isfinite(obs).sum():.1f} %)")

    filas = []
    m_obs = _metricas(var, obs[ok], obs[ok])
    filas.append({"método": "obs", "RMSE": 0.0, "CRPS": 0.0,
                  **{k: v for k, (v, _) in m_obs.items()}})

    for nombre, campo in campos:
        m = _metricas(var, obs[ok], campo[ok])
        filas.append({"método": nombre,
                      "RMSE": float(np.sqrt(np.mean((campo[ok] - obs[ok]) ** 2))),
                      "CRPS": float(np.mean(np.abs(campo[ok] - obs[ok]))),
                      **{k: v for k, (_, v) in m.items()}})

    m_c = _metricas(var, obs[ok], agrupado[:, ok].ravel())
    crps = float(ps.crps_ensemble(
        obs[ok], np.moveaxis(agrupado[:, ok], 0, -1)).mean())
    detalles = ([] if censura_secos is None else ["secos censurados"]) + (
        ["centrado"] if centrar_residuo else [])
    etiqueta_c = f"Difusión ({miembros} miembros" + (
        ", " + ", ".join(detalles) if detalles else "") + ")"
    filas.append({"método": etiqueta_c,
                  "RMSE": float(np.sqrt(np.mean(
                      (media_ens[ok] - obs[ok]) ** 2))),
                  "CRPS": crps,
                  **{k: v for k, (_, v) in m_c.items()}})

    tabla = pd.DataFrame(filas).set_index("método")
    variante = (espacio
                + (f"_censura{censura_secos:g}" if censura_secos is not None else "")
                + ("_centrado" if centrar_residuo else ""))
    salida = RUTA_RESULTADOS / f"f3_difusion_prueba_{var}_{variante}.csv"
    tabla.round(3).to_csv(salida)
    print(f"\nGuardado en {salida}\n")
    print(tabla.round(3).to_string())
    return tabla


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("variable", choices=["pr", "tasmax", "tasmin"])
    p.add_argument("--miembros", type=int, default=8)
    p.add_argument("--pasos", type=int, default=25)
    p.add_argument("--checkpoint", default=None,
                   help="nombre del .pt en modelos/ (por defecto difusion_<var>.pt)")
    p.add_argument("--censura-secos", type=float, default=None, metavar="UMBRAL",
                   help="aplicar el residuo solo donde pred_b >= UMBRAL (mm/d)")
    p.add_argument("--centrar-residuo", action="store_true",
                   help="restar la media entre miembros por celda-día (residuo de media 0)")
    args = p.parse_args()
    evaluar(args.variable, args.miembros, args.pasos,
            checkpoint=args.checkpoint, censura_secos=args.censura_secos,
            centrar_residuo=args.centrar_residuo)
