"""Entrenamiento de la difusión sobre los residuos de la U-Net.

Flujo:
1. Reconstruye el periodo de entrenamiento con la U-Net congelada.
2. El objetivo de la difusión es el residuo restante: objetivo − U-Net,
   normalizado por su desviación estándar (del entrenamiento).
3. Condición (4 canales): [entrada gruesa, DEM, rugosidad, predicción U-Net].
4. Pérdida DDPM estándar: MSE entre el ruido real y el predicho (enmascarada).

El muestreo/evaluación (ensambles, CRPS, p99) vive en evaluar_difusion.py.

Uso:
    python -m cmip6cusco.downscaling.entrenar_difusion pr --epocas 2   # humo
    python -m cmip6cusco.downscaling.entrenar_difusion pr --epocas 40  # completo
"""

import argparse

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader, TensorDataset

from cmip6cusco.config import RAIZ_REPO, cargar_config
from cmip6cusco.downscaling.difusion import DenoiserUNet, ProgramaRuido
from cmip6cusco.downscaling.entrenar import RUTA_MODELOS, _cortar
from cmip6cusco.downscaling.unet import UNet

RUTA_PROCESSED = RAIZ_REPO / "data" / "processed"
RUTA_INTERIM = RAIZ_REPO / "data" / "interim"
RUTA_RESULTADOS = RAIZ_REPO / "resultados"


def preparar(var: str, config, espacio: str = "log1p"):
    """Condiciones y residuos U-Net normalizados para entrenar la difusión.

    espacio: "crudo" (residuo en mm/d) o "log1p" (residuo en log(1+mm/d)).
    En crudo, el ruido residual de los días secos es ~simétrico y el clip a
    ≥0 lo rectifica en llovizna espuria (medido: media +26 %). En log1p el
    cero es un punto natural y el ruido se vuelve multiplicativo — la
    variante correcta para precipitación.
    """
    pares = xr.open_dataset(RUTA_PROCESSED / f"f3_pares_{var}.nc")
    est = xr.open_dataset(RUTA_INTERIM / "covariables_estaticas.nc")
    mascara = xr.open_dataarray(RUTA_INTERIM / "mascara_comun.nc").values

    punto = torch.load(RUTA_MODELOS / f"unet_{var}.pt", weights_only=False)
    stats_b = punto["stats"]
    unet = UNet(3)
    unet.load_state_dict(punto["estado"])
    unet.eval()

    dem_n = ((est.dem_media.values - stats_b["dem_media"]) / stats_b["dem_stdv"]).astype("float32")
    rug_n = ((est.dem_std.values - stats_b["rug_media"]) / stats_b["rug_std"]).astype("float32")

    def bloque(anios):
        ent = _cortar(pares.entrada, anios).values
        obj = _cortar(pares.objetivo, anios).values
        x_var = ((ent - stats_b["ent_media"]) / stats_b["ent_std"]).astype("float32")
        x_cond3 = np.stack([x_var,
                            np.broadcast_to(dem_n, x_var.shape).copy(),
                            np.broadcast_to(rug_n, x_var.shape).copy()], axis=1)
        # limpiar NaN de todo el stack antes de pasarlo por la U-Net
        # (mismo bug que en entrenar.py: los bordes del DEM traen NaN y un
        # solo NaN envenena la convolución del dominio entero)
        x_cond3 = np.nan_to_num(x_cond3)
        with torch.no_grad():
            xt = torch.from_numpy(x_cond3)
            res_b = np.concatenate([unet(xt[i:i + 64]).numpy()[:, 0]
                                    for i in range(0, len(xt), 64)])
        pred_b = ent + res_b * stats_b["res_std"]
        if var == "pr":
            pred_b = np.clip(pred_b, 0, None)
        if espacio == "log1p":
            residuo_c = np.log1p(obj) - np.log1p(pred_b)
        else:
            residuo_c = obj - pred_b        # lo que la U-Net no explica
        return x_cond3, pred_b, residuo_c

    cond_tr, pred_tr, res_tr = bloque(config["periodos"]["entrenamiento"])
    std_c = float(np.nanstd(res_tr))
    if not np.isfinite(std_c) or std_c == 0:
        raise RuntimeError(f"std del residuo U-Net inválida ({std_c}): "
                           "revisar NaN en condiciones o checkpoint de la U-Net")

    def tensores(cond3, pred_b, residuo):
        pred_n = ((pred_b - stats_b["ent_media"]) / stats_b["ent_std"]).astype("float32")
        cond = np.concatenate([cond3, pred_n[:, None]], axis=1)
        x0 = np.nan_to_num(residuo / std_c)[:, None].astype("float32")
        return (torch.from_numpy(np.nan_to_num(cond)), torch.from_numpy(x0))

    stats_c = {**stats_b, "res_c_std": std_c, "espacio": espacio}
    return tensores(cond_tr, pred_tr, res_tr), stats_c, mascara


def entrenar(var: str, epocas: int, batch: int, lr: float, pasos: int,
             espacio: str = "log1p"):
    config = cargar_config()
    torch.manual_seed(config["semilla"])
    np.random.seed(config["semilla"])

    (cond_tr, x0_tr), stats, mascara = preparar(var, config, espacio)
    mask_t = torch.from_numpy(mascara.astype("float32"))[None, None]
    print(f"[{var}] difusión ({espacio}): {len(x0_tr)} días de entrenamiento | "
          f"std del residuo U-Net: {stats['res_c_std']:.3f}")

    modelo = DenoiserUNet(canales_condicion=4)
    programa = ProgramaRuido(pasos)
    optim = torch.optim.Adam(modelo.parameters(), lr=lr)
    cargador = DataLoader(TensorDataset(cond_tr, x0_tr), batch_size=batch,
                          shuffle=True)

    RUTA_MODELOS.mkdir(exist_ok=True)
    curva = []
    for epoca in range(1, epocas + 1):
        modelo.train()
        perdida_epoca = 0.0
        for cond_b, x0_b in cargador:
            t = torch.randint(0, programa.pasos, (len(x0_b),))
            ruido = torch.randn_like(x0_b)
            x_t = programa.ensuciar(x0_b, t, ruido)
            eps_pred = modelo(x_t, cond_b, t)
            perdida = (((eps_pred - ruido) ** 2) * mask_t).sum() / (
                mask_t.sum() * len(x0_b))
            optim.zero_grad()
            perdida.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            optim.step()
            perdida_epoca += float(perdida.detach()) * len(x0_b)
        perdida_epoca /= len(x0_tr)
        curva.append({"época": epoca, "pérdida_ddpm": perdida_epoca})
        print(f"[{var}] época {epoca:3d} | pérdida DDPM {perdida_epoca:.4f}")

        torch.save({"estado": modelo.state_dict(), "stats": stats,
                    "config": {"var": var, "pasos": pasos, "espacio": espacio}},
                   RUTA_MODELOS / f"difusion_{var}.pt")

    pd.DataFrame(curva).to_csv(
        RUTA_RESULTADOS / f"f3_difusion_curva_{var}.csv", index=False)
    print(f"[{var}] entrenamiento terminado; checkpoint en modelos/difusion_{var}.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("variable", choices=["pr", "tasmax", "tasmin"])
    p.add_argument("--epocas", type=int, default=40)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--pasos", type=int, default=400)
    p.add_argument("--espacio", choices=["crudo", "log1p"], default="log1p")
    args = p.parse_args()
    entrenar(args.variable, args.epocas, args.batch, args.lr, args.pasos,
             args.espacio)
