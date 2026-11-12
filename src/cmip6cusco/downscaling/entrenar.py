"""Entrenamiento de la U-Net de downscaling.

Flujo:
1. Carga los pares de data/processed/f3_pares_{var}.nc y las covariables
   estáticas; corta entrenamiento/validación según configs/dominio.yml.
2. Normaliza con estadísticos del entrenamiento (nunca de la validación).
3. La red aprende el residuo normalizado; en la reconstrucción se
   desnormaliza y se suma a la entrada gruesa (y pr se recorta a ≥ 0).
4. Reporta por época la pérdida y el RMSE de validación contra el listón
   de la interpolación pura (RMSE de la entrada sola); si la red no baja de
   ese listón, no aporta nada.
5. Guarda el mejor checkpoint en modelos/ (fuera de git) y la curva de
   entrenamiento en resultados/ (versionada).

Uso:
    python -m cmip6cusco.downscaling.entrenar pr --epocas 30
    python -m cmip6cusco.downscaling.entrenar tasmax --epocas 2   # humo
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader, TensorDataset

from cmip6cusco.config import RAIZ_REPO, cargar_config
from cmip6cusco.downscaling.unet import UNet, perdida_l1_pinball

RUTA_PROCESSED = RAIZ_REPO / "data" / "processed"
RUTA_INTERIM = RAIZ_REPO / "data" / "interim"
RUTA_MODELOS = RAIZ_REPO / "modelos"
RUTA_RESULTADOS = RAIZ_REPO / "resultados"


def _cortar(da: xr.DataArray, anios: list[int]) -> xr.DataArray:
    return da.sel(time=slice(str(anios[0]), str(anios[1])))


def preparar_tensores(var: str, config):
    """Tensores (X, y, máscara) de entrenamiento y validación + estadísticos."""
    pares = xr.open_dataset(RUTA_PROCESSED / f"f3_pares_{var}.nc")
    est = xr.open_dataset(RUTA_INTERIM / "covariables_estaticas.nc")
    mascara = xr.open_dataarray(RUTA_INTERIM / "mascara_comun.nc").values

    splits = {}
    for nombre in ["entrenamiento", "validacion"]:
        ent = _cortar(pares.entrada, config["periodos"][nombre]).values
        obj = _cortar(pares.objetivo, config["periodos"][nombre]).values
        splits[nombre] = (ent, obj)

    ent_tr, obj_tr = splits["entrenamiento"]
    residuo_tr = obj_tr - ent_tr

    # estadísticos solo del entrenamiento (sobre celdas válidas)
    valida = np.isfinite(residuo_tr)
    stats = {
        "ent_media": float(np.nanmean(ent_tr)),
        "ent_std": float(np.nanstd(ent_tr)),
        "res_std": float(np.nanstd(residuo_tr[valida])),
        "dem_media": float(est.dem_media.mean()),
        "dem_stdv": float(est.dem_media.std()),
        "rug_media": float(est.dem_std.mean()),
        "rug_std": float(est.dem_std.std()),
    }

    dem_n = ((est.dem_media.values - stats["dem_media"]) / stats["dem_stdv"]).astype("float32")
    rug_n = ((est.dem_std.values - stats["rug_media"]) / stats["rug_std"]).astype("float32")
    mask_t = torch.from_numpy(mascara.astype("float32"))[None]

    def a_tensores(ent, obj):
        x_var = (ent - stats["ent_media"]) / stats["ent_std"]
        x = np.stack([
            x_var,
            np.broadcast_to(dem_n, x_var.shape).copy(),
            np.broadcast_to(rug_n, x_var.shape).copy(),
        ], axis=1).astype("float32")                       # (n, 3, 50, 50)
        # limpiar NaN de todos los canales (los bordes realineados del DEM
        # traen ~0.2 % de NaN; un solo NaN envenena la red entera)
        x = np.nan_to_num(x)
        y = (obj - ent) / stats["res_std"]                  # residuo normalizado
        y = np.nan_to_num(y)[:, None].astype("float32")     # (n, 1, 50, 50)
        return torch.from_numpy(x), torch.from_numpy(y)

    x_tr, y_tr = a_tensores(ent_tr, obj_tr)
    x_va, y_va = a_tensores(*splits["validacion"])
    return (x_tr, y_tr), (x_va, y_va), splits, stats, mask_t


def rmse_reconstruccion(modelo, x, ent_crudo, obj_crudo, stats, mask_t, var,
                        lote=64):
    """RMSE del campo reconstruido (entrada + residuo predicho) vs objetivo."""
    modelo.eval()
    errores, n = 0.0, 0
    mascara = mask_t.numpy().astype(bool)[0]
    with torch.no_grad():
        for i in range(0, len(x), lote):
            res = modelo(x[i:i + lote]).numpy()[:, 0] * stats["res_std"]
            rec = ent_crudo[i:i + lote] + res
            if var == "pr":
                rec = np.clip(rec, 0, None)  # la lluvia no puede ser negativa
            err = (rec - obj_crudo[i:i + lote])[:, mascara]
            errores += np.nansum(err ** 2)
            n += np.isfinite(err).sum()
    return float(np.sqrt(errores / n))


def entrenar(var: str, epocas: int, batch: int, lr: float, peso_pinball: float):
    config = cargar_config()
    torch.manual_seed(config["semilla"])
    np.random.seed(config["semilla"])

    (x_tr, y_tr), (x_va, y_va), splits, stats, mask_t = preparar_tensores(var, config)
    ent_va, obj_va = splits["validacion"]

    # listón: ¿qué RMSE tiene la entrada gruesa sola en validación?
    mascara = mask_t.numpy().astype(bool)[0]
    base = ent_va[:, mascara] - obj_va[:, mascara]
    rmse_base = float(np.sqrt(np.nanmean(base ** 2)))
    print(f"[{var}] listón (interpolación pura, validación): RMSE = {rmse_base:.3f}")

    modelo = UNet(canales_entrada=3)
    optim = torch.optim.Adam(modelo.parameters(), lr=lr)
    cargador = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch,
                          shuffle=True)

    RUTA_MODELOS.mkdir(exist_ok=True)
    RUTA_RESULTADOS.mkdir(exist_ok=True)
    mejor_rmse, curva = np.inf, []
    for epoca in range(1, epocas + 1):
        modelo.train()
        perdida_epoca = 0.0
        for xb, yb in cargador:
            optim.zero_grad()
            pred = modelo(xb)
            perdida = perdida_l1_pinball(pred, yb, mask_t, peso_pinball)
            perdida.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            optim.step()
            perdida_epoca += float(perdida.detach()) * len(xb)
        perdida_epoca /= len(x_tr)

        rmse_val = rmse_reconstruccion(modelo, x_va, ent_va, obj_va, stats,
                                       mask_t, var)
        marca = ""
        if rmse_val < mejor_rmse:
            mejor_rmse = rmse_val
            torch.save({"estado": modelo.state_dict(), "stats": stats,
                        "config": {"var": var, "peso_pinball": peso_pinball}},
                       RUTA_MODELOS / f"unet_{var}.pt")
            marca = " ← mejor, guardado"
        curva.append({"época": epoca, "pérdida_train": perdida_epoca,
                      "RMSE_val": rmse_val, "RMSE_listón": rmse_base})
        print(f"[{var}] época {epoca:3d} | pérdida {perdida_epoca:.4f} | "
              f"RMSE val {rmse_val:.3f} (listón {rmse_base:.3f}){marca}")

    pd.DataFrame(curva).to_csv(
        RUTA_RESULTADOS / f"f3_unet_curva_{var}.csv", index=False)
    print(f"[{var}] mejor RMSE val: {mejor_rmse:.3f} "
          f"({100 * (rmse_base - mejor_rmse) / rmse_base:+.1f} % vs listón)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("variable", choices=["pr", "tasmax", "tasmin"])
    p.add_argument("--epocas", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--peso-pinball", type=float, default=0.0)
    args = p.parse_args()
    entrenar(args.variable, args.epocas, args.batch, args.lr, args.peso_pinball)
