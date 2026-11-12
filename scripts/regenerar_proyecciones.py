"""Regenera el QDM de las proyecciones, los índices ETCCDI, el ensamble y
las figuras, por partes. Si se corta, retoma donde quedó.

Los QDM se guardan en `data/interim/cmip6_qdm/` y no se borran.

Cada paso corre en su propio subproceso para que la RAM (~6 GB) se libere
entre combinaciones. Si se corta, volver a lanzarlo: salta lo ya hecho
(QDM existentes y tablas de índices ya escritas). Un QDM que no abre o está
incompleto se borra y se rehace.

Uso:  python scripts/regenerar_proyecciones.py
Log:  data/logs/regenerar_proyecciones.log
"""

import datetime as dt
import subprocess
import sys
import time

import xarray as xr

from cmip6cusco.config import RAIZ_REPO, cargar_config

RUTA_QDM = RAIZ_REPO / "data" / "interim" / "cmip6_qdm"
RUTA_DIARIO = RAIZ_REPO / "data" / "interim" / "cmip6_diario"
RUTA_RES = RAIZ_REPO / "resultados"
LOG = RAIZ_REPO / "data" / "logs" / "regenerar_proyecciones.log"


def log(msg: str) -> None:
    linea = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(linea, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(linea + "\n")


def correr(args: list[str], etiqueta: str) -> bool:
    """Corre un paso en su propio subproceso y deja su salida en el log."""
    t0 = time.time()
    log(f"INICIO {etiqueta}")
    with LOG.open("a", encoding="utf-8") as f:
        r = subprocess.run([sys.executable, *args], cwd=RAIZ_REPO,
                           stdout=f, stderr=subprocess.STDOUT)
    minutos = (time.time() - t0) / 60
    estado = "OK" if r.returncode == 0 else f"FALLÓ (código {r.returncode})"
    log(f"{estado} {etiqueta} [{minutos:.1f} min]")
    return r.returncode == 0


def qdm_completo(ruta) -> bool:
    """El archivo abre y cubre el periodo futuro. El umbral es el caso más
    corto: UKESM1-0-LL (calendario de 360 días) en ssp585 llega a 2099,
    85 años = 30 600 días; los demás tienen más."""
    try:
        with xr.open_dataarray(ruta) as da:
            return da.sizes.get("time", 0) >= 85 * 360
    except Exception:
        return False


def limpiar_qdm_rotos(gcm: str, ssp: str, variables: list[str]) -> None:
    for v in variables:
        ruta = RUTA_QDM / f"{gcm}_{v}_{ssp}.nc"
        if ruta.exists() and not qdm_completo(ruta):
            log(f"   QDM incompleto, se borra y se rehace: {ruta.name}")
            ruta.unlink()


def mantener_despierta() -> None:
    """En Windows, pide que la PC no entre en suspensión mientras corre este
    proceso (SetThreadExecutionState). Se libera sola al terminar; no cambia
    la configuración de energía."""
    if sys.platform == "win32":
        import ctypes
        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            es_continuous | es_system_required)


def main() -> None:
    mantener_despierta()
    cfg = cargar_config()
    combos = [(g, s) for g in cfg["gcms"] for s in cfg["escenarios"]]
    log(f"=== Regeneración de proyecciones: {len(combos)} combinaciones ===")
    fallas = []

    for i, (g, s) in enumerate(combos, 1):
        csv = RUTA_RES / f"f5_etccdi_{g}_{s}.csv"
        if csv.exists():
            log(f"[{i}/{len(combos)}] {g} {s}: ya hecho, se salta")
            continue
        # dos intentos: con ~6 GB de RAM, las fallas por memoria dependen de
        # qué más esté abierto en la PC y suelen no repetirse
        for intento in (1, 2):
            limpiar_qdm_rotos(g, s, cfg["variables"])
            ok = correr(["-m", "cmip6cusco.ajuste_sesgo.qdm", "proyectar", g, s],
                        f"[{i}/{len(combos)}] QDM {g} {s}")
            ok = ok and correr(["-m", "cmip6cusco.indices.proyecciones", g, s],
                               f"[{i}/{len(combos)}] índices {g} {s}")
            if ok and csv.exists():
                break
            if intento == 1:
                log(f"   se reintenta {g} {s}")
        else:
            fallas.append(f"{g} {s}")

    if fallas:
        log(f"Combinaciones con falla ({len(fallas)}): {', '.join(fallas)}")
        log("Se detiene antes del ensamble: revisar el log y volver a lanzar.")
        return

    pasos = [
        (["-m", "cmip6cusco.indices.ensamble"], "ensamble"),
        (["-m", "figuras.f5_mapas_cambio", "calcular"], "mapas: calcular"),
        (["-m", "figuras.f5_mapas_cambio", "dibujar"], "mapas: dibujar"),
        (["-m", "figuras.f5_cambios_ensamble", "2100"], "figura de cambios 2100"),
        (["-m", "figuras.f5_cambios_ensamble", "2050"], "figura de cambios 2050"),
        (["-m", "figuras.f5_demo_difusion", "--recompute"], "demo de la difusión"),
    ]
    for args, etiqueta in pasos:
        if not correr(args, etiqueta):
            fallas.append(etiqueta)
    log("=== FIN ===" + (f" con fallas: {', '.join(fallas)}" if fallas else " sin fallas"))


if __name__ == "__main__":
    main()
