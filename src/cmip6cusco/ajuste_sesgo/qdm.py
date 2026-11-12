"""Quantile Delta Mapping (QDM, Cannon et al. 2015).

Qué hace: corrige la distribución del GCM contra las observaciones, cuantil
por cuantil y mes por mes, conservando la señal de cambio del modelo (por eso
"Delta"). El quantile mapping simple, en cambio, distorsiona la tendencia.

Validación:
- GCM diario interpolado (bilineal) a la grilla canónica 0.1° y enmascarado.
- Calibración 1981-2004 (24 años) / validación independiente 2005-2014.
- pr: multiplicativo (kind="*"), con jitter bajo 0.05 mm/d para el problema
  de los ceros (llovizna); tasmax/tasmin: aditivo (kind="+").
- 50 cuantiles, agrupados por mes (group="time.month").

Métricas de validación (dominio enmascarado, 2005-2014): sesgo de la media,
frecuencia de días húmedos (≥1 mm), percentiles extremos (p95 diario de P,
p05/p95 de T) y Perkins Skill Score de la distribución diaria completa.

Uso:  python -m cmip6cusco.ajuste_sesgo.qdm MPI-ESM1-2-HR
"""

import sys

import numpy as np
import pandas as pd
import xarray as xr
from xsdba import QuantileDeltaMapping
from xsdba.processing import jitter_under_thresh

from cmip6cusco.config import RAIZ_REPO, cargar_config
from cmip6cusco.utils.grilla import alinear_a_grilla, grilla_canonica

RUTA_INTERIM = RAIZ_REPO / "data" / "interim"
RUTA_RESULTADOS = RAIZ_REPO / "resultados"
RUTA_QDM = RUTA_INTERIM / "cmip6_qdm"   # proyecciones corregidas

PERIODO_CAL = slice("1981", "2004")
PERIODO_VAL = slice("2005", "2014")
PERIODO_HIST = slice("1981", "2014")    # calibración del QDM para las proyecciones

ARCHIVOS_OBS = {
    "pr": ("piscop_v3_prec_diaria_cusco.nc", "precipitation", "mm/d"),
    "tasmax": ("piscot_v12_tmax_diaria_cusco.nc", "tmax", "degC"),
    "tasmin": ("piscot_v12_tmin_diaria_cusco.nc", "tmin", "degC"),
}


def _cargar_obs(var: str, config, mascara) -> xr.DataArray:
    archivo, nombre, unidades = ARCHIVOS_OBS[var]
    da = xr.open_dataset(RUTA_INTERIM / archivo)[nombre]
    da = alinear_a_grilla(da, config).where(mascara == 1)
    da.attrs["units"] = unidades
    return da.sel(time=slice("1981", "2014")).astype("float32")


def _cargar_gcm(gcm: str, var: str, config, mascara,
                experimento: str = "historical") -> xr.DataArray:
    """Diario del GCM interpolado a la grilla canónica y enmascarado."""
    ruta = RUTA_INTERIM / "cmip6_diario" / f"{gcm}_{var}_{experimento}.nc"
    if not ruta.exists():
        raise FileNotFoundError(
            f"Falta {ruta.name}: correr primero "
            f"'python -m cmip6cusco.datos.cmip6_diario {gcm} {experimento}'")
    da = xr.open_dataarray(ruta)
    lats, lons = grilla_canonica(config)
    da = da.interp(lat=lats, lon=lons, method="linear").astype("float32")
    da = da.where(mascara == 1)
    da.attrs["units"] = "mm/d" if var == "pr" else "degC"
    return da


def _armonizar_tiempo(gcm_da: xr.DataArray,
                      obs: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Deja GCM y obs sobre el mismo eje de tiempo (exigencia de xsdba).

    Dos desajustes reales: (1) los GCM estampan el día a las 12:00 (punto
    medio) y las obs a las 00:00 -> se lleva todo a las 00:00 con floor('D');
    (2) algunos GCM usan calendarios sin bisiestos (noleap) o de 360 días ->
    se convierten las obs al calendario del GCM (nunca al revés: inventar
    días en el GCM sería fabricar datos).
    """
    gcm_da = gcm_da.assign_coords(time=gcm_da.indexes["time"].floor("D"))
    obs = obs.assign_coords(time=obs.indexes["time"].floor("D"))

    calendario = gcm_da.time.dt.calendar
    if calendario not in ("standard", "gregorian", "proleptic_gregorian"):
        alineacion = "year" if "360" in calendario else "date"
        obs = obs.convert_calendar(calendario, align_on=alineacion)

    comun = np.intersect1d(gcm_da.time.values, obs.time.values)
    if len(comun) < 0.99 * min(gcm_da.sizes["time"], obs.sizes["time"]):
        raise ValueError(
            f"Solape temporal sospechosamente bajo ({len(comun)} días) entre "
            f"GCM (cal. {calendario}) y obs: revisar antes de continuar.")
    return gcm_da.sel(time=comun), obs.sel(time=comun)


def pss(a: np.ndarray, b: np.ndarray, bins: np.ndarray) -> float:
    """Perkins Skill Score: solape de los histogramas (1 = distribuciones idénticas)."""
    fa, _ = np.histogram(a[np.isfinite(a)], bins=bins, density=False)
    fb, _ = np.histogram(b[np.isfinite(b)], bins=bins, density=False)
    fa = fa / fa.sum()
    fb = fb / fb.sum()
    return float(np.minimum(fa, fb).sum())


def _metricas(var: str, obs: np.ndarray, mod: np.ndarray) -> dict:
    """Métricas de validación agrupadas (valores diarios apilados del dominio)."""
    obs, mod = obs[np.isfinite(obs)], mod[np.isfinite(mod)]
    if var == "pr":
        bins = np.concatenate([np.arange(0, 60, 1), [np.inf]])
        return {
            "media (mm/d)": (obs.mean(), mod.mean()),
            "frec. húmedos ≥1mm (%)": (100 * (obs >= 1).mean(), 100 * (mod >= 1).mean()),
            "p95 (mm/d)": (np.quantile(obs, 0.95), np.quantile(mod, 0.95)),
            "p99 (mm/d)": (np.quantile(obs, 0.99), np.quantile(mod, 0.99)),
            "PSS": (1.0, pss(obs, mod, bins)),
        }
    bins = np.arange(-15, 40, 0.5)
    return {
        "media (°C)": (obs.mean(), mod.mean()),
        "p05 (°C)": (np.quantile(obs, 0.05), np.quantile(mod, 0.05)),
        "p95 (°C)": (np.quantile(obs, 0.95), np.quantile(mod, 0.95)),
        "PSS": (1.0, pss(obs, mod, bins)),
    }


def ajustar_gcm(gcm: str) -> pd.DataFrame | None:
    config = cargar_config()
    salida = RUTA_RESULTADOS / f"f2_qdm_validacion_{gcm}.csv"
    if salida.exists():
        print(f"Ya existe {salida.name}; nada que hacer.")
        return None
    np.random.seed(config["semilla"])  # jitter de pr reproducible
    mascara = xr.open_dataarray(RUTA_INTERIM / "mascara_comun.nc")
    filas = []

    for var in ["pr", "tasmax", "tasmin"]:
        print(f"{var}: cargando obs y GCM...")
        try:
            gcm_da = _cargar_gcm(gcm, var, config, mascara)
        except FileNotFoundError as e:
            print(f"   OMITIDO {var}: {e}")
            continue
        obs = _cargar_obs(var, config, mascara)
        gcm_da, obs = _armonizar_tiempo(gcm_da, obs)

        ref = obs.sel(time=PERIODO_CAL)
        hist = gcm_da.sel(time=PERIODO_CAL)
        sim_val = gcm_da.sel(time=PERIODO_VAL)
        obs_val = obs.sel(time=PERIODO_VAL)

        kind = "*" if var == "pr" else "+"
        if var == "pr":  # jitter: rompe el empate de los ceros para el mapeo
            ref = jitter_under_thresh(ref, thresh="0.05 mm/d")
            hist = jitter_under_thresh(hist, thresh="0.05 mm/d")
            sim_val = jitter_under_thresh(sim_val, thresh="0.05 mm/d")

        print(f"   entrenando QDM (50 cuantiles × 12 meses, kind '{kind}')...")
        qdm = QuantileDeltaMapping.train(
            ref, hist, nquantiles=50, group="time.month", kind=kind)
        ajustado_val = qdm.adjust(sim_val, interp="linear")

        print("   métricas de validación (2005-2014)...")
        m_crudo = _metricas(var, obs_val.values.ravel(), sim_val.values.ravel())
        m_qdm = _metricas(var, obs_val.values.ravel(), ajustado_val.values.ravel())
        for metrica, (v_obs, v_crudo) in m_crudo.items():
            filas.append({
                "variable": var, "métrica": metrica, "obs": v_obs,
                "GCM crudo": v_crudo, "GCM+QDM": m_qdm[metrica][1],
            })

    tabla = pd.DataFrame(filas).set_index(["variable", "métrica"])
    RUTA_RESULTADOS.mkdir(exist_ok=True)
    tabla.round(3).to_csv(salida)
    print(f"\nGuardado en {salida}\n")
    print(tabla.round(2).to_string())
    return tabla


def proyectar_gcm(gcm: str, ssp: str, sobrescribir: bool = False) -> None:
    """Aplica el QDM (calibrado en 1981-2014) a las proyecciones SSP.

    Produce el GCM corregido en la grilla 0.1° para 2015-2100, que es la
    entrada del downscaling. A diferencia de ajustar_gcm(), que solo valida,
    aquí se escriben los campos corregidos.

    Trend-preserving (Cannon 2015): qdm.adjust() ubica cada día futuro en la
    distribución del propio periodo futuro, de modo que la señal de cambio del
    modelo (más cálido, cambios en extremos de P) no se borra; solo se corrige el
    sesgo de la distribución cuantil por cuantil.
    """
    config = cargar_config()
    np.random.seed(config["semilla"])  # jitter de pr reproducible
    mascara = xr.open_dataarray(RUTA_INTERIM / "mascara_comun.nc")
    RUTA_QDM.mkdir(parents=True, exist_ok=True)

    for var in config["variables"]:
        salida = RUTA_QDM / f"{gcm}_{var}_{ssp}.nc"
        if salida.exists() and not sobrescribir:
            print(f"   ya existe {salida.name}; omito")
            continue

        try:
            hist = _cargar_gcm(gcm, var, config, mascara, "historical")
            fut = _cargar_gcm(gcm, var, config, mascara, ssp)
        except FileNotFoundError as e:
            print(f"   OMITIDO {var}/{ssp}: {e}")
            continue

        obs = _cargar_obs(var, config, mascara)
        # el par de calibración exige el mismo eje de tiempo (obs y GCM histórico);
        # al futuro solo se le lleva la marca al inicio del día (no solapa obs)
        hist, obs = _armonizar_tiempo(hist, obs)
        fut = fut.assign_coords(time=fut.indexes["time"].floor("D"))

        ref = obs.sel(time=PERIODO_HIST)
        hist = hist.sel(time=PERIODO_HIST)
        kind = "*" if var == "pr" else "+"
        if var == "pr":
            ref = jitter_under_thresh(ref, thresh="0.05 mm/d")
            hist = jitter_under_thresh(hist, thresh="0.05 mm/d")
            fut = jitter_under_thresh(fut, thresh="0.05 mm/d")

        print(f"{var}/{ssp}: QDM (50 cuantiles × 12 meses, kind '{kind}')"
              f" entrenado 1981-2014, corrigiendo 2015-2100...")
        qdm = QuantileDeltaMapping.train(
            ref, hist, nquantiles=50, group="time.month", kind=kind)
        corr = qdm.adjust(fut, interp="linear")
        if var == "pr":
            corr = corr.clip(min=0)

        corr.name = var
        corr.attrs.update(
            units="mm/d" if var == "pr" else "degC", gcm=gcm, escenario=ssp,
            metodo="QDM 50 cuantiles mensual", ref_entrenamiento="1981-2014",
            trend_preserving="Cannon et al. 2015")
        corr.to_netcdf(salida, encoding={var: {"zlib": True, "complevel": 4}})
        mb = salida.stat().st_size / 1e6
        # cordura: fin de siglo (2071-2100) vs obs — chequea que la señal de
        # cambio se conserve. Se usa np.nanmean y no xr.mean(): la reducción
        # de xarray (bottleneck) sobre float32 gigante pierde precisión con
        # suma ingenua; numpy usa suma pareada y da el valor fiel.
        m_obs = float(np.nanmean(obs.sel(time=PERIODO_HIST).values))
        m_fin = float(np.nanmean(corr.sel(time=slice("2071", "2100")).values))
        signo = (f"P fin-siglo {m_fin:.2f} vs obs {m_obs:.2f} mm/d" if var == "pr"
                 else f"ΔT fin-siglo−obs {m_fin - m_obs:+.2f} °C (obs {m_obs:.2f})")
        print(f"   OK {salida.name}: {dict(corr.sizes)} | {mb:.0f} MB | {signo}")


def corregir_periodo(gcm: str, var: str, experimento: str, config, mascara,
                     anios: tuple[int, int] | None = None) -> xr.DataArray:
    """Devuelve el GCM corregido por QDM (calibrado en 1981-2014) sin escribirlo.

    Es el mismo QDM de proyectar_gcm. Sirve para el baseline histórico de los
    índices (experimento='historical', anios=REF), de modo que la U-Net tenga
    un baseline calculado igual que el futuro. Ajustar el histórico da ~obs.
    """
    np.random.seed(config["semilla"])  # jitter de pr reproducible
    hist = _cargar_gcm(gcm, var, config, mascara, "historical")
    sim = _cargar_gcm(gcm, var, config, mascara, experimento)
    obs = _cargar_obs(var, config, mascara)
    hist, obs = _armonizar_tiempo(hist, obs)
    sim = sim.assign_coords(time=sim.indexes["time"].floor("D"))

    ref = obs.sel(time=PERIODO_HIST)
    hist = hist.sel(time=PERIODO_HIST)
    kind = "*" if var == "pr" else "+"
    if var == "pr":
        ref = jitter_under_thresh(ref, thresh="0.05 mm/d")
        hist = jitter_under_thresh(hist, thresh="0.05 mm/d")
        sim = jitter_under_thresh(sim, thresh="0.05 mm/d")

    qdm = QuantileDeltaMapping.train(
        ref, hist, nquantiles=50, group="time.month", kind=kind)
    corr = qdm.adjust(sim, interp="linear")
    if var == "pr":
        corr = corr.clip(min=0)
    if anios is not None:
        corr = corr.sel(time=slice(str(anios[0]), str(anios[1])))
    corr.name = var
    corr.attrs["units"] = "mm/d" if var == "pr" else "degC"
    return corr


def resumen() -> pd.DataFrame:
    """Consolida las validaciones por GCM en una tabla comparativa (crudo vs QDM)."""
    filas = []
    for ruta in sorted(RUTA_RESULTADOS.glob("f2_qdm_validacion_*.csv")):
        gcm = ruta.stem.replace("f2_qdm_validacion_", "")
        t = pd.read_csv(ruta).set_index(["variable", "métrica"])

        def v(var, met, col):
            try:
                return float(t.loc[(var, met), col])
            except KeyError:
                return np.nan

        fila = {"GCM": gcm}
        for etiqueta, col in [("crudo", "GCM crudo"), ("QDM", "GCM+QDM")]:
            obs_m = v("pr", "media (mm/d)", "obs")
            obs_p99 = v("pr", "p99 (mm/d)", "obs")
            fila[f"sesgo P % ({etiqueta})"] = 100 * (v("pr", "media (mm/d)", col) - obs_m) / obs_m
            fila[f"err p99 % ({etiqueta})"] = 100 * (v("pr", "p99 (mm/d)", col) - obs_p99) / obs_p99
            fila[f"PSS pr ({etiqueta})"] = v("pr", "PSS", col)
            fila[f"sesgo Tmax ({etiqueta})"] = v("tasmax", "media (°C)", col) - v("tasmax", "media (°C)", "obs")
            fila[f"err p05 Tmin ({etiqueta})"] = v("tasmin", "p05 (°C)", col) - v("tasmin", "p05 (°C)", "obs")
        filas.append(fila)

    tabla = pd.DataFrame(filas).set_index("GCM")
    tabla.loc["MEDIA ENSAMBLE"] = tabla.mean()
    salida = RUTA_RESULTADOS / "f2_qdm_resumen.csv"
    tabla.round(2).to_csv(salida)
    print(f"Resumen guardado en {salida}\n")
    print(tabla.round(2).to_string())
    return tabla


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            "Uso:\n"
            "  python -m cmip6cusco.ajuste_sesgo.qdm <GCM|todos>          "
            "# validación: calibración 1981-2004, prueba 2005-2014\n"
            "  python -m cmip6cusco.ajuste_sesgo.qdm resumen             "
            "# consolida las validaciones\n"
            "  python -m cmip6cusco.ajuste_sesgo.qdm proyectar <GCM|todos> [ssp...] "
            "# corrige las proyecciones")
    objetivo = sys.argv[1]
    if objetivo == "resumen":
        resumen()
    elif objetivo == "proyectar":
        config = cargar_config()
        gcm_arg = sys.argv[2] if len(sys.argv) > 2 else "todos"
        gcms = config["gcms"] if gcm_arg == "todos" else [gcm_arg]
        ssps = sys.argv[3:] or config["escenarios"]
        for g in gcms:
            print(f"===== {g} =====")
            for ssp in ssps:
                proyectar_gcm(g, ssp)
    else:
        gcms = cargar_config()["gcms"] if objetivo == "todos" else [objetivo]
        for g in gcms:
            print(f"===== {g} =====")
            ajustar_gcm(g)
