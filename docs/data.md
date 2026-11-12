# Data sources

No raw data is stored in the repository. Each source is downloaded by a script in `src/cmip6cusco/datos/` into `data/raw/`, cropped to the domain (11–16 °S, 70–75 °W) into `data/interim/`, and the national file is deleted.

| Dataset | Use | Resolution | Period | Script |
|---|---|---|---|---|
| CMIP6, 10 GCMs (pr, tasmax, tasmin, daily and monthly) | Input to bias adjustment and downscaling | 0.7°–2.8° | 1981–2100 | `cmip6_nube.py`, `cmip6_diario.py` |
| PISCOp v3.0 (SENAMHI) | Observed precipitation | 0.1° | 1981–2025 | `observaciones.py piscop_v3` |
| PISCOt v1.2 (SENAMHI) | Observed Tmax and Tmin | 0.1° | 1981–2020 | `observaciones.py piscot_tmax`, `piscot_tmin` |
| RAIN4PE | Second precipitation reference | 0.1° | 1981–2015 | `observaciones.py rain4pe` |
| DEM of PISCOt v1.2 | Elevation and roughness for the networks | 0.01° | — | `observaciones.py` |
| BASD-CMIP6-PE | Published product used for comparison | 0.1° | 1850–2100 | `basd_cmip6_pe.py` |

## Access

- **CMIP6:** Pangeo catalogue on Google Cloud (`https://storage.googleapis.com/cmip6/pangeo-cmip6.json`), read as zarr with `intake-esm` and `xarray` without downloading whole files. Only the cropped domain is transferred. GCMs: CanESM5, UKESM1-0-LL, IPSL-CM6A-LR, CNRM-CM6-1, CNRM-ESM2-1, EC-Earth3, MPI-ESM1-2-HR, MRI-ESM2-0, MIROC6, GFDL-ESM4; scenarios SSP1-2.6, SSP3-7.0 and SSP5-8.5.
- **PISCOp v3.0:** Figshare, article 32411886 (`PISCOp_d.nc`, 1.5 GB). The time axis comes without CF metadata and is rebuilt from 1981-01-01, checking the expected length.
- **PISCOt v1.2:** Figshare, article 22712365 (Huerta et al. 2023), official 0.1° files (`tmax_daily_1981_2020_010.nc`, `tmin_daily_1981_2020_010.nc`, about 0.65 GB each).
- **RAIN4PE:** Fernandez-Palomino et al. (2022), GFZ Data Services, [doi:10.5880/pik.2020.010](https://doi.org/10.5880/pik.2020.010).
- **BASD-CMIP6-PE:** Fernandez-Palomino et al. (2024), GFZ Data Services, [doi:10.5880/pik.2023.001](https://doi.org/10.5880/pik.2023.001). Files come by decade (about 240 MB each) and are processed one at a time.

## Conventions

- NetCDF with dimensions `(time, lat, lon)` and latitude in ascending order.
- Units: pr in mm/day, tasmax and tasmin in °C.
- GCM calendars without leap days or with 360 days: observations are converted to the GCM calendar for calibration, never the other way round.
- The 0.1° products do not share exact cell centres (PISCOt is shifted by up to 0.025°); `utils/grilla.py` aligns them to one grid by nearest neighbour without interpolating values.

## Running on Windows

- Slow reads from the Pangeo catalogue ("Could not determine bucket type … DNS query cancelled") come from the gRPC DNS resolver. Set `GRPC_DNS_RESOLVER=native` before running the download scripts (in PowerShell: `$env:GRPC_DNS_RESOLVER="native"`).
- "OMP Error #15 … libiomp5md.dll already initialized" when PyTorch and NumPy share a process: set `KMP_DUPLICATE_LIB_OK=TRUE`.
- xESMF is not available on conda-forge for Windows; regridding uses `xarray-regrid`.
