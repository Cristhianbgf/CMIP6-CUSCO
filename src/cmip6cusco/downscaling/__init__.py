"""Downscaling con deep learning.

- dataset: pares de entrenamiento (obs agregada a ~1° y obs a 0.1°) y
  covariables del DEM.
- unet: U-Net determinística (pérdida L1 + cuantílica).
- difusion: difusión residual tipo CorrDiff; la U-Net da la media y la
  difusión genera los residuos.
- entrenar, entrenar_difusion: entrenamiento con semilla fija.
- evaluar, evaluar_difusion: métricas en el periodo de prueba.
- aplicar: U-Net sobre las proyecciones corregidas con QDM.
"""
