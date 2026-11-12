"""U-Net residual para downscaling de la escala gruesa a 0.1°.

Diseño:
- La red no predice el campo completo: predice el residuo fino
  (objetivo − entrada), que es lo que la escala gruesa no sabe. La entrada
  gruesa re-interpolada ya lleva el 62-93 % de la varianza;
  aprender solo el residuo estabiliza el entrenamiento y garantiza que la
  red nunca "olvide" la señal de gran escala. (Misma lógica que CorrDiff.)
- Canales de entrada: [variable gruesa, elevación media, rugosidad] — la
  orografía es la covariable que explica la estructura fina en los Andes.
- Pérdida: L1 enmascarada (las 20 celdas sin obs no puntúan) + término
  cuantílico opcional (pinball) para reforzar la cola de la distribución.

El dominio es 50×50; la U-Net necesita lados divisibles por 8 -> se rellena
por reflexión a 64×64 y se recorta la salida.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BloqueConv(nn.Module):
    """Dos convoluciones 3×3 con BatchNorm y ReLU (bloque estándar de U-Net)."""

    def __init__(self, c_ent: int, c_sal: int):
        super().__init__()
        self.red = nn.Sequential(
            nn.Conv2d(c_ent, c_sal, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_sal), nn.ReLU(inplace=True),
            nn.Conv2d(c_sal, c_sal, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_sal), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.red(x)


class UNet(nn.Module):
    """U-Net pequeña (3 niveles, ~1.7 M parámetros con base=32)."""

    def __init__(self, canales_entrada: int = 3, base: int = 32):
        super().__init__()
        self.enc1 = BloqueConv(canales_entrada, base)
        self.enc2 = BloqueConv(base, base * 2)
        self.enc3 = BloqueConv(base * 2, base * 4)
        self.fondo = BloqueConv(base * 4, base * 8)
        self.sube3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = BloqueConv(base * 8, base * 4)
        self.sube2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = BloqueConv(base * 4, base * 2)
        self.sube1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = BloqueConv(base * 2, base)
        self.salida = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        alto, ancho = x.shape[-2:]
        # rellenar por reflexión hasta múltiplo de 8 (50 -> 64)
        pad_h, pad_w = (-alto) % 8, (-ancho) % 8
        pad_h += 8 if pad_h < 8 else 0  # margen extra para bordes limpios
        pad_w += 8 if pad_w < 8 else 0
        x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2,
                      pad_h // 2, pad_h - pad_h // 2), mode="reflect")

        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        f = self.fondo(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([self.sube3(f), e3], dim=1))
        d2 = self.dec2(torch.cat([self.sube2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.sube1(d2), e1], dim=1))
        y = self.salida(d1)

        # recortar el relleno
        i0, j0 = pad_h // 2, pad_w // 2
        return y[..., i0:i0 + alto, j0:j0 + ancho]


def perdida_l1_pinball(pred: torch.Tensor, objetivo: torch.Tensor,
                       mascara: torch.Tensor, peso_pinball: float = 0.0,
                       cuantil: float = 0.95) -> torch.Tensor:
    """L1 enmascarada + término pinball opcional para la cola alta.

    La pérdida pinball penaliza asimétricamente: si peso_pinball > 0, la red
    recibe presión extra para no subestimar los valores altos (extremos).
    """
    err = pred - objetivo
    # la máscara es (1, 50, 50): el divisor debe contar todas las celdas
    # válidas del lote (máscara × tamaño de lote), no las de un solo campo
    divisor = mascara.sum() * err.shape[0]
    l1 = (err.abs() * mascara).sum() / divisor
    if peso_pinball == 0.0:
        return l1
    pinball = torch.where(err < 0, -cuantil * err, (1 - cuantil) * err)
    pinball = (pinball * mascara).sum() / divisor
    return l1 + peso_pinball * pinball
