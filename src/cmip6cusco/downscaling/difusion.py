"""Difusión condicional sobre el residuo de la U-Net.

Como en CorrDiff (Mardani et al. 2025): la U-Net predice la media del campo
fino y le falta la variabilidad, por eso subestima el p99 (-9 % medido).
Este módulo aprende la distribución del residuo
(objetivo − U-Net) con un modelo de difusión condicionado en la entrada
gruesa, la orografía y la propia predicción de la U-Net. En aplicación se
generan n muestras por día: un ensamble probabilístico cuya cola debería
recuperar lo que la media suaviza.

Implementación: DDPM estándar (Ho et al. 2020) con programa coseno
(Nichol & Dhariwal 2021) y muestreo DDIM determinístico (Song et al. 2021)
para inferencia rápida. Denoiser: U-Net pequeña con incrustación sinusoidal
del paso de tiempo inyectada en cada bloque.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------- programa de ruido -------------------------

class ProgramaRuido:
    """Programa coseno de varianzas del proceso de difusión (T pasos)."""

    def __init__(self, pasos: int = 400, s: float = 0.008):
        t = torch.linspace(0, pasos, pasos + 1)
        f = torch.cos((t / pasos + s) / (1 + s) * math.pi / 2) ** 2
        alfas_acum = (f / f[0]).clamp(1e-5, 1.0)
        self.pasos = pasos
        self.alfas_acum = alfas_acum[1:]                      # ᾱ_t, t=1..T
        self.raiz_alfas = self.alfas_acum.sqrt()
        self.raiz_1m_alfas = (1 - self.alfas_acum).sqrt()

    def ensuciar(self, x0: torch.Tensor, t: torch.Tensor, ruido: torch.Tensor):
        """q(x_t | x_0): mezcla señal y ruido según el paso t de cada muestra."""
        a = self.raiz_alfas[t][:, None, None, None]
        b = self.raiz_1m_alfas[t][:, None, None, None]
        return a * x0 + b * ruido


# --------------------------- denoiser U-Net ---------------------------

def incrustacion_sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Codifica el paso de difusión t como vector sinusoidal (como en DDPM)."""
    mitad = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(mitad, device=t.device) / mitad)
    ang = t[:, None].float() * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], dim=1)


class BloqueT(nn.Module):
    """Bloque conv doble que suma la incrustación de tiempo tras la 1ª conv."""

    def __init__(self, c_ent: int, c_sal: int, dim_t: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_ent, c_sal, 3, padding=1)
        self.conv2 = nn.Conv2d(c_sal, c_sal, 3, padding=1)
        self.norma1 = nn.GroupNorm(8, c_sal)
        self.norma2 = nn.GroupNorm(8, c_sal)
        self.proy_t = nn.Linear(dim_t, c_sal)

    def forward(self, x, emb_t):
        h = F.silu(self.norma1(self.conv1(x)))
        h = h + self.proy_t(emb_t)[:, :, None, None]
        return F.silu(self.norma2(self.conv2(h)))


class DenoiserUNet(nn.Module):
    """U-Net condicional: entrada = [x_t (1 canal) | condición (n canales)]."""

    def __init__(self, canales_condicion: int = 4, base: int = 32,
                 dim_t: int = 128):
        super().__init__()
        self.dim_t = dim_t
        self.mlp_t = nn.Sequential(nn.Linear(dim_t, dim_t), nn.SiLU(),
                                   nn.Linear(dim_t, dim_t))
        c_in = 1 + canales_condicion
        self.enc1 = BloqueT(c_in, base, dim_t)
        self.enc2 = BloqueT(base, base * 2, dim_t)
        self.enc3 = BloqueT(base * 2, base * 4, dim_t)
        self.fondo = BloqueT(base * 4, base * 8, dim_t)
        self.sube3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.dec3 = BloqueT(base * 8, base * 4, dim_t)
        self.sube2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = BloqueT(base * 4, base * 2, dim_t)
        self.sube1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = BloqueT(base * 2, base, dim_t)
        self.salida = nn.Conv2d(base, 1, 1)

    def forward(self, x_t, condicion, t):
        emb = self.mlp_t(incrustacion_sinusoidal(t, self.dim_t))
        x = torch.cat([x_t, condicion], dim=1)

        alto, ancho = x.shape[-2:]
        pad_h, pad_w = (-alto) % 8 + 8, (-ancho) % 8 + 8
        x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2,
                      pad_h // 2, pad_h - pad_h // 2), mode="reflect")

        e1 = self.enc1(x, emb)
        e2 = self.enc2(F.max_pool2d(e1, 2), emb)
        e3 = self.enc3(F.max_pool2d(e2, 2), emb)
        f = self.fondo(F.max_pool2d(e3, 2), emb)
        d3 = self.dec3(torch.cat([self.sube3(f), e3], 1), emb)
        d2 = self.dec2(torch.cat([self.sube2(d3), e2], 1), emb)
        d1 = self.dec1(torch.cat([self.sube1(d2), e1], 1), emb)
        y = self.salida(d1)

        i0, j0 = pad_h // 2, pad_w // 2
        return y[..., i0:i0 + alto, j0:j0 + ancho]


# ------------------------------ muestreo ------------------------------

@torch.no_grad()
def muestrear_ddim(modelo: DenoiserUNet, programa: ProgramaRuido,
                   condicion: torch.Tensor, n_pasos: int = 50,
                   semilla: int | None = None) -> torch.Tensor:
    """Genera residuos: parte de ruido puro y lo de-ruidiza en n_pasos DDIM."""
    if semilla is not None:
        torch.manual_seed(semilla)
    lote = condicion.shape[0]
    forma = (lote, 1, condicion.shape[-2], condicion.shape[-1])
    x = torch.randn(forma)

    indices = torch.linspace(programa.pasos - 1, 0, n_pasos).long()
    for i, t in enumerate(indices):
        t_lote = torch.full((lote,), int(t), dtype=torch.long)
        eps = modelo(x, condicion, t_lote)
        a_t = programa.alfas_acum[t]
        x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        # En los primeros pasos ᾱ~0 y la división amplifica ×100s
        # cualquier error del denoiser -> colas explotadas. Se recorta x0 al
        # rango plausible del residuo normalizado (±8σ), práctica estándar.
        x0 = x0.clamp(-8.0, 8.0)
        if i + 1 < len(indices):
            a_prev = programa.alfas_acum[indices[i + 1]]
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        else:
            x = x0
    return x
