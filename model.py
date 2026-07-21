"""El generador: un unico modelo condicional  mapa semantico -> CTA.

La variante de CoW y el sitio de oclusion NO son entradas aprendidas aparte:
son ediciones del mapa semantico. Por eso un solo modelo cubre cualquier
combinacion variante x oclusion, incluidas las que no aparecen en los datos.

SPADE difusion 3D, sin atencion -> totalmente convolucional, asi que se entrena
con parches de 96^3 y se muestrea sobre el volumen entero.
"""
import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.nets import SPADEDiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler

import common as C


def build(channels=(32, 64, 128, 192), spade_ch=64):
    # sin atencion -> totalmente convolucional: se entrena con parches de 64^3 y
    # se muestrea el volumen entero. Medido: 30.6 GB con bs=8 a 64^3 (el SPADE
    # proyecta los 41 canales del mapa en cada bloque, y eso es lo que pesa).
    return SPADEDiffusionModelUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        label_nc=C.N_CLASSES,                 # el mapa semantico entra por SPADE
        channels=channels,
        attention_levels=(False,) * len(channels),
        num_res_blocks=2,
        norm_num_groups=32,
        spade_intermediate_channels=spade_ch,
    )


def scheduler(num_train_timesteps=1000):
    # v-prediction, no epsilon. Con eps-prediction, a t alto el termino de senal
    # (sqrt(alpha_cumprod)=0.012) es despreciable y al modelo le sale gratis
    # predecir eps ~= x_t; entonces x0_pred ~= 0 y el muestreo desde ruido puro
    # devuelve un volumen plano (medido: contraste vaso-parenquima -1 HU).
    # Con v-prediction, el objetivo a t alto tiende a -x0: el modelo esta OBLIGADO
    # a predecir la imagen, que es justo lo que el muestreo necesita para arrancar.
    return DDPMScheduler(num_train_timesteps=num_train_timesteps,
                         schedule="scaled_linear_beta",
                         beta_start=0.0015, beta_end=0.0195,
                         prediction_type="v_prediction")


def onehot(sem, device):
    """(B,D,H,W) uint8 -> (B,41,D,H,W) float."""
    x = torch.as_tensor(sem, device=device, dtype=torch.long)
    return F.one_hot(x, C.N_CLASSES).permute(0, 4, 1, 2, 3).float()


@torch.no_grad()
def sample(net, sched, sem, device, steps=50, patch=128, overlap=0.5, bs=4,
           seed=None, progress=None):
    """Difusion en ventana deslizante sobre el volumen entero.

    En CADA paso de denoising se predice el ruido por parches y se promedia en
    los solapes con pesos gaussianos, manteniendo un unico x_t global. Asi el
    resultado es coherente en todo el volumen, no parche a parche.

    Sin gradientes cabe un parche mayor que en entrenamiento (la red es
    convolucional, no depende del tamano), y eso reduce mucho el numero de pasadas.
    """
    net.eval()
    shape = sem.shape                                   # (D,H,W)
    patch = min(patch, *shape)
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(int(seed))
    x = torch.randn((1, 1, *shape), device=device, generator=g)

    seg = onehot(sem[None], device)
    sched.set_timesteps(steps)

    # pesos gaussianos del parche (evitan costuras en los solapes)
    ax = [torch.exp(-0.5 * ((torch.arange(patch, device=device) - (patch - 1) / 2)
                            / (0.25 * patch)) ** 2) for _ in range(3)]
    w = (ax[0][:, None, None] * ax[1][None, :, None] * ax[2][None, None, :])[None, None]

    step = max(1, int(patch * (1 - overlap)))
    starts = [sorted({*range(0, dim - patch + 1, step), dim - patch}) for dim in shape]
    coords = [(z, y, xx) for z in starts[0] for y in starts[1] for xx in starts[2]]

    for i, t in enumerate(sched.timesteps):
        eps = torch.zeros_like(x)
        acc = torch.zeros_like(x)
        for k in range(0, len(coords), bs):
            chunk = coords[k:k + bs]
            xb = torch.cat([x[:, :, z:z + patch, y:y + patch, u:u + patch]
                            for z, y, u in chunk])
            sb = torch.cat([seg[:, :, z:z + patch, y:y + patch, u:u + patch]
                            for z, y, u in chunk])
            tt = torch.full((len(chunk),), int(t), device=device, dtype=torch.long)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                e = net(x=xb, timesteps=tt, seg=sb).float()
            for j, (z, y, u) in enumerate(chunk):
                eps[:, :, z:z + patch, y:y + patch, u:u + patch] += e[j:j + 1] * w
                acc[:, :, z:z + patch, y:y + patch, u:u + patch] += w
        eps = eps / acc.clamp_min(1e-8)
        x, _ = sched.step(eps, t, x, generator=g)
        if progress and (i % 10 == 0 or i == len(sched.timesteps) - 1):
            progress(i + 1, len(sched.timesteps))

    # la difusion vive en el espacio normalizado; se vuelve a [-1,1]
    out = C.denormalize(x[0, 0].float())
    return out.clamp(-1, 1).cpu().numpy()
