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
from monai.networks.schedulers import DDIMScheduler, DDPMScheduler

import common as C
import tejidos as T


def build(channels=(32, 64, 128, 192), spade_ch=64, label_nc=None):
    # sin atencion -> totalmente convolucional: se entrena con parches de 64^3 y
    # se muestrea el volumen entero. Medido: 30.6 GB con bs=8 a 64^3 (el SPADE
    # proyecta los 41 canales del mapa en cada bloque, y eso es lo que pesa).
    return SPADEDiffusionModelUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        label_nc=label_nc or T.N_CLASSES_IN,  # el mapa semantico entra por SPADE
        channels=channels,
        attention_levels=(False,) * len(channels),
        num_res_blocks=2,
        norm_num_groups=32,
        spade_intermediate_channels=spade_ch,
    )


BETAS = dict(schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195,
             prediction_type="v_prediction")

# CLIP_SAMPLE TIENE QUE IR A False. El scheduler por defecto recorta la muestra a
# [-1,1], pero la difusion vive en el espacio NORMALIZADO (common.normalize), y
# ahi los datos no caben en [-1,1]:
#     40 HU (parenquima) -> -0.12 |  232 HU (vaso) -> +0.65 |  900 HU (hueso) -> +3.34
# Con el recorte el techo de salida es to_hu(denormalize(1)) = 318 HU: el hueso es
# INALCANZABLE y el craneo no se genera. Medido: hueso p95 = 310 HU con recorte,
# 900 HU sin el.
CLIP = dict(clip_sample=False)


def scheduler(num_train_timesteps=1000):
    """El de ENTRENAMIENTO (add_noise / get_velocity). Para muestrear: `sampler()`.

    v-prediction, no epsilon. Con eps-prediction, a t alto el termino de senal
    (sqrt(alpha_cumprod)=0.012) es despreciable y al modelo le sale gratis
    predecir eps ~= x_t; entonces x0_pred ~= 0 y el muestreo desde ruido puro
    devuelve un volumen plano (medido: contraste vaso-parenquima -1 HU).
    Con v-prediction, el objetivo a t alto tiende a -x0: el modelo esta OBLIGADO
    a predecir la imagen, que es justo lo que el muestreo necesita para arrancar.
    """
    return DDPMScheduler(num_train_timesteps=num_train_timesteps, **BETAS, **CLIP)


def sampler(num_train_timesteps=1000):
    """El de MUESTREO: DDIM. No es intercambiable con DDPM aqui.

    DDPM diezmado a 50 pasos borra los vasos. Medido sobre el mismo checkpoint y
    el mismo parche, contraste vaso-parenquima (la CTA real da +259 HU):

        DDPM 50 pasos, con solapes    +24.4 HU
        DDPM 50 pasos, parche unico   +24.2 HU     <- no son los solapes
        DDIM 50 pasos                +217.1 HU     <- es el sampler
        DDIM 200 pasos               +218.x HU     <- 50 pasos ya bastan

    Los vasos son de 1-2 voxeles y el ruido que DDPM reinyecta en cada paso
    (el termino sigma_t * z, que DDIM no tiene) se los come. El modelo estaba
    bien desde la epoca 107; lo que fallaba era como se leia.
    """
    return DDIMScheduler(num_train_timesteps=num_train_timesteps, **BETAS, **CLIP)


def onehot(sem, device):
    """(B,D,H,W) uint8 -> (B,41,D,H,W) float. Solo las clases de `sem`."""
    x = torch.as_tensor(sem, device=device, dtype=torch.long)
    return F.one_hot(x, C.N_CLASSES).permute(0, 4, 1, 2, 3).float()


def labelmap(sem, tis, device):
    """(sem, tis) -> (B,44,D,H,W). El tensor de SPADE, MULTI-HOT, no one-hot.

    Los 41 primeros canales son el `sem` de siempre; los 3 ultimos son encefalo /
    extracraneal / LCR de `tejidos`. Un voxel de ventriculo enciende a la vez el
    canal 1 (`SOFT`) y el 43 (`LCR`): el canal que el modelo ya tenia aprendido
    sigue diciendo lo mismo, y el nuevo anade informacion encima. Es lo que hace
    que el warm-start sea exacto — con los canales nuevos a cero la entrada es
    identica a la de antes.

    `tis=None` -> los 3 canales nuevos a cero (equivale al modelo de la epoca 107).
    """
    x = onehot(sem, device)
    if tis is None:
        z = torch.zeros((x.shape[0], T.N_TIS - 1, *x.shape[2:]), device=device)
    else:
        t = torch.as_tensor(tis, device=device, dtype=torch.long)
        z = F.one_hot(t, T.N_TIS).permute(0, 4, 1, 2, 3).float()[:, 1:]  # fuera el 0
    return torch.cat([x, z], dim=1)


def warm_start(net, state, verbose=True):
    """Carga un checkpoint con MENOS canales de etiqueta, sin perder nada.

    Ampliar el espacio de clases cambia la forma de las convoluciones de entrada
    de SPADE (label_nc pasa de 41 a 44). En vez de entrenar de cero, se copian los
    41 canales aprendidos y los 3 nuevos se inicializan A CERO: la salida es
    entonces IDENTICA a la del checkpoint viejo, y el modelo aprende a usar los
    canales nuevos encima de lo que ya sabia.

    Se empareja por forma, no por nombre, para no depender de como llame MONAI a
    las capas SPADE: cualquier tensor que solo difiera en la dimension 1 (canales
    de entrada) se copia con relleno de ceros.
    """
    propio = net.state_dict()
    nuevos, iguales, saltados = [], 0, []
    for k, v in propio.items():
        if k not in state:
            saltados.append((k, "ausente")); continue
        w = state[k]
        if w.shape == v.shape:
            v.copy_(w); iguales += 1
        elif (len(w.shape) == len(v.shape) and w.shape[0] == v.shape[0]
              and w.shape[2:] == v.shape[2:] and v.shape[1] > w.shape[1]):
            v.zero_()
            v[:, :w.shape[1]].copy_(w)
            nuevos.append((k, tuple(w.shape), tuple(v.shape)))
        else:
            saltados.append((k, f"{tuple(w.shape)} vs {tuple(v.shape)}"))
    net.load_state_dict(propio)
    if verbose:
        print(f"warm start: {iguales} tensores copiados tal cual, "
              f"{len(nuevos)} ampliados con ceros, {len(saltados)} sin emparejar")
        for k, a, b in nuevos[:4]:
            print(f"    ampliado {k}  {a} -> {b}")
        if len(nuevos) > 4:
            print(f"    ... y {len(nuevos)-4} mas")
        for k, why in saltados[:6]:
            print(f"    SIN EMPAREJAR {k}  ({why})")
    return {"iguales": iguales, "ampliados": len(nuevos), "saltados": len(saltados)}


@torch.no_grad()
def sample(net, sched, sem, device, tis=None, steps=50, patch=128, overlap=0.5,
           bs=4, seed=None, progress=None, guidance=1.3):
    """Difusion en ventana deslizante sobre el volumen entero.

    En CADA paso de denoising se predice el ruido por parches y se promedia en
    los solapes con pesos gaussianos, manteniendo un unico x_t global. Asi el
    resultado es coherente en todo el volumen, no parche a parche.

    Sin gradientes cabe un parche mayor que en entrenamiento (la red es
    convolucional, no depende del tamano), y eso reduce mucho el numero de pasadas.

    ## guidance: la intensidad del vaso se fija AQUI, no en el entrenamiento

    El modelo coloca los vasos en su sitio pero a ~66% de la intensidad real: la
    difusion regresa a la media en estructuras de 1-2 voxeles y las suaviza. Subir
    el peso de la perdida (vessel_w 4->12, 100 epocas) no lo movio. Lo que SI lo
    mueve es amplificar la condicion en el muestreo, estilo classifier-free:

        e = e_sin + guidance * (e_con - e_sin)

    donde `e_sin` es la prediccion con el MISMO mapa pero con los vasos borrados
    (a SOFT). El modelo no se entreno con dropout de condicion, asi que `e_sin` no
    es un incondicional aprendido, pero funciona igual. Medido sobre un parche fijo
    (contraste vaso-parenquima, real +179 HU):

        guidance 1.0   +119 HU   66%   <- muestreo normal, vasos palidos
        guidance 1.3   +191 HU  107%   <- iguala el real, parenquima limpio
        guidance 1.5   +245 HU  137%   <- vasos ya mas brillantes que el real
        guidance 3.0   +860 HU  480%   <- reventados + parenquima granuloso

    1.3 es el punto donde la intensidad casa con la real sin ensuciar el fondo.
    guidance=1.0 desactiva el segundo forward (coste x1 en vez de x2).
    """
    net.eval()
    shape = sem.shape                                   # (D,H,W)
    patch = min(patch, *shape)
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(int(seed))
    x = torch.randn((1, 1, *shape), device=device, generator=g)

    tis_b = None if tis is None else tis[None]
    seg = labelmap(sem[None], tis_b, device)
    seg_u = None
    if guidance != 1.0:
        sem0 = sem.copy()
        sem0[(sem0 >= C.VESSEL0) & (sem0 <= C.VESSEL0 + C.N_VESSEL - 1)] = C.SOFT
        seg_u = labelmap(sem0[None], tis_b, device)     # "sin vasos" -> incondicional
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
                if seg_u is not None:
                    sbu = torch.cat([seg_u[:, :, z:z + patch, y:y + patch, u:u + patch]
                                     for z, y, u in chunk])
                    eu = net(x=xb, timesteps=tt, seg=sbu).float()
                    e = eu + guidance * (e - eu)
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
