"""Entrena el generador unico: mapa semantico -> CTA.

Un solo modelo para todo. La variante de CoW y el sitio de oclusion no son
entradas del modelo: son ediciones del mapa semantico que entra por SPADE. Por
eso generaliza a combinaciones variante x oclusion que no existen en los datos.

Datos (209 casos): TopAneu 58 (arbol completo, GT), TopCoW 125 (pseudo), ISLES 26
(CTA con oclusion real -> el modelo aprende como se ve de verdad un cut-off).

La entrada de SPADE es MULTI-HOT de 44 canales: los 41 de `sem` de siempre mas los
3 de `tejidos` (encefalo / extracraneal / LCR). Ver `model.labelmap`.

## Aumento con oclusiones

El modelo apenas habia visto combinaciones variante x oclusion. Con probabilidad
`--oclu_p` cada parche recibe una receta precalculada por `oclusiones.py`: se
apagan los segmentos distales SOBRE EL PARCHE (`aplicar_oclusion`), de modo que el
vaso deja de verse a partir del punto ocluido. NO se pinta ningun coagulo: la
oclusion es la ausencia de contraste distal, no el signo hiperdenso. La receta se
precalcula porque `cow_edit.occlude` cuesta 1-2 s; materializarla es barato.

Uso: python oclusiones.py --workers 4 --por-donante 6      # una vez, antes
     python train_gen.py --resume [--epochs 800] [--patch 96] [--bs 3]

NO correr los dos a la vez: es lo que estaba en marcha en los dos BSOD del
22/07/2026. Ver `vigilar.ps1`.

## Vigilar el contraste, no la val loss

La val loss de difusion NO dice si los vasos estan. El modelo llevaba desde la
epoca 107 pintandolos bien y nadie lo sabia porque se miraba solo la MSE. Cada
`--mon` epocas se muestrea de verdad un parche fijo, se mide el contraste
vaso-parenquima en HU y se guarda un PNG en `mon/`. Eso es lo que hay que mirar.

## No pasarse de VRAM: en Windows NO da OOM, se traga la RAM del sistema

Medido en la RTX 6000 Ada (48 GB), pico de `max_memory_allocated` y ms por
iteracion:

    parche 64^3  bs 6     21.7 GB       462 ms
    parche 64^3  bs 8     28.7 GB       614 ms
    parche 80^3  bs 4     28.3 GB       609 ms
    parche 96^3  bs 3     37.0 GB       808 ms     <- el maximo que cabe a 96^3
    parche 96^3  bs 6     73.1 GB    55 324 ms     <- DESBORDA

Los 96^3 con bs=6 piden 73 GB en una tarjeta de 48. En Linux eso seria un OOM
limpio; en Windows el driver WDDM deja a CUDA desbordar a RAM del sistema y tirar
por PCIe, asi que el entrenamiento ARRANCA y parece que va -- solo que 68 veces
mas lento (46 min por epoca) y machacando el driver. El equipo hizo BSOD
`HYPERVISOR_ERROR 0x00020001` dos veces el 22/07/2026 con esa configuracion.

Por eso `--bs` vale 3 por defecto y hay una comprobacion en la primera iteracion
que aborta si el pico se acerca a la VRAM fisica. No la quites.

## No reventar la GPU

Muestrear DENTRO del bucle es exactamente lo que tumbo el entrenamiento en la
epoca 107 con `CUDA illegal memory access`. Se hace secuencial y en el mismo
proceso -- nunca en paralelo con el paso de entrenamiento -- con `eval()`,
`no_grad()`, fuera del contexto de autocast del entrenamiento y liberando cache
antes y despues. Ver `monitorizar`.
"""
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import common as C
import model as M
import tejidos as T
import hybrid as H
import oclusiones as O

ROOT = Path(__file__).parent
DEV = "cuda"


def load_all():
    data = []
    for ds in ("topaneu", "topcow", "isles"):
        for f in sorted((ROOT / "sem" / ds).glob("*.npz")):
            s = np.load(f)
            if "tis" not in s:
                raise SystemExit(f"{f.name} no tiene 'tis'. Ejecuta antes: "
                                 f"python tejidos.py --workers 6")
            p = np.load(ROOT / "prep" / ds / f.name)
            data.append((f.stem, ds, p["img"].astype(np.float32), s["sem"], s["tis"],
                         O.cargar(f.stem)))
    return data


def sample_patch(img, sem, tis, P, rng, con_lo=False):
    """70% de los parches centrados en un vaso: son <1% del volumen pero son
    lo unico que importa aqui."""
    ves = (sem >= C.VESSEL0) & (sem <= C.VESSEL0 + C.N_VESSEL - 1)
    if rng.random() < 0.70 and ves.any():
        idx = np.array(np.nonzero(ves))
        c = idx[:, rng.integers(idx.shape[1])]
    else:
        c = np.array([rng.integers(s) for s in img.shape])
    lo = np.clip(c - P // 2, 0, np.array(img.shape) - P)
    s = tuple(slice(l, l + P) for l in lo)
    if con_lo:
        return img[s], sem[s], tis[s], lo
    return img[s], sem[s], tis[s]


def aplicar_oclusion(pi, ps, pt, lo, shape, receta, rng):
    """Aplica una receta de `oclusiones.py` SOBRE EL PARCHE ya recortado.

    La receta se precalcula una vez por donante porque `cow_edit.occlude` cuesta
    1-2 s (geodesica sobre el vaso) y no se puede pagar por parche. Aqui solo se
    materializa: apagar los segmentos distales, que es barato.

    Una oclusion, para este generador, es simplemente que el vaso DEJA DE VERSE a
    partir del punto ocluido. NO se pinta ningun coagulo: no queremos el signo
    hiperdenso, solo la ausencia de contraste distal. (Se probo pintar el trombo a
    58 HU y el modelo ni lo aprendia ni hacia falta; ver el historico en git.)

    Devuelve el parche sin tocar si la oclusion cae entera fuera de el.
    """
    apagados = np.asarray(receta["apagados"], np.uint8)

    ves = np.zeros(ps.shape, np.uint8)
    m = (ps >= C.VESSEL0) & (ps <= C.VESSEL0 + C.N_VESSEL - 1)
    ves[m] = ps[m] - C.VESSEL0 + 1
    lost = (ves > 0) & np.isin(ves, apagados)

    if not lost.any():
        return pi, ps, pt

    ves2 = ves.copy()
    ves2[lost] = 0
    hu = H.render(C.to_hu(pi), ves, ves2, None, rng=rng)   # thr=None: sin coagulo
    pi = C.to_unit(hu).astype(np.float32)

    ps = ps.copy()
    ps[lost] = C.SOFT          # un vaso sin contraste ya no es un vaso
    return pi, ps, pt


def augment(img, sem, tis, rng):
    if rng.random() < 0.5:
        img = img[::-1].copy()
        sem = sem[::-1].copy()
        tis = tis[::-1].copy()
        v = (sem >= C.VESSEL0) & (sem <= C.VESSEL0 + C.N_VESSEL - 1)
        sem[v] = C.VESSEL0 + C.LR_SWAP[sem[v] - C.VESSEL0 + 1] - 1   # swap R<->L
    if rng.random() < 0.3:                       # variacion del realce de contraste
        img = img * rng.normal(1.0, 0.05) + rng.normal(0.0, 0.03)
    return img.astype(np.float32), sem.astype(np.uint8), tis.astype(np.uint8)


# --- monitorizacion -----------------------------------------------------------

def _contraste(hu, sem):
    """mediana(vaso) - mediana(parenquima en un anillo de 2-6 vox alrededor)."""
    from scipy import ndimage as ndi
    ves = (sem >= C.VESSEL0) & (sem <= C.VESSEL0 + C.N_VESSEL - 1)
    if ves.sum() < 50:
        return float("nan")
    anillo = ndi.binary_dilation(ves, iterations=6) & ~ndi.binary_dilation(ves, iterations=2)
    if anillo.sum() < 50:
        return float("nan")
    return float(np.median(hu[ves]) - np.median(hu[anillo]))


@torch.no_grad()
def monitorizar(net, ep, ref, out_dir, steps=50):
    """Muestrea un parche fijo, mide el contraste y guarda un PNG.

    Secuencial y en el mismo proceso: solapar esto con el entrenamiento es lo que
    provoca el `CUDA illegal memory access`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img, sem, tis = ref
    P = sem.shape[0]
    torch.cuda.empty_cache()
    modo = net.training
    net.eval()
    try:
        sch = M.sampler()
        sch.set_timesteps(steps)
        g = torch.Generator(device=DEV); g.manual_seed(0)
        x = torch.randn((1, 1, P, P, P), device=DEV, generator=g)
        seg = M.labelmap(sem[None], tis[None], DEV)
        for t in sch.timesteps:
            tt = torch.full((1,), int(t), device=DEV, dtype=torch.long)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                e = net(x=x, timesteps=tt, seg=seg).float()
            x, _ = sch.step(e, t, x, generator=g)
        hu = C.to_hu(C.denormalize(x[0, 0].float()).clamp(-1, 1).cpu().numpy())
    finally:
        net.train(modo)
        torch.cuda.empty_cache()

    cg, cr = _contraste(hu, sem), _contraste(C.to_hu(img), sem)
    z = P // 2
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.9))
    for a, (v, t) in zip(axs, ((hu, f"generado  ep {ep}"), (C.to_hu(img), "real"))):
        a.imshow(np.rot90(v[:, :, z]), cmap="gray", vmin=-50, vmax=250)
        a.set_title(t, fontsize=10)
    axs[2].imshow(np.rot90(np.clip(hu, 0, 400)[:, :, z] - np.clip(C.to_hu(img), 0, 400)[:, :, z]),
                  cmap="coolwarm", vmin=-200, vmax=200)
    axs[2].set_title("diferencia", fontsize=10)
    for a in axs:
        a.axis("off")
    fig.suptitle(f"epoca {ep} — contraste vaso-parenquima {cg:+.0f} HU "
                 f"(real {cr:+.0f} HU)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"ep{ep:04d}.png", dpi=95)
    plt.close(fig)
    return cg, cr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--bs", type=int, default=3,
                    help="OJO: con patch 96 el limite son 3 en una tarjeta de 48 GB. "
                         "Ver la tabla en el docstring")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--vessel_w", type=float, default=4.0,
                    help="peso extra de la perdida dentro de los vasos")
    ap.add_argument("--oclu_p", type=float, default=0.35,
                    help="probabilidad de aplicar una receta de oclusion al parche "
                         "(0 = desactivado). Requiere haber corrido oclusiones.py")
    ap.add_argument("--mon", type=int, default=10,
                    help="cada cuantas epocas muestrear y guardar PNG en mon/")
    ap.add_argument("--mon_patch", type=int, default=96)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--from", dest="desde", default="gen_last.pt",
                    help="checkpoint del que reanudar (dentro de ckpt/)")
    ap.add_argument("--warmup", type=int, default=500,
                    help="pasos de rampa lineal del LR. IMPRESCINDIBLE al reanudar "
                         "con los momentos de AdamW reiniciados (ver abajo)")
    a = ap.parse_args()
    (ROOT / "ckpt").mkdir(exist_ok=True)
    mon_dir = ROOT / "mon"; mon_dir.mkdir(exist_ok=True)

    data = load_all()
    rng = np.random.default_rng(0)
    val = {d[0] for d in data[::12]}
    tr = [d for d in data if d[0] not in val]
    va = [d for d in data if d[0] in val]
    from collections import Counter
    con_rec = sum(1 for d in tr if d[5])
    print(f"train {len(tr)} | val {len(va)}  {Counter(d[1] for d in tr)}", flush=True)
    print(f"oclusiones: {con_rec}/{len(tr)} donantes con receta, p={a.oclu_p}"
          f"{'  (corre oclusiones.py)' if not con_rec else ''}", flush=True)

    net = M.build().to(DEV)
    sched = M.scheduler()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-6)
    lrs = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    scaler = torch.amp.GradScaler()
    ep0, best = 0, 1e9

    ck = ROOT / "ckpt" / "gen_last.pt"
    desde = ROOT / "ckpt" / a.desde
    warmup = 0
    if a.resume and desde.exists():
        st = torch.load(desde, map_location=DEV, weights_only=False)
        # El checkpoint de la epoca 107 tiene 41 canales de etiqueta y la red
        # ahora tiene 44. `warm_start` copia los 41 aprendidos y pone los 3
        # nuevos a cero: verificado bit a bit identico (max |dif| = 0.0 en fp64).
        info = M.warm_start(net, st["model"])
        # El estado de AdamW (exp_avg, exp_avg_sq) tiene la forma VIEJA de los 24
        # pesos ampliados, asi que no se puede reutilizar si hubo ampliacion: se
        # reinician los momentos. El scheduler si, que solo guarda escalares.
        if info["ampliados"] == 0 and "opt" in st:
            opt.load_state_dict(st["opt"])
        else:
            # MEDIDO, y cuesta caro descubrirlo: con los momentos a cero el PRIMER
            # paso de AdamW vale lr*sign(g) para los 40M de parametros a la vez, en
            # una direccion coherente. Seis pasos a lr=1e-4 bastaron para tirar el
            # contraste vaso-parenquima de +217 a +18 HU sin que la val loss lo
            # dejara claro. La rampa lo evita: los primeros pasos son ~0.
            warmup = a.warmup
            print(f"   ({info['ampliados']} tensores ampliados -> momentos de AdamW "
                  f"reiniciados; rampa de LR de {warmup} pasos)", flush=True)
        # OJO con el scheduler: su estado guarda el `T_max` con el que se creo. Si
        # una corrida corta lo dejo en T_max=110 y ahora se pide --epochs 800, al
        # cargarlo el coseno queda EN SU FINAL y el LR arranca en ~0 sin avisar.
        # Solo se reutiliza si el horizonte coincide; si no, se avanza a mano.
        if "lrs" in st and st["lrs"].get("T_max") == a.epochs:
            lrs.load_state_dict(st["lrs"])
        else:
            for _ in range(st["epoch"] + 1):
                lrs.step()
            print(f"   (horizonte distinto: scheduler avanzado a mano, "
                  f"lr {lrs.get_last_lr()[0]:.2e})", flush=True)
        ep0, best = st["epoch"] + 1, st.get("best", 1e9)
        print(f"reanudado desde {a.desde} en la epoca {ep0} (best {best:.4f})", flush=True)

    # parche de referencia FIJO para la monitorizacion: mismo caso, mismo sitio,
    # misma semilla en cada medida, para que la serie sea comparable.
    ref_case = va[0] if va else tr[0]
    r = np.random.default_rng(123)
    ref = sample_patch(ref_case[2], ref_case[3], ref_case[4], a.mon_patch, r)
    print(f"monitor: {ref_case[0]} parche {a.mon_patch}^3 -> {mon_dir}", flush=True)
    # El parche de referencia cambia si cambia el caso, el tamano o el contenido de
    # `sem`, y entonces el contraste "real" cambia con el -- la serie de un run NO es
    # comparable con la de otro. Ha pasado ya (unas corridas con real=272 HU y otras
    # con 179.2 en el mismo fichero). Por eso se anota el caso y, sobre todo, la
    # FRACCION del real, que es lo unico comparable entre corridas.
    hist = ROOT / "mon" / "contraste.csv"
    if not hist.exists():
        hist.write_text("epoca,caso,loss,val,contraste_gen,contraste_real,frac\n")

    nT = sched.num_train_timesteps
    paso = 0
    for ep in range(ep0, a.epochs):
        net.train()
        tl, t0 = 0.0, time.time()
        # CosineAnnealingLR calcula el LR siguiente de forma RECURSIVA a partir del
        # lr actual del grupo, asi que la rampa no puede dejarlo mutado al llegar a
        # `lrs.step()`: se guarda el valor del scheduler y se restaura al final.
        lr_ep = [g["lr"] for g in opt.param_groups]
        for _ in range(a.iters):
            if warmup:
                fw = min(1.0, (paso + 1) / warmup)
                for gp, b in zip(opt.param_groups, lr_ep):
                    gp["lr"] = b * fw
            paso += 1
            xs, ss, ts = [], [], []
            for _ in range(a.bs):
                _, _, img, sem, tis, rec = tr[rng.integers(len(tr))]
                pi, ps, pt, lo = sample_patch(img, sem, tis, a.patch, rng, con_lo=True)
                if rec and rng.random() < a.oclu_p:
                    m = rec[rng.integers(len(rec))]
                    pi, ps, pt = aplicar_oclusion(pi, ps, pt, lo, img.shape, m, rng)
                pi, ps, pt = augment(pi, ps, pt, rng)
                xs.append(pi[None]); ss.append(ps); ts.append(pt)
            x0 = C.normalize(torch.from_numpy(np.stack(xs)).to(DEV))
            sem_t = torch.from_numpy(np.stack(ss).astype(np.int64)).to(DEV)
            seg = M.labelmap(np.stack(ss), np.stack(ts), DEV)

            noise = torch.randn_like(x0)
            t = torch.randint(0, nT, (x0.shape[0],), device=DEV)
            xt = sched.add_noise(x0, noise, t)
            target = sched.get_velocity(x0, noise, t)     # v-prediction

            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = net(x=xt, timesteps=t, seg=seg)
                # los vasos son finisimos: sin este peso, la MSE los ignora
                w = torch.ones_like(x0)
                ves = ((sem_t >= C.VESSEL0) &
                       (sem_t <= C.VESSEL0 + C.N_VESSEL - 1))[:, None]
                w = w + a.vessel_w * ves.float()
                loss = (w * (pred.float() - target) ** 2).sum() / w.sum()

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += loss.item()

            # En Windows pasarse de VRAM no da OOM: el driver desborda a RAM del
            # sistema y sigue, 68 veces mas lento y tumbando el equipo a BSOD. Como
            # no salta ninguna excepcion, hay que mirarlo a mano una vez.
            if paso == 1:
                pico = torch.cuda.max_memory_allocated()
                _, vram = torch.cuda.mem_get_info()
                print(f"   VRAM pico {pico/2**30:.1f} GB de {vram/2**30:.1f} GB",
                      flush=True)
                if pico > 0.90 * vram:
                    raise SystemExit(
                        f"\nABORTADO: el paso pide {pico/2**30:.1f} GB y la tarjeta "
                        f"tiene {vram/2**30:.1f} GB.\nNo daria OOM -- desbordaria a "
                        f"RAM del sistema y dejaria el equipo inservible.\n"
                        f"Baja --bs (a 96^3 el maximo son 3) o --patch. "
                        f"Ver la tabla en el docstring.")
        for gp, b in zip(opt.param_groups, lr_ep):   # deshacer la rampa
            gp["lr"] = b
        lrs.step()
        tl /= a.iters

        vl = float("nan")
        if (ep + 1) % 20 == 0 or ep == a.epochs - 1:
            net.eval()
            vl, nb = 0.0, 0
            g = torch.Generator(device=DEV).manual_seed(0)
            with torch.no_grad():
                for _, _, img, sem, tis, _ in va:
                    for _ in range(4):
                        pi, ps, pt = sample_patch(img, sem, tis, a.patch,
                                                  np.random.default_rng(nb))
                        x0 = C.normalize(torch.from_numpy(pi[None, None]).to(DEV))
                        seg = M.labelmap(ps[None], pt[None], DEV)
                        noise = torch.randn(x0.shape, device=DEV, generator=g)
                        t = torch.randint(0, nT, (1,), device=DEV, generator=g)
                        xt = sched.add_noise(x0, noise, t)
                        target = sched.get_velocity(x0, noise, t)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            pred = net(x=xt, timesteps=t, seg=seg)
                        vl += F.mse_loss(pred.float(), target).item()
                        nb += 1
            vl /= nb
            print(f"ep {ep+1:4d} loss {tl:.4f} val {vl:.4f} ({time.time()-t0:.0f}s)",
                  flush=True)
            if vl < best:
                best = vl
                torch.save({"model": net.state_dict(), "epoch": ep, "val": vl},
                           ROOT / "ckpt" / "gen_best.pt")
                print(f"   -> gen_best.pt (val {vl:.4f})", flush=True)
        else:
            print(f"ep {ep+1:4d} loss {tl:.4f} ({time.time()-t0:.0f}s)", flush=True)

        if a.mon and ((ep + 1) % a.mon == 0 or ep == a.epochs - 1):
            cg, cr = monitorizar(net, ep + 1, ref, mon_dir)
            print(f"   contraste vaso-parenquima {cg:+.0f} HU  (real {cr:+.0f} HU)"
                  f"  -> mon/ep{ep+1:04d}.png", flush=True)
            with hist.open("a") as fh:
                fh.write(f"{ep+1},{ref_case[0]},{tl:.5f},{vl:.5f},"
                         f"{cg:.1f},{cr:.1f},{cg/cr if cr else float('nan'):.3f}\n")

        torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                    "lrs": lrs.state_dict(), "epoch": ep, "best": best}, ck)


if __name__ == "__main__":
    main()
