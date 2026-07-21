"""El puente: de segmentos arteriales apagados a territorios de parenquima.

`cow_edit` decide QUE VASOS se quedan sin flujo, a partir de la variante y del sitio
de oclusion (grafo hemodinamico). El atlas de `lesion2ncct` dice DONDE esta cada
territorio. Entre los dos no habia nada: esto es lo que faltaba.

Sustituye a `run_case.OCC_TERR`, que era una tabla fija oclusion -> territorio. En
ella la variante NO cambiaba el territorio infartado (solo modulaba, via
`lesionfreq_COW*`, donde caia la lesion DENTRO del territorio), asi que una ICA
ocluida infartaba lo mismo con Acom que sin el. Aqui el territorio SALE del grafo,
que es justo lo que se buscaba:

    ICA con Acom permeable   -> la ACA se rescata -> NO se infarta
    ICA sin Acom             -> la ACA cae        -> se infarta MCA + ACA
    ICA cervical, CoW completo -> no cae nada     -> NCCT SIN infarto

Territorios del atlas (`atlas/atlas_territories.nii.gz`, 182x218x182, voxel a voxel
con el NCCT del banco). Los ids se han verificado por centroide -- impar = izquierda,
par = derecha:

    1 ACA izq        2 ACA der
    3 MCA izq        4 MCA der
    5 PCA izq        6 PCA der
    7 fosa post izq  8 fosa post der
    9 profundo izq  10 profundo der     (perforantes, ~13 ml cada uno)

Los ids 9 y 10 `OCC_TERR` no los usaba nunca. Son los que permiten distinguir una
oclusion de M1 (se lleva por delante las lenticuloestriadas: infarto que incluye
los ganglios basales) de una de M2 (que las respeta). Con la tabla vieja, un M2
ocluido podia infartar el nucleo lenticular, que es un error clinico.

Aviso honesto sobre 9/10: por centroide (~12-15 mm de la linea media, a la altura
de los ganglios basales) y por volumen (~13 ml) es el territorio de las perforantes
proximales. No hay nombre en el atlas, asi que no se puede afirmar si separa
lenticuloestriadas de coroidea anterior. Ambas son perforantes proximales de la
M1/ICA y ambas se respetan en una oclusion distal, que es el comportamiento que
importa aqui.
"""
import os
import sys

import numpy as np
import nibabel as nib
from scipy import ndimage

L2N = r"C:\Desactivar_Respaldo\lesion2ncct_portable\lesion2ncct_bundle"
if L2N not in sys.path:
    sys.path.insert(0, L2N)
import run_case as R2          # reutilizamos su colocador de formas y su banco LDM

ATLAS_DIR = os.path.join(L2N, "atlas")

TERR = {"L": {"aca": 1, "mca": 3, "pca": 5, "post": 7, "deep": 9},
        "R": {"aca": 2, "mca": 4, "pca": 6, "post": 8, "deep": 10}}
DEEP = (TERR["L"]["deep"], TERR["R"]["deep"])

NOMBRE = {1: "ACA izq", 2: "ACA der", 3: "MCA izq", 4: "MCA der",
          5: "PCA izq", 6: "PCA der", 7: "fosa post izq", 8: "fosa post der",
          9: "profundo izq", 10: "profundo der"}


def _tabla():
    """vaso -> [(territorio, volumen tipico del infarto, en ml)].

    El volumen es el del infarto que produce perder ESE segmento: cuanto mas
    proximal, mas territorio. Calibrado sobre los numeros de `OCC_TERR`, que es
    contra lo que se ajustaron el banco de formas y el Gen3D (M1 55 ml, M2 25,
    ACA 20, PCA 22, basilar 30).
    """
    v = {}
    for s in ("L", "R"):
        t = TERR[s]
        v[f"{s}-M1"] = [(t["mca"], 55), (t["deep"], 13)]   # + perforantes
        v[f"{s}-M2"] = [(t["mca"], 25)]                    # respeta el profundo
        v[f"{s}-M3"] = [(t["mca"], 10)]
        v[f"{s}-A1A2"] = [(t["aca"], 20)]
        v[f"{s}-A3"] = [(t["aca"], 10)]
        v[f"{s}-P1P2"] = [(t["pca"], 22)]
        v[f"{s}-P3P4"] = [(t["pca"], 10)]
        v[f"{s}-AChA"] = [(t["deep"], 8)]
        v[f"{s}-SCA"] = [(t["post"], 12)]
        v[f"{s}-AICA"] = [(t["post"], 10)]
        v[f"{s}-PICA"] = [(t["post"], 15)]
        v[f"{s}-VA"] = [(t["post"], 15)]
    v["BA"] = [(TERR["L"]["post"], 15), (TERR["R"]["post"], 15)]   # tronco + ambas cerebelosas
    v["3rd-A2"] = [(TERR["L"]["aca"], 10), (TERR["R"]["aca"], 10)]  # ACA azigos
    v["3rd-A3"] = [(TERR["L"]["aca"], 8), (TERR["R"]["aca"], 8)]
    # Sin territorio propio: son conductos (ICA) o colaterales (Pcom, Acom) y la
    # oftalmica no irriga parenquima. Su parenquima lo llevan sus hijos, y de
    # apagarlos ya se encarga el grafo.
    for n in ("R-ICA-C1-C5", "L-ICA-C1-C5", "R-ICA-C6-C7", "L-ICA-C6-C7",
              "R-Pcom", "L-Pcom", "Acom", "R-OA", "L-OA"):
        v[n] = []
    return v


VESSEL_TERR = _tabla()

# Un trombo en el M1 mas alla de esta fraccion del recorrido deja las perforantes
# proximales a el -> siguen perfundidas -> no hay infarto profundo.
PERF_FRAC = 0.7


def territorios(apagados, artery=None, frac=None, distal_muerto=True):
    """Territorios infartados y volumen (ml), a partir de lo que apago el grafo.

    - Cada segmento de `apagados` aporta su(s) territorio(s): se quedo sin flujo.
    - La arteria OCLUIDA aporta el suyo si su porcion distal se quedo sin contraste
      (`distal_muerto`, o sea `not info["distal_retrogrado"]`). Si una colateral la
      rellena en sentido retrogrado, no hay infarto por ese lado.
    - Excepcion, el territorio profundo de un M1 ocluido: las perforantes nacen A LO
      LARGO del M1 y no tienen etiqueta propia, asi que el grafo no puede decidir por
      ellas. Se decide con `frac`: trombo proximal (<= PERF_FRAC) las deja sin flujo;
      trombo pegado a la bifurcacion, no.

    Cada territorio se queda con el volumen del segmento MAS PROXIMAL que lo alimenta:
    perder el M1 infarta todo el territorio de la ACM (55 ml), aunque el M2 y el M3
    esten tambien en la lista con sus 25 y 10.

    Devuelve {id_territorio: ml}. Vacio si no cae nada -- p. ej. una ICA cervical
    ocluida con el CoW completo: la oclusion asintomatica clasica, y el NCCT sale SIN
    infarto. El volumen va POR TERRITORIO, no agregado: `synth_mask` lo necesita para
    repartirlo entre los hemisferios (ver alli).
    """
    segs = list(apagados)
    if artery and distal_muerto:
        segs.append(artery)

    vol = {}
    for s in segs:
        for tid, ml in VESSEL_TERR.get(s, []):
            if (s == artery and s.endswith("-M1") and tid in DEEP
                    and frac is not None and frac > PERF_FRAC):
                continue                       # trombo distal: las perforantes se salvan
            vol[tid] = max(vol.get(tid, 0), ml)
    return vol


def _atlas():
    at = np.asanyarray(nib.load(os.path.join(ATLAS_DIR, "atlas_territories.nii.gz"))
                       .dataobj).squeeze().astype(np.int16)
    tis = np.asanyarray(nib.load(os.path.join(ATLAS_DIR, "tissue.nii.gz"))
                        .dataobj).squeeze()
    return at, tis


# --- pool de NCCT de CONTROL (fondos sanos desacoplados del donante CTA) ----------
# Un NCCT de control (p.ej. SinoCT leido como normal, registrado a MNI con
# `register_to_mni.py`) NO trae CTA/CoW: no puede ser donante de oclusion, pero si el
# FONDO de cerebro sobre el que cae el infarto (que lo situa el atlas MNI, no el paciente).
# Vacio por defecto; se llena al ingerir el dataset de control. Override por env.
CONTROL_DIR = os.environ.get("CONTROL_NCCT_DIR", os.path.join(L2N, "control_ncct"))


def control_pool():
    """Lista de NCCT de control disponibles (en MNI 182^3). Vacia si no hay ninguno."""
    if not os.path.isdir(CONTROL_DIR):
        return []
    import glob
    return sorted(glob.glob(os.path.join(CONTROL_DIR, "*_ncct.nii.gz")))


def bank_ncct_pool():
    """Los NCCT del banco -> [(pid, cow_id, path)]. TODOS valen como fondo.

    El NCCT de un donante y su CTA viven en espacios distintos (MNI vs slab canonico) y no
    se superponen nunca, asi que el cerebro de fondo no tiene por que ser el del donante del
    CTA: lo que ata el caso es el TERRITORIO (el atlas MNI), no la identidad. Desacoplarlo
    multiplica la variedad -- la familia 0 pasa de 2 cerebros a los 9 del banco -- sin tocar
    un dato. Incluye a `pat_cow0_02`, descartado como donante por tener las etiquetas de VASOS
    corruptas: su cerebro no tiene nada malo y como fondo es perfectamente valido.
    """
    import csv
    import glob
    fam = {}
    man = os.path.join(R2.BANK, "manifest.csv")
    if os.path.isfile(man):
        with open(man, encoding="utf-8") as f:
            fam = {r["pid"]: int(r["cow_id"]) for r in csv.DictReader(f)}
    out = []
    for p in sorted(glob.glob(os.path.join(R2.BANK, "*_ncct.nii.gz"))):
        pid = os.path.basename(p).replace("_ncct.nii.gz", "")
        out.append((pid, fam.get(pid, 0), p))
    return out


def pick_ncct_base(seed, source="banco", variante=""):
    """Elige el NCCT base. -> (hu, affine, nombre, cow_id, en_dominio) o None.

    `source`:
      "banco"   los 9 del banco, desacoplados del donante del CTA (por defecto).
      "control" el pool externo de controles (SinoCT...); si esta vacio, cae al banco.

    La eleccion depende de (semilla, VARIANTE): con la semilla fija, cambiar de variante
    cambia tambien de cerebro, que es lo que se espera al pasearse por el menu -- si solo
    dependiera de la semilla, todas las variantes saldrian sobre el mismo cerebro. No rompe
    nada comparable: la comparacion entre variantes se hace sobre el mismo ARBOL (`pid`), y
    el fondo del NCCT ya va desacoplado del arbol.

    Devuelve tambien el `cow_id` DEL CEREBRO (no el del CTA): es la condicion con la que se
    entreno el Gen3D para ese cerebro, asi que pasarla mantiene al generador en dominio.
    `en_dominio` dice si se puede usar el Gen3D (banco si; control externo no).

    El espejo L-R duplica la variedad (9 -> 18 fondos). La mascara NO se refleja: sale del
    atlas, que manda sobre el lado.
    """
    rng = np.random.default_rng([int(seed), 99, sum(map(ord, str(variante)))])
    if source == "control":
        pool = control_pool()
        if pool:
            p = pool[int(rng.integers(len(pool)))]
            im = nib.load(p)
            hu = np.asanyarray(im.dataobj).astype(np.float32)
            if rng.random() < 0.5:
                hu = hu[::-1].copy()
            return hu, im.affine, os.path.basename(p), 0, False

    pool = bank_ncct_pool()
    if not pool:
        return None
    pid, cow, p = pool[int(rng.integers(len(pool)))]
    im = nib.load(p)
    hu = np.asanyarray(im.dataobj).astype(np.float32)
    esp = bool(rng.random() < 0.5)
    if esp:
        hu = hu[::-1].copy()          # eje 0 del NCCT del banco (LAS) = izq-der
    return hu, im.affine, f"{pid}{' (espejo)' if esp else ''}", cow, True


def pick_control(seed):
    """Compat: el pool externo de control. -> (hu, affine, nombre) o None."""
    r = pick_ncct_base(seed, source="control")
    if r is None or r[4]:            # None, o cayo al banco (en_dominio=True) -> no hay control
        return None
    return r[0], r[1], r[2]


# --- fases del infarto ------------------------------------------------------------
# (objetivo_HU, alpha_nucleo): el tejido se INTERPOLA hacia `objetivo`, no se le resta una
# cantidad fija. `alpha` es cuanto se acerca en el nucleo (el borde va a menos).
#
# Un infarto no "resta densidad": el edema citotoxico mete agua y el tejido CONVERGE hacia
# la densidad del agua. Interpolar reproduce solo el signo cardinal -- la perdida de
# diferenciacion gris-blanca -- porque la gris (38 HU) tiene mas recorrido hasta el objetivo
# que la blanca (26): con objetivo 25 y alpha 0.6, gris -> 30.2 y blanca -> 25.4, o sea que
# la diferencia cae de 12 a 4.8 HU. Eso es exactamente lo que se ve en un TC precoz.
#
# Y ademas NUNCA CLIPA. La formula vieja (`base - (10+12*core)*strength`) restaba 15-33 HU a
# ciegas: la blanca quedaba en negativo, el clip a 0 la aplanaba y el 23% del infarto salia
# como un agujero a 0 HU -- de ahi el aspecto de borron liso a densidad de LCR.
# Calibrado sobre el tejido REAL de estos cerebros (medido, no de libro):
#   sustancia blanca ~23.6 HU | sustancia gris ~40.5 HU | diferencia ~17 HU
FASES = {
    "hiperagudo": (30.0, 0.35),   # < 6 h   sutil, tipo ASPECTS: apenas se intuye
    "agudo":      (26.0, 0.60),   # 6-24 h  perdida gris-blanca franca
    "subagudo":   (21.0, 0.80),   # 1-7 d   hipodensidad evidente
    "cronico":    (11.0, 0.95),   # > 1 mes encefalomalacia, casi LCR
}
FASE_DEF = "agudo"


def _insertar_infarto(base, mask, strength=1.0, fase=FASE_DEF):
    """Inserta el infarto interpolando el parenquima hacia la densidad de la fase.

    `strength` modula la conspicuidad (1.0 = la de la fase; <1 mas sutil, >1 mas marcado).
    El borde se difumina con el mapa de distancia: el nucleo llega al alpha de la fase y la
    periferia se queda a medias, que es como se ve la penumbra.
    """
    if not mask.any():
        return base.astype(np.float32)
    obj, a_core = FASES.get(fase, FASES[FASE_DEF])

    # perfil nucleo->borde: 1 en el centro, se desvanece en los ultimos ~4 mm
    dt = ndimage.distance_transform_edt(mask).astype(np.float32)
    perfil = np.clip(dt / 4.0, 0, 1)                     # 0 en el borde, 1 a 4 vox adentro
    perfil = ndimage.gaussian_filter(perfil, 1.0)
    alpha = np.clip(perfil * a_core * strength, 0, 0.98)  # nunca 1: siempre queda textura

    out = base.astype(np.float32).copy()
    sel = mask & (base > 5)          # solo parenquima: el LCR y el hueso no se infartan
    lerp = base[sel] + (obj - base[sel]) * alpha[sel]
    # UNIDIRECCIONAL: un infarto solo puede bajar la densidad, nunca subirla. Sin esto, en
    # las fases precoces (objetivo 30 > blanca 23.6) la interpolacion ACLARABA la sustancia
    # blanca y el hiperagudo salia mas brillante que el cerebro sano. Con el minimo, la gris
    # (40.5) cae hacia el objetivo y la blanca se queda quieta hasta que el objetivo baja de
    # ella -- que es justo como progresa de verdad: primero se borra la gris, luego todo.
    out[sel] = np.minimum(base[sel], lerp)
    return out.astype(np.float32)


def _oscurece(base, mask, strength):
    """Compat: inserta el infarto con la fase por defecto."""
    return _insertar_infarto(base, mask, strength=strength, fase=FASE_DEF)


def infer_ncct(pid, mask, cow_id, dev, strength, espejo=False, base=None, aff=None,
               use_gen=True, fase=FASE_DEF):
    """Inserta el infarto en un NCCT base. Dos modos:

    - MISMO PACIENTE (por defecto, `base=None`): carga `{pid}_ncct` del banco y usa el Gen3D
      (dominio de los 9 cerebros del banco). `espejo` refleja L-R el CEREBRO base -- y NO la
      mascara: el territorio lo pone el atlas, asi que si el CTA se reflejo, "R-M1" sigue
      siendo la derecha del volumen reflejado y "MCA der" del atlas tambien; reflejar la
      mascara mandaria el infarto al lado contrario de la oclusion.

    - CONTROL DESACOPLADO (`base` = array HU en MNI, `aff` su affine, `use_gen=False`): el
      fondo es un cerebro de control (otro escaner). No se pasa por el Gen3D (fuera de
      dominio); la hipodensidad se inserta con `_oscurece`. No hay `espejo`: no hay CTA con
      el que acoplar, y el lado del infarto ya lo fija el atlas.

    Caso sano (mascara vacia): se devuelve el base tal cual -- sin Gen3D no hay hipodensidad
    alucinada, el NCCT sale limpio (vale para ambos modos).
    """
    import torch
    import torch.nn.functional as F

    if base is None:
        mi = nib.load(os.path.join(R2.BANK, f"{pid}_ncct.nii.gz"))
        ncct = np.asanyarray(mi.dataobj).astype(np.float32)
        aff = mi.affine
        if espejo:
            ncct = ncct[::-1].copy()          # eje 0 del NCCT del banco (LAS) = izq-der
    else:
        ncct = np.asarray(base, np.float32)   # control: fondo tal cual, sin espejo

    if not mask.any():
        return aff, ncct                      # sano: base limpio, sin generador

    if not use_gen:                           # base fuera de dominio: sin Gen3D
        return aff, _insertar_infarto(ncct, mask, strength, fase)

    base01 = np.clip(ncct / R2.HU_HI, 0, 1).astype(np.float32)

    G = R2.Gen3D().to(dev).eval()
    G.load_state_dict(torch.load(R2.WEIGHTS, map_location=dev)["G"])
    b = torch.from_numpy(base01)[None, None].to(dev)
    m = torch.from_numpy(mask.astype(np.float32))[None, None].to(dev)
    sh = base01.shape
    pp = (0, (16 - sh[2] % 16) % 16, 0, (16 - sh[1] % 16) % 16, 0, (16 - sh[0] % 16) % 16)
    with torch.no_grad():
        fp = G(F.pad(b, pp), F.pad(m, pp), torch.tensor([cow_id], device=dev))
    out = (fp[0, 0, :sh[0], :sh[1], :sh[2]].cpu().numpy() * R2.HU_HI).astype(np.float32)
    return aff, _insertar_infarto(out, mask, strength, fase)


def _intracraneal_ncct(nc):
    """Mascara SUAVE [0,1] del contenido intracraneal del NCCT (ventana [0,80]).

    El craneo del NCCT esta clampeado al techo de la ventana (~80 HU), no a >300 como en
    el CTA, asi que se detecta distinto: se rellena la cabeza, se le quita esa cascara de
    hueso brillante y se queda la cavidad. Sirve para que la deformacion mueva el cerebro
    pero NO el craneo (que si no sale ondulado).
    """
    head = ndimage.binary_fill_holes(nc > 5)
    skull = ndimage.binary_dilation(nc >= 78, iterations=1)
    cav = head & ~skull
    lab, n = ndimage.label(cav)
    if n == 0:
        return np.ones(nc.shape, np.float32)
    big = lab == (1 + int(np.argmax(np.bincount(lab.ravel())[1:])))
    big = ndimage.binary_erosion(big, iterations=2)
    return ndimage.gaussian_filter(big.astype(np.float32), 4.0)


def deformar(ncct, mask, rng, alpha=3.0, sigma=14.0):
    """Deformacion elastica del NCCT y su mascara con EL MISMO campo.

    Sin esto los cerebros del banco se repiten tal cual. `alpha` es el desplazamiento
    tipico en voxeles: hay que RENORMALIZAR tras el filtrado (si no, el gaussiano aplasta
    la amplitud ~350x y no deforma nada; la misma trampa que documenta `randomize_anatomy`).

    El campo se modula por la mascara intracraneal: el cerebro se deforma (variedad) pero
    el CRANEO queda rigido. Si no, el hueso sale ondulado y no casa con el craneo liso del
    CTA del mismo paciente.
    """
    im = _intracraneal_ncct(ncct)
    g = []
    for _ in range(3):
        f = ndimage.gaussian_filter(rng.normal(0, 1, ncct.shape), sigma).astype(np.float32)
        f *= alpha / (f.std() + 1e-8)
        g.append(f * im)
    idx = np.indices(ncct.shape, dtype=np.float32)
    co = [idx[i] + g[i] for i in range(3)]
    nc = ndimage.map_coordinates(ncct, co, order=1, mode="nearest").astype(np.float32)
    mk = ndimage.map_coordinates(mask.astype(np.uint8), co, order=0, mode="nearest") > 0
    return nc, mk


def _un_blob(terr_ids, vol, cow_id, rng, shape, at, tis):
    """Una lesion conexa dentro de `terr_ids` (que han de ser del MISMO hemisferio)."""
    allowed = np.isin(at, list(terr_ids)) & np.isin(tis, (1, 2))
    if allowed.sum() == 0:
        allowed = np.isin(at, list(terr_ids))
    if allowed.sum() == 0:
        return np.zeros(shape, bool)

    # prior de donde caen las lesiones en esta familia de CoW
    lf = os.path.join(ATLAS_DIR, f"lesionfreq_COW{cow_id}.nii.gz")
    p = (np.asanyarray(nib.load(lf).dataobj).squeeze() * allowed
         if os.path.exists(lf) else allowed.astype(float)).astype(np.float64)
    if p.sum() <= 0:
        p = allowed.astype(np.float64)
    p /= p.sum()
    seed_vox = np.unravel_index(rng.choice(p.size, p=p.ravel()), p.shape)
    target = int(np.clip(vol * 1000, 1, allowed.sum()))

    bank = R2._load_bank()
    if bank:
        crop = bank[int(rng.integers(len(bank)))]
        m = R2._place_shape(crop, seed_vox, allowed, rng, target)
        if m is not None and m.sum() >= 200:
            return R2._largest_cc(ndimage.binary_closing(m, iterations=1) & allowed)

    # fallback: radio "ruidoso" desde la semilla, recortado al territorio
    seedm = np.zeros(shape, bool)
    seedm[seed_vox] = True
    edt = ndimage.distance_transform_edt(~seedm).astype(np.float32)
    field = ndimage.gaussian_filter(rng.standard_normal(shape).astype(np.float32), 3.0)
    field = (field - field.mean()) / (field.std() + 1e-6)
    eff = edt * np.clip(1.0 + 0.6 * field, 0.15, None)
    vals = np.sort(eff[allowed])
    R = vals[min(target, len(vals) - 1)]
    cand = allowed & (eff <= R)
    lab, _ = ndimage.label(cand)
    mask = (lab == lab[seed_vox]) if lab[seed_vox] > 0 else cand
    return ndimage.binary_closing(mask, iterations=1) & allowed


def synth_mask(vol_terr, cow_id, seed, shape):
    """Mascara del infarto en el grid del atlas/NCCT (MNI, LAS 182x218x182).

    `vol_terr` = {id_territorio: ml}, lo que devuelve `territorios()`.

    Misma receta que `run_case.synth_mask` -- forma realista del banco LDM colocada en
    el territorio, con fallback procedural -- con dos diferencias:

    1. El territorio y el volumen los decide el grafo, no la tabla fija `OCC_TERR`.
    2. Se coloca UNA LESION POR HEMISFERIO, no una global. `run_case` terminaba en
       `_largest_cc`, asi que un infarto bilateral -- una basilar ocluida se lleva las
       dos PCA y las dos fosas posteriores -- colapsaba a la componente conexa mayor y
       salia en un solo lado: el manifest decia "bilateral" y la imagen mostraba medio.
       Los ids del atlas son todos lateralizados (impar izquierda, par derecha), asi
       que basta con agrupar por paridad y colocar una forma en cada grupo.

    Devuelve (mascara, volumen_ml_real).
    """
    if not vol_terr or sum(vol_terr.values()) <= 0:
        return np.zeros(shape, bool), 0.0

    rng = np.random.default_rng(seed)
    at, tis = _atlas()

    # sigma 0.3, no la 0.5 de run_case: con 0.5 la cola es tan larga que un M2 (base
    # 25 ml) sale a 69, y el volumen deja de ser verdad de terreno util.
    jit = float(np.clip(rng.lognormal(0.0, 0.3), 0.4, 2.5))

    mask = np.zeros(shape, bool)
    for par in (1, 0):                                   # 1 = izquierda, 0 = derecha
        lado = {t: v for t, v in vol_terr.items() if t % 2 == par}
        if not lado:
            continue
        vol = min(sum(lado.values()) * jit, 250.0)
        mask |= _un_blob(list(lado), vol, cow_id, rng, shape, at, tis)

    return mask, float(mask.sum()) / 1000.0              # 1 mm iso -> 1000 vox = 1 ml
