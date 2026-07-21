"""(variante de CoW + sitio de oclusion) -> CTA con la oclusion + NCCT con el infarto.

El caso completo, del MISMO paciente, derivado de un solo grafo hemodinamico:

    variante + arteria
        |
        +-- cow_edit.occlude  ->  que vasos se quedan sin flujo  ("apagados")
        |        |
        |        +-- hybrid.render     -> CTA: el arbol apagado pierde el contraste
        |        |                              y el trombo aparece hiperdenso
        |        +-- territorios       -> que parenquima se infarta
        |                 |
        |                 +-- Gen3D    -> NCCT: hipodensidad en ese territorio
        v
    los dos consistentes por construccion: el territorio infartado en el NCCT es el
    del arbol que se apago en el CTA.

Lo que cambia frente a `lesion2ncct/run_case.py`: alli el territorio salia de una
tabla fija (`OCC_TERR`) que ignoraba la variante. Aqui sale del grafo, asi que la
variante cambia el infarto -- que es el punto:

    caso.py --variante "CoW completo" --arteria R-ICA-C6-C7 --frac 0.9   -> infarta MCA (el Acom rescata la ACA)
    caso.py --variante "Acom ausente" --arteria R-ICA-C6-C7 --frac 0.9   -> infarta MCA + ACA
    caso.py --variante "CoW completo" --arteria R-ICA-C1-C5              -> NO infarta nada

Uso:
    python bank.py                                  (una vez: prepara las anatomias)
    python caso.py --variante "Acom ausente" --arteria R-ICA-C6-C7 --frac 0.9 --out caso1

Salida en --out:
    cta_oclusion.nii.gz   CTA con la oclusion   (slab canonico 0.6 mm del donante)
    ncct_infarto.nii.gz   NCCT con el infarto   (MNI 1 mm)
    mask.nii.gz           mascara del infarto   (MNI)
    vasos.nii.gz          arbol editado         (canonico)
    caso.json             variante, apagados, territorios, volumen, HU medidas

CTA y NCCT son del mismo paciente pero viven en espacios distintos (canonico vs MNI),
igual que en lesion2ncct: el atlas de territorios solo existe en MNI y el NCCT del
banco esta en MNI, mientras que el arbol arterial se edita en el slab canonico.
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
import common as C
import cow_edit as E
import hybrid as H
import territorios as T
import deface as D        # quita la cara del CTA y del NCCT (ambos son donantes reales)

ROOT = Path(__file__).parent
BANK_SEM = ROOT / "bank_sem"

# La familia de CoW del banco (0 completo, 1 A1 hipoplasica, 2 PCA fetal) elige el
# donante y condiciona el Gen3D. La variante FINA la aplica `cow_edit` sobre las
# etiquetas, asi que no hace falta que el banco la traiga ya: para "Acom ausente" se
# parte de un donante completo y el Acom se borra.
FAMILIA = {
    "CoW completo": 0, "Acom ausente": 0,
    "Pcom derecha ausente": 0, "Pcom izquierda ausente": 0, "Pcom bilateral ausente": 0,
    "A1 hipoplasica derecha": 1, "A1 hipoplasica izquierda": 1,
    "A1 derecha ausente": 1, "A1 izquierda ausente": 1,
    "PCA fetal derecha": 2, "PCA fetal izquierda": 2, "PCA fetal bilateral": 2,
}


def donantes(familia=None):
    """[(pid, cow_id, path), ...] del banco ya preparado."""
    out = []
    for p in sorted(BANK_SEM.glob("*.npz")):
        cow = int(np.load(p)["cow_id"])
        if familia is None or cow == familia:
            out.append((p.stem, cow, p))
    return out


def _campo(shape, rng, alpha, sigma=16.0, factor=4, mod=None):
    """Campo de desplazamiento suave. `alpha` = desplazamiento tipico, en voxeles.

    `mod` (mascara [0,1] del mismo shape) escala el desplazamiento: donde vale 0 no hay
    movimiento. Se usa para dejar el craneo quieto y deformar solo el cerebro.

    Se genera a 1/4 de resolucion y se reescala. Filtrar ruido con sigma=16 sobre el slab
    entero (12,6 M de voxeles) cuesta segundos POR EJE, y aqui hay que poder redibujar el
    campo varias veces si rompe el arbol; a baja resolucion sale ~60x mas barato y el
    campo es equivalente, porque a esa escala es suave por construccion.

    Hay que RENORMALIZAR tras el filtrado: un gaussiano aplasta la amplitud del ruido
    ~350x y sin esto el desplazamiento seria de centesimas de voxel (la misma trampa que
    documenta `hybrid.randomize_anatomy`).
    """
    small = tuple(max(4, s // factor) for s in shape)
    g = []
    for _ in range(3):
        f = ndi.gaussian_filter(rng.normal(0, 1, small), sigma / factor).astype(np.float32)
        f = ndi.zoom(f, [s / t for s, t in zip(shape, small)], order=1)
        f = f[:shape[0], :shape[1], :shape[2]]
        f *= alpha / (f.std() + 1e-8)
        g.append(f)
    if mod is not None:                        # anula el desplazamiento fuera del cerebro
        g = [f * mod for f in g]
    idx = np.indices(shape, dtype=np.float32)
    return [idx[i] + g[i] for i in range(3)]


def _intracraneal(hu):
    """Mascara SUAVE [0,1] del contenido intracraneal, para modular la deformacion.

    El cerebro se deforma (da variedad al arbol arterial, que es lo que se busca) pero el
    craneo NO: una deformacion elastica del hueso lo ondula y se ve raro. Multiplicando el
    campo por esta mascara -- 1 dentro, 0 en el hueso, con una transicion suave -- el
    craneo queda rigido y solo se mueve lo de dentro.
    """
    head = ndi.binary_fill_holes(hu > -300)
    skull = ndi.binary_dilation(hu > 300, iterations=1)
    cav = head & ~skull
    lab, n = ndi.label(cav)
    if n == 0:
        return np.ones(hu.shape, np.float32)
    big = lab == (1 + int(np.argmax(np.bincount(lab.ravel())[1:])))
    big = ndi.binary_erosion(big, iterations=2)           # aleja la transicion del hueso
    return ndi.gaussian_filter(big.astype(np.float32), 4.0)


def _warp_vasos(ves, co, margen=8):
    """Deforma el mapa de vasos SIN fragmentarlo.

    Deformar las etiquetas por vecino mas proximo (que es lo que hace
    `hybrid.randomize_anatomy`, y le vale porque alli el arbol solo se pinta) parte los
    vasos finos: son de 1-2 voxeles de calibre, y una vertebral de 114 pasos geodesicos
    se queda en 20. Aqui el arbol alimenta un GRAFO que necesita conectividad y longitud,
    asi que eso no vale: `occlude` rechaza el segmento por corto.

    El truco es no interpolar la mascara binaria sino su MAPA DE DISTANCIA CON SIGNO, que
    es suave: se deforma con interpolacion lineal y se vuelve a umbralizar en 0. Asi el
    tubo se desplaza entero en vez de descomponerse en trozos.

    Se hace sobre la caja de cada vaso (con margen), que si no las 36 transformadas de
    distancia sobre el slab entero cuestan minutos. Donde dos vasos se pisan tras
    deformar gana el que quede mas adentro.
    """
    out = np.zeros(ves.shape, np.uint8)
    mejor = np.full(ves.shape, np.inf, np.float32)
    for vid, sl in enumerate(ndi.find_objects(ves.astype(np.int32)), 1):
        if sl is None:
            continue
        sl = tuple(slice(max(0, s.start - margen), min(n, s.stop + margen))
                   for s, n in zip(sl, ves.shape))
        sub = ves[sl] == vid
        if not sub.any():
            continue
        sdf = (ndi.distance_transform_edt(~sub)
               - ndi.distance_transform_edt(sub)).astype(np.float32)
        ini = [s.start for s in sl]
        loc = [co[i][sl] - ini[i] for i in range(3)]
        w = ndi.map_coordinates(sdf, loc, order=1, mode="constant", cval=1e3)

        o, b = out[sl], mejor[sl]
        gana = (w < 0) & (w < b)
        o[gana], b[gana] = vid, w[gana]
        out[sl], mejor[sl] = o, b
    return out


def _fragmentado(ves, umbral=0.7):
    """Segmentos ocluibles partidos en trozos: la componente conexa mayor no llega a
    `umbral` de sus voxeles.

    OJO: hay que compararlo con el ORIGINAL, no con cero. El pseudo-etiquetado ya entrega
    vasos partidos de fabrica -- en el banco, el R-M2 (que se bifurca) viene roto en
    todos los donantes. Un criterio absoluto rechaza cualquier deformacion y el generador
    acaba devolviendo siempre el donante intacto, que fue exactamente lo que paso.
    """
    cajas = ndi.find_objects(ves.astype(np.int32))     # bbox de cada etiqueta, una pasada
    malos = []
    for n in C.OCCLUDABLE:
        vid = E.V[n]
        sl = cajas[vid - 1] if vid - 1 < len(cajas) else None
        if sl is None:
            continue
        m = ves[sl] == vid                             # solo su caja: el CC sale barato
        tot = int(m.sum())
        if tot < 60:
            continue
        lab, k = ndi.label(m)
        if k <= 1:
            continue
        if int(np.bincount(lab.ravel())[1:].max()) < umbral * tot:
            malos.append(n)
    return malos


# Amplitudes de deformacion a probar, de mas a menos (voxeles de 0.6 mm). Se baja un
# escalon si la arteria pedida se queda inocluible: algun segmento del banco esta justo
# en el limite -- el BA de `pat_cow0_00` arrastra un fragmento satelite pegado a la
# vertebral, la semilla de `_seed_of` cae ahi y su geodesica es de 4 pasos en vez de ~30,
# que es el minimo exacto que exige `occlude`. Con 0.0 se usa el donante tal cual.
ESCALA = (3.5, 2.5, 1.5, 0.8, 0.0)


def otro_paciente(hu, ves, seed, pid, alpha, intentos=3):
    """Convierte al donante en otro paciente sin romperle el arbol.

    Reflejo L-R (que intercambia las etiquetas lateralizadas, asi que "R-M1" sigue siendo
    la derecha anatomica del volumen nuevo) + deformacion elastica, con el MISMO campo a
    imagen y arbol. El arbol se deforma con `_warp_vasos`, que no lo fragmenta.

    El reflejo depende solo de (seed, pid), no de `alpha`: asi, aunque a un caso haya que
    bajarle la amplitud, sigue siendo el mismo paciente que su pareja.
    """
    rng = np.random.default_rng([seed, sum(map(ord, pid))])
    espejo = bool(rng.random() < 0.5)
    if espejo:
        hu, ves = C.flip_lr(hu, ves)
    if alpha <= 0:
        return hu, ves, espejo, rng

    im = _intracraneal(hu)                    # el craneo no se deforma; el cerebro si
    roto = set(_fragmentado(ves))            # lo que YA venia partido del pseudo-etiquetado
    for _ in range(intentos):
        co = _campo(hu.shape, rng, alpha, mod=im)
        v2 = _warp_vasos(ves, co)
        if set(_fragmentado(v2)) - roto:     # solo cuenta la rotura NUEVA
            continue
        h2 = ndi.map_coordinates(hu, co, order=1, mode="nearest").astype(np.float32)
        return h2, v2, espejo, rng
    return hu, ves, espejo, rng


def candidatos(variante, arteria, seed, pid=None):
    """Donantes compatibles, en orden: admiten la variante y tienen la arteria.

    Se devuelve la LISTA, no el primero, porque `occludable` filtra por volumen de
    vaso pero no por longitud: un donante puede tener un M1 de 200 vox y solo 2 pasos
    geodesicos, y entonces `occlude` lo rechaza por corto. Ahi hay que pasar al
    siguiente donante, no abortar el caso.
    """
    fam = FAMILIA[variante]
    rng = np.random.default_rng(seed)
    cand = donantes(fam) or donantes()          # si la familia no tiene, cualquiera
    if pid is not None:
        cand = [c for c in cand if c[0] == pid] or cand
    if not cand:
        raise SystemExit("banco vacio: corre primero  python bank.py")

    out, fallos = [], []
    for i in rng.permutation(len(cand)):
        p_id, cow, p = cand[i]
        d = np.load(p)
        ves = d["ves"]
        if variante not in E.available_variants(ves):
            fallos.append(f"{p_id}: no admite '{variante}'")
        elif arteria and arteria not in E.occludable(ves):
            fallos.append(f"{p_id}: no tiene {arteria}")
        else:
            out.append((p_id, cow, d))
    if not out:
        raise SystemExit("ningun donante compatible:\n  " + "\n  ".join(fallos))
    return out


def generar(variante="CoW completo", arteria=None, frac=None, seed=0, fuerza=1.5,
            out=None, quiet=False, pid=None, variar=True, ncct_source="banco",
            fase=T.FASE_DEF):
    """Un caso completo. Devuelve el dict que se escribe en caso.json.

    `pid` fija la anatomia del CTA. Uselo para comparar: dos casos que solo se diferencian
    en la variante (o en el sitio de oclusion) solo demuestran algo si son el MISMO arbol.

    `variar` convierte al donante del CTA en otro paciente (reflejo L-R + deformacion
    elastica del ARBOL), para que no salga identico en todos los casos: el banco solo tiene
    9 anatomias. La aleatorizacion depende SOLO de (seed, pid) -- va con su propio rng --
    asi que dos casos que comparten los dos siguen siendo el mismo arbol y se pueden
    comparar. Solo afecta al CTA.

    `ncct_source` de donde sale el CEREBRO de fondo del NCCT:
      "banco"   (por defecto) cualquiera de los 9 del banco, DESACOPLADO del donante del CTA.
      "mismo"   el del donante del CTA (el comportamiento de la reforma 2026).
      "control" el pool externo de controles sanos (SinoCT...); si esta vacio, cae al banco.

    Sobre el desacople -- el CTA (slab canonico 0.6 mm) y el NCCT (MNI 1 mm) viven en
    espacios distintos y NUNCA se superponen voxel a voxel, asi que "mismo paciente" era una
    etiqueta, no una alineacion: no compraba nada y ataba el NCCT a los 2 donantes de la
    familia 0 (de ahi que saliera siempre el mismo cerebro, y de ahi la deformacion elastica
    que lo emborronaba). Lo que de verdad ata el caso -- que el infarto caiga en el
    territorio del arbol apagado -- lo garantiza el atlas MNI y se mantiene intacto con
    cualquier fondo.
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    if arteria and frac is None:
        frac = float(rng.uniform(0.35, 0.9))

    # Primer (donante, amplitud) que admita la oclusion. `candidatos` filtra por volumen
    # de vaso, pero `occlude` ademas exige una geodesica de >= 4 pasos, que la
    # deformacion puede acortar: si pasa, se baja un escalon de `ESCALA` antes de
    # descartar el donante.
    errs, hallado = [], None
    for pid_, cow, d in candidatos(variante, arteria, seed, pid):
        ves_d = d["ves"]
        hu_d = C.to_hu(d["img"].astype(np.float32))
        for alpha in (ESCALA if variar else (0.0,)):
            hu0, ves0, espejo, rng_a = otro_paciente(hu_d, ves_d, seed, pid_, alpha)
            ves_v, cambios = E.apply_variant(ves0, variante)
            if not arteria:
                hallado = (pid_, cow, d, hu0, ves0, espejo, rng_a, alpha,
                           ves_v, cambios, ves_v, None, {})
                break
            try:
                ves2, thr, oinfo = E.occlude(ves_v, arteria, frac=frac, rng=rng)
            except ValueError as e:
                errs.append(f"{pid_} (alpha {alpha}): {e}")
                continue
            hallado = (pid_, cow, d, hu0, ves0, espejo, rng_a, alpha,
                       ves_v, cambios, ves2, thr, oinfo)
            break
        if hallado:
            break
    if not hallado:
        raise ValueError(f"ningun donante puede ocluir {arteria}: " + "; ".join(errs))

    (pid_, cow, d, hu0, ves0, espejo, rng_a, alpha,
     ves_v, cambios, ves2, thr, oinfo) = hallado

    pid = pid_
    hu = hu0                      # el ya reflejado/deformado, NO el del banco
    if not quiet:
        print(f"anatomia {pid} (familia CoW {cow}{', espejo' if espejo else ''}, "
              f"deform {alpha}) | {variante} | oclusion {arteria or 'ninguna'} | {dev}",
              flush=True)

    # --- CTA: el arbol apagado pierde el contraste ---------------------------
    hu2 = H.render(hu, ves0, ves2, thr, rng=rng)

    # De-identificacion del CTA: el sustrato es un donante REAL (espejado + deformado) con
    # su cara y craneo. Se quita la cara -- todo lo anterior al cerebro, sin tocar ningun
    # vaso -- y se re-centra la affine para borrar la posicion de mesa. `aff` (canonica al
    # origen) va a TODAS las salidas del espacio canonico (cta/vasos/trombo/mascara).
    hu2, cara_cta = D.deface_head(hu2, ves2)
    aff = D.deid_affine()

    # --- el puente: vasos apagados -> territorio de parenquima ---------------
    # `oinfo["apagados"]` incluye segmentos que YA estaban desconectados antes de
    # ocluir: tipicamente basura del pseudo-etiquetado (un 3rd-A3 sin su 3rd-A2 nace
    # huerfano y nunca fue alcanzable desde una fuente). En `hybrid` eso era inocuo
    # -- solo significaba "no pintes ese vaso" -- pero aqui se traduciria en
    # territorio infartado. Solo cuenta lo que pasa de perfundido a NO perfundido.
    base = E.reachable(ves_v)
    apagados = [s for s in oinfo.get("apagados", []) if s in base]
    huerfanos = [s for s in oinfo.get("apagados", []) if s not in base]

    vol_terr = {}
    if arteria:
        vol_terr = T.territorios(apagados, artery=arteria, frac=frac,
                                 distal_muerto=not oinfo["distal_retrogrado"])
    terr = sorted(vol_terr)

    # --- NCCT: la hipodensidad en ese territorio -----------------------------
    # Si el CTA se ha reflejado, el cerebro del NCCT tambien (`espejo`), para que la
    # oclusion y el infarto no acaben en hemisferios opuestos. La MASCARA no se refleja:
    # sale del atlas, que no se toca.
    ncct_ref = nib.load(os.path.join(T.R2.BANK, f"{pid}_ncct.nii.gz"))
    mask, vol = T.synth_mask(vol_terr, cow, seed, ncct_ref.shape)

    # El cerebro de fondo va DESACOPLADO del donante del CTA (`ncct_source="banco"`). El NCCT
    # y el CTA viven en espacios distintos y no se superponen nunca, asi que atarlos al mismo
    # donante no compraba nada y costaba carisimo: la familia 0 solo tiene 2 donantes, de ahi
    # que el NCCT "siempre fuera el mismo". Lo que ata el caso es el TERRITORIO (atlas MNI),
    # y eso se mantiene sea cual sea el fondo. Con "mismo" se recupera el comportamiento viejo.
    #
    # NO se deforma. La deformacion elastica era el parche para fingir variedad con 2 cerebros
    # y emborronaba el parenquima (aspecto "deconstruido": surcos estirados, craneo abollado)
    # ademas de separar el cerebro del atlas del que sale la mascara. Con 9 fondos reales no
    # hace falta el parche.
    if ncct_source == "mismo":
        aff_n, ncct = T.infer_ncct(pid, mask, cow, dev, fuerza, espejo=espejo, fase=fase)
        ncct_base = f"{pid} (mismo paciente)"
    else:
        got = T.pick_ncct_base(seed, source=ncct_source, variante=variante)
        if got is None:
            raise ValueError("no hay ningun NCCT base disponible (banco vacio)")
        base, aff_b, bname, bcow, en_dominio = got
        if ncct_source == "control" and en_dominio and not quiet:
            print("AVISO: pool de control vacio; se usa el banco.", flush=True)
        # cow_id del CEREBRO (no el del CTA): es la condicion con la que se entreno el Gen3D
        # para ese cerebro. Un control externo esta fuera de dominio -> sin Gen3D.
        aff_n, ncct = T.infer_ncct(pid, mask, bcow, dev, fuerza, base=base, aff=aff_b,
                                   use_gen=en_dominio, fase=fase)
        ncct_base = bname

    # De-identificacion del NCCT: tambien es un cerebro REAL (del banco/control), con cara
    # y craneo. Se quita la cara en su ventana de cerebro ([0,80]: aire=10, hueso=60,
    # relleno=0), protegiendo el INFARTO para no borrarlo (la mascara es el ground truth).
    # La affine NO se re-centra: el NCCT vive en MNI, una plantilla estandar sin PHI.
    ncct, cara_ncct = D.deface_head(ncct, (mask > 0).astype(np.uint8),
                                    fill=0.0, air=10.0, bone=60.0)

    # --- salidas -------------------------------------------------------------
    C.save_nifti(out / "cta_oclusion.nii.gz", hu2.astype(np.float32), aff)
    C.save_nifti(out / "vasos.nii.gz", ves2.astype(np.int16), aff)
    # el arbol SANO de este paciente (ya reflejado/deformado): es la referencia contra
    # la que se pinta el rojo/verde. El de `bank_sem` ya no vale, no es esta anatomia.
    C.save_nifti(out / "vasos_sano.nii.gz", ves0.astype(np.int16), aff)
    # el trombo, para poder senalar la oclusion en el MIP
    C.save_nifti(out / "trombo.nii.gz",
                 (thr if thr is not None else np.zeros(ves2.shape, bool)).astype(np.uint8), aff)
    nib.save(nib.Nifti1Image(ncct, aff_n), out / "ncct_infarto.nii.gz")
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), aff_n), out / "mask.nii.gz")
    # las regiones de cara borrada, para que la de-identificacion sea auditable/reversible
    C.save_nifti(out / "deface_mask.nii.gz", cara_cta.astype(np.uint8), aff)
    nib.save(nib.Nifti1Image(cara_ncct.astype(np.uint8), aff_n), out / "deface_mask_ncct.nii.gz")

    v = ves2 > 0
    lost = (ves0 > 0) & (ves2 == 0)
    info = {
        "anatomia": pid, "familia_cow": cow, "espejo": espejo, "deformacion": alpha,
        "ncct_base": ncct_base,
        "defaced": True, "affine_recentrada": True,
        "deface_voxeles_cta": int(cara_cta.sum()), "deface_voxeles_ncct": int(cara_ncct.sum()),
        "fase": fase,
        "variante": variante,
        "cambios_variante": cambios, "arteria": arteria, "frac": frac,
        "apagados": apagados,
        "huerfanos_pseudolabel": huerfanos,   # ya desconectados antes de ocluir: no cuentan
        "relleno_retrogrado": oinfo.get("distal_retrogrado"),
        "ramas_rescatadas": oinfo.get("ramas_distales_rescatadas", []),
        "territorios": [T.NOMBRE[t] for t in terr],
        "territorios_id": terr,
        "ml_por_territorio": {T.NOMBRE[t]: v for t, v in sorted(vol_terr.items())},
        "volumen_infarto_ml": round(vol, 1),      # medido sobre la mascara, no el pedido
        "HU_vaso_permeable": round(float(hu2[v].mean()), 0) if v.any() else None,
        "HU_vaso_apagado": round(float(hu2[lost].mean()), 0) if lost.any() else None,
        "HU_trombo": round(float(hu2[thr].mean()), 0) if thr is not None and thr.any() else None,
        "HU_infarto": round(float(ncct[mask].mean()), 1) if mask.any() else None,
        "HU_contralateral": None,
    }
    if mask.any():   # el contraste que de verdad se ve: infarto vs su espejo sano
        info["HU_contralateral"] = round(float(ncct[mask[::-1]].mean()), 1)

    (out / "caso.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    if not quiet:
        print(json.dumps(info, indent=1, ensure_ascii=False))
        if not terr and arteria:
            print("\n-> Sin territorio infartado: el CoW rescata todo el arbol distal. "
                  "El NCCT sale sano (oclusion asintomatica).")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variante", default="CoW completo", choices=list(FAMILIA))
    ap.add_argument("--arteria", default=None,
                    help="sitio de oclusion (%s). Sin esto, caso sano." % ", ".join(C.OCCLUDABLE))
    ap.add_argument("--frac", type=float, default=None,
                    help="posicion del trombo a lo largo del segmento, 0=proximal 1=distal")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fuerza", type=float, default=1.5,
                    help="cuanto se ve la hipodensidad del infarto")
    ap.add_argument("--pid", default=None, help="fija la anatomia (p.ej. pat_cow0_00)")
    ap.add_argument("--no-variar", action="store_true",
                    help="usa el donante tal cual, sin reflejo ni deformacion")
    ap.add_argument("--fase", default=T.FASE_DEF, choices=tuple(T.FASES),
                    help="fase temporal del infarto (densidad y conspicuidad)")
    ap.add_argument("--ncct-source", default="banco", choices=("banco", "mismo", "control"),
                    help="cerebro de fondo del NCCT: banco (9, desacoplado), mismo (donante "
                         "del CTA), control (pool externo de sanos)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    generar(a.variante, a.arteria, a.frac, a.seed, a.fuerza, a.out,
            pid=a.pid, variar=not a.no_variar, ncct_source=a.ncct_source, fase=a.fase)


if __name__ == "__main__":
    main()
