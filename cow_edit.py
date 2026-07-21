"""Edicion del mapa de vasos: variante de CoW + sitio de oclusion.

Este modulo es el que acopla las dos cosas. El polinomio de Willis se trata como
un GRAFO de flujo: las fuentes son las dos ICA cervicales y las dos vertebrales,
y un segmento solo se opacifica si sigue siendo alcanzable desde alguna fuente.

Consecuencia (lo que se busca): ocluir la ICA derecha con Acom presente deja la
ACA derecha rellena por colateral cruzada -> conserva contraste; sin Acom, la
ACA derecha tambien se apaga. La variante cambia el resultado de la oclusion,
igual que en la clinica.

El trombo se situa a una fraccion del recorrido del segmento, y las ramas que
salen PROXIMALES al trombo siguen perfundidas (p. ej. una oclusion de ICA distal
al origen del Pcom se comporta distinto de una proximal).
"""
import numpy as np
from scipy import ndimage as ndi

import common as C

V = C.VID

# --- grafo de flujo DIRIGIDO -------------------------------------------------
# El arbol arterial transmite en sentido anterogrado. Solo las comunicantes
# (Acom, Pcom) son bidireccionales: son las que rescatan territorios.
#
# Esto es lo que hace que una oclusion en T carotideo deje el MCA vacio aunque
# el Acom rellene la ACA (el flujo NO vuelve retrogrado por el terminus), y que
# una oclusion de ICA proximal al Pcom SI rellene todo el hemisferio.
FLOW = [
    ("R-ICA-C1-C5", "R-ICA-C6-C7"), ("L-ICA-C1-C5", "L-ICA-C6-C7"),
    ("R-ICA-C6-C7", "R-M1"), ("R-M1", "R-M2"), ("R-M2", "R-M3"),
    ("L-ICA-C6-C7", "L-M1"), ("L-M1", "L-M2"), ("L-M2", "L-M3"),
    ("R-ICA-C6-C7", "R-A1A2"), ("R-A1A2", "R-A3"),
    ("L-ICA-C6-C7", "L-A1A2"), ("L-A1A2", "L-A3"),
    ("R-ICA-C6-C7", "R-AChA"), ("L-ICA-C6-C7", "L-AChA"),
    ("R-ICA-C6-C7", "R-OA"), ("L-ICA-C6-C7", "L-OA"),
    ("R-VA", "BA"), ("L-VA", "BA"),
    ("R-VA", "R-PICA"), ("L-VA", "L-PICA"),
    ("BA", "R-AICA"), ("BA", "L-AICA"),
    ("BA", "R-SCA"), ("BA", "L-SCA"),
    ("BA", "R-P1P2"), ("BA", "L-P1P2"),
    ("R-P1P2", "R-P3P4"), ("L-P1P2", "L-P3P4"),
    ("3rd-A2", "3rd-A3"),
]
# comunicantes: llevan flujo en los dos sentidos (colateralidad primaria)
COMM = [
    ("Acom", "R-A1A2"), ("Acom", "L-A1A2"), ("Acom", "3rd-A2"),
    ("R-Pcom", "R-P1P2"), ("R-Pcom", "R-ICA-C6-C7"),
    ("L-Pcom", "L-P1P2"), ("L-Pcom", "L-ICA-C6-C7"),
]
EDGES = FLOW + COMM                     # vecindad anatomica (sin direccion)
SOURCES = ["R-ICA-C1-C5", "L-ICA-C1-C5", "R-VA", "L-VA"]


def _directed(nodes):
    """Adyacencia dirigida entre los segmentos presentes."""
    adj = {}
    for a, b in FLOW:
        if a in nodes and b in nodes:
            adj.setdefault(a, set()).add(b)
    for a, b in COMM:
        if a in nodes and b in nodes:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    return adj

# padre en el arbol de flujo -> define la direccion proximal->distal del segmento
PARENT = {
    "R-ICA-C6-C7": "R-ICA-C1-C5", "L-ICA-C6-C7": "L-ICA-C1-C5",
    "R-M1": "R-ICA-C6-C7", "L-M1": "L-ICA-C6-C7",
    "R-M2": "R-M1", "L-M2": "L-M1", "R-M3": "R-M2", "L-M3": "L-M2",
    "R-A1A2": "R-ICA-C6-C7", "L-A1A2": "L-ICA-C6-C7",
    "R-A3": "R-A1A2", "L-A3": "L-A1A2",
    "BA": "R-VA", "R-P1P2": "BA", "L-P1P2": "BA",
    "R-P3P4": "R-P1P2", "L-P3P4": "L-P1P2",
    "R-Pcom": "R-P1P2", "L-Pcom": "L-P1P2",
    "R-SCA": "BA", "L-SCA": "BA", "R-AICA": "BA", "L-AICA": "BA",
    "R-PICA": "R-VA", "L-PICA": "L-VA",
    "R-AChA": "R-ICA-C6-C7", "L-AChA": "L-ICA-C6-C7",
    "R-OA": "R-ICA-C6-C7", "L-OA": "L-ICA-C6-C7",
    "Acom": "R-A1A2", "3rd-A2": "Acom", "3rd-A3": "3rd-A2",
}

# Ramas que nacen EN EL EXTREMO distal del segmento (la bifurcacion terminal),
# no a lo largo de el. Un trombo en cualquier punto del segmento las deja siempre
# distales: el M1 y la ACA nacen en el terminus carotideo, no a media ICA.
# Las demas ramas (Pcom, AChA, OA, SCA, AICA, PICA, Acom) son colaterales que
# nacen a lo largo, y ahi si se mide el origen sobre el vaso.
TERMINAL_CHILDREN = {
    "R-ICA-C1-C5": ["R-ICA-C6-C7"], "L-ICA-C1-C5": ["L-ICA-C6-C7"],
    "R-ICA-C6-C7": ["R-M1", "R-A1A2"], "L-ICA-C6-C7": ["L-M1", "L-A1A2"],
    "R-M1": ["R-M2"], "L-M1": ["L-M2"], "R-M2": ["R-M3"], "L-M2": ["L-M3"],
    "R-A1A2": ["R-A3"], "L-A1A2": ["L-A3"],
    "R-VA": ["BA"], "L-VA": ["BA"],
    "BA": ["R-P1P2", "L-P1P2"],
    "R-P1P2": ["R-P3P4"], "L-P1P2": ["L-P3P4"],
    "3rd-A2": ["3rd-A3"],
}

ADJ = {}
for a, b in EDGES:
    ADJ.setdefault(a, set()).add(b)
    ADJ.setdefault(b, set()).add(a)


def present(ves, name, min_vox=20):
    return int((ves == V[name]).sum()) >= min_vox


def _bfs(adj, sources):
    seen, stack = set(sources), list(sources)
    while stack:
        for m in adj.get(stack.pop(), ()):
            if m not in seen:
                seen.add(m)
                stack.append(m)
    return seen


def reachable(ves, split=None):
    """Segmentos presentes alcanzables por el flujo desde alguna fuente.

    `split` = (artery, {vecino: "prox"|"dist"}) parte el segmento ocluido en dos
    nodos SIN arista entre ellos (el trombo no deja pasar). Cada rama vecina se
    engancha al lado donde realmente nace. Asi, un trombo proximal al origen del
    Pcom deja que el Pcom rellene la ICA distal (y el M1); uno distal, no.
    """
    nodes = {n for n in V if present(ves, n)}
    if split is None:
        return _bfs(_directed(nodes), {s for s in SOURCES if s in nodes})

    art, side = split
    P, D = art + "|prox", art + "|dist"
    adj = _directed(nodes - {art})
    for a, b in FLOW:                          # aristas dirigidas que tocan `art`
        if a not in nodes or b not in nodes:
            continue
        if a == art:                           # art -> b : sale del lado donde nace b
            adj.setdefault(P if side.get(b, "dist") == "prox" else D, set()).add(b)
        elif b == art:                         # a -> art : entra por donde a lo toca
            adj.setdefault(a, set()).add(P if side.get(a, "prox") == "prox" else D)
    for a, b in COMM:                          # comunicantes: bidireccionales
        if a not in nodes or b not in nodes:
            continue
        if art in (a, b):
            other = b if a == art else a
            end = P if side.get(other, "dist") == "prox" else D
            adj.setdefault(end, set()).add(other)
            adj.setdefault(other, set()).add(end)

    srcs = {s for s in SOURCES if s in nodes and s != art}
    if art in SOURCES:
        srcs.add(P)                            # la entrada proximal sigue abierta
    return _bfs(adj, srcs)


INFLOW = {}
for _a, _b in FLOW:
    INFLOW.setdefault(_b, []).append(_a)       # quien alimenta a cada segmento


def _seed_of(ves, name):
    """Voxel PROXIMAL del segmento: el que toca al vaso que lo alimenta.

    Hay que medir la distancia al conjunto del padre, no a su centroide: la ICA
    cervical es larga y curva, su centroide cae lejos de la union con el sifon y
    eso invertia la geodesica (el seed acababa en el extremo distal).
    """
    m = ves == V[name]
    idx = np.array(np.nonzero(m))
    par = np.zeros(m.shape, bool)
    for p in INFLOW.get(name, []):
        if present(ves, p):
            par |= ves == V[p]
    if par.any():
        dt = ndi.distance_transform_edt(~par)
        return idx[:, int(np.argmin(dt[m]))]
    return idx[:, int(np.argmin(idx[2]))]      # sin padre (fuente): el mas caudal


def geodesic(mask, seed):
    """Distancia geodesica dentro de `mask` desde `seed`, en pasos de cara (6-conn).

    Se usa 6-conectividad a proposito: da bastantes mas pasos que la 26-conn a lo
    largo del vaso, y hace falta esa resolucion para decidir si una rama nace
    proximal o distal al trombo (la ICA C6-C7 son ~14 pasos en 26-conn, muy poco).
    """
    dist = np.full(mask.shape, -1, np.int32)
    cur = np.zeros(mask.shape, bool)
    cur[tuple(seed)] = True
    dist[cur] = 0
    st = ndi.generate_binary_structure(3, 1)
    d = 0
    while True:
        nxt = ndi.binary_dilation(cur, st) & mask & (dist < 0)
        if not nxt.any():
            break
        d += 1
        dist[nxt] = d
        cur = nxt
    return dist


def occlude(ves, artery, frac=0.5, rng=None, thrombus_mm=4.0):
    """Ocluye `artery` a la fraccion `frac` de su recorrido proximal->distal.

    Devuelve (ves_ocluido, mascara_trombo, info). Borra del mapa:
      - la porcion distal al trombo del propio segmento,
      - todo segmento que deje de ser alcanzable desde una fuente (el arbol
        distal se apaga, salvo que una colateral del CoW lo rescate).
    """
    rng = rng or np.random.default_rng()
    if not present(ves, artery):
        raise ValueError(f"{artery} no esta presente en este caso")

    ves = ves.copy()
    m = ves == V[artery]
    seed = _seed_of(ves, artery)
    gd = geodesic(m, seed)
    valid = gd >= 0
    dmax = int(gd[valid].max())
    if dmax < 4:
        raise ValueError(f"{artery} demasiado corto para ocluir ({dmax} vox)")
    cut = max(2, min(dmax - 1, int(round(frac * dmax))))

    # cada rama nace proximal o distal al trombo: eso decide si sigue perfundida
    term = set(TERMINAL_CHILDREN.get(artery, ()))
    side = {}
    for nb in ADJ.get(artery, ()):
        if not present(ves, nb):
            continue
        if nb in term:
            side[nb] = "dist"                  # nace en la bifurcacion terminal
            continue
        if nb in INFLOW.get(artery, []):
            side[nb] = "prox"                  # de aqui viene el flujo
            continue
        nbm = ndi.binary_dilation(ves == V[nb], ndi.generate_binary_structure(3, 3))
        touch = gd[nbm & valid]                # colateral: se mide donde nace
        origin = float(np.percentile(touch, 10)) if touch.size else np.inf
        side[nb] = "prox" if origin <= cut else "dist"

    nvox = max(1, int(round(thrombus_mm / C.SPACING)))
    thr_band = valid & (gd > cut) & (gd <= cut + nvox)
    ves[thr_band] = 0                          # el coagulo no capta contraste

    keep = reachable(ves, split=(artery, side))
    dist_ok = (artery + "|dist") in keep       # relleno retrogrado (p.ej. via Pcom)
    if not dist_ok:
        ves[valid & (gd > cut)] = 0            # sin contraste distal al trombo

    lost = []
    for name, vid in V.items():
        if name == artery or name in keep or not present(ves, name):
            continue
        ves[ves == vid] = 0                    # arbol desconectado de toda fuente
        lost.append(name)

    rescued = sorted(n for n in ADJ.get(artery, ())
                     if present(ves, n) and n in keep and side.get(n) == "dist")
    info = {"artery": artery, "frac": float(frac), "cut": cut, "len_vox": dmax,
            "apagados": sorted(lost), "distal_retrogrado": bool(dist_ok),
            "ramas_distales_rescatadas": rescued}
    return ves, thr_band, info


# --- variantes de CoW --------------------------------------------------------
# Cada variante es una lista de operaciones sobre el mapa de vasos.
VARIANTS = {
    "CoW completo":            [],
    "Pcom derecha ausente":    [("del", "R-Pcom")],
    "Pcom izquierda ausente":  [("del", "L-Pcom")],
    "Pcom bilateral ausente":  [("del", "R-Pcom"), ("del", "L-Pcom")],
    "Acom ausente":            [("del", "Acom")],
    "PCA fetal derecha":       [("fetal", "R")],
    "PCA fetal izquierda":     [("fetal", "L")],
    "PCA fetal bilateral":     [("fetal", "R"), ("fetal", "L")],
    "A1 hipoplasica derecha":  [("hypo", "R-A1A2")],
    "A1 hipoplasica izquierda": [("hypo", "L-A1A2")],
    "A1 derecha ausente":      [("delprox", "R-A1A2")],
    "A1 izquierda ausente":    [("delprox", "L-A1A2")],
}


def _erode_segment(ves, name, keep_frac=0.5):
    """Adelgaza un segmento (hipoplasia): erosiona su lumen."""
    m = ves == V[name]
    e = ndi.binary_erosion(m, ndi.generate_binary_structure(3, 1))
    if e.sum() < 0.15 * m.sum():
        return ves
    ves[m & ~e] = 0
    return ves


def _del_proximal(ves, name, frac=0.45):
    """Borra la mitad proximal de un segmento (A1 ausente = solo queda A2)."""
    m = ves == V[name]
    gd = geodesic(m, _seed_of(ves, name))
    valid = gd >= 0
    dmax = int(gd[valid].max())
    ves[valid & (gd <= int(frac * dmax))] = 0
    return ves


def apply_variant(ves, variant):
    """Aplica la variante al mapa de vasos. Devuelve (ves, info)."""
    if variant not in VARIANTS:
        raise ValueError(f"variante desconocida: {variant}")
    ves = ves.copy()
    done = []
    for op, arg in VARIANTS[variant]:
        if op == "del":
            if present(ves, arg):
                ves[ves == V[arg]] = 0
                done.append(f"borrado {arg}")
        elif op == "hypo":
            if present(ves, arg):
                ves = _erode_segment(ves, arg)
                done.append(f"hipoplasia {arg}")
        elif op == "delprox":
            if present(ves, arg):
                ves = _del_proximal(ves, arg)
                done.append(f"A1 ausente ({arg})")
        elif op == "fetal":
            # PCA fetal: la PCA se llena desde la ICA por un Pcom grande y el
            # P1 (origen del P1P2 en la BA) es hipoplasico/ausente.
            side = arg
            pcom, p1 = f"{side}-Pcom", f"{side}-P1P2"
            if present(ves, p1):
                m = ves == V[p1]
                gd = geodesic(m, _seed_of(ves, p1))
                valid = gd >= 0
                ves[valid & (gd <= int(0.3 * int(gd[valid].max())))] = 0   # quita P1
                done.append(f"P1 {side} ausente (fetal)")
            if present(ves, pcom):              # engrosa el Pcom
                d = ndi.binary_dilation(ves == V[pcom],
                                        ndi.generate_binary_structure(3, 1))
                ves[d & (ves == 0)] = V[pcom]
                done.append(f"Pcom {side} engrosado")
    return ves, done


def available_variants(ves):
    """Variantes aplicables a este caso (necesitan que el segmento exista)."""
    out = []
    for name, ops in VARIANTS.items():
        ok = True
        for op, arg in ops:
            seg = f"{arg}-Pcom" if op == "fetal" else arg
            if op == "fetal":
                ok &= present(ves, f"{arg}-Pcom") and present(ves, f"{arg}-P1P2")
            elif not present(ves, seg):
                ok = False
        if ok:
            out.append(name)
    return out


def occludable(ves):
    return [a for a in C.OCCLUDABLE if present(ves, a, min_vox=60)]
