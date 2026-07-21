"""De-identificacion de la CTA: quita la cara y borra las coordenadas de mesa.

El riesgo de privacidad del pipeline hibrido no esta en la cabecera (el NIfTI sale
limpio: descrip/aux_file/db_name vacios, y el npz de origen solo trae sem/ves/affine/
src) sino en los VOXELES: el sustrato es un donante REAL -- espejado y deformado, pero
con su CARA y su CRANEO intactos. La superficie facial de un TC se reconstruye y se
reconoce, asi que un reflejo + una deformacion de ~1.5 mm no de-identifican. Este modulo
si.

Dos operaciones, ninguna toca el grafo ni el render:

  deface_head(hu, ves) -> (hu_sin_cara, mascara)
      Borra todo lo que queda ANTERIOR al cerebro (piel, nariz, ojos, hueso facial),
      dejando intacto lo intracraneal. Como el corte es ESTRICTAMENTE anterior al borde
      del cerebro nunca toca el parenquima; y la mascara de vasos, dilatada, se protege
      aparte para no cortar la A3/OA anteriores ni la ICA cervical. Reversible: devuelve
      la mascara de lo borrado, que se guarda como un NIfTI aparte.

  deid_affine() -> affine re-centrada
      La affine del donante arrastra su posicion de mesa en el escaner (la traslacion
      mundo de la adquisicion real; p. ej. z=+1452 mm). Se sustituye por la affine
      canonica centrada en el origen -- mismo espaciado y orientacion RAS, sin las
      coordenadas del paciente. Hay que aplicarla a TODAS las salidas que comparten
      espacio (cta/vasos/trombo/mascara) para no descuadrarlas entre si.
"""
import numpy as np
from scipy import ndimage as ndi

import common as C

AIR_FILL = -1000.0   # HU de aire: la cara borrada queda negra


def _head(vol, air=-50.0):
    """La cabeza solida (piel hacia dentro), como mayor componente conexo.

    OJO con el umbral de aire, que depende de la ESCALA de intensidad:
      - CTA canonico: HU recortado a la ventana (-100, 900), fondo -100 -> air=-50.
        Un `>-300` es cierto en todo el volumen y da la caja entera.
      - NCCT (MNI): ventana de cerebro [0, 80], fondo 0 -> air=10.
    `fill_holes` rellena senos y cavidades aereas internas para dejar la cabeza maciza."""
    solid = ndi.binary_fill_holes(vol > air)
    lab, n = ndi.label(solid)
    if n == 0:
        return solid
    big = 1 + int(np.argmax(np.bincount(lab.ravel())[1:]))
    return lab == big


def _intracranial(vol, air=-50.0, bone=300.0):
    """Mascara binaria de la cavidad intracraneal: el mayor componente conexo de
    (cabeza & ~craneo). `bone` marca el hueso -- y en el CTA tambien el contraste
    brillante, lo que solo abre agujeritos finos alrededor de los vasos: el mayor
    componente sigue siendo el parenquima. En el NCCT [0,80] el hueso satura a ~80,
    asi que bone=60."""
    skull = ndi.binary_dilation(vol > bone, iterations=1)
    cav = _head(vol, air) & ~skull
    lab, n = ndi.label(cav)
    if n == 0:
        return np.zeros(vol.shape, bool)
    big = 1 + int(np.argmax(np.bincount(lab.ravel())[1:]))
    return ndi.binary_erosion(lab == big, iterations=2)


def deface_head(vol, ves, gap=4, vessel_margin=2, fill=AIR_FILL, air=-50.0, bone=300.0):
    """Borra la cara. Devuelve (vol_defaced, mascara_bool_de_lo_borrado).

    `y` es el eje P->A (indice creciente = anterior = hacia la cara). Para cada columna
    (x,z) que tenga cerebro, el borde anterior del cerebro es el mayor indice `y` de la
    cavidad; todo lo que queda por delante (y > borde + `gap`) es cara y se borra. Por ser
    estrictamente anterior a la cavidad nunca alcanza el parenquima. Los voxeles de vaso
    (dilatados `vessel_margin`) se excluyen para no cortar arterias anteriores (A3, OA) ni
    la ICA cervical -- en el NCCT no hay vasos etiquetados, se pasa `ves` a ceros.

    `fill`/`air`/`bone` dependen de la escala de intensidad: CTA canonico HU (-1000/-50/300),
    NCCT en ventana de cerebro (0/10/60). Donde no hay cerebro en la columna (slices por
    debajo de la cavidad) no se borra nada: solo se de-identifica lo que rodea al cerebro.
    """
    brain = _intracranial(vol, air, bone)
    ny = vol.shape[1]
    yidx = np.arange(ny)[None, :, None]                 # (1, y, 1)

    has = brain.any(axis=1)                             # (x, z): columna con cerebro
    ant = np.where(brain, yidx, -1).max(axis=1)        # (x, z): borde anterior, -1 si no hay
    face = has[:, None, :] & (yidx > ant[:, None, :] + gap)

    protect = ndi.binary_dilation(ves > 0, iterations=vessel_margin) if vessel_margin > 0 else ves > 0
    face &= ~protect

    out = vol.copy()
    out[face] = fill
    return out.astype(np.float32), face


def deid_affine():
    """Affine canonica centrada en el origen (sin la posicion de mesa del donante).
    Conserva espaciado y orientacion RAS del grid canonico."""
    return C.target_affine(np.zeros(3)).astype(np.float32)
