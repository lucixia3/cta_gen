# cta_gen — ictus sintético con variante de CoW y sitio de oclusión

Genera el caso a partir de dos parámetros: **la variante del polígono de Willis** y **el
sitio de oclusión**. `hybrid.py` da la angio-TC; `caso.py` da además el **NCCT con el
infarto** del mismo paciente. Cada llamada produce un paciente distinto.

```
donante --> [variante] --> [oclusión + hemodinámica] --> árbol editado --> CTA
```

```powershell
$py = "C:\Desactivar_Respaldo\ISLES_train_package\.venv\Scripts\python.exe"

& $py hybrid.py --variant "Acom ausente" --artery R-ICA-C6-C7 --frac 0.9
& $py batch.py --n 200 --out dataset      # dataset + manifest.csv (verdad de terreno)
& $py samples2d.py                        # láminas 2D: sano | ocluido | árbol
& $py qc.py --labels                      # valida la hemodinámica (sin GPU)
```

## Lo que hace que funcione: el grafo hemodinámico

`cow_edit.py` trata el polígono como un **grafo de flujo dirigido**. Las fuentes son
las dos carótidas cervicales y las dos vertebrales; un segmento solo se opacifica si
sigue siendo **alcanzable desde una fuente**. Las comunicantes (Acom, Pcom) son las
únicas aristas bidireccionales: son las que rescatan territorios.

El segmento ocluido se parte en dos nodos (proximal / distal) sin arista entre ellos
—el trombo— y cada rama se engancha al lado donde realmente nace. Por eso **el mismo
vaso ocluido en dos puntos distintos da dos cuadros clínicos distintos**.

De ahí sale sola la clínica, sin reglas ad hoc:

| Oclusión | CoW completo | Variante |
|---|---|---|
| ICA en el T carotídeo | MCA muerto, ACA rescatada por el Acom | **sin Acom**: cae también la ACA |
| ICA proximal al Pcom | no cae nada (relleno retrógrado) | **sin Pcom**: cae todo el hemisferio |
| Basilar | solo las cerebelosas; las PCA se salvan | **sin Pcom**: cae todo el territorio posterior |
| Carótida cervical | nada (la oclusión asintomática clásica) | |
| M1 | se apagan M2 y M3 | |

## El caso completo: CTA + NCCT del mismo paciente (`caso.py`)

`hybrid.py` da la CTA. `caso.py` da además el **NCCT con el infarto**, derivado del
**mismo grafo**: el territorio que se infarta es el del árbol que se apagó.

```powershell
& $py bank.py                                                     # una vez (GPU, ~2 min)
& $py caso.py --variante "Acom ausente" --arteria R-ICA-C6-C7 --frac 0.9 --out casos/B
& $py batch_casos.py --out dataset_casos       # lote de 18 casos + manifest.csv
& $py qc_caso.py dataset_casos/*/              # ¿el infarto cae en el lado de la oclusión?
& $py samples_caso.py --dir dataset_casos      # lámina: árbol apagado | NCCT con el infarto
& $py mip_caso.py --dir dataset_casos          # MIP angiográfico con la oclusión señalada
```

Sale `cta_oclusion.nii.gz` (slab canónico), `ncct_infarto.nii.gz` + `mask.nii.gz` (MNI),
`vasos.nii.gz`, `trombo.nii.gz` y un `caso.json` con la variante, los apagados, los
territorios y las HU.

`mip_caso.py` da el angiograma como se mira de verdad: MIP axial y coronal, el trombo
marcado con un círculo, y el árbol que ha perdido el contraste en rojo. El cráneo se
sustrae con `qc.bone_subtract` — sin eso el MIP de cabeza completa sale blanco.

**La variante decide el infarto**, que es el punto:

| Variante | Oclusión | Apaga | Infarto |
|---|---|---|---|
| CoW completo | ICA (sifón) | M1, M2, M3 | MCA + profundo — **81 ml** (el Acom rescata la ACA) |
| **Acom ausente** | ICA (sifón) | + A1A2 | **+ ACA — 105 ml** |
| CoW completo | ICA **cervical** | nada | **ninguno** (la oclusión asintomática) |
| CoW completo | **M1** | M2, M3 | MCA + **profundo** (perforantes) — 81 ml |
| CoW completo | **M2** | M3 | MCA sola, **respeta el profundo** — 30 ml |
| Pcom bilat. ausente | basilar | P1P2, P3P4, SCA... | ambas PCA + ambas fosas posteriores |

### Las tres piezas

**`bank.py`** — la anatomía sale del banco de `lesion2ncct_portable`: 9 cerebros con
CTA *y* NCCT **del mismo paciente**, y el NCCT alineado vóxel a vóxel con el atlas de
territorios. Es la única fuente que permite que las dos imágenes sean el mismo enfermo.
Hace dos cosas, y las dos son imprescindibles:

*Completa las etiquetas.* El banco trae solo las 15 clases de TopCoW: **no están las ICA
cervicales ni las vertebrales, que son las fuentes del grafo** — sobre esas etiquetas
`reachable()` arranca sin fuentes y apaga el árbol entero — ni el M2/M3 que hay que
apagar. Se completan a 36 clases con `ckpt/seg_best.pt`, igual que `pseudolabel.py` hace
con TopCoW. Se cachea en `bank_sem/`.

*Arregla la lateralidad.* **Un tercio del banco trae las etiquetas del polígono mal
lateralizadas.** En `pat_cow1_02` el `R-M1` del GT está 43 mm a la **izquierda**. Es un
fallo de los datos, y lo hereda igual `lesion2ncct/run_case.py`, que usa esas mismas
etiquetas: la oclusión la sitúa la etiqueta del donante y el infarto el atlas MNI (que sí
sabe dónde está la derecha), así que acaban en **hemisferios opuestos** — y cada imagen
por separado parece perfecta.

El juez es el propio segmentador: se entrenó con TopAneu, cuya convención R/L es la
buena, y predice también las 15 clases del polígono. Se le pregunta, clase a clase, si los
vóxeles que el GT llama `R-M1` él los llama R o L. **Hay que votar por clase, no en
agregado**: la ICA y la Acom viven pegadas a la línea media y ahí el segmentador (Dice
0.71) se equivoca de lado a menudo, así que el voto global da empates engañosos. Con el
voto por clase:

| | |
|---|---|
| todas las clases en contra | espejo global limpio → se endereza con `LR_SWAP` (1 donante) |
| las clases se contradicen | GT **incoherente** → se **descarta** (3 donantes) |

El caso incoherente es real: en `pat_cow0_02` el M1 está espejado (`0/722`) pero la P1P2
no (`763/0`). Un `LR_SWAP` arreglaría una y rompería la otra, así que no hay forma de
garantizar el lado y el donante no vale. Quedan **6 de 9**, dos por familia de CoW.

Tampoco sirve estimar la línea media para decidir esto: ni del árbol (el pseudo-etiquetado
no cubre igual los dos lados y la corre varios mm) ni de la cabeza (el slab canónico está
**dentro** del cráneo, la máscara es todo unos y no hay borde que centrar).

**`territorios.py`** — el puente que faltaba: `apagados` (nombres de segmento) → ids del
atlas. Sustituye a la tabla `OCC_TERR` de lesion2ncct, que era fija
(oclusión → territorio) y **en la que la variante no cambiaba el territorio infartado**:
una ICA ocluida infartaba lo mismo con Acom que sin él. Aquí el territorio sale del grafo.

Tres cosas que la tabla vieja no hacía:
- Usa los territorios **9 y 10** del atlas (perforantes profundas, ~13 ml/lado), que
  `OCC_TERR` ignoraba. Son los que separan un M1 ocluido (se lleva las lenticuloestriadas)
  de un M2 (que las respeta). Con la tabla vieja, un M2 podía infartar el núcleo lenticular.
- Cada territorio se queda con el volumen del segmento **más proximal** que lo alimenta,
  así que perder el M1 infarta todo el territorio de la ACM aunque el M2 también esté en
  la lista.
- **Coloca una lesión por hemisferio.** `run_case.synth_mask` colocaba una sola forma y
  terminaba en `_largest_cc`, así que un infarto bilateral —una basilar ocluida se lleva
  las dos PCA y las dos fosas posteriores— colapsaba a la componente conexa mayor: el
  manifest decía "bilateral" y la imagen enseñaba medio. Como los ids del atlas son todos
  lateralizados (impar izquierda, par derecha), basta agrupar por paridad. Comprobado:
  los dos basilares del lote salen 9145/8970 y 23603/22267 vóxeles izq/der, y los
  unilaterales siguen con **0** vóxeles al otro lado.

**`caso.py`** — encadena: variante → oclusión → `apagados` → CTA (`hybrid.render`) y
territorios → máscara → NCCT (`Gen3D` de lesion2ncct). Y convierte al donante en otro
paciente, que si no los casos salen todos con el mismo árbol (solo hay 6 anatomías):
reflejo L-R + deformación elástica, con `--no-variar` para desactivarlo.

### Deformar el árbol sin romperlo

`hybrid.randomize_anatomy` deforma las etiquetas por **vecino más próximo**, y le vale
porque allí el árbol solo se pinta. Aquí alimenta un **grafo** que necesita conectividad y
longitud, y eso lo rompe: los vasos son de 1-2 vóxeles de calibre, y una vertebral de 114
pasos geodésicos se queda en **20** — `occlude` la rechaza por corta.

`caso._warp_vasos` no interpola la máscara binaria sino su **mapa de distancia con signo**,
que es suave: se deforma con interpolación lineal y se vuelve a umbralizar en 0, así que el
tubo se desplaza entero. Con eso la vertebral aguanta (113 pasos) hasta amplitudes de 6
vóxeles. Se hace sobre la caja de cada vaso, que si no las 36 transformadas de distancia
sobre el slab entero cuestan minutos.

Dos trampas más, por si se toca:

- **La validación hay que hacerla contra el original, no contra cero.** El
  pseudo-etiquetado ya entrega vasos partidos de fábrica (el `R-M2`, que se bifurca, viene
  roto en todos los donantes). Con un criterio absoluto se rechaza *cualquier* campo y el
  generador acaba devolviendo siempre el donante intacto — que fue exactamente lo que pasó,
  y no se nota: los casos salen bien, solo que todos con el mismo árbol.
- **La aleatorización depende solo de `(seed, pid)`**, con su propio `rng`. Por eso dos
  casos que comparten los dos siguen siendo el mismo cerebro (Dice 1.000) y se pueden
  comparar, aunque ocluyan arterias distintas. Es lo que permite tener variedad *y* pares
  limpios a la vez.

### Un detalle que costó: qué cuenta como "apagado"

`cow_edit` lista en `apagados` **todo** segmento no alcanzable, incluidos los que ya
nacían huérfanos del pseudo-etiquetado (un `3rd-A3` sin su `3rd-A2` nunca fue alcanzable
desde una fuente). En `hybrid` eso era inocuo — significaba "no pintes ese vaso" — pero
aquí se traducía en **territorio infartado**: hasta la carótida cervical, que no debe
infartar nada, salía con 19 ml de ACA bilateral. Solo cuenta lo que pasa de perfundido a
no perfundido, así que `caso.py` compara contra `reachable()` **antes** de ocluir y los
descarta (van a `huerfanos_pseudolabel` en el JSON).

### Lateralidad

`qc_caso.py` comprueba que el infarto cae en el hemisferio del vaso ocluido. Es lo que
más fácil se rompe y lo que menos se nota: el CTA vive en el slab canónico y el NCCT en
MNI, que tienen **el eje x al revés**, y un caso con el infarto en el lado contrario
parece perfecto en cualquier visor.

Cuidado al medirlo, que aquí hay dos trampas encadenadas. En MNI `x = 0` es la línea media,
pero **en el slab canónico no**: el grid se centra en el centroide del polígono, así que el
paciente puede estar 20 mm descentrado y "derecha = x > 0" da falsos positivos. Y la línea
media **no se puede estimar con el centroide del árbol**, porque el pseudo-etiquetado no
cubre igual los dos lados y la corre lo bastante como para que un M1 perfectamente colocado
caiga del lado equivocado (pasó). Lo que sí es invariante es comparar cada arteria **con su
par contralateral**: `¿está R-M1 a la derecha de L-M1?`.

Los 18 casos del lote pasan: 10/10 con infarto lateralizable en el lado correcto, 0
equivocados, y los 7 pares R/L del árbol coherentes en los 18.

### Limitaciones propias de esta parte

- El territorio profundo de un **M1 ocluido** depende de dónde caiga el trombo, pero las
  perforantes **no tienen etiqueta propia**, así que el grafo no puede decidir por ellas:
  se usa una regla sobre `frac` (`PERF_FRAC = 0.7`), no anatomía.
- Los ids 9/10 del atlas **no vienen con nombre**. Por centroide (~12-15 mm de la línea
  media, a la altura de los ganglios basales) y volumen (~13 ml) son perforantes
  proximales, pero no se puede afirmar si separan lenticuloestriadas de coroidea anterior.
- El infarto es una **hipodensidad establecida**, no el sutil agudo (delta ~-15 HU con
  `--fuerza 1.5`; con el default el core baja a ~10 HU, bastante oscuro — bájalo a ~0.7
  si se quiere algo más precoz).
- Solo **6 anatomías** (2 por familia de CoW) tras descartar las de lateralidad
  incoherente. El reflejo L-R y la deformación elástica dan variedad, pero el parénquima
  de fondo sigue saliendo de esos 6 cerebros.
- El **CTA se deforma y el NCCT también, pero con campos independientes** (viven en
  espacios distintos). Siguen siendo el mismo paciente en el sentido que importa —el mismo
  donante y la misma lógica de territorio— pero no son comparables vóxel a vóxel, cosa que
  nunca lo fueron.
- CTA y NCCT quedan en **espacios distintos** (canónico vs MNI), como en lesion2ncct: el
  atlas y el NCCT del banco solo existen en MNI, y el árbol se edita en el canónico.
- El `BA` de algún donante arrastra un fragmento satélite pegado a la vertebral;
  `cow_edit._seed_of` planta ahí la semilla y su geodésica sale de 4 pasos en vez de ~30,
  justo el mínimo que exige `occlude`. El trombo se coloca en ese fragmento. El resultado
  clínico sale bien (el grafo es topológico), pero la geometría del trombo no es la del
  basilar. Si se toca, mirar `_seed_of`.

## Dos renderizadores

**`hybrid.py` — el que se usa.** Toma la CTA REAL del donante y le aplica el árbol
editado. Calidad angiográfica real, CPU, ~10 s.

- Vasos permeables **232 HU**, vaso ocluido **~40 HU**, trombo **~58 HU**.
- La supresión no deja fantasma: la zona borrada queda a **±3 HU** del tejido vecino
  y con su misma textura, por debajo del ruido del propio TC (±17 HU).
- `randomize_anatomy`: donante aleatorio + reflejo L-R (intercambiando las etiquetas
  lateralizadas) + deformación elástica aplicada con el **mismo campo** a imagen y
  etiquetas. Cada llamada es otro paciente.

**`generate.py` — difusión, APARCADO.** SPADE-diffusion 3D `mapa semántico -> CTA`.
Entrenado hasta la época 107 (`ckpt/gen_last.pt`, reanudable con `train_gen.py
--resume`). Daba solo **+14 HU** de contraste frente a los +248 del híbrido. Solo
tiene sentido retomarlo para un modo sintético puro (cráneo y parénquima también
generados).

## Datos (209 CTA)

| Fuente | n | Etiquetas | Aporta |
|---|---|---|---|
| **TopAneu** `topaneu_deployment` | 58 CT | 36 clases, GT | El árbol **completo** (M1/M2/M3, A1-A3, P1-P4, VA, SCA, AICA, PICA, AChA, OA) |
| **TopCoW** `Dataset008_Willis` | 125 CT | 15 + pseudo | Variabilidad anatómica |
| **ISLES-TUM** `TopCoW_reference\CTA_ISLES2024_TUM` | 26 CT | 15 + pseudo | CTA con oclusión real |

Las 15 clases de TopCoW coinciden exactamente con las 15 primeras de TopAneu.

**Por qué el SPADE anterior (`phase1_best.pt`) daba vasos incoherentes**: se entrenaba
con el label de TopCoW, que solo describe el polígono, así que el modelo tenía que
*inventar* todo el árbol distal. TopAneu cierra ese agujero — y es también lo que
permite que al ocluir un M1 se apaguen el M2 y el M3, cosa que el híbrido de PYTHIA
no podía hacer.

**Por qué ISLES sirve sin tener anotado el sitio de oclusión**: el vaso ocluido sale
truncado de la segmentación (no hay contraste que segmentar), así que el par
(etiqueta, imagen) es consistente y el sitio se *lee* comparando cada territorio con
su contralateral. Localizado en **17 de 26** casos.

## Preparación de datos (ya hecha)

```powershell
& $py prep.py all --workers 6   # 209 casos -> prep/   (~15 min, CPU)
& $py train_seg.py              # segmentador 36 clases, Dice val 0.714  (~1 h, GPU)
& $py pseudolabel.py            # completa TopCoW+ISLES -> sem/
```

El segmentador **no se despliega**: solo sirve para completar las etiquetas.

## Salidas

`dataset/` — por cada caso: la CTA (`.nii.gz`), su mapa de vasos (`_vasos.nii.gz`) y
una fila en `manifest.csv` con variante, arteria, posición del trombo, **territorios
apagados**, relleno retrógrado y las HU medidas. El nombre lleva donante y semilla:
cualquier caso es reproducible.

## Limitaciones honestas

- El flujo es **alcanzabilidad en un grafo**, no una simulación de perfusión: no hay
  caudales ni resistencias. Un Pcom filiforme rescata igual que uno robusto.
- **No se modelan colaterales piales**. En una oclusión de M1 el M2/M3 se apaga del
  todo; en la realidad puede haber relleno tardío y tenue. Es fiel a un CTA de fase
  arterial precoz, no a uno tardío.
- `A1A2` es un único segmento: cuando el Acom rellena la ACA, en la realidad rellena
  el A2 y el A1 queda como muñón. La etiqueta no los distingue.
- El fondo (cráneo, parénquima) procede de una CTA real deformada. No es sintético
  puro — para eso haría falta terminar la difusión.
- Los donantes de TopAneu son pacientes con aneurismas; alguno puede verse en la
  imagen.

## Gotchas

- **No solapar muestreo y entrenamiento en la GPU**: provoca `CUDA illegal memory
  access`.
- **Difusión: v-prediction obligatorio.** Con eps-prediction el muestreo devuelve un
  volumen plano (los datos tienen std 0.21, muy lejos de N(0,1), y al modelo le sale
  gratis el atajo eps≈x_t, que implica x0≈0). Ver `model.scheduler()`.
- La geodésica del vaso usa **6-conectividad** a propósito (la de 26 da solo ~14
  pasos en la ICA C6-C7, insuficiente para situar el trombo).
- El MIP de cabeza completa sale blanco: hay que sustraer todo lo brillante que no
  sea vaso etiquetado, no solo la clase `BONE` (ver `qc.bone_subtract`).
