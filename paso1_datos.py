"""Paso 1: descargar EuroSAT, mirarlo y fijar la particion.

    python paso1_datos.py

Salidas:
  datos/                    el conjunto descargado (unos 90 MB)
  particion.json            que imagenes van a entrenamiento, validacion y prueba
  figuras/ejemplos.png      tres ejemplos de cada clase
"""
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import EuroSAT

sys.stdout.reconfigure(encoding="utf-8")
AQUI = Path(__file__).parent
SEMILLA = 0   # misma semilla = mismo reparto siempre

# 1. Descarga (solo la primera vez) y carga. Cada elemento es (imagen, clase).
datos = EuroSAT(root=AQUI / "datos", download=True)
clases = datos.classes
etiquetas = np.array(datos.targets)
print(f"{len(datos)} imagenes, {len(clases)} clases")

# 2. Cuantas imagenes hay de cada clase: si alguna tuviera muy pocas, lo
#    sabriamos antes de entrenar.
cuenta = Counter(etiquetas.tolist())
for i, nombre in enumerate(clases):
    print(f"  {nombre:22s} {cuenta[i]:5d}")

# 3. Como es una imagen por dentro
img, clase = datos[0]
arr = np.array(img)
print(f"\nUna imagen: tamano {arr.shape} (alto, ancho, canales), tipo {arr.dtype}, "
      f"valores de {arr.min()} a {arr.max()}, clase '{clases[clase]}'")

# 4. Reparto 80/10/10 ESTRATIFICADO: cada clase se reparte en esas
#    proporciones por separado, para que las tres partes tengan la misma mezcla
#    de clases que el conjunto completo.
rng = np.random.default_rng(SEMILLA)
particion = {"entrenamiento": [], "validacion": [], "prueba": []}
for c in range(len(clases)):
    idx = np.where(etiquetas == c)[0]
    rng.shuffle(idx)
    n = len(idx)
    n_ent, n_val = int(0.8 * n), int(0.1 * n)
    particion["entrenamiento"] += idx[:n_ent].tolist()
    particion["validacion"] += idx[n_ent:n_ent + n_val].tolist()
    particion["prueba"] += idx[n_ent + n_val:].tolist()

with open(AQUI / "particion.json", "w") as f:
    json.dump({"semilla": SEMILLA, "clases": clases, **particion}, f)
print("\nParticion guardada en particion.json:")
for k, v in particion.items():
    print(f"  {k:14s} {len(v):5d} imagenes")

# 5. Tres ejemplos de cada clase, para verlas con nuestros propios ojos
(AQUI / "figuras").mkdir(exist_ok=True)
fig, ejes = plt.subplots(3, len(clases), figsize=(2 * len(clases), 6.5))
for c, nombre in enumerate(clases):
    for fila, i in enumerate(np.where(etiquetas == c)[0][:3]):
        ejes[fila, c].imshow(datos[i][0])
        ejes[fila, c].axis("off")
    ejes[0, c].set_title(nombre, fontsize=9)
fig.tight_layout()
fig.savefig(AQUI / "figuras" / "ejemplos.png", dpi=110)
print("Ejemplos guardados en figuras/ejemplos.png")
