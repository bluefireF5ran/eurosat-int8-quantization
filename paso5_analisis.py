"""Paso 5: donde pierde acierto el modelo de 8 bits y cuando gana velocidad.

    python paso5_analisis.py

Usa las predicciones guardadas en medidas.json (pasos 3 y 4) y los modelos de
modelos/. Salidas:
  figuras/confusion.png       matrices de confusion de 32 y 8 bits
  figuras/velocidad_lote.png  tiempo por imagen segun el tamano del lote
  analisis.json               las cifras de todo lo anterior
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov

sys.stdout.reconfigure(encoding="utf-8")
AQUI = Path(__file__).parent
m = json.load(open(AQUI / "medidas.json"))
clases = json.load(open(AQUI / "particion.json"))["clases"]
K = len(clases)
reales = np.array(m["fp32"]["reales"])
p32, p8 = np.array(m["fp32"]["predicciones"]), np.array(m["int8"]["predicciones"])

# ---------- A. Donde se pierde acierto ----------
def confusion(pred):
    """Fila = clase real, columna = clase predicha."""
    M = np.zeros((K, K), dtype=int)
    for r, p in zip(reales, pred):
        M[r, p] += 1
    return M

C32, C8 = confusion(p32), confusion(p8)
print("Acierto por clase (32 bits -> 8 bits):")
por_clase = {}
for c, nombre in enumerate(clases):
    a32, a8 = C32[c, c] / C32[c].sum(), C8[c, c] / C8[c].sum()
    por_clase[nombre] = {"fp32": a32, "int8": a8}
    print(f"  {nombre:22s} {a32:6.1%} -> {a8:6.1%}  ({100 * (a8 - a32):+.1f})")

# Confusiones mas frecuentes en el modelo de 8 bits (fuera de la diagonal)
fuera = [(C8[r, p], clases[r], clases[p]) for r in range(K) for p in range(K) if r != p and C8[r, p] > 0]
print("\nConfusiones mas frecuentes con 8 bits (real -> predicha):")
for n, r, p in sorted(fuera, reverse=True)[:6]:
    print(f"  {n:3d}  {r} -> {p}")

# Imagenes que cambian de prediccion al cuantizar
empeoran = int(((p32 == reales) & (p8 != reales)).sum())
mejoran = int(((p32 != reales) & (p8 == reales)).sum())
cambian = int((p32 != p8).sum())
print(f"\nImagenes cuya prediccion cambia al cuantizar: {cambian} de {len(reales)} "
      f"({empeoran} empeoran, {mejoran} mejoran)")

(AQUI / "figuras").mkdir(exist_ok=True)
fig, ejes = plt.subplots(1, 2, figsize=(15, 6.5))
for ax, M, titulo in [(ejes[0], C32, "32 bits"), (ejes[1], C8, "8 bits")]:
    ax.imshow(np.log1p(M), cmap="Blues")
    for r in range(K):
        for p in range(K):
            if M[r, p]:
                ax.text(p, r, M[r, p], ha="center", va="center", fontsize=8,
                        color="white" if r == p else "black")
    ax.set_xticks(range(K)), ax.set_yticks(range(K))
    ax.set_xticklabels(clases, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(clases, fontsize=8)
    ax.set_xlabel("predicha"), ax.set_ylabel("real")
    ax.set_title(f"{titulo}: {np.trace(M) / M.sum():.2%} de acierto")
fig.tight_layout()
fig.savefig(AQUI / "figuras" / "confusion.png", dpi=110)
plt.close(fig)

# ---------- B. Velocidad segun el tamano del lote ----------
# El modelo se convirtio para lotes de 1 imagen; reshape lo adapta a otros
# tamanos. Medimos el tiempo de cada llamada y lo dividimos por el lote.
nucleo = ov.Core()
LOTES = [1, 8, 32, 128]
velocidad = {}
for nombre in ("fp32", "int8"):
    velocidad[nombre] = {}
    for b in LOTES:
        modelo = nucleo.read_model(AQUI / "modelos" / f"{nombre}.xml")
        modelo.reshape([b, 3, 64, 64])
        compilado = nucleo.compile_model(modelo, "CPU")
        x = np.random.default_rng(0).standard_normal((b, 3, 64, 64)).astype(np.float32)
        for _ in range(20):
            compilado(x)
        tiempos = []
        for _ in range(max(50, 2000 // b)):
            t0 = time.perf_counter()
            compilado(x)
            tiempos.append((time.perf_counter() - t0) * 1000)
        velocidad[nombre][b] = float(np.median(tiempos)) / b
print("\nTiempo por imagen (ms) segun el tamano del lote:")
print("  lote      32 bits    8 bits    8 bits es")
for b in LOTES:
    a, c = velocidad["fp32"][b], velocidad["int8"][b]
    print(f"  {b:4d}   {a:8.4f}  {c:8.4f}   {a / c:.2f} veces mas rapido")

fig, ax = plt.subplots(figsize=(7, 4.5))
for nombre, etiqueta in (("fp32", "32 bits"), ("int8", "8 bits")):
    ax.plot(LOTES, [velocidad[nombre][b] for b in LOTES], "o-", label=etiqueta)
ax.set_xscale("log", base=2), ax.set_yscale("log")
ax.set_xticks(LOTES), ax.set_xticklabels(LOTES)
ax.set_xlabel("imagenes por llamada (lote)"), ax.set_ylabel("tiempo por imagen (ms)")
ax.grid(alpha=0.3), ax.legend()
fig.tight_layout()
fig.savefig(AQUI / "figuras" / "velocidad_lote.png", dpi=110)

json.dump({"por_clase": por_clase, "cambian": cambian, "empeoran": empeoran,
           "mejoran": mejoran, "velocidad_ms_por_imagen": velocidad},
          open(AQUI / "analisis.json", "w"), indent=1)
print("\nFiguras en figuras/ y cifras en analisis.json")
