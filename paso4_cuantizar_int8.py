"""Paso 4: cuantizar el modelo a 8 bits con NNCF y medir lo mismo que en el paso 3.

    python paso4_cuantizar_int8.py

Salidas:
  modelos/int8.xml + int8.bin   el modelo cuantizado
  medidas.json                  se le anaden las medidas del modelo de 8 bits
"""
import json
import sys
import time
from pathlib import Path

import nncf
import numpy as np
import openvino as ov
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import EuroSAT

sys.stdout.reconfigure(encoding="utf-8")
AQUI = Path(__file__).parent
MEDIA, DESV = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
N_CALIBRACION = 300

part = json.load(open(AQUI / "particion.json"))
t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEDIA, DESV)])
nucleo = ov.Core()

# 1. Calibracion: 300 imagenes del ENTRENAMIENTO (nunca de la prueba), elegidas
#    al azar con semilla fija. NNCF las pasa por la red para ver en que rango
#    se mueve cada capa.
rng = np.random.default_rng(0)
idx_cal = rng.choice(part["entrenamiento"], N_CALIBRACION, replace=False).tolist()
cal = Subset(EuroSAT(AQUI / "datos", transform=t), idx_cal)
calibracion = nncf.Dataset(DataLoader(cal, batch_size=1), lambda lote: lote[0].numpy())

# 2. Cuantizar el modelo de 32 bits del paso 3
fp32 = nucleo.read_model(AQUI / "modelos" / "fp32.xml")
int8 = nncf.quantize(fp32, calibracion, subset_size=N_CALIBRACION)
ov.save_model(int8, AQUI / "modelos" / "int8.xml")
tam_mb = (AQUI / "modelos" / "int8.bin").stat().st_size / 1e6
print(f"\nModelo de 8 bits: {tam_mb:.2f} MB de pesos")

# 3. Medir exactamente igual que en el paso 3
compilado = nucleo.compile_model(int8, "CPU")
prueba = Subset(EuroSAT(AQUI / "datos", transform=t), part["prueba"])
predicciones, reales = [], []
for x, y in DataLoader(prueba, batch_size=1):
    predicciones.append(int(compilado(x.numpy())[0].argmax()))
    reales.append(int(y))
predicciones, reales = np.array(predicciones), np.array(reales)
acierto = float((predicciones == reales).mean())
print(f"Acierto en prueba: {acierto:.4f} ({(predicciones == reales).sum()} de {len(reales)})")

x = prueba[0][0].unsqueeze(0).numpy()
for _ in range(50):
    compilado(x)
tiempos = []
for _ in range(1000):
    t0 = time.perf_counter()
    compilado(x)
    tiempos.append((time.perf_counter() - t0) * 1000)
lat = float(np.median(tiempos))
print(f"Latencia por imagen en CPU: mediana {lat:.3f} ms "
      f"(percentil 90: {np.percentile(tiempos, 90):.3f} ms)")

# 4. Guardar y comparar con el modelo de 32 bits y con las hipotesis
medidas = json.load(open(AQUI / "medidas.json"))
medidas["int8"] = {"acierto": acierto, "tamano_mb": tam_mb, "latencia_ms": lat,
                   "predicciones": predicciones.tolist(), "reales": reales.tolist(),
                   "calibracion": idx_cal}
json.dump(medidas, open(AQUI / "medidas.json", "w"))

a, b = medidas["fp32"], medidas["int8"]
print("\n                32 bits     8 bits     cambio")
print(f"  acierto      {a['acierto']:7.2%}   {b['acierto']:7.2%}   {100 * (b['acierto'] - a['acierto']):+.2f} puntos")
print(f"  tamano       {a['tamano_mb']:6.2f} MB  {b['tamano_mb']:6.2f} MB  {a['tamano_mb'] / b['tamano_mb']:.2f} veces menor")
print(f"  latencia     {a['latencia_ms']:6.3f} ms  {b['latencia_ms']:6.3f} ms  {a['latencia_ms'] / b['latencia_ms']:.2f} veces mas rapido")
print("\nTus hipotesis: precision 85-90 %, 4 veces mas pequeno, mucho mas rapido.")
