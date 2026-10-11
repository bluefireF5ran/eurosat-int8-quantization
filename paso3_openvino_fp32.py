"""Paso 3: convertir el modelo a OpenVINO y medir la referencia en 32 bits.

    python paso3_openvino_fp32.py

Salidas:
  modelos/fp32.xml + fp32.bin   el modelo en formato OpenVINO (estructura + pesos)
  medidas.json                  acierto, tamano y latencia del modelo de 32 bits
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import EuroSAT
from torchvision.models import mobilenet_v3_small

sys.stdout.reconfigure(encoding="utf-8")
AQUI = Path(__file__).parent
MEDIA, DESV = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# 1. Reconstruir la red y cargar los pesos entrenados en el paso 2
part = json.load(open(AQUI / "particion.json"))
modelo = mobilenet_v3_small()
modelo.classifier[3] = nn.Linear(modelo.classifier[3].in_features, len(part["clases"]))
modelo.load_state_dict(torch.load(AQUI / "modelos" / "mobilenet_fp32.pt", map_location="cpu"))
modelo.eval()

# 2. Convertir a OpenVINO. Necesita una entrada de ejemplo para conocer la
#    forma: 1 imagen, 3 canales, 64 x 64 pixeles.
ov_modelo = ov.convert_model(modelo, example_input=torch.zeros(1, 3, 64, 64))
ov.save_model(ov_modelo, AQUI / "modelos" / "fp32.xml", compress_to_fp16=False)
tam_mb = (AQUI / "modelos" / "fp32.bin").stat().st_size / 1e6
print(f"Modelo OpenVINO de 32 bits: {tam_mb:.2f} MB de pesos")

# 3. Compilarlo para la CPU
nucleo = ov.Core()
compilado = nucleo.compile_model(AQUI / "modelos" / "fp32.xml", "CPU")

# 4. Acierto en la PRUEBA (las 2.700 imagenes reservadas)
t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEDIA, DESV)])
prueba = Subset(EuroSAT(AQUI / "datos", transform=t), part["prueba"])
predicciones, reales = [], []
for x, y in DataLoader(prueba, batch_size=1):
    salida = compilado(x.numpy())[0]          # puntuacion de cada una de las 10 clases
    predicciones.append(int(salida.argmax()))
    reales.append(int(y))
predicciones, reales = np.array(predicciones), np.array(reales)
acierto = float((predicciones == reales).mean())
print(f"Acierto en prueba: {acierto:.4f} ({(predicciones == reales).sum()} de {len(reales)})")

# 5. Latencia de UNA imagen en la CPU: calentamiento y despues la mediana de
#    muchas repeticiones con la misma imagen.
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

json.dump({"fp32": {"acierto": acierto, "tamano_mb": tam_mb, "latencia_ms": lat,
                    "predicciones": predicciones.tolist(), "reales": reales.tolist()}},
          open(AQUI / "medidas.json", "w"))
print("Guardado en medidas.json")
