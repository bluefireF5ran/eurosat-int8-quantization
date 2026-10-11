"""Paso 2: afinar una MobileNetV3-Small preentrenada en ImageNet sobre EuroSAT.

    python paso2_entrenar.py

Salidas:
  modelos/mobilenet_fp32.pt     los pesos de la mejor epoca segun la validacion
  entrenamiento.csv             perdida y acierto por epoca
"""
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import EuroSAT
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

sys.stdout.reconfigure(encoding="utf-8")
AQUI = Path(__file__).parent
EPOCAS = 10
LOTE = 128          # imagenes por paso de entrenamiento
TASA = 1e-3         # tasa de aprendizaje
torch.manual_seed(0)
disp = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Transformaciones. Normalizar con la media y desviacion de ImageNet,
#    porque la red preentrenada "espera" imagenes asi.
MEDIA, DESV = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
t_entrenar = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomChoice([transforms.RandomRotation((a, a)) for a in (0, 90, 180, 270)]),
    transforms.ToTensor(),                       # 0..255 -> 0..1, y canales primero
    transforms.Normalize(MEDIA, DESV),
])
t_evaluar = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEDIA, DESV)])

# 2. Datos, con el reparto fijo del paso 1
part = json.load(open(AQUI / "particion.json"))
ent = Subset(EuroSAT(AQUI / "datos", transform=t_entrenar), part["entrenamiento"])
val = Subset(EuroSAT(AQUI / "datos", transform=t_evaluar), part["validacion"])
c_ent = DataLoader(ent, batch_size=LOTE, shuffle=True, num_workers=0)
c_val = DataLoader(val, batch_size=256, num_workers=0)

# 3. El modelo: MobileNetV3-Small con pesos de ImageNet. Su ultima capa decide
#    entre 1000 clases de ImageNet; la cambiamos por una de 10, las nuestras.
modelo = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
modelo.classifier[3] = nn.Linear(modelo.classifier[3].in_features, len(part["clases"]))
modelo = modelo.to(disp)
n_pesos = sum(p.numel() for p in modelo.parameters())
print(f"MobileNetV3-Small: {n_pesos / 1e6:.2f} millones de pesos, en {disp}")

perdida = nn.CrossEntropyLoss()
optim = torch.optim.AdamW(modelo.parameters(), lr=TASA)
# La tasa de aprendizaje baja poco a poco hasta casi cero al final (coseno)
plan = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCAS)


def acierto(cargador):
    modelo.eval()
    bien = total = 0
    with torch.no_grad():
        for x, y in cargador:
            pred = modelo(x.to(disp)).argmax(1)
            bien += (pred.cpu() == y).sum().item()
            total += len(y)
    return bien / total


# 4. Entrenamiento: en cada epoca, una pasada por todo el entrenamiento y
#    despues medir en validacion. Guardamos el modelo solo si mejora.
(AQUI / "modelos").mkdir(exist_ok=True)
mejor, filas = 0.0, []
for epoca in range(1, EPOCAS + 1):
    t0 = time.time()
    modelo.train()
    suma = n = 0
    for x, y in c_ent:
        x, y = x.to(disp), y.to(disp)
        optim.zero_grad()
        l = perdida(modelo(x), y)
        l.backward()
        optim.step()
        suma += l.item() * len(y)
        n += len(y)
    plan.step()
    acc = acierto(c_val)
    filas.append({"epoca": epoca, "perdida_ent": round(suma / n, 4), "acierto_val": round(acc, 4)})
    marca = ""
    if acc > mejor:
        mejor = acc
        torch.save(modelo.state_dict(), AQUI / "modelos" / "mobilenet_fp32.pt")
        marca = "  <- mejor hasta ahora, guardado"
    print(f"epoca {epoca:2d}  perdida {suma / n:.4f}  acierto validacion {acc:.4f}  "
          f"({time.time() - t0:.0f} s){marca}")

with open(AQUI / "entrenamiento.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(filas[0]))
    w.writeheader()
    w.writerows(filas)
print(f"\nMejor acierto en validacion: {mejor:.4f}. Modelo en modelos/mobilenet_fp32.pt")
