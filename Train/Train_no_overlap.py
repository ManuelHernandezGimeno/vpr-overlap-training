import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/workspace/mhernang/VPR"))

# Localizar la librería oficial de Mapillary SLS y modificar msls.py

MAPILLARY_SLS_ROOT = Path(
    os.environ.get("MAPILLARY_SLS_ROOT", "/opt/mapillary_sls")
)

if not MAPILLARY_SLS_ROOT.exists():
    raise FileNotFoundError(
        f"No se encuentra mapillary_sls en {MAPILLARY_SLS_ROOT}. "
        "Debe estar clonado en el Docker o montado en el proyecto."
    )

msls_file = MAPILLARY_SLS_ROOT / "mapillary_sls" / "datasets" / "msls.py"

if not msls_file.exists():
    raise FileNotFoundError(f"No se encuentra msls.py en: {msls_file}")

text = msls_file.read_text()

text = text.replace(
    "self.pIdx = np.asarray(self.pIdx)",
    "self.pIdx = np.asarray(self.pIdx, dtype=object)"
)

text = text.replace(
    "self.nonNegIdx = np.asarray(self.nonNegIdx)",
    "self.nonNegIdx = np.asarray(self.nonNegIdx, dtype=object)"
)

msls_file.write_text(text)

print("msls.py modificado correctamente")

# Añadir repo al path
sys.path.insert(0, str(MAPILLARY_SLS_ROOT))

# Importar MSLS DESPUÉS de modificar el archivo
from mapillary_sls.datasets.msls import MSLS

print("MSLS importado correctamente")

# Importar librerías

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import re
import time
import gc
import json
import pandas as pd

#DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformaciones para entrenamiento y la validación
#Se reescala a 256x256 y después aleatoriamente a 224x224. Se ajusta la escala y se modifican parámetros
#aleatoriamente. 10% de posibilidad de que pase a escala de grises. También desenfoque gaussiano aleatorio.
#Finalmente, convierte la imagen a tensor de PyTorch y se normaliza porque porque la ResNet50 preentrenada
#fue entrenada originalmente con imágenes normalizadas así.
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1
    ),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# La validación no debe usar aumentos aleatorios,
# porque quieres medir el rendimiento real del modelo de forma estable
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# DATASET DE ENTRENAMIENTO: TODAS LAS CIUDADES TRAIN DE MSLS
# Carpeta raíz del dataset MSLS con las imágenes originales
MSLS_ROOT = Path(os.environ.get("MSLS_ROOT", PROJECT_ROOT / "data" / "mapillary"))
if not MSLS_ROOT.exists():
    raise FileNotFoundError(f"No se encuentra el dataset MSLS en: {MSLS_ROOT}")

train_cities = "trondheim,london,boston,melbourne,amsterdam,helsinki,tokyo,toronto,saopaulo,moscow,zurich,paris,bangkok,budapest,austin,berlin,ottawa,phoenix,goa,amman,nairobi,manila"  

train_dataset = MSLS(
    root_dir=MSLS_ROOT,                   # Ruta donde tienes guardado el dataset MSLS
    cities=train_cities,                  # Elijo las ciudades de entrenamiento del dataset
    transform=train_transform,            # Aplico las transformaciones anteriores a cada imagen
    mode="train",                         # <- importante porque vamos a entrenar el modelo
    nNeg=5,                               # nº negativos por query
    posDistThr=25,                        # threshold GPS para positivos
    negDistThr=25,                        # threshold negativos
    cached_queries=5000,                  #Cuántas queries se cargan en cada subconjunto de entrenamiento
    cached_negatives=5000,               #Cuántos negativos se almacenan para hacer minería de negativos.
)

# Evita pedir más negativos de los que existen
train_dataset.cached_negatives = min(
    train_dataset.cached_negatives,
    len(train_dataset.dbImages)
)

print("Dataset de entrenamiento cargado.")
print("Ciudades de entrenamiento:", train_cities)
print("Queries train:", len(train_dataset.qImages))
print("Database train:", len(train_dataset.dbImages))
print("cached_queries:", train_dataset.cached_queries)
print("cached_negatives:", train_dataset.cached_negatives)

# ============================================================
# DATASET DE VALIDACIÓN CIUDAD POR CIUDAD
# ============================================================

VAL_CITIES = ["cph", "sf"]
val_cities = ",".join(VAL_CITIES)

def make_val_packs_citywise(cities, root_dir, transform):
    """
    Carga MSLS ciudad por ciudad.

    Cada val_pack contiene:
        - query_paths de una sola ciudad
        - database_paths de esa misma ciudad
        - pIdx local de esa ciudad

    Esto evita el problema detectado al usar MSLS(cities="cph,sf").
    """

    val_packs = []

    for city in cities:

        ds = MSLS(
            root_dir=root_dir,
            cities=city,
            transform=transform,
            mode="val",
            posDistThr=25,
            negDistThr=25,
        )

        pIdx = np.asarray(ds.pIdx, dtype=object)

        val_pack = {
            "city": city,
            "query_paths": np.asarray(ds.qImages),
            "database_paths": np.asarray(ds.dbImages),
            "pIdx": pIdx,
        }

        print(
            f"Val {city}: "
            f"queries={len(val_pack['query_paths'])}, "
            f"database={len(val_pack['database_paths'])}, "
            f"queries con positivos={sum(len(p) > 0 for p in pIdx)}"
        )

        val_packs.append(val_pack)

    return val_packs


val_packs = make_val_packs_citywise(
    cities=VAL_CITIES,
    root_dir=MSLS_ROOT,
    transform=val_transform
)

print("Dataset de validación cargado ciudad por ciudad.")
print("Ciudades de validación:", val_cities)
print("Queries val:", sum(len(pack["query_paths"]) for pack in val_packs))
print("Database val:", sum(len(pack["database_paths"]) for pack in val_packs))
print("Queries con positivos:", sum(len(p) > 0 for pack in val_packs for p in pack["pIdx"]))

# Dataset auxiliar para cargar imágenes
def center_path(path_string):
    parts = str(path_string).split(",")
    return parts[len(parts) // 2]

class ImagePathDataset(Dataset):
    def __init__(self, image_paths, image_keys, transform=None):
        self.image_paths = list(image_paths)
        self.image_keys = np.asarray(image_keys)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = center_path(self.image_paths[idx])

        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        key = int(self.image_keys[idx])

        return img, key

# Función para extraer descriptores

def compute_descriptors(model,image_paths,image_keys,transform,device,batch_size=32,num_workers=2,desc_name="descriptors"):
    """
    Calcula descriptores para una lista de imágenes.
    """
    dataset = ImagePathDataset(
        image_paths=image_paths,
        image_keys=image_keys,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2
    )

    descriptors_list = []
    keys_list = []

    model.eval()

    print(f"Calculando {desc_name}: {len(dataset)} imágenes")

    with torch.no_grad():
        for images, keys in tqdm(loader, desc=desc_name):
            images = images.to(device, non_blocking=True)

            descriptors = model(images)

            descriptors_list.append(descriptors.cpu())
            keys_list.append(keys.cpu().numpy())

    descriptors = torch.cat(descriptors_list, dim=0)
    keys = np.concatenate(keys_list, axis=0)

    print(f"{desc_name} calculados: {descriptors.shape}")

    return descriptors, keys

# Función para generar predicciones top-10

def compute_predictions_l2_vectorized(
    query_descriptors,
    query_keys,
    db_descriptors,
    db_keys,
    top_k=10,
    device="cuda",
    distance_batch_size=64
):
    """
    Calcula top-k por distancia L2 entre queries y database.

    query_descriptors: [Nq, D]
    db_descriptors: [Ndb, D]
    """

    model_device = torch.device(device)

    db_descriptors = db_descriptors.to(model_device)
    query_descriptors = query_descriptors.to(model_device)

    all_predictions = []

    print("Calculando distancias L2 query-database...")

    with torch.no_grad():
        for start in tqdm(range(0, query_descriptors.shape[0], distance_batch_size),
                          desc="L2 search"):

            end = min(start + distance_batch_size, query_descriptors.shape[0])

            q_batch = query_descriptors[start:end]

            # Distancias [num_queries_batch, num_database]
            distances = torch.cdist(q_batch, db_descriptors, p=2)

            # Menor distancia = mejor candidato
            top_indices = torch.topk(
                distances,
                k=top_k,
                dim=1,
                largest=False
            ).indices.cpu().numpy()

            for i, inds in enumerate(top_indices):
                q_key = query_keys[start + i]
                top_db_keys = db_keys[inds]

                row = np.concatenate([
                    np.array([q_key], dtype=np.int64),
                    top_db_keys.astype(np.int64)
                ])

                all_predictions.append(row)

    predictions = np.stack(all_predictions, axis=0)

    return predictions

# Importamos modelo de VPR

# Defino una nueva capa Generalized Mean Pooling.
# Convierte un mapa de características convolucional en un único descriptor global.
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):           # 2 parámetros: p (controla el tipo de pooling) y eps (valor pequeño para evitar problemas numéricos)
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)   # Define p como un parámetro entrenable(aprender el tipo de pooling conveniente)
        self.eps = eps                             # Guardar eps para evitar elevar a una potencia números demasiado cercanos a cero.

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)      # Primero limita el mínimo de x a eps. Luego eleva cada valor a p.
        x = F.avg_pool2d(x, kernel_size=(x.size(-2), x.size(-1)))   # Aplica average pooling. Comprime cada canal a un único valor
        x = x.pow(1.0 / self.p)
        return x                                   # Aplica la raíz correspondiente para completar la operación GeM.


# Modelo que produce un descriptor global para cada imagen.
class ResNet50GeM(nn.Module):
    def __init__(self, out_dim=512):               # El descriptor final tendrá 512 dimensiones.
        super().__init__()

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)         # Carga una ResNet50 preentrenada en ImageNet.
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])      # Quito las 2 últimas capas

        self.pool = GeM()                                                   # Añado la capa GeM
        self.fc = nn.Linear(2048, out_dim)                                  # Añado una capa lineal que reduce el descriptor de 2048 dimensiones a 512.

        # Necesario para MSLS update_subcache(net=model)
        self.meta = {"outputdim": out_dim}

    def forward(self, x):
        x = self.backbone(x)                      # La imagen pasa por la ResNet50 sin las capas finales
        x = self.pool(x)                          # Aplica GeM
        x = torch.flatten(x, 1)                   # Aplana el tensor
        x = self.fc(x)                            # Aplica la capa lineal
        x = F.normalize(x, p=2, dim=1)            # Normaliza cada descriptor para que tenga norma 1 porque luego vas a comparar descriptores
        return x


model = ResNet50GeM(out_dim=512).to(device)       # Crea el modelo y lo mueve a GPU

optimizer = torch.optim.Adam(                     # Crea el optimizador Adam que actualiza los pesos durante el entrenamiento.
    model.parameters(),                           # Entrenamos todos los parámetros
    lr=1e-5,                                      # Tasa de aprendizaje pequeña porque el modelo está preentrenado
    weight_decay=1e-6                             # Regularización para evitar sobreajuste
)

# Definimos Triplet loss estándar

def msls_triplet_loss(descriptors, nNeg, margin=0.1):
    """
    descriptors: [B, 2+nNeg, D]
    0: query
    1: positive
    2...: negatives
    """

    q = descriptors[:, 0, :]          # Extraemos los descriptores de la query, los positivos y los negativos
    p = descriptors[:, 1, :]
    n = descriptors[:, 2:, :]

    d_pos = torch.norm(q - p, dim=1)  # Calcula la distancia entre cada query y su positivo

    losses = []                       # Crea una lista vacía donde se guardará la pérdida asociada a cada negativo

    for i in range(nNeg):
        d_neg = torch.norm(q - n[:, i, :], dim=1)         # Calcula la distancia entre cada query y sus negativos
        losses.append(F.relu(d_pos - d_neg + margin))     # Calcula la triplet loss: d_pos + margin < d_neg. Si ya se cumple esa condición, la pérdida vale 0.

    return torch.stack(losses, dim=1).mean()              # Junta todas las pérdidas y calcula la media.

# Definimos la función para el cálculo del Recall y para la evaluación

def recall(ranks, pidx, ks):
    """
    ranks: matriz [num_queries, top_k]
           Cada fila contiene los ids/keys de database recuperados para una query.

    pidx: lista o array de positivos.
          pidx[qidx] contiene los ids/keys correctos de database para esa query.

    ks: lista de valores de K, por ejemplo [1, 5, 10].
    """

    recall_at_k = np.zeros(len(ks), dtype=np.float32)

    for qidx in range(ranks.shape[0]):

        positives = pidx[qidx]

        if len(positives) == 0:
            continue

        for i, k in enumerate(ks):

            retrieved = ranks[qidx, :k]

            if np.sum(np.in1d(retrieved, positives)) > 0:
                recall_at_k[i:] += 1
                break

    valid_queries = sum(len(p) > 0 for p in pidx)

    if valid_queries == 0:
        return recall_at_k

    recall_at_k /= valid_queries

    return recall_at_k


def eval_vpr(query_keys, positive_keys, predictions, ks=[1, 5, 10]):
    """
    query_keys: ids/keys de las queries evaluadas.
    positive_keys: lista de positivos para cada query, en el mismo orden que query_keys.
    predictions: matriz [num_queries, 1 + top_k]
                 columna 0: query_key
                 columnas 1...: database_keys predichos.
    ks: recall@k que quieres calcular.
    """
    # asegurarse de que el positivo y las predicciones son del mismo elemento
    pred_queries = predictions[:, 0:1]
    pred2gt = [np.where(pred_queries[:, 0] == key)[0][0]for key in query_keys]

    # cambio orden para encajar con ground truth
    predictions = predictions[pred2gt, 1:]

    recall_at_k = recall(predictions, positive_keys, ks)

    metrics = {}

    for i, k in enumerate(ks):
        metrics[f"recall@{k}"] = recall_at_k[i]

    return metrics

# Función completa de evaluación

def validate_vpr_from_pack(
    model,
    val_pack,
    transform,
    device,
    num_val_queries=None,
    num_val_db=None,
    ks=[1, 5, 10],
    descriptor_batch_size=128,
    distance_batch_size=128,
    num_workers=8
):
    print("\nIniciando validación...")

    db_paths = val_pack["database_paths"]
    query_paths = val_pack["query_paths"]
    positive_keys_all = val_pack["pIdx"]

    if num_val_db is not None:
        db_paths = db_paths[:num_val_db]

        positive_keys_all = np.array([
            np.array([p for p in positives if p < num_val_db], dtype=np.int64)
            for positives in positive_keys_all
        ], dtype=object)

    db_keys = np.arange(len(db_paths), dtype=np.int64)
    query_keys_all = np.arange(len(query_paths), dtype=np.int64)

    valid_query_indices = np.array(
        [i for i, p in enumerate(positive_keys_all) if len(p) > 0],
        dtype=np.int64
    )

    print("Database total:", len(db_paths))
    print("Queries totales:", len(query_paths))
    print("Queries con positivos:", len(valid_query_indices))

    #La selección de queries para la evaluación es aleatoria
    rng = np.random.default_rng(42)

    if num_val_queries is not None:
        valid_query_indices = rng.choice(
            valid_query_indices,
            size=min(num_val_queries, len(valid_query_indices)),
            replace=False
        )
    selected_query_paths = query_paths[valid_query_indices]
    selected_query_keys = query_keys_all[valid_query_indices]
    selected_positive_keys = positive_keys_all[valid_query_indices]

    print("Queries usadas en esta validación:", len(selected_query_paths))

    # 1. Descriptores de database
    db_descriptors, db_keys_out = compute_descriptors(
        model=model,
        image_paths=db_paths,
        image_keys=db_keys,
        transform=transform,
        device=device,
        batch_size=descriptor_batch_size,
        num_workers=num_workers,
        desc_name="database descriptors"
    )

    # 2. Descriptores de queries
    query_descriptors, query_keys_out = compute_descriptors(
        model=model,
        image_paths=selected_query_paths,
        image_keys=selected_query_keys,
        transform=transform,
        device=device,
        batch_size=descriptor_batch_size,
        num_workers=num_workers,
        desc_name="query descriptors"
    )

    # 3. Top-k por L2
    predictions = compute_predictions_l2_vectorized(
        query_descriptors=query_descriptors,
        query_keys=query_keys_out,
        db_descriptors=db_descriptors,
        db_keys=db_keys_out,
        top_k=max(ks),
        device=device,
        distance_batch_size=distance_batch_size
    )

    # 4. Recall
    metrics = eval_vpr(
        query_keys=query_keys_out,
        positive_keys=selected_positive_keys,
        predictions=predictions,
        ks=ks
    )

    print("Validación terminada.")

    return metrics

def validate_vpr_citywise(
    model,
    val_packs,
    transform,
    device,
    num_val_queries=None,
    num_val_db=None,
    ks=[1, 5, 10],
    descriptor_batch_size=512,
    distance_batch_size=512,
    num_workers=8
):
    """
    Valida el modelo ciudad por ciudad.

    Cada ciudad se evalúa contra su propia database y su propio pIdx local.
    Después se calcula una métrica global ponderada por el número de queries
    válidas de cada ciudad.
    """

    city_metrics = []
    total_valid_queries = 0

    accum = {
        f"recall@{k}": 0.0
        for k in ks
    }

    for val_pack in val_packs:

        city = val_pack["city"]
        pIdx = val_pack["pIdx"]

        num_valid_queries = int(sum(len(p) > 0 for p in pIdx))

        if num_valid_queries == 0:
            print(f"[WARNING] {city}: no hay queries válidas.")
            continue

        print("\n" + "=" * 60)
        print(f"VALIDANDO CIUDAD: {city}")
        print("=" * 60)

        metrics_city = validate_vpr_from_pack(
            model=model,
            val_pack=val_pack,
            transform=transform,
            device=device,
            num_val_queries=num_val_queries,
            num_val_db=num_val_db,
            ks=ks,
            descriptor_batch_size=descriptor_batch_size,
            distance_batch_size=distance_batch_size,
            num_workers=num_workers
        )

        city_row = {
            "city": city,
            "num_valid_queries": num_valid_queries,
        }

        for k in ks:
            key = f"recall@{k}"
            city_row[key] = float(metrics_city[key])
            accum[key] += float(metrics_city[key]) * num_valid_queries

        city_metrics.append(city_row)
        total_valid_queries += num_valid_queries

        print(
            f"{city}: "
            f"queries={num_valid_queries}, "
            f"R@1={metrics_city['recall@1']:.4f}, "
            f"R@5={metrics_city['recall@5']:.4f}, "
            f"R@10={metrics_city['recall@10']:.4f}"
        )

    if total_valid_queries == 0:
        raise RuntimeError("No hay queries válidas en ninguna ciudad de validación.")

    global_metrics = {
        key: accum[key] / total_valid_queries
        for key in accum.keys()
    }

    global_metrics["num_valid_queries"] = total_valid_queries

    print("\n" + "=" * 60)
    print("MÉTRICAS GLOBALES CORREGIDAS")
    print("=" * 60)
    print(f"Queries válidas totales: {total_valid_queries}")
    print(f"R@1:  {global_metrics['recall@1']:.4f}")
    print(f"R@5:  {global_metrics['recall@5']:.4f}")
    print(f"R@10: {global_metrics['recall@10']:.4f}")

    return global_metrics, city_metrics

# DIRECTORIOS Y ARCHIVOS DE SALIDA

OUTPUT_DIR = Path(os.environ.get("OUTPUTNEW2_ROOT", PROJECT_ROOT / "outputs"/ "Resultados" ))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_FILE = OUTPUT_DIR / "best_model.ckpt"
BEST_METRICS_FILE = OUTPUT_DIR / "best_metrics.npy"

METRICS_JSON = OUTPUT_DIR / "metrics_history.json"
METRICS_CSV = OUTPUT_DIR / "metrics_history.csv"

# Funciones para checkpoints y métricas

def get_epoch_from_checkpoint_name(path):
    """
    Extrae el número de época de nombres como:
    epoch_001.ckpt
    epoch_025.ckpt
    """
    match = re.search(r"epoch_(\d+)", path.name)

    if match is None:
        return -1

    return int(match.group(1))

def find_latest_epoch_checkpoint(checkpoint_dir):
    """
    Busca el último checkpoint de época guardado.
    """
    checkpoint_dir = Path(checkpoint_dir)

    candidates = list(checkpoint_dir.glob("epoch_*.ckpt"))

    if len(candidates) == 0:
        return None

    candidates = sorted(
        candidates,
        key=lambda p: get_epoch_from_checkpoint_name(p)
    )

    return candidates[-1]

def load_metrics_history():
    """
    Carga el histórico de métricas si existe.
    """
    if METRICS_JSON.exists():
        with open(METRICS_JSON, "r") as f:
            history = json.load(f)

        print(f"Histórico cargado desde: {METRICS_JSON}")
        print(f"Épocas registradas: {len(history)}")
    else:
        history = []
        print("No existe histórico previo. Se crea uno nuevo.")

    return history


def save_metrics_history(metrics_history):
    """
    Guarda métricas en JSON y CSV.
    """
    metrics_history = sorted(metrics_history, key=lambda x: x["epoch"])

    with open(METRICS_JSON, "w") as f:
        json.dump(metrics_history, f, indent=4)

    df = pd.DataFrame(metrics_history)
    df.to_csv(METRICS_CSV, index=False)

    print(f"Métricas guardadas en:")
    print(METRICS_JSON)
    print(METRICS_CSV)


def get_best_recall_from_history(metrics_history, min_delta=0.0):
    """
    Devuelve el mejor R@1 siguiendo el mismo criterio de mejora
    usado por el early stopping.
    """
    best_r1 = -1.0

    history = sorted(metrics_history, key=lambda x: x["epoch"])

    for record in history:
        r1 = record.get("recall@1", None)

        if r1 is None:
            continue

        r1 = float(r1)

        if r1 > best_r1 + min_delta:
            best_r1 = r1

    return best_r1


def count_epochs_without_improvement(metrics_history, min_delta=0.0):
    """
    Calcula cuántas épocas consecutivas lleva sin mejorar R@1,
    usando el mismo min_delta que el early stopping.
    """
    if len(metrics_history) == 0:
        return 0

    history = sorted(metrics_history, key=lambda x: x["epoch"])

    best_r1 = -1.0
    count = 0

    for record in history:
        r1 = record.get("recall@1", None)

        if r1 is None:
            continue

        r1 = float(r1)

        if r1 > best_r1 + min_delta:
            best_r1 = r1
            count = 0
        else:
            count += 1

    return count


def load_checkpoint_for_resume(model, optimizer, device):
    """
    Carga exclusivamente el último checkpoint de época.
    No carga best_model.ckpt para continuar entrenamiento.
    """

    latest_epoch_ckpt = find_latest_epoch_checkpoint(CHECKPOINT_DIR)

    if latest_epoch_ckpt is None:
        print("No hay checkpoints de época previos. Se empieza desde ImageNet.")
        return model, optimizer, 0

    print("Cargando último checkpoint de época:")
    print(latest_epoch_ckpt)

    checkpoint = torch.load(latest_epoch_ckpt, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print("Pesos del modelo cargados correctamente.")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Estado del optimizador cargado correctamente.")

    last_epoch = int(checkpoint.get("epoch", 0))
    print(f"Última época cargada: {last_epoch}")

    return model, optimizer, last_epoch

def keep_last_n_checkpoints(checkpoint_dir, n_keep=10):
    """
    Mantiene solo los n_keep checkpoints de época más recientes.
    Borra los checkpoints epoch_XXX.ckpt más antiguos.
    No afecta a best_model.ckpt.
    """

    checkpoint_dir = Path(checkpoint_dir)

    checkpoints = list(checkpoint_dir.glob("epoch_*.ckpt"))

    if len(checkpoints) <= n_keep:
        return

    checkpoints = sorted(
        checkpoints,
        key=lambda p: get_epoch_from_checkpoint_name(p)
    )

    checkpoints_to_delete = checkpoints[:-n_keep]

    for ckpt_path in checkpoints_to_delete:
        try:
            ckpt_path.unlink()
            print(f"Checkpoint antiguo borrado: {ckpt_path}")
        except Exception as e:
            print(f"No se pudo borrar {ckpt_path}: {e}")

# BUCLE DE ENTRENAMIENTO COMPLETO CON EARLY STOPPING
import time
import gc

# Configuración general
max_epochs = 100
patience = 20            # <- número de épocas para el criterio de parada
min_delta = 0.001

batch_size = 32
nNeg = train_dataset.nNeg
ks = [1, 5, 10]

descriptor_batch_size = 512
distance_batch_size = 512
num_workers_train = 8
num_workers_val = 8

num_val_queries=None
num_val_db=None

best_recall_at_1 = -1.0
epochs_without_improvement = 0

# Cargar histórico de métricas
metrics_history = load_metrics_history()

# Cargar último checkpoint entrenado, si existe
model, optimizer, last_epoch = load_checkpoint_for_resume(
    model=model,
    optimizer=optimizer,
    device=device
)

# Si ya hay histórico, recuperamos mejor R@1 y paciencia acumulada
best_recall_at_1 = get_best_recall_from_history(
    metrics_history,
    min_delta=min_delta
)
epochs_without_improvement = count_epochs_without_improvement(
    metrics_history,
    min_delta=min_delta
)


# El entrenamiento continúa desde la época siguiente
start_epoch = last_epoch + 1

print("\n============================================================")
print("INICIO DEL ENTRENAMIENTO COMPLETO")
print("============================================================")
print(f"Epoch inicial: {start_epoch}")
print(f"Máximo de épocas: {max_epochs}")
print(f"Patience: {patience}")
print(f"Mejor R@1 inicial: {best_recall_at_1:.4f}")
print(f"Épocas sin mejora acumuladas: {epochs_without_improvement}")
print(f"Batch size train: {batch_size}")
print(f"nNeg: {nNeg}")
if num_val_queries is None:
    print("Validación: todas las queries de validación")
else:
    print(f"Validación: {num_val_queries} queries aleatorias")

if num_val_db is None:
    print("Database validación: completa")
else:
    print(f"Database validación limitada a: {num_val_db}")
print(f"Salida: {OUTPUT_DIR}")
print("============================================================")

for current_epoch in range(start_epoch, max_epochs + 1):

    print("\n============================================================")
    print(f"EPOCH {current_epoch}/{max_epochs}")
    print("============================================================")

    epoch_start_time = time.time()

    # --------------------------------------------------------
    # ENTRENAMIENTO
    # --------------------------------------------------------
    print("\nInicializando nueva época en MSLS...")
    train_dataset.new_epoch()
    print(f"Número de subsets MSLS: {train_dataset.nCacheSubset}")

    epoch_loss = 0.0
    num_batches = 0

    for subset_idx in range(train_dataset.nCacheSubset):

        print("\n------------------------------------------------------------")
        print(f"Epoch {current_epoch} | Subset {subset_idx + 1}/{train_dataset.nCacheSubset}")
        print("------------------------------------------------------------")

        gc.collect()
        torch.cuda.empty_cache()

        model.eval()

        mining_start_time = time.time()

        print("Ejecutando update_subcache para hard negative mining...")

        with torch.no_grad():
            train_dataset.update_subcache(net=model)

        mining_time = time.time() - mining_start_time

        print(f"update_subcache terminado en {mining_time / 60:.2f} min")

        if len(train_dataset.triplets) == 0:
            print(f"Subset {subset_idx + 1}: no se encontraron tripletas. Se omite.")
            continue

        print(f"Tripletas encontradas: {len(train_dataset.triplets)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers_train,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=2
        )

        model.train()

        subset_loss = 0.0
        subset_batches = 0

        for batch_idx, (images, target) in enumerate(train_loader):

            images = images.to(device, non_blocking=True)

            B, T, C, H, W = images.shape
            images = images.view(B * T, C, H, W)

            descriptors = model(images)
            descriptors = descriptors.view(B, T, -1)

            loss = msls_triplet_loss(
                descriptors=descriptors,
                nNeg=nNeg,
                margin=0.1
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            subset_loss += loss.item()
            subset_batches += 1

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {current_epoch} | "
                    f"Subset {subset_idx + 1}/{train_dataset.nCacheSubset} | "
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_subset_loss = subset_loss / max(subset_batches, 1)

        print(
            f"Subset {subset_idx + 1}/{train_dataset.nCacheSubset} terminado | "
            f"Avg subset loss: {avg_subset_loss:.4f}"
        )

    avg_train_loss = epoch_loss / max(num_batches, 1)

    print("\n------------------------------------------------------------")
    print(f"Train loss media epoch {current_epoch}: {avg_train_loss:.4f}")
    print(f"Batches entrenados: {num_batches}")
    print("------------------------------------------------------------")

    # --------------------------------------------------------
    # VALIDACIÓN
    # --------------------------------------------------------
    print("\n============================================================")
    print(f"VALIDACIÓN EPOCH {current_epoch}")
    print("============================================================")

    gc.collect()
    torch.cuda.empty_cache()

    model.eval()

    val_metrics, val_city_metrics = validate_vpr_citywise(
        model=model,
        val_packs=val_packs,
        transform=val_transform,
        device=device,
        num_val_queries=num_val_queries,
        num_val_db=num_val_db,
        ks=ks,
        descriptor_batch_size=descriptor_batch_size,
        distance_batch_size=distance_batch_size,
        num_workers=num_workers_val
    )

    recall_at_1 = float(val_metrics["recall@1"])
    recall_at_5 = float(val_metrics["recall@5"])
    recall_at_10 = float(val_metrics["recall@10"])

    print("\nResultados validación:")
    print(f"R@1:  {recall_at_1:.4f}")
    print(f"R@5:  {recall_at_5:.4f}")
    print(f"R@10: {recall_at_10:.4f}")

    epoch_time = time.time() - epoch_start_time

    # --------------------------------------------------------
    # REGISTRO DE MÉTRICAS Y CRITERIO DE MEJORA
    # --------------------------------------------------------
    epoch_checkpoint_path = CHECKPOINT_DIR / f"epoch_{current_epoch:03d}.ckpt"

    epoch_record = {
        "epoch": current_epoch,
        "train_loss": float(avg_train_loss),
        "recall@1": recall_at_1,
        "recall@5": recall_at_5,
        "recall@10": recall_at_10,
        "num_val_queries": num_val_queries,
        "num_val_db": num_val_db,
        "num_valid_queries": int(val_metrics["num_valid_queries"]),
        "epoch_time_min": epoch_time / 60.0,
        "num_train_batches": int(num_batches),
        "checkpoint": str(epoch_checkpoint_path),
        "best_so_far": False,
    }

    for city_row in val_city_metrics:
        city = city_row["city"]

        epoch_record[f"{city}_num_valid_queries"] = int(city_row["num_valid_queries"])
        epoch_record[f"{city}_recall@1"] = float(city_row["recall@1"])
        epoch_record[f"{city}_recall@5"] = float(city_row["recall@5"])
        epoch_record[f"{city}_recall@10"] = float(city_row["recall@10"])

    improved = recall_at_1 > best_recall_at_1 + min_delta

    if improved:

        print("\nNuevo mejor modelo encontrado.")
        print(f"R@1 anterior: {best_recall_at_1:.4f}")
        print(f"R@1 nuevo:    {recall_at_1:.4f}")

        best_recall_at_1 = recall_at_1
        epochs_without_improvement = 0
        epoch_record["best_so_far"] = True

    else:

        epochs_without_improvement += 1

        print("\nNo mejora R@1 de forma suficiente.")
        print(f"Mejor R@1 hasta ahora: {best_recall_at_1:.4f}")
        print(f"R@1 actual: {recall_at_1:.4f}")
        print(f"Min delta: {min_delta:.4f}")
        print(f"Épocas consecutivas sin mejora: {epochs_without_improvement}/{patience}")

    epoch_record["best_recall@1_so_far"] = float(best_recall_at_1)
    epoch_record["epochs_without_improvement"] = int(epochs_without_improvement)

    # --------------------------------------------------------
    # GUARDAR CHECKPOINT DE LA ÉPOCA
    # --------------------------------------------------------
    torch.save({
        "epoch": current_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": float(avg_train_loss),
        "val_metrics": {
            "recall@1": recall_at_1,
            "recall@5": recall_at_5,
            "recall@10": recall_at_10,
            "num_valid_queries": int(val_metrics["num_valid_queries"]),
        },
        "val_city_metrics": val_city_metrics,
        "validation_mode": "independent_citywise",
        "model_name": "ResNet50GeM",
        "descriptor_dim": model.meta["outputdim"],
        "train_cities": train_cities,
        "val_cities": val_cities,
        "nNeg": nNeg,
        "posDistThr": train_dataset.posDistThr if hasattr(train_dataset, "posDistThr") else None,
        "negDistThr": train_dataset.negDistThr if hasattr(train_dataset, "negDistThr") else None,
        "best_recall@1": best_recall_at_1,
        "early_stopping_patience": patience,
        "early_stopping_min_delta": min_delta,
        "epochs_without_improvement": epochs_without_improvement,
    }, epoch_checkpoint_path)

    print(f"Checkpoint de época guardado en: {epoch_checkpoint_path}")

    keep_last_n_checkpoints(
        checkpoint_dir=CHECKPOINT_DIR,
        n_keep=6
    )

    # --------------------------------------------------------
    # GUARDAR BEST MODEL SI MEJORA R@1
    # --------------------------------------------------------
    if improved:
        torch.save({
            "epoch": current_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": float(avg_train_loss),
            "val_metrics": {
                "recall@1": recall_at_1,
                "recall@5": recall_at_5,
                "recall@10": recall_at_10,
                "num_valid_queries": int(val_metrics["num_valid_queries"]),
            },
            "val_city_metrics": val_city_metrics,
            "validation_mode": "independent_citywise",
            "model_name": "ResNet50GeM",
            "descriptor_dim": model.meta["outputdim"],
            "train_cities": train_cities,
            "val_cities": val_cities,
            "nNeg": nNeg,
            "posDistThr": train_dataset.posDistThr if hasattr(train_dataset, "posDistThr") else None,
            "negDistThr": train_dataset.negDistThr if hasattr(train_dataset, "negDistThr") else None,
            "best_recall@1": best_recall_at_1,
            "source_checkpoint": str(epoch_checkpoint_path),
            "early_stopping_patience": patience,
            "early_stopping_min_delta": min_delta,
        }, BEST_MODEL_FILE)

        np.save(
            BEST_METRICS_FILE,
            {
                "epoch": current_epoch,
                "recall@1": recall_at_1,
                "recall@5": recall_at_5,
                "recall@10": recall_at_10,
                "num_valid_queries": int(val_metrics["num_valid_queries"]),
                "val_city_metrics": val_city_metrics,
                "validation_mode": "independent_citywise",
                "source_checkpoint": str(epoch_checkpoint_path),
            },
            allow_pickle=True
        )

        print(f"Best model guardado en: {BEST_MODEL_FILE}")

    # --------------------------------------------------------
    # GUARDAR HISTÓRICO DE MÉTRICAS
    # --------------------------------------------------------
    metrics_history = [
        r for r in metrics_history
        if int(r["epoch"]) != current_epoch
    ]

    metrics_history.append(epoch_record)
    save_metrics_history(metrics_history)

    print("\nResumen epoch:")
    print(f"Epoch: {current_epoch}")
    print(f"Train loss: {avg_train_loss:.4f}")
    print(f"R@1: {recall_at_1:.4f}")
    print(f"R@5: {recall_at_5:.4f}")
    print(f"R@10: {recall_at_10:.4f}")
    print(f"Tiempo epoch: {epoch_time / 60.0:.2f} min")

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------
    if epochs_without_improvement >= patience:

        print("\n============================================================")
        print("EARLY STOPPING ACTIVADO")
        print(f"R@1 no ha mejorado en {patience} épocas consecutivas.")
        print(f"Mejor R@1 alcanzado: {best_recall_at_1:.4f}")
        print(f"Best model guardado en: {BEST_MODEL_FILE}")
        print("============================================================")

        break

print("\n============================================================")
print("ENTRENAMIENTO FINALIZADO")
print(f"Mejor R@1: {best_recall_at_1:.4f}")
print(f"Best model: {BEST_MODEL_FILE}")
print(f"Últimos checkpoints en: {CHECKPOINT_DIR}")
print(f"Métricas: {METRICS_CSV}")
print("============================================================")
