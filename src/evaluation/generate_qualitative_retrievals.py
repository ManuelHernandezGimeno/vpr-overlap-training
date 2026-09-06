import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50


# ============================================================
# CONFIGURACION
# ============================================================

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/workspace/mhernang/VPR"))
MSLS_ROOT = Path(os.environ.get("MSLS_ROOT", PROJECT_ROOT / "data" / "mapillary"))
MAPILLARY_SLS_ROOT = Path(os.environ.get("MAPILLARY_SLS_ROOT", "/opt/mapillary_sls"))

VAL_CITIES = ["cph", "sf"]
VAL_CITIES_NAME = ",".join(VAL_CITIES)

TOP_K = 10
N_TOP_VISUAL = 5

N_CORRECT_EXAMPLES = 6
N_INCORRECT_EXAMPLES = 6
N_CITY_EXAMPLES = 4

DESCRIPTOR_BATCH_SIZE = 512
DISTANCE_BATCH_SIZE = 512
NUM_WORKERS = 16

RANDOM_SEED = 42

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analisis_cualitativo_recuperaciones"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_CANDIDATES = [
    PROJECT_ROOT / "outputs" / "Resultados_O" / "best_model.ckpt"
]

CHECKPOINT_FILE = None

for candidate in CHECKPOINT_CANDIDATES:
    if candidate.exists():
        CHECKPOINT_FILE = candidate
        break

if CHECKPOINT_FILE is None:
    raise FileNotFoundError(
        "No se ha encontrado best_model.ckpt. Revisa CHECKPOINT_CANDIDATES."
    )

print("Checkpoint usado:")
print(CHECKPOINT_FILE)


# ============================================================
# PREPARAR MSLS
# ============================================================

if not MSLS_ROOT.exists():
    raise FileNotFoundError(f"No existe MSLS_ROOT: {MSLS_ROOT}")

if not MAPILLARY_SLS_ROOT.exists():
    raise FileNotFoundError(f"No existe MAPILLARY_SLS_ROOT: {MAPILLARY_SLS_ROOT}")

msls_file = MAPILLARY_SLS_ROOT / "mapillary_sls" / "datasets" / "msls.py"

if not msls_file.exists():
    raise FileNotFoundError(f"No se encuentra msls.py en: {msls_file}")

text = msls_file.read_text()

if "self.pIdx = np.asarray(self.pIdx, dtype=object)" not in text:
    text = text.replace(
        "self.pIdx = np.asarray(self.pIdx)",
        "self.pIdx = np.asarray(self.pIdx, dtype=object)"
    )

if "self.nonNegIdx = np.asarray(self.nonNegIdx, dtype=object)" not in text:
    text = text.replace(
        "self.nonNegIdx = np.asarray(self.nonNegIdx)",
        "self.nonNegIdx = np.asarray(self.nonNegIdx, dtype=object)"
    )

msls_file.write_text(text)

print("msls.py preparado correctamente.")

sys.path.insert(0, str(MAPILLARY_SLS_ROOT))

from mapillary_sls.datasets.msls import MSLS

print("MSLS importado correctamente.")


# ============================================================
# TRANSFORMACION DE VALIDACION
# ============================================================

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# UTILIDADES
# ============================================================

def center_path(path_string):
    parts = str(path_string).split(",")
    return parts[len(parts) // 2]


def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(make_json_serializable(v) for v in obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()

    return obj


def row_value_is_valid(value):
    if value is None:
        return False

    try:
        return not pd.isna(value)
    except Exception:
        return True


# ============================================================
# MODELO
# ============================================================

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.avg_pool2d(x, kernel_size=(x.size(-2), x.size(-1)))
        x = x.pow(1.0 / self.p)
        return x


class ResNet50GeM(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()

        # No se cargan pesos ImageNet para evitar descargas.
        # El checkpoint entrenado cargara todos los pesos.
        backbone = resnet50(weights=None)

        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = GeM()
        self.fc = nn.Linear(2048, out_dim)

        self.meta = {"outputdim": out_dim}

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

checkpoint = torch.load(CHECKPOINT_FILE, map_location="cpu")

model = ResNet50GeM(out_dim=512)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("Modelo cargado correctamente.")

if isinstance(checkpoint, dict):
    if "epoch" in checkpoint:
        print("Epoch del checkpoint:", checkpoint["epoch"])

    if "val_metrics" in checkpoint:
        print("Metricas guardadas en checkpoint:")
        print(checkpoint["val_metrics"])

    if "best_recall@1" in checkpoint:
        print("best_recall@1 guardado:", checkpoint["best_recall@1"])


# ============================================================
# DATASET AUXILIAR PARA CARGAR IMAGENES
# ============================================================

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


# ============================================================
# DESCRIPTORES Y BUSQUEDA
# ============================================================

def compute_descriptors(
    model,
    image_paths,
    image_keys,
    transform,
    device,
    batch_size=512,
    num_workers=16,
    desc_name="descriptors"
):
    dataset = ImagePathDataset(
        image_paths=image_paths,
        image_keys=image_keys,
        transform=transform
    )

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    loader = DataLoader(**loader_kwargs)

    descriptors_list = []
    keys_list = []

    model.eval()

    print(f"Calculando {desc_name}: {len(dataset)} imagenes")

    with torch.inference_mode():
        for images, keys in tqdm(loader, desc=desc_name):
            images = images.to(device, non_blocking=True)

            descriptors = model(images)

            descriptors_list.append(descriptors.cpu())
            keys_list.append(keys.cpu().numpy())

    descriptors = torch.cat(descriptors_list, dim=0)
    keys = np.concatenate(keys_list, axis=0)

    print(f"{desc_name} calculados: {descriptors.shape}")

    return descriptors, keys


def compute_predictions_l2_vectorized(
    query_descriptors,
    query_keys,
    db_descriptors,
    db_keys,
    top_k=10,
    device="cuda",
    distance_batch_size=512
):
    """
    Calcula Top-K por distancia L2.

    Estructura:
        predictions[:, 0]  -> query_key local
        predictions[:, 1:] -> indices locales de database
    """

    model_device = torch.device(device)

    db_descriptors_gpu = db_descriptors.to(model_device)
    query_descriptors_gpu = query_descriptors.to(model_device)

    all_predictions = []

    effective_top_k = min(top_k, len(db_keys))

    print("Calculando distancias L2 query-database...")

    with torch.inference_mode():
        for start in tqdm(
            range(0, query_descriptors_gpu.shape[0], distance_batch_size),
            desc="L2 search"
        ):

            end = min(start + distance_batch_size, query_descriptors_gpu.shape[0])

            q_batch = query_descriptors_gpu[start:end]

            distances = torch.cdist(q_batch, db_descriptors_gpu, p=2)

            top_indices = torch.topk(
                distances,
                k=effective_top_k,
                dim=1,
                largest=False
            ).indices.cpu().numpy()

            for i, inds in enumerate(top_indices):
                q_key = int(query_keys[start + i])
                top_db_keys = db_keys[inds].astype(np.int64)

                row = np.concatenate([
                    np.array([q_key], dtype=np.int64),
                    top_db_keys
                ])

                all_predictions.append(row)

    predictions = np.stack(all_predictions, axis=0)

    return predictions


# ============================================================
# METRICAS
# ============================================================

def recall(ranks, pidx, ks):
    recall_at_k = np.zeros(len(ks), dtype=np.float32)

    for qidx in range(ranks.shape[0]):

        positives = pidx[qidx]

        if len(positives) == 0:
            continue

        for i, k in enumerate(ks):
            retrieved = ranks[qidx, :k]

            if np.sum(np.isin(retrieved, positives)) > 0:
                recall_at_k[i:] += 1
                break

    valid_queries = sum(len(p) > 0 for p in pidx)

    if valid_queries == 0:
        return recall_at_k

    recall_at_k /= valid_queries

    return recall_at_k


def eval_vpr(query_keys, positive_keys, predictions, ks=[1, 5, 10]):
    """
    Evalua Recall@K usando pIdx local de una unica ciudad.
    """

    pred_queries = predictions[:, 0]

    pred2gt = [
        np.where(pred_queries == key)[0][0]
        for key in query_keys
    ]

    ordered_predictions = predictions[pred2gt, 1:]

    recall_at_k = recall(ordered_predictions, positive_keys, ks)

    metrics = {}

    for i, k in enumerate(ks):
        metrics[f"recall@{k}"] = float(recall_at_k[i])

    return metrics


# ============================================================
# EVALUACION INDEPENDIENTE POR CIUDAD
# ============================================================

def evaluate_city(
    city,
    query_global_offset,
    db_global_offset
):
    """
    Evalua una ciudad de validacion de forma independiente.

    Importante:
        - Las queries de city buscan solo en database de city.
        - pIdx se usa en indices locales de city.
        - Los indices globales solo se guardan como informacion adicional.
    """

    print("\n============================================================")
    print(f"EVALUANDO CIUDAD: {city}")
    print("============================================================")

    ds = MSLS(
        root_dir=MSLS_ROOT,
        cities=city,
        transform=val_transform,
        mode="val",
        posDistThr=25,
        negDistThr=25,
    )

    query_paths = np.asarray(ds.qImages)
    db_paths = np.asarray(ds.dbImages)
    pIdx = np.asarray(ds.pIdx, dtype=object)

    print(f"{city}: queries={len(query_paths)}")
    print(f"{city}: database={len(db_paths)}")
    print(f"{city}: queries con positivos={sum(len(p) > 0 for p in pIdx)}")

    db_keys = np.arange(len(db_paths), dtype=np.int64)
    query_keys_all = np.arange(len(query_paths), dtype=np.int64)

    valid_query_indices = np.array(
        [i for i, p in enumerate(pIdx) if len(p) > 0],
        dtype=np.int64
    )

    selected_query_paths = query_paths[valid_query_indices]
    selected_query_keys = query_keys_all[valid_query_indices]
    selected_positive_keys = pIdx[valid_query_indices]

    print(f"{city}: queries usadas en evaluacion={len(selected_query_paths)}")

    db_descriptors, db_keys_out = compute_descriptors(
        model=model,
        image_paths=db_paths,
        image_keys=db_keys,
        transform=val_transform,
        device=device,
        batch_size=DESCRIPTOR_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        desc_name=f"{city} database descriptors"
    )

    query_descriptors, query_keys_out = compute_descriptors(
        model=model,
        image_paths=selected_query_paths,
        image_keys=selected_query_keys,
        transform=val_transform,
        device=device,
        batch_size=DESCRIPTOR_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        desc_name=f"{city} query descriptors"
    )

    predictions = compute_predictions_l2_vectorized(
        query_descriptors=query_descriptors,
        query_keys=query_keys_out,
        db_descriptors=db_descriptors,
        db_keys=db_keys_out,
        top_k=TOP_K,
        device=device,
        distance_batch_size=DISTANCE_BATCH_SIZE
    )

    metrics = eval_vpr(
        query_keys=query_keys_out,
        positive_keys=selected_positive_keys,
        predictions=predictions,
        ks=[1, 5, 10]
    )

    print(f"\nMetricas {city}:")
    print(f"R@1:  {metrics['recall@1']:.4f}")
    print(f"R@5:  {metrics['recall@5']:.4f}")
    print(f"R@10: {metrics['recall@10']:.4f}")

    query_key_to_desc_pos = {
        int(key): i
        for i, key in enumerate(query_keys_out)
    }

    rows = []

    for pred in predictions:

        q_local_idx = int(pred[0])
        q_global_idx = int(query_global_offset + q_local_idx)

        topk_local = [int(x) for x in pred[1:]]

        positives_local = [int(x) for x in pIdx[q_local_idx]]
        positives_local_set = set(positives_local)

        positives_global = [
            int(db_global_offset + p)
            for p in positives_local
        ]

        top1_local_idx = topk_local[0]
        top1_global_idx = int(db_global_offset + top1_local_idx)

        top1_correct = top1_local_idx in positives_local_set

        first_correct_rank = None

        for rank, db_local_idx in enumerate(topk_local, start=1):
            if db_local_idx in positives_local_set:
                first_correct_rank = rank
                break

        # Positivo oficial mas cercano en espacio de descriptores.
        # Solo afecta a la visualizacion del GT naranja.
        nearest_positive_local_idx = None
        nearest_positive_global_idx = None
        nearest_positive_desc_distance = None

        valid_positive_candidates = [
            int(p) for p in positives_local
            if 0 <= int(p) < len(db_paths)
        ]

        if len(valid_positive_candidates) > 0:
            q_desc_pos = query_key_to_desc_pos[q_local_idx]

            q_desc = query_descriptors[q_desc_pos].unsqueeze(0)
            cand_desc = db_descriptors[valid_positive_candidates]

            dists = torch.cdist(q_desc, cand_desc, p=2).squeeze(0)

            best_local = int(torch.argmin(dists).item())

            nearest_positive_local_idx = int(valid_positive_candidates[best_local])
            nearest_positive_global_idx = int(db_global_offset + nearest_positive_local_idx)
            nearest_positive_desc_distance = float(dists[best_local].item())

        row = {
            "query_idx": q_global_idx,
            "query_global_idx": q_global_idx,
            "query_local_idx": q_local_idx,
            "query_city": city,
            "query_path": center_path(query_paths[q_local_idx]),

            "num_positives": len(positives_local),

            "top1_idx": top1_global_idx,
            "top1_global_idx": top1_global_idx,
            "top1_local_idx": top1_local_idx,
            "top1_city": city,
            "top1_path": center_path(db_paths[top1_local_idx]),

            "top1_correct": bool(top1_correct),
            "first_correct_rank": first_correct_rank,
            "correct_in_top10": first_correct_rank is not None,

            "positives_local": json.dumps(sorted(positives_local)),
            "positives_global": json.dumps(sorted(positives_global)),

            "nearest_positive_idx": nearest_positive_global_idx,
            "nearest_positive_global_idx": nearest_positive_global_idx,
            "nearest_positive_local_idx": nearest_positive_local_idx,
            "nearest_positive_city": city if nearest_positive_local_idx is not None else None,
            "nearest_positive_path": center_path(db_paths[nearest_positive_local_idx]) if nearest_positive_local_idx is not None else None,
            "nearest_positive_desc_distance": nearest_positive_desc_distance,
        }

        for rank, db_local_idx in enumerate(topk_local, start=1):
            db_global_idx = int(db_global_offset + db_local_idx)

            row[f"top{rank}_idx"] = db_global_idx
            row[f"top{rank}_global_idx"] = db_global_idx
            row[f"top{rank}_local_idx"] = int(db_local_idx)
            row[f"top{rank}_city"] = city
            row[f"top{rank}_path"] = center_path(db_paths[db_local_idx])
            row[f"top{rank}_correct"] = bool(db_local_idx in positives_local_set)

        rows.append(row)

    city_summary = {
        "city": city,
        "num_queries_total": int(len(query_paths)),
        "num_queries_evaluated": int(len(rows)),
        "num_database": int(len(db_paths)),
        "query_global_offset": int(query_global_offset),
        "db_global_offset": int(db_global_offset),
        "recall@1": float(metrics["recall@1"]),
        "recall@5": float(metrics["recall@5"]),
        "recall@10": float(metrics["recall@10"]),
        "top1_correct": int(sum(r["top1_correct"] for r in rows)),
        "top1_incorrect": int(sum(not r["top1_correct"] for r in rows)),
        "correct_in_top10": int(sum(r["correct_in_top10"] for r in rows)),
    }

    return rows, city_summary, len(query_paths), len(db_paths)


all_rows = []
city_summaries = []

query_global_offset = 0
db_global_offset = 0

for city in VAL_CITIES:
    city_rows, city_summary, num_queries_city, num_db_city = evaluate_city(
        city=city,
        query_global_offset=query_global_offset,
        db_global_offset=db_global_offset
    )

    all_rows.extend(city_rows)
    city_summaries.append(city_summary)

    query_global_offset += num_queries_city
    db_global_offset += num_db_city


# ============================================================
# DATAFRAME Y METRICAS GLOBALES
# ============================================================

df = pd.DataFrame(all_rows)

r1_visual = df["top1_correct"].mean()
r5_visual = df[[f"top{i}_correct" for i in range(1, 6)]].any(axis=1).mean()
r10_visual = df[[f"top{i}_correct" for i in range(1, 11)]].any(axis=1).mean()

print("\n============================================================")
print("METRICAS GLOBALES CORREGIDAS")
print("============================================================")
print(f"Ciudades: {VAL_CITIES_NAME}")
print(f"Queries evaluadas: {len(df)}")
print(f"Database total informativa: {db_global_offset}")
print(f"R@1 global:  {r1_visual:.4f}")
print(f"R@5 global:  {r5_visual:.4f}")
print(f"R@10 global: {r10_visual:.4f}")
print("Top-1 correctos:", int(df["top1_correct"].sum()))
print("Top-1 fallos:", int((~df["top1_correct"]).sum()))
print("Correcto en Top-10:", int(df["correct_in_top10"].sum()))

csv_path = OUTPUT_DIR / "recuperaciones_resumen.csv"
df.to_csv(csv_path, index=False)
print(f"CSV guardado en: {csv_path}")

df_city_metrics = pd.DataFrame(city_summaries)

city_metrics_path = OUTPUT_DIR / "metricas_por_ciudad.csv"
df_city_metrics.to_csv(city_metrics_path, index=False)

print(f"Metricas por ciudad guardadas en: {city_metrics_path}")

print("\nMetricas por ciudad:")
for row in city_summaries:
    print(
        f"{row['city']}: "
        f"queries={row['num_queries_evaluated']}, "
        f"R@1={row['recall@1']:.4f}, "
        f"R@5={row['recall@5']:.4f}, "
        f"R@10={row['recall@10']:.4f}"
    )

if isinstance(checkpoint, dict) and "val_metrics" in checkpoint:
    print("\nMetricas guardadas en checkpoint:")
    print(checkpoint["val_metrics"])
    print(
        "[INFO] Es normal que no coincidan si el checkpoint fue seleccionado "
        "con la validacion antigua MSLS(cities='cph,sf')."
    )


# ============================================================
# CREAR FIGURAS
# ============================================================

def load_thumbnail(path, size=(224, 160)):
    img = Image.open(path).convert("RGB")
    img = ImageOps.contain(img, size)

    canvas = Image.new("RGB", size, "white")

    x = (size[0] - img.size[0]) // 2
    y = (size[1] - img.size[1]) // 2

    canvas.paste(img, (x, y))

    return canvas


def draw_cell(draw, x, y, w, h, label, status):
    if status == "query":
        color = (30, 90, 200)
    elif status == "correct":
        color = (0, 160, 0)
    elif status == "wrong":
        color = (210, 0, 0)
    elif status == "gt":
        color = (220, 150, 0)
    else:
        color = (0, 0, 0)

    draw.rectangle([x, y, x + w, y + h], outline=color, width=5)
    draw.text((x + 5, y + h + 4), label, fill=color)


def save_retrieval_grid(row, save_path, title=None, n_top=5, include_gt=True):
    thumb_w, thumb_h = 224, 160
    text_h = 50
    gap = 12
    margin = 16

    q_global = int(row["query_global_idx"])
    q_local = int(row["query_local_idx"])
    q_city = str(row["query_city"])

    items = []

    items.append({
        "path": row["query_path"],
        "label": f"Query {q_city} {q_local} | g {q_global}",
        "status": "query"
    })

    shown_global_indices = set()

    for rank in range(1, n_top + 1):
        db_global = int(row[f"top{rank}_global_idx"])
        db_local = int(row[f"top{rank}_local_idx"])
        db_city = str(row[f"top{rank}_city"])
        correct = bool(row[f"top{rank}_correct"])

        shown_global_indices.add(db_global)

        items.append({
            "path": row[f"top{rank}_path"],
            "label": f"Top-{rank} | {db_city} {db_local} | g {db_global}",
            "status": "correct" if correct else "wrong"
        })

    if include_gt and row_value_is_valid(row.get("nearest_positive_global_idx", None)):
        gt_global = int(row["nearest_positive_global_idx"])

        if gt_global not in shown_global_indices:
            gt_city = str(row["nearest_positive_city"])
            gt_local = int(row["nearest_positive_local_idx"])
            gt_dist = row.get("nearest_positive_desc_distance", None)

            if row_value_is_valid(gt_dist):
                gt_label = f"GT | {gt_city} {gt_local} | g {gt_global} | d={float(gt_dist):.3f}"
            else:
                gt_label = f"GT | {gt_city} {gt_local} | g {gt_global}"

            items.append({
                "path": row["nearest_positive_path"],
                "label": gt_label,
                "status": "gt"
            })

    n = len(items)

    title_h = 40 if title is not None else 0

    canvas_w = margin * 2 + n * thumb_w + (n - 1) * gap
    canvas_h = margin * 2 + title_h + thumb_h + text_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    if title is not None:
        draw.text((margin, margin), title, fill=(0, 0, 0))

    y_img = margin + title_h

    for i, item in enumerate(items):
        x = margin + i * (thumb_w + gap)

        thumb = load_thumbnail(item["path"], size=(thumb_w, thumb_h))
        canvas.paste(thumb, (x, y_img))

        draw_cell(
            draw=draw,
            x=x,
            y=y_img,
            w=thumb_w,
            h=thumb_h,
            label=item["label"],
            status=item["status"]
        )

    canvas.save(save_path)
    print(f"Figura guardada: {save_path}")


def save_montage(rows_df, save_path, title, n_top=5, include_gt=True):
    temp_paths = []

    for i, (_, row) in enumerate(rows_df.iterrows(), start=1):
        temp_path = OUTPUT_DIR / f"_temp_{i:02d}.png"

        save_retrieval_grid(
            row=row,
            save_path=temp_path,
            title=f"{title} | ejemplo {i}",
            n_top=n_top,
            include_gt=include_gt
        )

        temp_paths.append(temp_path)

    imgs = [Image.open(p).convert("RGB") for p in temp_paths]

    if len(imgs) == 0:
        return

    width = max(img.width for img in imgs)
    height = sum(img.height for img in imgs)

    montage = Image.new("RGB", (width, height), "white")

    y = 0

    for img in imgs:
        montage.paste(img, (0, y))
        y += img.height

    montage.save(save_path)

    print(f"Montaje guardado: {save_path}")

    for p in temp_paths:
        p.unlink(missing_ok=True)


rng = np.random.default_rng(RANDOM_SEED)

df_correct = df[df["top1_correct"] == True].copy()
df_incorrect = df[df["top1_correct"] == False].copy()

# ------------------------------------------------------------
# Montajes globales
# ------------------------------------------------------------

if len(df_correct) > 0:
    n = min(N_CORRECT_EXAMPLES, len(df_correct))

    correct_indices = rng.choice(
        df_correct.index.to_numpy(),
        size=n,
        replace=False
    )

    df_correct_sel = df.loc[correct_indices]

    save_montage(
        rows_df=df_correct_sel,
        save_path=OUTPUT_DIR / "recuperaciones_correctas_top1.png",
        title="Recuperacion correcta: Top-1 correcto",
        n_top=N_TOP_VISUAL,
        include_gt=False
    )

    for i, (_, row) in enumerate(df_correct_sel.iterrows(), start=1):
        save_retrieval_grid(
            row=row,
            save_path=OUTPUT_DIR / f"correcta_{i:02d}.png",
            title="Recuperacion correcta: Top-1 correcto",
            n_top=N_TOP_VISUAL,
            include_gt=False
        )
else:
    print("[WARNING] No hay ejemplos correctos Top-1.")


if len(df_incorrect) > 0:
    n = min(N_INCORRECT_EXAMPLES, len(df_incorrect))

    incorrect_indices = rng.choice(
        df_incorrect.index.to_numpy(),
        size=n,
        replace=False
    )

    df_incorrect_sel = df.loc[incorrect_indices]

    save_montage(
        rows_df=df_incorrect_sel,
        save_path=OUTPUT_DIR / "recuperaciones_incorrectas_top1.png",
        title="Recuperacion incorrecta segun MSLS: Top-1 incorrecto",
        n_top=N_TOP_VISUAL,
        include_gt=True
    )

    for i, (_, row) in enumerate(df_incorrect_sel.iterrows(), start=1):
        save_retrieval_grid(
            row=row,
            save_path=OUTPUT_DIR / f"incorrecta_{i:02d}.png",
            title="Recuperacion incorrecta segun MSLS: Top-1 incorrecto",
            n_top=N_TOP_VISUAL,
            include_gt=True
        )
else:
    print("[WARNING] No hay ejemplos incorrectos Top-1.")


# ------------------------------------------------------------
# Montajes por ciudad
# ------------------------------------------------------------

for city in VAL_CITIES:
    df_city = df[df["query_city"] == city].copy()

    df_city_correct = df_city[df_city["top1_correct"] == True].copy()
    df_city_incorrect = df_city[df_city["top1_correct"] == False].copy()

    if len(df_city_correct) > 0:
        n = min(N_CITY_EXAMPLES, len(df_city_correct))

        idx = rng.choice(
            df_city_correct.index.to_numpy(),
            size=n,
            replace=False
        )

        save_montage(
            rows_df=df.loc[idx],
            save_path=OUTPUT_DIR / f"recuperaciones_correctas_top1_{city}.png",
            title=f"{city}: recuperacion correcta Top-1",
            n_top=N_TOP_VISUAL,
            include_gt=False
        )

    if len(df_city_incorrect) > 0:
        n = min(N_CITY_EXAMPLES, len(df_city_incorrect))

        idx = rng.choice(
            df_city_incorrect.index.to_numpy(),
            size=n,
            replace=False
        )

        save_montage(
            rows_df=df.loc[idx],
            save_path=OUTPUT_DIR / f"recuperaciones_incorrectas_top1_{city}.png",
            title=f"{city}: recuperacion incorrecta Top-1",
            n_top=N_TOP_VISUAL,
            include_gt=True
        )


# ------------------------------------------------------------
# Caso adicional 1:
# Top-1 falla, pero hay positivo dentro de Top-10
# ------------------------------------------------------------

df_partial_fail = df[
    (df["top1_correct"] == False) &
    (df["correct_in_top10"] == True)
].copy()

if len(df_partial_fail) > 0:
    row = df_partial_fail.iloc[0]

    save_retrieval_grid(
        row=row,
        save_path=OUTPUT_DIR / "caso_top1_falla_pero_acierta_en_top10.png",
        title=f"Top-1 falla, pero hay positivo en Top-{int(row['first_correct_rank'])}",
        n_top=N_TOP_VISUAL,
        include_gt=True
    )
else:
    print("[INFO] No hay casos donde falle Top-1 pero acierte en Top-10.")


# ------------------------------------------------------------
# Caso adicional 2:
# Fallo completo en Top-10
# ------------------------------------------------------------

df_full_fail = df[
    (df["top1_correct"] == False) &
    (df["correct_in_top10"] == False)
].copy()

if len(df_full_fail) > 0:
    row = df_full_fail.iloc[0]

    save_retrieval_grid(
        row=row,
        save_path=OUTPUT_DIR / "caso_fallo_completo_top10.png",
        title="Fallo completo: no hay positivo en Top-10",
        n_top=N_TOP_VISUAL,
        include_gt=True
    )
else:
    print("[INFO] No hay casos de fallo completo en Top-10.")


# ============================================================
# RESUMEN JSON
# ============================================================

summary = {
    "checkpoint": str(CHECKPOINT_FILE),
    "checkpoint_epoch": checkpoint.get("epoch", None) if isinstance(checkpoint, dict) else None,
    "val_cities": VAL_CITIES_NAME,
    "evaluation_mode": "independent_citywise",
    "important_note": (
        "Cada ciudad se evalua contra su propia database. "
        "Los indices globales son solo informativos."
    ),
    "num_queries_evaluated": int(len(df)),
    "num_database_total_informative": int(db_global_offset),
    "recall@1_visual": float(r1_visual),
    "recall@5_visual": float(r5_visual),
    "recall@10_visual": float(r10_visual),
    "top1_correct": int(df["top1_correct"].sum()),
    "top1_incorrect": int((~df["top1_correct"]).sum()),
    "correct_in_top10": int(df["correct_in_top10"].sum()),
    "city_metrics_csv": str(city_metrics_path),
    "csv": str(csv_path),
}

if isinstance(checkpoint, dict) and "val_metrics" in checkpoint:
    summary["checkpoint_val_metrics"] = make_json_serializable(checkpoint["val_metrics"])

summary_path = OUTPUT_DIR / "resumen_metricas.json"

with open(summary_path, "w") as f:
    json.dump(make_json_serializable(summary), f, indent=4)

print(f"Resumen de metricas guardado en: {summary_path}")


print("\n============================================================")
print("PROCESO TERMINADO")
print("Resultados en:")
print(OUTPUT_DIR)
print("Archivos principales:")
print("- recuperaciones_resumen.csv")
print("- metricas_por_ciudad.csv")
print("- resumen_metricas.json")
print("- recuperaciones_correctas_top1.png")
print("- recuperaciones_incorrectas_top1.png")
print("- recuperaciones_correctas_top1_cph.png")
print("- recuperaciones_incorrectas_top1_cph.png")
print("- recuperaciones_correctas_top1_sf.png")
print("- recuperaciones_incorrectas_top1_sf.png")
print("- correcta_XX.png")
print("- incorrecta_XX.png")
print("- caso_top1_falla_pero_acierta_en_top10.png")
print("- caso_fallo_completo_top10.png")
print("============================================================")
