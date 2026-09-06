# ============================================================
# CLASIFICACIÓN DE IMÁGENES DEL MISMO LUGAR EN MSLS
# Basado en la lógica de Mapillary MSLS msls.py
# + exportación de positivos a .npy
# + visualización de query y positivos directos
#
# ESTE SCRIPT:
# 1) Lee ciudades del dataset MSLS
# 2) Encuentra positivos directos query -> database
# 3) Guarda resultados por ciudad en carpetas separadas
# 4) Exporta la información a .npy con estructura estilo MSLS
# 5) Permite visualizar una query y sus positivos para comprobar
#    que todo está funcionando correctamente
# ============================================================
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/workspace/mhernang/VPR"))

from os.path import join
import math
import torch
import random
import sys
from tqdm import tqdm
from torch.utils.data import Dataset
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

# Carpeta raíz del dataset
ROOT_DIR = Path(os.environ.get("MSLS_ROOT", "/workspace/mhernang/VPR/data/mapillary"))
if not ROOT_DIR.exists():
    raise FileNotFoundError(f"No se encuentra el dataset MSLS en: {ROOT_DIR}")

# Split a usar: "train" o "val"

# Lista de ciudades a procesar
TRAIN_CITIES = [
    "trondheim", "london", "boston", "melbourne", "amsterdam", "helsinki",
    "tokyo", "toronto", "saopaulo", "moscow", "zurich", "paris", "bangkok",
    "budapest", "austin", "berlin", "ottawa", "phoenix", "goa", "amman",
    "nairobi", "manila"
]

VAL_CITIES = ["cph", "sf"]

SPLITS = {
    "train": TRAIN_CITIES,
    "val": VAL_CITIES,
}

# Distancia máxima (metros) para considerar dos imágenes positivas
# Igual que el parámetro posDistThr de MSLS
POS_DIST_THR = 25

# Si True, elimina panorámicas
EXCLUDE_PANOS = True

# Longitud de secuencia:
# 1 = imagen individual (im2im)
SEQ_LENGTH = 1

# Carpeta de salida
OUTPUT_DIR = Path(os.environ.get("POSITIVES_ROOT", PROJECT_ROOT / "positives"))
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ============================================================
# FUNCIONES AUXILIARES RELACIONADAS CON MSLS
# ============================================================

def get_subdir_for_city(city: str) -> str:
    """
    En MSLS algunas ciudades están en /test
    y otras en /train_val

    Devuelve la subcarpeta correcta.
    """
    default_test_cities = {
        "miami", "athens", "buenosaires",
        "stockholm", "bengaluru", "kampala"
    }

    return "test" if city in default_test_cities else "train_val"

def arange_as_seq(data: pd.DataFrame, path: str, seq_length: int = 1):
    """
    Construye secuencias válidas de imágenes.

    Si seq_length = 1:
        cada imagen se considera una secuencia.

    Si seq_length > 1:
        exige que:
        - pertenezcan a la misma sequence_key
        - frame_number sea consecutivo
    """
    seq_info = pd.read_csv(join(path, "seq_info.csv"), index_col=0)

    seq_keys = []
    seq_idxs = []

    for idx in data.index:

        # Evita salirse de rango
        if idx < (seq_length // 2) or idx >= (len(seq_info) - seq_length // 2):
            continue

        # Índices vecinos que forman la secuencia
        seq_idx = np.arange(-seq_length // 2, seq_length // 2) + 1 + idx
        seq = seq_info.iloc[seq_idx]

        # Todas deben pertenecer a la misma secuencia
        same_sequence = len(np.unique(seq["sequence_key"])) == 1

         # Si la secuencia tiene varias imágenes, deben ser consecutivas
        consecutive = True
        if seq_length > 1:
            consecutive = (seq["frame_number"].diff()[1:] == 1).all()

        if same_sequence and consecutive:

            # Se construye un string con las rutas de la secuencia.
            # Si seq_length = 1, el string contendrá una única ruta.
            seq_key = ",".join(
                [join(path, "images", key + ".jpg") for key in seq["key"]]
            )

            seq_keys.append(seq_key)
            seq_idxs.append(seq_idx)

    return seq_keys, np.asarray(seq_idxs)

def filter_by_center_frame(seq_keys, seq_idxs, valid_frames):
    """
    Conserva solo secuencias cuyo frame central
    pertenece a valid_frames.
    """
    keys = []
    idxs = []

    valid_set = set(valid_frames)

    for key, idx in zip(seq_keys, seq_idxs):
        center = idx[len(idx) // 2]

        if center in valid_set:
            keys.append(key)
            idxs.append(idx)

    return keys, np.asarray(idxs)

# ============================================================
# CONSTRUCCIÓN DE POSITIVOS DIRECTOS POR CIUDAD
# ============================================================

def build_city_msls_style(root_dir: str,
                          city: str,
                          mode: str = "train",
                          seq_length: int = 1,
                          pos_dist_thr: float = 10.0,
                          exclude_panos: bool = True):
    """
    Construye una salida compatible conceptualmente con MSLS:
    - query y database separados
    - pIdx: positivos directos por query
    - place_id local por query (se reinicia en cada ciudad)
    """

    subdir = get_subdir_for_city(city)
    city_root = join(root_dir, subdir, city)

    query_root = join(city_root, "query")
    db_root = join(city_root, "database")

    required_files = [
        join(query_root, "postprocessed.csv"),
        join(query_root, "raw.csv"),
        join(query_root, "seq_info.csv"),
        join(db_root, "postprocessed.csv"),
        join(db_root, "raw.csv"),
        join(db_root, "seq_info.csv"),
    ]

    for file_path in required_files:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"No existe el archivo requerido: {file_path}")

    # Leer CSVs
    # load query data
    q_data = pd.read_csv(join(query_root, "postprocessed.csv"), index_col=0)
    q_raw = pd.read_csv(join(query_root, "raw.csv"), index_col=0)

    # load database data
    db_data = pd.read_csv(join(db_root, "postprocessed.csv"), index_col=0)
    db_raw = pd.read_csv(join(db_root, "raw.csv"), index_col=0)

    # Construir secuencias válidas
    q_seq_keys, q_seq_idxs = arange_as_seq(q_data, query_root, seq_length)
    db_seq_keys, db_seq_idxs = arange_as_seq(db_data, db_root, seq_length)

    # Filtrar panorámicas
    if exclude_panos:
        q_non_pano = np.where((q_raw["pano"] == False).values)[0]
        db_non_pano = np.where((db_raw["pano"] == False).values)[0]

        q_seq_keys, q_seq_idxs = filter_by_center_frame(q_seq_keys, q_seq_idxs, q_non_pano)
        db_seq_keys, db_seq_idxs = filter_by_center_frame(db_seq_keys, db_seq_idxs, db_non_pano)

    unique_q_frames = np.unique(q_seq_idxs)
    unique_db_frames = np.unique(db_seq_idxs)

    # Si tras el filtrado no queda nada, se devuelve vacío
    if len(unique_q_frames) == 0 or len(unique_db_frames) == 0:
        return pd.DataFrame(), pd.DataFrame(), {}

    # Filtrar solo frames usados
    q_data_f = q_data.loc[unique_q_frames].copy()
    db_data_f = db_data.loc[unique_db_frames].copy()

    # Coordenadas UTM
    utm_q = q_data_f[["easting", "northing"]].values.reshape(-1, 2)
    utm_db = db_data_f[["easting", "northing"]].values.reshape(-1, 2)

    # Positivos directos query->database
    neigh = NearestNeighbors(algorithm="brute")
    neigh.fit(utm_db)
    _, indices = neigh.radius_neighbors(utm_q, pos_dist_thr)

    # --------------------------------------------------------
    # Construir tablas query y database
    # --------------------------------------------------------
    query_records = []
    db_records = []

    for i, key in enumerate(q_seq_keys):
        center_frame = int(q_seq_idxs[i][len(q_seq_idxs[i]) // 2])
        center_image = key.split(",")[len(key.split(",")) // 2]

        query_records.append({
            "city": city,
            "split": mode,
            "role": "query",
            "place_id": i,                  # <- se reinicia por ciudad
            "local_idx": i,                 # índice local de query
            "image_path": key,
            "center_image_path": center_image,
            "center_frame_idx": center_frame
        })

    for i, key in enumerate(db_seq_keys):
        center_frame = int(db_seq_idxs[i][len(db_seq_idxs[i]) // 2])
        center_image = key.split(",")[len(key.split(",")) // 2]

        db_records.append({
            "city": city,
            "split": mode,
            "role": "database",
            "local_idx": i,                 # índice local de database
            "image_path": key,
            "center_image_path": center_image,
            "center_frame_idx": center_frame
        })

    q_df = pd.DataFrame(query_records)
    db_df = pd.DataFrame(db_records)

    # Índices locales tipo MSLS
    qIdx = np.arange(len(q_df), dtype=np.int64)
    dbIdx = np.arange(len(db_df), dtype=np.int64)

    # --------------------------------------------------------
    # Construir pIdx: positivos directos por query
    # pIdx[i] = array de índices locales de database positivos para la query i
    # --------------------------------------------------------
    pIdx = []

    for q_idx in range(len(q_seq_keys)):
        q_frames = q_seq_idxs[q_idx]
        q_unique_positions = np.where(np.isin(unique_q_frames, q_frames))[0]

        positive_db_positions = np.unique([
            p for pos in indices[q_unique_positions] for p in pos
        ])

        if len(positive_db_positions) == 0:
            pIdx.append(np.array([], dtype=np.int64))
            continue

        db_frame_ids = unique_db_frames[positive_db_positions]

        db_seq_idxs_found = np.where(
            np.isin(db_seq_idxs, db_frame_ids).reshape(db_seq_idxs.shape)
        )[0]

        db_seq_idxs_found = np.unique(db_seq_idxs_found).astype(np.int64)
        pIdx.append(db_seq_idxs_found)

    pIdx = np.array(pIdx, dtype=object)

    # Diccionario de salida
    city_pack = {
        "city": city,
        "mode": mode,
        "qIdx": qIdx,                                   # queries válidas
        "dbIdx": dbIdx,                                 # database válida
        "pIdx": pIdx,                                   # positivos directos por query
        "query_paths": q_df["center_image_path"].to_numpy(),
        "database_paths": db_df["center_image_path"].to_numpy(),
        "query_place_id": q_df["place_id"].to_numpy()   # 0..Nq-1 por ciudad
    }

    return q_df, db_df, city_pack

# ============================================================
# EJECUCIÓN PRINCIPAL POR CIUDAD
# ============================================================
for split_name, city_list in SPLITS.items():
    for city in city_list:

        print("\n================================")
        print(f"Procesando ciudad: {city} ({split_name})")
        print("================================")

        # Carpeta de salida por ciudad
        city_output_dir = join(OUTPUT_DIR, split_name, city)
        Path(city_output_dir).mkdir(parents=True, exist_ok=True)

        # Archivos de salida
        query_csv = join(city_output_dir, f"query_nodes_{split_name}.csv")
        database_csv = join(city_output_dir, f"database_nodes_{split_name}.csv")
        positives_npy = join(city_output_dir, f"positives_{split_name}.npy")
        summary_csv = join(city_output_dir, f"summary_{split_name}.csv")

        if Path(positives_npy).exists():
            print(f"{city} ({split_name}) ya procesada. Se omite.")
            continue

        try:
            q_df, db_df, city_pack = build_city_msls_style(
                root_dir=ROOT_DIR,
                city=city,
                mode=split_name,
                seq_length=SEQ_LENGTH,
                pos_dist_thr=POS_DIST_THR,
                exclude_panos=EXCLUDE_PANOS
            )

            if q_df.empty or db_df.empty:
                print(f"Sin datos válidos para {city}")
                continue

            # Guardar tablas
            q_df.to_csv(query_csv, index=False)
            db_df.to_csv(database_csv, index=False)

            # Guardar .npy estilo MSLS
            np.save(positives_npy, city_pack, allow_pickle=True)

            # Resumen por query/place_id
            summary_df = pd.DataFrame({
                "place_id": city_pack["query_place_id"],
                "query_path": city_pack["query_paths"],
                "num_positives": [len(x) for x in city_pack["pIdx"]]
            })

            summary_df.to_csv(summary_csv, index=False)

            print("Archivos guardados:")
            print(query_csv)
            print(database_csv)
            print(positives_npy)
            print(summary_csv)

            print("\nResumen:")
            print(f"Ciudad: {city}")
            print(f"Split: {split_name}")
            print(f"Queries/place_id: {len(q_df)}")
            print(f"Database imágenes: {len(db_df)}")
            print(f"Media de positivos por query: {np.mean([len(x) for x in city_pack['pIdx']]):.2f}")

            # Ejemplo
            example_place_id = 0
            print(f"\nEjemplo place_id/query {example_place_id}:")
            print("Query:", city_pack["query_paths"][example_place_id])
            print("Positivos locales DB:", city_pack["pIdx"][example_place_id])

        except Exception as e:
            print(f"Error procesando {city} ({split_name}): {e}")
            continue
