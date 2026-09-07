# 1) Instalar VGGT y versiones requeridas de librerías en Docker
# 2) Cargar librerías
import os
import sys
import json
import gc
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", DEFAULT_PROJECT_ROOT)
)
VGGT_ROOT = os.environ.get("VGGT_ROOT", "/opt/vggt")
sys.path.append(VGGT_ROOT)

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

print("Imports correctos")

# 3) Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
if device == "cuda":
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
else:
    dtype = torch.float32

# Cargar modelo
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model.eval()

# CONFIGURACIÓN
# Carpeta raíz del dataset MSLS con las imágenes originales
MSLS_ROOT = Path(os.environ.get("MSLS_ROOT", PROJECT_ROOT / "data" / "msls"))
if not MSLS_ROOT.exists():
    raise FileNotFoundError(f"No se encuentra el dataset MSLS en: {MSLS_ROOT}")

# Carpeta donde están las subcarpetas por ciudad con positives_train.npy
POSITIVES_ROOT = Path(os.environ.get("TRAIN_POSITIVES_ROOT", PROJECT_ROOT / "positives" / "train"))

# Carpeta donde se guardarán profundidades, intrínsecos y resultados
OUTPUT_ROOT = Path(os.environ.get("OVERLAP_ROOT", PROJECT_ROOT / "overlaps" / "vggt"))

MAX_IMAGES_PER_BLOCK = 20
SAVE_DEPTH_IMAGES = False
SAVE_INTRINSICS = True

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Función para guardar el mapa de profundidad

def save_depth_as_image(depth, save_path):
    """
    Guarda el mapa de profundidad como imagen normalizada.
    """
    depth = depth.detach().cpu().squeeze().float().numpy()

    d_min = np.nanmin(depth)
    d_max = np.nanmax(depth)

    if d_max - d_min < 1e-8:
        depth_norm = np.zeros_like(depth)
    else:
        depth_norm = (depth - d_min) / (d_max - d_min)

    depth_img = (depth_norm * 255).astype(np.uint8)
    Image.fromarray(depth_img).save(save_path)

# Función para calcular rotación y traslación relativa

def relative_pose(E_ref, E_pos):
    """
    Calcula la rotación y traslación relativa entre referencia y positivo.

    Las extrínsecas tienen forma [R|t], mundo -> cámara.
    """
    R_ref = E_ref[:, :3]
    t_ref = E_ref[:, 3]

    R_pos = E_pos[:, :3]
    t_pos = E_pos[:, 3]

    R_rel = R_pos @ R_ref.T
    t_rel = t_pos - R_rel @ t_ref

    return R_rel, t_rel

# Calculo del overlap entre imágenes

# Defino una función para proyectar los puntos de una cámara en la otra
def project_points(points_3d, K, E):
    R = E[:, :3]
    t = E[:, 3:]

    # mundo -> la otra cámara
    points_cam = (R @ points_3d.T + t).T  # [N, 3]

    # Filtro para quedarme solo puntos delante de la cámara
    z = points_cam[:, 2]
    valid_z = z > 1e-6

    points_cam_valid = points_cam[valid_z]
    if points_cam_valid.shape[0] == 0:
        return None, valid_z

    # proyección pinhole (pasar a coordenadas homogéneas de imagen y después a 2D del pixel )
    pixels_h = (K @ points_cam_valid.T).T  # [M, 3]
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]

    return pixels, valid_z

def compute_overlap_and_mask(depth_map, intrinsic, extrinsic, img_idx_src=0, img_idx_tgt=1):
    # depth, K, E de cada imagen
    depth = depth_map[0, img_idx_src].detach().squeeze()
    K_src = intrinsic[0, img_idx_src].detach()
    E_src = extrinsic[0, img_idx_src].detach()
    K_tgt = intrinsic[0, img_idx_tgt].detach()
    E_tgt = extrinsic[0, img_idx_tgt].detach()

    H, W = depth.shape
    # malla de píxeles
    ys, xs = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij"
    )

    # intrínsecos
    fx = K_src[0, 0]
    fy = K_src[1, 1]
    cx = K_src[0, 2]
    cy = K_src[1, 2]

    # puntos 3D en coords cámara source
    z = depth.reshape(-1)
    x = (xs.reshape(-1) - cx) * z / fx
    y = (ys.reshape(-1) - cy) * z / fy

    pts_cam_src = torch.stack([x, y, z], dim=1)  # [N, 3]

    # cámara source -> proyecto en coordenadas mundo
    # Si se cumple Xc = R Xw + t  => Xw = R^T (Xc - t)
    R_src = E_src[:, :3]
    t_src = E_src[:, 3]
    pts_world = (R_src.T @ (pts_cam_src - t_src).T).T

    # proyectar en target (la otra cámara)
    pixels_tgt, valid_z = project_points(pts_world, K_tgt, E_tgt)

    # máscara final en la imagen source, inicialmente todo False
    mask_flat = torch.zeros(H * W, dtype=torch.bool, device=device)

    if pixels_tgt is None:
        return 0.0, mask_flat.reshape(H, W)

    # índices de los puntos source que estaban delante de la target
    valid_indices = torch.nonzero(valid_z, as_tuple=False).squeeze(1)

    u = pixels_tgt[:, 0]
    v = pixels_tgt[:, 1]

    # Calculamos que pixeles de una imagen están contenidos en la otra también
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)

     # marcar como True solo los píxeles source cuyos puntos caen dentro de la target
    mask_flat[valid_indices[inside]] = True
    mask = mask_flat.reshape(H, W)

    # ratio
    overlap_ratio = mask.float().mean().item()

    return overlap_ratio, mask

# Función para detectar lugares de más de X imagenes y dividirlos

def split_positives_into_balanced_blocks(positive_paths, max_total_images_per_block):
    """
    Divide los positivos en bloques equilibrados.

    La referencia NO se incluye aquí.
    Cada bloque tendrá como máximo:
        max_total_images_per_block - 1 positivos

    porque la referencia se añadirá después en cada bloque.
    """

    max_positives_per_block = max_total_images_per_block - 1

    n = len(positive_paths)

    if n == 0:
        return []

    # Si caben todos los positivos con la referencia en un único bloque
    if n <= max_positives_per_block:
        return [positive_paths]

    # Número mínimo de bloques necesarios
    num_blocks = int(np.ceil(n / max_positives_per_block))

    # Reparto equilibrado
    base_size = n // num_blocks
    remainder = n % num_blocks

    blocks = []
    start = 0

    for block_idx in range(num_blocks):
        # Los primeros "remainder" bloques tienen un elemento más
        block_size = base_size + (1 if block_idx < remainder else 0)

        block = positive_paths[start:start + block_size]
        blocks.append(block)

        start += block_size

    return blocks

# Funciones para detectar lugares creados y crear el identificador

def is_place_done(place_output_dir):
    """
    Devuelve True si el lugar ya fue procesado completamente.
    """
    done_file = place_output_dir / "DONE.txt"
    return done_file.exists()


def mark_place_as_done(place_output_dir):
    """
    Crea un archivo marcador para indicar que el lugar se procesó correctamente.
    """
    done_file = place_output_dir / "DONE.txt"

    with open(done_file, "w") as f:
        f.write("DONE\n")

# Bucle principal

# Busca dentro de POSITIVES_ROOT las carpeta de las ciudades
# y crea una lista ordenada alfabéticamente
TARGET_CITIES = ["trondheim", "london", "boston", "melbourne", "amsterdam","helsinki",
              "tokyo","toronto","saopaulo","moscow","zurich","paris","bangkok",
              "budapest","austin","berlin","ottawa","phoenix","goa","amman","nairobi","manila"]


city_dirs = [POSITIVES_ROOT / city for city in TARGET_CITIES]

# Empieza un bucle que procesa una ciudad cada vez hasta acabarlas
for city_dir in city_dirs:
    if not city_dir.exists():
          print(f"[ERROR] No existe la carpeta de la ciudad: {city_dir}")
          continue

    # Construye la ruta al archivo .npy de la ciudad correspondiente
    # y compruebo si existe
    positives_file = city_dir / "positives_train.npy"

    if not positives_file.exists():
        print(f"[SKIP] {city_dir.name}: no existe positives_train.npy")
        continue
    # Cargamos el .npy y extraemos metadatos
    data = np.load(positives_file, allow_pickle=True).item()

    city_name = data["city"]
    mode = data["mode"]

    query_paths = data["query_paths"]
    database_paths = data["database_paths"]
    pIdx = data["pIdx"]                     # Contiene los índices positivos
    query_place_id = data["query_place_id"] # Contiene el identificador de lugar asociado a cada query

    print("\n==============================")
    print(f"Ciudad: {city_name}")
    print(f"Modo: {mode}")
    print(f"Queries: {len(query_paths)}")
    print(f"Database: {len(database_paths)}")
    print("==============================")

    # Construye la ruta donde se guardarán los resultados de esa ciudad
    city_output_dir = OUTPUT_ROOT / city_name
    city_output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Cada query representa una imagen de referencia.
    # Sus positivos se obtienen de database_paths usando pIdx.
    # Se extrae el identificador del lugar asociado a esa query
    # en string para usarlo en nombres de carpetas y claves del diccionario.
    # --------------------------------------------------------
    for q_i in range(len(query_paths)):

        ref_path = Path(query_paths[q_i])
        positive_indices = list(pIdx[q_i])

        place_id = str(query_place_id[q_i])

        # Carpeta de salida del lugar
        place_output_dir = city_output_dir / f"place_{place_id}"

        if not ref_path.exists():
            print(f"[SKIP] {city_name}/place_{place_id}: no existe la referencia {ref_path}")
            continue

        # Si ya está procesado, saltarlo
        if is_place_done(place_output_dir):
            print(f"[DONE] {city_name}/place_{place_id}: ya procesado")
            continue

        # Si no hay positivos, solo hay imagen de referencia.
        # No hace falta ejecutar VGGT.
        if len(positive_indices) == 0:
            print(f"[SKIP] {city_name}/place_{place_id}: solo referencia, sin positivos")
            continue

        # Lista de positivos
        positive_paths = [Path(database_paths[db_i]) for db_i in positive_indices]

        # Eliminar duplicados manteniendo orden
        positive_paths = list(dict.fromkeys(positive_paths))

        # Crear bloques equilibrados de positivos
        positive_blocks = split_positives_into_balanced_blocks(
            positive_paths,
            max_total_images_per_block=MAX_IMAGES_PER_BLOCK
        )

        print(
            f"[BLOCKS] {city_name}/place_{place_id}: "
            f"1 referencia + {len(positive_paths)} positivos -> "
            f"{len(positive_blocks)} bloque(s)"
        )

        # Carpeta de salida ya creada
        place_output_dir.mkdir(parents=True, exist_ok=True)

        # Asumimos que el lugar se completará correctamente.
        # Si algún bloque falla o se salta, lo cambiaremos a False.
        place_completed = True

        # Procesar cada bloque por separado
        for block_idx, positive_block in enumerate(positive_blocks):

            print(
                f"[BLOCK] {city_name}/place_{place_id}/block_{block_idx}: "
                f"1 referencia + {len(positive_block)} positivos"
            )

            # Referencia fija + positivos de este bloque
            all_image_paths = [ref_path] + positive_block

            # Comprobar que las imágenes existen
            existing_image_paths = []

            for img_path in all_image_paths:
                if img_path.exists():
                    existing_image_paths.append(img_path)
                else:
                    print(f"[WARNING] No existe: {img_path}")

            if len(existing_image_paths) < 2:
                print(
                    f"[SKIP] {city_name}/place_{place_id}/block_{block_idx}: "
                    f"menos de 2 imágenes existentes"
                )
                place_completed = False
                continue

            print(
                f"[OK] {city_name}/place_{place_id}/block_{block_idx}: "
                f"{len(existing_image_paths)} imágenes"
            )

            image_names = [str(p) for p in existing_image_paths]

            # ====================================================
            # Cargar y preprocesar imágenes
            # ====================================================

            images = load_and_preprocess_images(image_names).to(device)
            images = images.unsqueeze(0)

            # ====================================================
            # BLOQUE DE INFERENCIA
            # ====================================================

            with torch.no_grad():

                if device == "cuda":

                    with torch.cuda.amp.autocast(dtype=dtype):
                        tokens, ps_idx = model.aggregator(images)

                        pose_enc = model.camera_head(tokens)[-1]

                        extrinsic, intrinsic = pose_encoding_to_extri_intri(
                            pose_enc,
                            images.shape[-2:]
                        )

                        depth_map, depth_conf = model.depth_head(
                            tokens,
                            images,
                            ps_idx
                        )

                else:

                    tokens, ps_idx = model.aggregator(images)

                    pose_enc = model.camera_head(tokens)[-1]

                    extrinsic, intrinsic = pose_encoding_to_extri_intri(
                        pose_enc,
                        images.shape[-2:]
                    )

                    depth_map, depth_conf = model.depth_head(
                        tokens,
                        images,
                        ps_idx
                    )

            # ====================================================
            # GUARDAR DEPTH E INTRÍNSECO POR IMAGEN
            # ====================================================
            # Recorrer cada imagen procesada y su índice.
            for idx, img_path in enumerate(existing_image_paths):

                # Obtiene el nombre del archivo sin extensión.
                stem = img_path.stem

                # Guarda la profundidad solo si se activa explícitamente
                if SAVE_DEPTH_IMAGES:
                    save_depth_as_image(
                        depth_map[0, idx],
                        place_output_dir / f"d_{stem}.png"
                    )

                # Guarda la matriz intrínseca
                if SAVE_INTRINSICS:
                    np.save(
                        place_output_dir / f"i_{stem}.npy",
                        intrinsic[0, idx].detach().cpu().numpy()
                    )

            # ====================================================
            # REFERENCIA + POSITIVOS
            # ====================================================
            # Indica que la primera imagen de la lista es la referencia
            # y obtiene su nombre y su matriz extrínseca.
            ref_idx = 0
            ref_name = existing_image_paths[ref_idx].name
            E_ref = extrinsic[0, ref_idx].detach()

            # Bucle recorriendo todos los positivos para calcular los overlap con la referencia.
            for pos_idx in range(1, len(existing_image_paths)):

                pos_path = existing_image_paths[pos_idx]
                pos_name = pos_path.name
                pos_stem = pos_path.stem

                # overlap referencia -> positivo
                overlap_ref_to_pos, _ = compute_overlap_and_mask(
                    depth_map,
                    intrinsic,
                    extrinsic,
                    ref_idx,
                    pos_idx
                )

                # overlap positivo -> referencia
                overlap_pos_to_ref, _ = compute_overlap_and_mask(
                    depth_map,
                    intrinsic,
                    extrinsic,
                    pos_idx,
                    ref_idx
                )

                # Nos quedamos con el menor
                overlap_min = min(overlap_ref_to_pos, overlap_pos_to_ref)

                # Pose relativa referencia -> positivo
                E_pos = extrinsic[0, pos_idx].detach()
                R_rel, t_rel = relative_pose(E_ref, E_pos)

                # o_nombredelarchivo

                overlap_info = {
                    "reference_image": ref_name,
                    "reference_path": str(existing_image_paths[ref_idx]),

                    "positive_image": pos_name,
                    "positive_path": str(pos_path),

                    "block_index": block_idx,

                    "relative_translation": t_rel.detach().cpu().numpy(),
                    "relative_rotation": R_rel.detach().cpu().numpy(),

                    "overlap_ref_to_pos": overlap_ref_to_pos,
                    "overlap_pos_to_ref": overlap_pos_to_ref,
                    "overlap": overlap_min
                }

                # Guardar diccionario como o_nombredelarchivo.npy
                np.save(
                    place_output_dir / f"o_{pos_stem}.npy",
                    overlap_info,
                    allow_pickle=True
                )


            # Liberar memoria GPU borrando variables grandes y limpiando la caché
            del images, tokens, ps_idx, pose_enc
            del extrinsic, intrinsic, depth_map, depth_conf

            if device == "cuda":
                torch.cuda.empty_cache()

        # Al final del bucle de bloques marcamos el procesado
        if place_completed:
            mark_place_as_done(place_output_dir)
            print(f"[DONE] {city_name}/place_{place_id}: procesado completo")
        else:
            print(f"[INCOMPLETE] {city_name}/place_{place_id}: no se marca como terminado")
