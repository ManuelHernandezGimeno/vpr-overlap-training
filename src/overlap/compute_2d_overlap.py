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
from shapely.geometry import Polygon

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/workspace/mhernang/VPR"))
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
MSLS_ROOT = Path(os.environ.get("MSLS_ROOT", PROJECT_ROOT / "data" / "mapillary"))
if not MSLS_ROOT.exists():
    raise FileNotFoundError(f"No se encuentra el dataset MSLS en: {MSLS_ROOT}")

# Carpeta donde están las subcarpetas por ciudad con positives_train.npy
POSITIVES_ROOT = Path(os.environ.get("POSITIVESTRAIN_ROOT", PROJECT_ROOT / "positives" / "train"))

# Carpeta donde se guardarán profundidades, intrínsecos y resultados
OUTPUT_ROOT = Path(os.environ.get("OVERLAP2D_ROOT", PROJECT_ROOT / "overlaps" / "Overlap_2D"))

MAX_IMAGES_PER_BLOCK = 20
MAX_FOV_DISTANCE = 25

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Funciones para el calculo del overlap

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

def camera_center_from_extrinsic(E):
    """
    Calcula el centro de cámara en coordenadas mundo.

    E tiene forma [3, 4] y representa:
        Xc = R Xw + t
    Por tanto:
        C = -R^T t
    """
    R = E[:, :3]
    t = E[:, 3]

    C = -R.T @ t

    return C


def camera_forward_from_extrinsic(E):
    """
    Calcula el vector de avance de la cámara en coordenadas mundo.

    En tu código anterior ya asumías que los puntos delante de cámara
    tienen z > 0, por tanto el eje forward local es [0, 0, 1].
    """
    R = E[:, :3]

    forward_cam = torch.tensor(
        [0.0, 0.0, 1.0],
        dtype=E.dtype,
        device=E.device
    )

    forward_world = R.T @ forward_cam
    forward_world = forward_world / torch.norm(forward_world)

    return forward_world


def horizontal_fov_from_intrinsic(K, image_width):
    """
    Calcula el FOV horizontal usando la matriz intrínseca.

    Fórmula:
        hfov = 2 * atan(W / (2 * fx))
    """
    fx = K[0, 0]

    W = torch.tensor(
        float(image_width),
        dtype=K.dtype,
        device=K.device
    )

    hfov = 2.0 * torch.atan(W / (2.0 * fx))

    return hfov


def rotate_vector_y(v, angle):
    """
    Rota un vector 3D alrededor del eje Y.
    """
    c = torch.cos(angle)
    s = torch.sin(angle)

    zero = torch.zeros((), dtype=v.dtype, device=v.device)
    one = torch.ones((), dtype=v.dtype, device=v.device)

    R_y = torch.stack([
        torch.stack([ c,    zero,  s]),
        torch.stack([zero,  one,   zero]),
        torch.stack([-s,    zero,  c])
    ])

    return R_y @ v


def build_fov_polygon_2d(E, K, image_width, max_distance=25.0, num_arc_points=30):
    """
    Construye el FOV horizontal de una cámara como un sector circular 2D.

    Se trabaja en el plano XZ:
        punto 3D (x, y, z) -> punto 2D (x, z)

    El sector queda definido por:
        centro de cámara,
        arco entre dirección izquierda y derecha del FOV.
    """

    # Pasamos a float32 para evitar baja precisión si VGGT está usando float16/bfloat16
    E = E.float()
    K = K.float()

    # Centro de cámara en mundo
    C = camera_center_from_extrinsic(E)

    # Dirección frontal de la cámara en mundo
    forward = camera_forward_from_extrinsic(E)

    # Proyectar la dirección al plano XZ y anular componente Y
    forward_2d_3d = forward.clone()
    forward_2d_3d[1] = 0.0

    norm_forward = torch.norm(forward_2d_3d)

    if norm_forward < 1e-8:
        # Si la cámara mira totalmente hacia arriba/abajo, no hay FOV horizontal fiable
        C_2d = (
            float(C[0].detach().cpu()),
            float(C[2].detach().cpu())
        )

        polygon = Polygon([C_2d, C_2d, C_2d])

        return polygon, {
            "camera_center_2d": C_2d,
            "hfov_rad": 0.0,
            "hfov_deg": 0.0,
            "valid_fov": False
        }

    forward_2d_3d = forward_2d_3d / norm_forward

    # FOV horizontal
    hfov = horizontal_fov_from_intrinsic(K, image_width)

    # Ángulos desde -hfov/2 hasta +hfov/2
    angles = torch.linspace(
        -hfov / 2.0,
        hfov / 2.0,
        steps=num_arc_points,
        dtype=E.dtype,
        device=E.device
    )

    # Punto central en 2D
    C_2d = (
        float(C[0].detach().cpu()),
        float(C[2].detach().cpu())
    )

    polygon_points = [C_2d]

    # Crear puntos del arco
    for angle in angles:
        dir_i = rotate_vector_y(forward_2d_3d, angle)
        dir_i = dir_i / torch.norm(dir_i)

        P_i = C + max_distance * dir_i

        P_i_2d = (
            float(P_i[0].detach().cpu()),
            float(P_i[2].detach().cpu())
        )

        polygon_points.append(P_i_2d)

    polygon = Polygon(polygon_points)

    return polygon, {
        "camera_center_2d": C_2d,
        "hfov_rad": float(hfov.detach().cpu()),
        "hfov_deg": float(torch.rad2deg(hfov).detach().cpu()),
        "valid_fov": True
    }

# Cálculo del overlap

def compute_fov_overlap_2d(
    intrinsic,
    extrinsic,
    img_idx_a=0,
    img_idx_b=1,
    image_width=None,
    max_distance=25.0
):
    """
    Calcula el overlap 2D entre los FOV horizontales de dos cámaras.

    Devuelve:
        overlap_iou = area_interseccion / area_union
        overlap_min = area_interseccion / max(area_a, area_b)
    """

    if image_width is None:
        raise ValueError("Debes pasar image_width, por ejemplo images.shape[-1]")

    E_a = extrinsic[0, img_idx_a].detach()
    E_b = extrinsic[0, img_idx_b].detach()

    K_a = intrinsic[0, img_idx_a].detach()
    K_b = intrinsic[0, img_idx_b].detach()

    poly_a, info_a = build_fov_polygon_2d(
        E_a,
        K_a,
        image_width=image_width,
        max_distance=max_distance,
        num_arc_points=30
    )

    poly_b, info_b = build_fov_polygon_2d(
        E_b,
        K_b,
        image_width=image_width,
        max_distance=max_distance,
        num_arc_points=30
    )

    if not poly_a.is_valid:
        poly_a = poly_a.buffer(0)

    if not poly_b.is_valid:
        poly_b = poly_b.buffer(0)

    inter = poly_a.intersection(poly_b)
    union = poly_a.union(poly_b)

    area_a = poly_a.area
    area_b = poly_b.area
    area_intersection = inter.area
    area_union = union.area

    if area_union > 1e-8:
        overlap_iou = area_intersection / area_union
    else:
        overlap_iou = 0.0

    max_area = max(area_a, area_b)

    if max_area > 1e-8:
        overlap_min = area_intersection / max_area
    else:
        overlap_min = 0.0

    return {
        "overlap_iou": overlap_iou,
        "overlap_min": overlap_min,

        "area_a": area_a,
        "area_b": area_b,
        "area_intersection": area_intersection,
        "area_union": area_union,

        "camera_a": info_a,
        "camera_b": info_b,

        "polygon_a": poly_a,
        "polygon_b": poly_b,
        "polygon_intersection": inter
    }

# Función para detectar lugares de más de 11 imagenes y dividirlos

def split_positives_into_balanced_blocks(positive_paths, max_total_images_per_block=11):
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

# Parámetros para visualizar el overlap de algunos casos
PRINT_OVERLAP_VALUES = True
MAX_OVERLAP_PRINTS = 20

# Bucle principal

# Busca dentro de POSITIVES_ROOT las carpeta de las ciudades
# y crea una lista ordenada alfabéticamente
TARGET_CITIES = ["trondheim", "london", "boston", "melbourne", "amsterdam","helsinki",
              "tokyo","toronto","saopaulo","moscow","zurich","paris","bangkok",
              "budapest","austin","berlin","ottawa","phoenix","goa","amman","nairobi","manila"]



city_dirs = [POSITIVES_ROOT / city for city in TARGET_CITIES]

#Defino cuantas imagenes veo su overlap
num_overlap_prints = 0

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

        # Si ya está procesado, saltarlo
        if is_place_done(place_output_dir):
            print(f"[DONE] {city_name}/place_{place_id}: ya procesado")
            continue

        if not ref_path.exists():
              print(f"[SKIP] {city_name}/place_{place_id}: no existe la referencia {ref_path}")
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
            missing_image = False

            for img_path in all_image_paths:
                if img_path.exists():
                    existing_image_paths.append(img_path)
                else:
                    print(f"[WARNING] No existe: {img_path}")
                    missing_image = True

            if missing_image:
                place_completed = False

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

                else:

                    tokens, ps_idx = model.aggregator(images)

                    pose_enc = model.camera_head(tokens)[-1]

                    extrinsic, intrinsic = pose_encoding_to_extri_intri(
                        pose_enc,
                        images.shape[-2:]
                    )

            # ====================================================
            # GUARDAR INTRÍNSECO POR IMAGEN
            # ====================================================
            # Recorrer cada imagen procesada y su índice.
            for idx, img_path in enumerate(existing_image_paths):

                # Obtiene el nombre del archivo sin extensión.
                stem = img_path.stem

                # Guarda la matriz intrínseca: i_nombredelarchivo.npy
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

                # Overlap 2D por intersección de FOV
                fov_result = compute_fov_overlap_2d(
                    intrinsic=intrinsic,
                    extrinsic=extrinsic,
                    img_idx_a=ref_idx,
                    img_idx_b=pos_idx,
                    image_width=images.shape[-1],
                    max_distance=MAX_FOV_DISTANCE
                )

                overlap_2d_iou = fov_result["overlap_iou"]
                overlap_2d_min = fov_result["overlap_min"]

                if PRINT_OVERLAP_VALUES and num_overlap_prints < MAX_OVERLAP_PRINTS:
                    print("\n[OVERLAP 2D]")
                    print(f"Ciudad: {city_name}")
                    print(f"Place: place_{place_id}")
                    print(f"Bloque: {block_idx}")
                    print(f"Overlap min: {overlap_2d_min:.4f}")
                    print(f"Overlap IoU: {overlap_2d_iou:.4f}")
                    print(f"Área referencia: {fov_result['area_a']:.4f}")
                    print(f"Área positivo: {fov_result['area_b']:.4f}")
                    print(f"Área intersección: {fov_result['area_intersection']:.4f}")
                    print(f"Área unión: {fov_result['area_union']:.4f}")
                    print(f"FOV ref: {fov_result['camera_a']['hfov_deg']:.2f} grados")
                    print(f"FOV pos: {fov_result['camera_b']['hfov_deg']:.2f} grados")
                    print("-" * 50)

                    num_overlap_prints += 1

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

                    "overlap_method": "fov_2d_iou",

                    # Valor principal usado para entrenamiento:
                    # overlap = área_intersección / área_unión
                    "overlap": overlap_2d_iou,

                    # Valores auxiliares
                    "fov_overlap_iou": overlap_2d_iou,
                    "fov_overlap_min": overlap_2d_min,

                    "fov_area_reference": fov_result["area_a"],
                    "fov_area_positive": fov_result["area_b"],
                    "fov_area_intersection": fov_result["area_intersection"],
                    "fov_area_union": fov_result["area_union"],

                    "reference_camera_2d": fov_result["camera_a"],
                    "positive_camera_2d": fov_result["camera_b"],

                    "max_fov_distance": MAX_FOV_DISTANCE
                }

                # Guardar diccionario como o_nombredelarchivo.npy
                np.save(
                    place_output_dir / f"o_{pos_stem}.npy",
                    overlap_info,
                    allow_pickle=True
                )


            # Liberar memoria GPU borrando variables grandes y limpiando la caché
            del images, tokens, ps_idx, pose_enc
            del extrinsic, intrinsic
            if device == "cuda":
                torch.cuda.empty_cache()

        # Al final del bucle de bloques marcamos el procesado
        if place_completed:
            mark_place_as_done(place_output_dir)
            print(f"[DONE] {city_name}/place_{place_id}: procesado completo")
        else:
            print(f"[INCOMPLETE] {city_name}/place_{place_id}: no se marca como terminado")
