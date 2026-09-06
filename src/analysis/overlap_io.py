from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm


def read_single_overlap_file(npy_path):
    """
    Lee un archivo o_*.npy de overlap y devuelve un diccionario plano.
    """

    npy_path = Path(npy_path)
    data = np.load(npy_path, allow_pickle=True).item()

    row = {
        "npy_path": str(npy_path),
        "reference_image": data.get("reference_image", None),
        "reference_path": data.get("reference_path", None),
        "positive_image": data.get("positive_image", None),
        "positive_path": data.get("positive_path", None),
        "block_index": data.get("block_index", None),
        "overlap_ref_to_pos": data.get("overlap_ref_to_pos", np.nan),
        "overlap_pos_to_ref": data.get("overlap_pos_to_ref", np.nan),
        "overlap": data.get("overlap", np.nan),
    }

    return row


def add_diagnostic_columns(df):
    """
    Añade columnas útiles para analizar casos extremos.
    """

    if len(df) == 0:
        return df

    df = df.copy()

    df["direction_diff"] = (
        df["overlap_ref_to_pos"] - df["overlap_pos_to_ref"]
    ).abs()

    df["is_extreme_1_0"] = (
        (
            (df["overlap_ref_to_pos"] == 1.0) &
            (df["overlap_pos_to_ref"] == 0.0)
        )
        |
        (
            (df["overlap_ref_to_pos"] == 0.0) &
            (df["overlap_pos_to_ref"] == 1.0)
        )
    )

    df["is_zero_overlap"] = df["overlap"] == 0.0

    return df


def get_city_names(overlap_root, cities):
    """
    Devuelve los nombres de ciudades a procesar.
    Si cities = ["all"], usa todas las carpetas dentro de overlap_root.
    """

    overlap_root = Path(overlap_root)

    if not overlap_root.exists():
        raise FileNotFoundError(
            f"No existe la carpeta raíz de overlaps: {overlap_root}"
        )

    if not overlap_root.is_dir():
        raise NotADirectoryError(
            f"La ruta de overlaps no es una carpeta: {overlap_root}"
        )

    if cities == ["all"]:
        city_names = sorted(
            p.name for p in overlap_root.iterdir()
            if p.is_dir()
        )
    else:
        city_names = list(cities)

    if len(city_names) == 0:
        raise RuntimeError(
            f"No se ha encontrado ninguna ciudad en {overlap_root}"
        )

    return city_names


def load_overlaps_for_city(
    overlap_root,
    city,
    max_places_per_city=None,
    max_files_per_place=None,
    require_done=True,
):
    """
    Lee todos los overlaps de una única ciudad.

    Devuelve:
        df: DataFrame con una fila por cada archivo o_*.npy.
    """

    overlap_root = Path(overlap_root)
    city_dir = overlap_root / city

    rows = []

    if not city_dir.exists():
        print(f"[SKIP] No existe ciudad: {city_dir}")
        return pd.DataFrame()

    place_dirs = sorted(
        [p for p in city_dir.iterdir()
        if p.is_dir() and p.name.startswith("place_")],
        key=lambda x: int(x.name.replace("place_", ""))
        if x.name.startswith("place_") else -1
    )

    if max_places_per_city is not None:
        place_dirs = place_dirs[:max_places_per_city]

    print("\n==============================")
    print(f"Ciudad: {city}")
    print(f"Places encontrados: {len(place_dirs)}")
    print("==============================")

    for place_dir in tqdm(place_dirs, desc=f"Leyendo {city}"):
        if require_done and not (place_dir / "DONE.txt").exists():
            print(f"[SKIP INCOMPLETE] {city}/{place_dir.name}: falta DONE.txt")
            continue

        place_name = place_dir.name

        try:
            place_id = int(place_name.replace("place_", ""))
        except Exception:
            place_id = None

        npy_files = sorted(place_dir.glob("o_*.npy"))

        if max_files_per_place is not None:
            npy_files = npy_files[:max_files_per_place]

        for npy_file in npy_files:

            try:
                row = read_single_overlap_file(npy_file)

                row["city"] = city
                row["place"] = place_name
                row["place_id"] = place_id
                row["file_name"] = npy_file.name

                rows.append(row)

            except Exception as e:
                print(f"[ERROR] {npy_file}: {e}")

    df = pd.DataFrame(rows)

    if len(df) == 0:
        print(f"No se ha leído ningún overlap para {city}.")
        return df

    for col in ["overlap", "overlap_ref_to_pos", "overlap_pos_to_ref"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = add_diagnostic_columns(df)

    return df


def summarize_overlaps_by_place(df):
    """
    Resume los overlaps por ciudad y lugar.
    """

    if len(df) == 0:
        return pd.DataFrame()

    summary = (
        df.groupby(["city", "place", "place_id"])
        .agg(
            num_positives=("overlap", "count"),
            overlap_mean=("overlap", "mean"),
            overlap_median=("overlap", "median"),
            overlap_min=("overlap", "min"),
            overlap_max=("overlap", "max"),
            overlap_std=("overlap", "std"),
            overlap_ref_to_pos_mean=("overlap_ref_to_pos", "mean"),
            overlap_pos_to_ref_mean=("overlap_pos_to_ref", "mean"),
            direction_diff_mean=("direction_diff", "mean"),
            extreme_1_0_count=("is_extreme_1_0", "sum"),
            zero_overlap_count=("is_zero_overlap", "sum"),
        )
        .reset_index()
        .sort_values(["city", "place_id"])
    )

    summary["zero_overlap_ratio"] = (
        summary["zero_overlap_count"] / summary["num_positives"]
    )

    summary["extreme_1_0_ratio"] = (
        summary["extreme_1_0_count"] / summary["num_positives"]
    )

    return summary


def save_city_overlap_cache(df, cache_dir, city):
    """
    Guarda los resultados de una ciudad.

    Crea:
        overlaps_<city>.pkl
        overlaps_<city>.csv
        summary_overlaps_<city>.pkl
        summary_overlaps_<city>.csv
    """

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_overlaps_by_place(df)

    overlaps_pkl = cache_dir / f"overlaps_{city}.pkl"
    overlaps_csv = cache_dir / f"overlaps_{city}.csv"

    summary_pkl = cache_dir / f"summary_overlaps_{city}.pkl"
    summary_csv = cache_dir / f"summary_overlaps_{city}.csv"

    df.to_pickle(overlaps_pkl)
    df.to_csv(overlaps_csv, index=False)

    summary.to_pickle(summary_pkl)
    summary.to_csv(summary_csv, index=False)

    print(f"[SAVE] {overlaps_pkl}")
    print(f"[SAVE] {overlaps_csv}")
    print(f"[SAVE] {summary_pkl}")
    print(f"[SAVE] {summary_csv}")

    return df, summary


def load_city_overlap_cache(cache_dir, city):
    """
    Carga overlaps_<city>.pkl.
    """

    cache_file = Path(cache_dir) / f"overlaps_{city}.pkl"

    if not cache_file.exists():
        raise FileNotFoundError(f"No existe la caché: {cache_file}")

    df = pd.read_pickle(cache_file)

    if "direction_diff" not in df.columns:
        df = add_diagnostic_columns(df)

    return df


def load_city_summary_cache(cache_dir, city):
    """
    Carga summary_overlaps_<city>.pkl.
    """

    cache_file = Path(cache_dir) / f"summary_overlaps_{city}.pkl"

    if not cache_file.exists():
        raise FileNotFoundError(f"No existe el resumen: {cache_file}")

    return pd.read_pickle(cache_file)


def load_selected_place_overlaps(
    overlap_root,
    city,
    place_ids,
    max_files_per_place=None
):
    """
    Lee overlaps solo de algunos places concretos.
    """

    overlap_root = Path(overlap_root)
    city_dir = overlap_root / city

    if not city_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de ciudad: {city_dir}")

    rows = []

    for place_id in tqdm(place_ids, desc=f"Leyendo places de {city}"):

        place_dir = city_dir / f"place_{place_id}"

        if not place_dir.exists():
            print(f"[SKIP] No existe: {place_dir}")
            continue

        npy_files = sorted(place_dir.glob("o_*.npy"))

        if max_files_per_place is not None:
            npy_files = npy_files[:max_files_per_place]

        print(f"{city}/place_{place_id}: {len(npy_files)} archivos overlap")

        for npy_file in npy_files:

            try:
                row = read_single_overlap_file(npy_file)

                row["city"] = city
                row["place"] = f"place_{place_id}"
                row["place_id"] = int(place_id)
                row["file_name"] = npy_file.name

                rows.append(row)

            except Exception as e:
                print(f"[ERROR] {npy_file}: {e}")

    df = pd.DataFrame(rows)

    if len(df) == 0:
        print("No se ha leído ningún overlap.")
        return df

    for col in ["overlap", "overlap_ref_to_pos", "overlap_pos_to_ref"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = add_diagnostic_columns(df)

    return df
