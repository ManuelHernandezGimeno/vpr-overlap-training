from pathlib import Path
import os
import time

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURACIÓN MANUAL
# ============================================================

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", DEFAULT_PROJECT_ROOT))


CACHE_DIR = Path(
    os.environ.get("CACHE_DIR", PROJECT_ROOT / "overlap_cache")
)

OUTPUT_DIR = CACHE_DIR / "global_analysis"
OUTPUT_CSV = OUTPUT_DIR / "global_city_summary.csv"

# Opciones:
#   ["all"] para analizar todas las ciudades con overlaps_<ciudad>.pkl
#   ["amsterdam"] para una ciudad
#   ["amsterdam", "london", "boston"] para varias
CITIES = ["all"]

# Número de lugares que se incluyen en las columnas de mayor y menor overlap.
TOP_N = 20

# Se considera cero exacto. Puedes usar, por ejemplo, 1e-12 si quieres
# considerar como cero valores numéricamente muy pequeños.
ZERO_TOLERANCE = 0.0


# ============================================================
# FUNCIONES
# ============================================================

def get_city_names(cache_dir, cities):
    """
    Obtiene las ciudades a analizar a partir de overlaps_<ciudad>.pkl.
    """
    cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de caché: {cache_dir}"
        )

    if cities == ["all"]:
        city_names = sorted(
            path.stem.replace("overlaps_", "", 1)
            for path in cache_dir.glob("overlaps_*.pkl")
        )
    else:
        city_names = list(cities)

    if len(city_names) == 0:
        raise RuntimeError(
            f"No se ha encontrado ninguna ciudad en {cache_dir}"
        )

    return city_names


def zero_mask(values, tolerance=0.0):
    """
    Devuelve una máscara booleana para valores iguales o próximos a cero.
    Los NaN nunca se consideran cero.
    """
    numeric = pd.to_numeric(values, errors="coerce")

    if tolerance == 0.0:
        return numeric.eq(0.0)

    return numeric.abs().le(tolerance)


def load_or_create_city_summary(df_pairs, cache_dir, city):
    """
    Carga summary_overlaps_<ciudad>.pkl si existe.
    Si no existe, reconstruye el resumen desde overlaps_<ciudad>.pkl.
    """
    summary_file = Path(cache_dir) / f"summary_overlaps_{city}.pkl"

    if summary_file.exists():
        summary = pd.read_pickle(summary_file)
    else:
        print(
            f"[WARNING] No existe {summary_file.name}. "
            "Se reconstruirá desde el archivo de pares."
        )

        summary = (
            df_pairs
            .groupby(["city", "place", "place_id"], dropna=False)
            .agg(
                num_positives=("overlap", "count"),
                overlap_mean=("overlap", "mean"),
                overlap_median=("overlap", "median"),
                overlap_min=("overlap", "min"),
                overlap_max=("overlap", "max"),
                overlap_std=("overlap", "std"),
            )
            .reset_index()
        )

    if "overlap_mean" not in summary.columns:
        raise KeyError(
            f"El resumen de {city} no contiene la columna overlap_mean."
        )

    return summary


def format_ranked_places(summary, ascending, top_n):
    """
    Convierte los TOP_N lugares ordenados en una única cadena para el CSV.
    Ejemplo:
        place_194: 0.944025 | place_327: 0.889464
    """
    ordered = (
        summary
        .sort_values(
            ["overlap_mean", "place_id"],
            ascending=[ascending, True],
            na_position="last",
        )
        .head(top_n)
    )

    values = []

    for _, row in ordered.iterrows():
        place = row.get("place", f"place_{row.get('place_id', '')}")
        overlap_mean = row["overlap_mean"]

        if pd.isna(overlap_mean):
            overlap_text = "NaN"
        else:
            overlap_text = f"{float(overlap_mean):.6f}"

        values.append(f"{place}: {overlap_text}")

    return " | ".join(values)


def analyze_city(cache_dir, city, top_n, zero_tolerance):
    """
    Calcula una fila de resumen para una ciudad.
    """
    pairs_file = Path(cache_dir) / f"overlaps_{city}.pkl"

    if not pairs_file.exists():
        raise FileNotFoundError(
            f"No existe el archivo de pares: {pairs_file}"
        )

    df_pairs = pd.read_pickle(pairs_file)

    if "overlap" not in df_pairs.columns:
        raise KeyError(
            f"El archivo {pairs_file.name} no contiene la columna overlap."
        )

    summary = load_or_create_city_summary(
        df_pairs=df_pairs,
        cache_dir=cache_dir,
        city=city,
    )

    overlaps = pd.to_numeric(df_pairs["overlap"], errors="coerce")

    num_pairs = len(df_pairs)
    num_nan = int(overlaps.isna().sum())
    num_zero = int(zero_mask(overlaps, zero_tolerance).sum())

    if num_nan > 0:
        print(
            f"[WARNING] {city}: hay {num_nan} pares con overlap NaN. "
            "Se incluyen en el número total de pares, pero no en la media."
        )

    percentage_zero = (
        100.0 * num_zero / num_pairs
        if num_pairs > 0
        else np.nan
    )

    overlap_weighted_mean = (
        float(overlaps.mean())
        if overlaps.notna().any()
        else np.nan
    )

    number_places = len(summary)

    number_places_mean_zero = int(
        zero_mask(summary["overlap_mean"], zero_tolerance).sum()
    )

    top_places = format_ranked_places(
        summary=summary,
        ascending=False,
        top_n=top_n,
    )

    bottom_places = format_ranked_places(
        summary=summary,
        ascending=True,
        top_n=top_n,
    )

    return {
        "ciudad": city,
        "numero_lugares": number_places,
        "numero_pares_query_positive": num_pairs,
        "numero_pares_overlap_cero": num_zero,
        "porcentaje_global_overlap_cero": percentage_zero,
        "overlap_medio_global_ponderado": overlap_weighted_mean,
        "numero_lugares_overlap_medio_cero": number_places_mean_zero,
        f"{top_n}_lugares_mayor_overlap_medio": top_places,
        f"{top_n}_lugares_menor_overlap_medio": bottom_places,
    }


# ============================================================
# EJECUCIÓN
# ============================================================

def main():
    start_time = time.perf_counter()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    city_names = get_city_names(
        cache_dir=CACHE_DIR,
        cities=CITIES,
    )

    print("========================================")
    print("Resumen global de overlaps por ciudad")
    print("CACHE_DIR:", CACHE_DIR)
    print("OUTPUT_CSV:", OUTPUT_CSV)
    print("CITIES:", city_names)
    print("TOP_N:", TOP_N)
    print("ZERO_TOLERANCE:", ZERO_TOLERANCE)
    print("========================================")

    rows = []

    for city in city_names:
        city_start = time.perf_counter()

        try:
            row = analyze_city(
                cache_dir=CACHE_DIR,
                city=city,
                top_n=TOP_N,
                zero_tolerance=ZERO_TOLERANCE,
            )
            rows.append(row)

            city_time = time.perf_counter() - city_start

            print(
                f"[OK] {city}: "
                f"{row['numero_lugares']} lugares, "
                f"{row['numero_pares_query_positive']} pares, "
                f"{row['porcentaje_global_overlap_cero']:.2f}% ceros, "
                f"{city_time:.2f} s"
            )

        except Exception as error:
            print(f"[ERROR] {city}: {error}")

    if len(rows) == 0:
        raise RuntimeError(
            "No se ha podido generar ningún resumen de ciudad."
        )

    result = (
        pd.DataFrame(rows)
        .sort_values("ciudad")
        .reset_index(drop=True)
    )

    # Formato adecuado para abrir directamente en Excel en configuración española:
    # separador ;, decimal con coma y codificación UTF-8 con BOM.
    result.to_csv(
        OUTPUT_CSV,
        index=False,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )

    total_time = time.perf_counter() - start_time

    print("\n========================================")
    print(f"[SAVE] {OUTPUT_CSV}")
    print("Ciudades procesadas:", len(result))
    print(f"Tiempo total: {total_time:.2f} segundos")
    print("========================================")

    print("\nResumen:")
    print(
        result[
            [
                "ciudad",
                "numero_lugares",
                "numero_pares_query_positive",
                "numero_pares_overlap_cero",
                "porcentaje_global_overlap_cero",
                "overlap_medio_global_ponderado",
                "numero_lugares_overlap_medio_cero",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
