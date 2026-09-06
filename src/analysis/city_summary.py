from pathlib import Path
import os

from overlap_io import (
    load_city_overlap_cache,
    summarize_overlaps_by_place,
)

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", DEFAULT_PROJECT_ROOT))


# ============================================================
# CONFIGURACIÓN MANUAL
# ============================================================
CACHE_DIR = Path(os.environ.get("CACHE_DIR", PROJECT_ROOT / "overlap_cache" ))

CITY = "amsterdam"

TOP_N = 30

SAVE_FILTERED_SUMMARY = True

# ============================================================
# EJECUCIÓN
# ============================================================

def main():

    df = load_city_overlap_cache(
        cache_dir=CACHE_DIR,
        city=CITY,
    )

    if len(df) == 0:
        print(f"No hay datos para la ciudad {CITY}")
        return

    print("========================================")
    print(f"Ciudad: {CITY}")
    print("Pares query-positive:", len(df))
    print("Places:", df["place_id"].nunique())
    print("========================================")

    print("\nEstadísticas overlap:")
    print(df["overlap"].describe())

    print("\nPorcentaje overlap = 0:")
    print((df["overlap"] == 0).mean() * 100)

    print("\nPorcentaje casos extremos 1/0 o 0/1:")
    print(df["is_extreme_1_0"].mean() * 100)

    summary = summarize_overlaps_by_place(df)

    print("\nPlaces con mayor overlap medio:")
    print(
        summary
        .sort_values("overlap_mean", ascending=False)
        .head(TOP_N)
        .to_string(index=False)
    )

    print("\nPlaces con menor overlap medio:")
    print(
        summary
        .sort_values("overlap_mean", ascending=True)
        .head(TOP_N)
        .to_string(index=False)
    )

    print("\nPlaces con más casos extremos 1/0 o 0/1:")
    print(
        summary
        .sort_values("extreme_1_0_ratio", ascending=False)
        .head(TOP_N)
        .to_string(index=False)
    )

    if SAVE_FILTERED_SUMMARY:
        out_csv = CACHE_DIR / f"summary_analysis_{CITY}.csv"
        summary.to_csv(out_csv, index=False)
        print(f"\n[SAVE] {out_csv}")


if __name__ == "__main__":
    main()
