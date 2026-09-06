# view_place.py

from pathlib import Path
import os

from overlap_io import load_city_overlap_cache

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", DEFAULT_PROJECT_ROOT))


# ============================================================
# CONFIGURACIÓN MANUAL
# ============================================================
CACHE_DIR = Path(os.environ.get("CACHE_DIR", PROJECT_ROOT / "overlap_cache" ))

CITY = "amsterdam"
PLACE_ID = 641

SORT_BY = "overlap"
ASCENDING = False

MAX_ROWS = None

SAVE_PLACE_CSV = True


# ============================================================
# EJECUCIÓN
# ============================================================

def main():

    df = load_city_overlap_cache(
        cache_dir=CACHE_DIR,
        city=CITY,
    )

    place_name = f"place_{PLACE_ID}"

    place_df = df[df["place"] == place_name].copy()

    if len(place_df) == 0:
        print(f"No hay datos para {CITY}/{place_name}")
        return

    place_df = place_df.sort_values(SORT_BY, ascending=ASCENDING)

    if MAX_ROWS is not None:
        place_df_show = place_df.head(MAX_ROWS)
    else:
        place_df_show = place_df

    print("========================================")
    print(f"Ciudad: {CITY}")
    print(f"Place: {place_name}")
    print(f"Número de positivos: {len(place_df)}")
    print(f"Overlap medio: {place_df['overlap'].mean():.6f}")
    print(f"Overlap máximo: {place_df['overlap'].max():.6f}")
    print(f"Overlap mínimo: {place_df['overlap'].min():.6f}")
    print(f"Casos extremos 1/0 o 0/1: {place_df['is_extreme_1_0'].sum()}")
    print("========================================")

    cols = [
        "reference_image",
        "positive_image",
        "overlap",
        "overlap_ref_to_pos",
        "overlap_pos_to_ref",
        "direction_diff",
        "is_extreme_1_0",
        "block_index",
        "npy_path",
    ]

    print(place_df_show[cols].to_string(index=False))

    if SAVE_PLACE_CSV:
        out_csv = CACHE_DIR / f"place_{CITY}_{PLACE_ID}.csv"
        place_df[cols].to_csv(out_csv, index=False)
        print(f"\n[SAVE] {out_csv}")


if __name__ == "__main__":
    main()
