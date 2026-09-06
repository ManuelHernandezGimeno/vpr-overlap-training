from pathlib import Path
import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from overlap_io import load_city_overlap_cache

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", DEFAULT_PROJECT_ROOT))


# ============================================================
# CONFIGURACIÓN MANUAL
# ============================================================
CACHE_DIR = Path(os.environ.get("CACHE_DIR", PROJECT_ROOT / "overlap_cache" ))
PLOTS_DIR = Path(os.environ.get("PLOTS_DIR", PROJECT_ROOT / "overlap_cache" / "plots" ))

CITY = "amsterdam"

BINS = 50

# ============================================================
# EJECUCIÓN
# ============================================================

def main():

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_city_overlap_cache(
        cache_dir=CACHE_DIR,
        city=CITY,
    )

    if len(df) == 0:
        print(f"No hay datos para {CITY}")
        return

    print("========================================")
    print(f"Ciudad: {CITY}")
    print(f"Número total de pares query-positive: {len(df)}")
    print(df["overlap"].describe())
    print("========================================")

    output_file = PLOTS_DIR / f"overlap_distribution_{CITY}.png"

    plt.figure(figsize=(8, 4))
    plt.hist(df["overlap"].dropna().values, bins=BINS)
    plt.xlabel("Overlap")
    plt.ylabel("Frecuencia")
    plt.title(f"Distribución de overlaps en {CITY}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()

    print(f"[SAVE] {output_file}")


if __name__ == "__main__":
    main()
