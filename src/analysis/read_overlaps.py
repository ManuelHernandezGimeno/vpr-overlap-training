from pathlib import Path
import os

from overlap_io import (
    get_city_names,
    load_overlaps_for_city,
    save_city_overlap_cache,
)

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/workspace/mhernang/VPR"))
# ============================================================
# CONFIGURACIÓN MANUAL
# ============================================================

OVERLAP_ROOT = Path(os.environ.get("OVERLAP_ROOT", PROJECT_ROOT / "overlaps" / "VGGT_Overlap"))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", PROJECT_ROOT / "Overlap_cache" ))

# Opciones:
#   ["all"] para todas las ciudades encontradas en OVERLAP_ROOT
#   ["amsterdam"] para una ciudad
#   ["amsterdam", "london", "boston"] para varias
CITIES = ["all"]

# Para pruebas puedes limitar.
# Para el procesamiento definitivo deja ambos en None.
MAX_PLACES_PER_CITY = None
MAX_FILES_PER_PLACE = None
REQUIRE_DONE = True

# Si False, se salta una ciudad cuando ya existe overlaps_<city>.pkl.
# Si True, vuelve a crear los archivos aunque ya existan.
FORCE_REBUILD = False

# ============================================================
# EJECUCIÓN
# ============================================================

def main():

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    city_names = get_city_names(
        overlap_root=OVERLAP_ROOT,
        cities=CITIES,
    )

    print("========================================")
    print("Construyendo cachés por ciudad")
    print("OVERLAP_ROOT:", OVERLAP_ROOT)
    print("CACHE_DIR:", CACHE_DIR)
    print("CITIES:", city_names)
    print("FORCE_REBUILD:", FORCE_REBUILD)
    print("========================================")

    for city in city_names:

        print("\n################################################")
        print(f"Procesando ciudad: {city}")
        print("################################################")

        expected_files = [
            CACHE_DIR / f"overlaps_{city}.pkl",
            CACHE_DIR / f"overlaps_{city}.csv",
            CACHE_DIR / f"summary_overlaps_{city}.pkl",
            CACHE_DIR / f"summary_overlaps_{city}.csv",
        ]

        if all(path.exists() for path in expected_files) and not FORCE_REBUILD:
            print(f"[SKIP] Ya existen todos los resultados para {city}")
            for path in expected_files:
                print(path)
            continue

        df_city = load_overlaps_for_city(
            overlap_root=OVERLAP_ROOT,
            city=city,
            max_places_per_city=MAX_PLACES_PER_CITY,
            max_files_per_place=MAX_FILES_PER_PLACE,
            require_done=REQUIRE_DONE,
        )

        print("\nDataFrame ciudad:")
        print(df_city.shape)

        if len(df_city) == 0:
            print(f"[WARNING] No se guarda caché vacía para {city}")
            continue

        print("\nEstadísticas de overlap:")
        print(df_city["overlap"].describe())

        print("Porcentaje overlap = 0:", (df_city["overlap"] == 0).mean() * 100)
        print("Porcentaje casos extremos 1/0 o 0/1:", df_city["is_extreme_1_0"].mean() * 100)

        _, summary_city = save_city_overlap_cache(
            df=df_city,
            cache_dir=CACHE_DIR,
            city=city,
        )

        print("\nResumen primeros places:")
        print(summary_city.head(20).to_string(index=False))

    print("\nProceso terminado.")


if __name__ == "__main__":
    main()
