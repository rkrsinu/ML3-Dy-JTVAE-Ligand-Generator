from pathlib import Path
import shutil
import math

# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DIR = Path(r"D:\2026\ML3\JTVAE\pair_oracle_cv")

OUTPUT_DIR = Path(
    r"D:\2026\ML3\JTVAE_final\ML3_JT_VAE_APP_DEVELOPMENT\oracle"
)

# Keep comfortably below 25 MB
PART_SIZE_MB = 20
PART_SIZE = PART_SIZE_MB * 1024 * 1024

MODELS = [
    "final_Ucal_extra_trees.joblib",
    "final_Ueff_extra_trees.joblib",
    "final_tio_extra_trees.joblib",
]

FEATURE_CONFIG = "feature_config.joblib"


# ============================================================
# SPLIT FUNCTION
# ============================================================

def split_file(source_file, output_dir, part_size):
    source_file = Path(source_file)
    output_dir = Path(output_dir)

    if not source_file.exists():
        raise FileNotFoundError(
            f"Source file does not exist:\n{source_file}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    file_size = source_file.stat().st_size
    n_parts = math.ceil(file_size / part_size)

    print()
    print("=" * 70)
    print(f"Splitting: {source_file.name}")
    print(f"Size     : {file_size / 1024 / 1024:.2f} MB")
    print(f"Parts    : {n_parts}")
    print("=" * 70)

    with open(source_file, "rb") as fin:

        for i in range(1, n_parts + 1):

            part_name = (
                f"{source_file.name}.part{i:02d}"
            )

            part_path = output_dir / part_name

            remaining = file_size - (i - 1) * part_size
            current_size = min(part_size, remaining)

            with open(part_path, "wb") as fout:

                remaining_bytes = current_size

                while remaining_bytes > 0:

                    chunk = fin.read(
                        min(1024 * 1024, remaining_bytes)
                    )

                    if not chunk:
                        break

                    fout.write(chunk)
                    remaining_bytes -= len(chunk)

            actual_size = part_path.stat().st_size

            print(
                f"{part_name:<55} "
                f"{actual_size / 1024 / 1024:8.2f} MB"
            )

            if actual_size > 25 * 1024 * 1024:
                raise RuntimeError(
                    f"ERROR: {part_path.name} exceeds 25 MB"
                )

    print("DONE")


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ORACLE MODEL SPLITTER")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Copy feature configuration
    # --------------------------------------------------------

    feature_source = SOURCE_DIR / FEATURE_CONFIG
    feature_destination = OUTPUT_DIR / FEATURE_CONFIG

    if not feature_source.exists():
        raise FileNotFoundError(
            f"Missing feature configuration:\n{feature_source}"
        )

    shutil.copy2(
        feature_source,
        feature_destination
    )

    print()
    print(
        f"Copied feature configuration: "
        f"{FEATURE_CONFIG}"
    )

    # --------------------------------------------------------
    # Split models
    # --------------------------------------------------------

    for model_name in MODELS:

        source = SOURCE_DIR / model_name

        split_file(
            source,
            OUTPUT_DIR,
            PART_SIZE
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL ORACLE DIRECTORY")
    print("=" * 70)

    total_size = 0

    for file in sorted(OUTPUT_DIR.iterdir()):

        if file.is_file():

            size = file.stat().st_size
            total_size += size

            print(
                f"{file.name:<55}"
                f"{size / 1024 / 1024:8.2f} MB"
            )

    print("-" * 70)

    print(
        f"Total: {total_size / 1024 / 1024:.2f} MB"
    )

    print()
    print("All oracle files are below 25 MB.")
    print("The original models have NOT been modified.")
    print()


if __name__ == "__main__":
    main()