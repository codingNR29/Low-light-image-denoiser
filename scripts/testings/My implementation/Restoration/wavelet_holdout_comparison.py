from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from PIL import Image

from skimage.restoration import denoise_wavelet
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parents[1]


COMPETITION_ROOT = (
    PROJECT_ROOT
    / "Competition repo"
    / "mora_sp_cup_2026"
)


NOISY_DIR = (
    COMPETITION_ROOT
    / "competition_data"
    / "public"
    / "noisy"
)


GT_DIR = (
    COMPETITION_ROOT
    / "competition_data"
    / "public"
    / "ground_truth"
)


# ============================================================
# 2. VALIDATION SET CREATED IN PARAMETER SWEEP
# ============================================================

VALIDATION_CSV = (
    SCRIPT_DIR
    / "wavelet_parameter_sweep"
    / "validation_60_images.csv"
)


# ============================================================
# 3. OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    SCRIPT_DIR
    / "wavelet_holdout_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 4. IMPORT ORGANIZER DEFECT CORRECTION
# ============================================================

ORGANIZER_DIR = (
    SCRIPT_DIR
    / "Organizer_baseline"
)

sys.path.insert(
    0,
    str(ORGANIZER_DIR)
)


from denoise import correct_defect_pixels


# ============================================================
# 5. CONFIGURATIONS TO TEST
# ============================================================

CONFIGURATIONS = [

    {
        "name":
            "db2_L3_Bayes",

        "wavelet":
            "db2",

        "level":
            3,

        "method":
            "BayesShrink"
    },

    {
        "name":
            "db4_L4_Bayes",

        "wavelet":
            "db4",

        "level":
            4,

        "method":
            "BayesShrink"
    },

    {
        "name":
            "sym4_L3_Bayes",

        "wavelet":
            "sym4",

        "level":
            3,

        "method":
            "BayesShrink"
    },

    {
        "name":
            "sym4_L4_Bayes",

        "wavelet":
            "sym4",

        "level":
            4,

        "method":
            "BayesShrink"
    }
]


# ============================================================
# 6. LOAD THE 60-IMAGE TUNING SET
# ============================================================

validation_df = pd.read_csv(
    VALIDATION_CSV
)


validation_ids = set(
    validation_df[
        "image"
    ].astype(int)
)


print(
    "Images used during parameter tuning:",
    len(validation_ids)
)


# ============================================================
# 7. CREATE THE 400-IMAGE HOLDOUT SET
# ============================================================

all_ids = set(
    range(
        1,
        461
    )
)


holdout_ids = sorted(
    all_ids
    -
    validation_ids
)


print(
    "Holdout images:",
    len(holdout_ids)
)


if len(holdout_ids) != 400:

    print(
        "WARNING: Expected 400 holdout images!"
    )


# Save the actual image list
pd.DataFrame({

    "image":
        holdout_ids

}).to_csv(

    OUTPUT_DIR
    / "holdout_400_images.csv",

    index=False
)


# ============================================================
# 8. METRIC FUNCTIONS
# ============================================================

def calculate_psnr(
    ground_truth,
    image
):

    return peak_signal_noise_ratio(
        ground_truth,
        image,
        data_range=255
    )


def calculate_ssim(
    ground_truth,
    image
):

    return structural_similarity(
        ground_truth,
        image,
        channel_axis=-1,
        data_range=255
    )


# ============================================================
# 9. COMPLETE RESTORATION FUNCTION
# ============================================================

def restore_image(
    image_rgb,
    wavelet_name,
    level,
    method
):

    # --------------------------------------------------------
    # Convert uint8 [0,255]
    # to float [0,1]
    # --------------------------------------------------------

    img_float = (
        image_rgb.astype(
            np.float32
        )
        / 255.0
    )


    # --------------------------------------------------------
    # Stage 1:
    # Organizer defect/outlier correction
    # --------------------------------------------------------

    corrected = correct_defect_pixels(
        img_float,
        threshold=0.25
    )


    # --------------------------------------------------------
    # Stage 2:
    # Wavelet denoising
    # --------------------------------------------------------

    denoised = denoise_wavelet(
        corrected,
        method=method,
        mode="soft",
        wavelet=wavelet_name,
        wavelet_levels=level,
        channel_axis=-1,
        rescale_sigma=True
    )


    # --------------------------------------------------------
    # Clip to valid image range
    # --------------------------------------------------------

    denoised = np.clip(
        denoised,
        0.0,
        1.0
    )


    # --------------------------------------------------------
    # Convert back to uint8
    # --------------------------------------------------------

    denoised_uint8 = (
        denoised
        * 255.0
    ).round().astype(
        np.uint8
    )


    return denoised_uint8


# ============================================================
# 10. STORAGE
# ============================================================

summary_results = []

all_image_results = []


# ============================================================
# 11. TEST EACH CONFIGURATION
# ============================================================

for config_number, config in enumerate(
    CONFIGURATIONS,
    start=1
):

    print()
    print("=" * 75)

    print(
        f"Configuration "
        f"{config_number}/"
        f"{len(CONFIGURATIONS)}"
    )

    print(
        "Name    :",
        config["name"]
    )

    print(
        "Wavelet :",
        config["wavelet"]
    )

    print(
        "Level   :",
        config["level"]
    )

    print(
        "Method  :",
        config["method"]
    )

    print("=" * 75)


    delta_psnr_values = []
    delta_ssim_values = []

    denoised_psnr_values = []
    denoised_ssim_values = []

    noisy_psnr_values = []
    noisy_ssim_values = []

    runtimes = []


    # ========================================================
    # 12. PROCESS THE 400 HOLDOUT IMAGES
    # ========================================================

    for count, image_number in enumerate(
        holdout_ids,
        start=1
    ):

        image_id = f"{image_number:03d}"


        noisy_path = (
            NOISY_DIR
            / f"{image_id}_noise.png"
        )


        gt_path = (
            GT_DIR
            / f"{image_id}.png"
        )


        # ----------------------------------------------------
        # Check files
        # ----------------------------------------------------

        if not noisy_path.exists():

            print(
                f"Missing noisy image: "
                f"{noisy_path}"
            )

            continue


        if not gt_path.exists():

            print(
                f"Missing GT image: "
                f"{gt_path}"
            )

            continue


        # ----------------------------------------------------
        # Load images
        # ----------------------------------------------------

        noisy = np.asarray(
            Image.open(
                noisy_path
            ).convert("RGB")
        )


        ground_truth = np.asarray(
            Image.open(
                gt_path
            ).convert("RGB")
        )


        # ----------------------------------------------------
        # Original noisy metrics
        # ----------------------------------------------------

        noisy_psnr = calculate_psnr(
            ground_truth,
            noisy
        )


        noisy_ssim = calculate_ssim(
            ground_truth,
            noisy
        )


        # ----------------------------------------------------
        # Restoration
        # ----------------------------------------------------

        start = time.perf_counter()


        denoised = restore_image(

            noisy,

            config["wavelet"],

            config["level"],

            config["method"]
        )


        elapsed = (
            time.perf_counter()
            - start
        )


        runtimes.append(
            elapsed
        )


        # ----------------------------------------------------
        # Denoised metrics
        # ----------------------------------------------------

        denoised_psnr = calculate_psnr(
            ground_truth,
            denoised
        )


        denoised_ssim = calculate_ssim(
            ground_truth,
            denoised
        )


        # ----------------------------------------------------
        # Improvements
        # ----------------------------------------------------

        delta_psnr = (
            denoised_psnr
            - noisy_psnr
        )


        delta_ssim = (
            denoised_ssim
            - noisy_ssim
        )


        # ----------------------------------------------------
        # Save values
        # ----------------------------------------------------

        noisy_psnr_values.append(
            noisy_psnr
        )

        noisy_ssim_values.append(
            noisy_ssim
        )

        denoised_psnr_values.append(
            denoised_psnr
        )

        denoised_ssim_values.append(
            denoised_ssim
        )

        delta_psnr_values.append(
            delta_psnr
        )

        delta_ssim_values.append(
            delta_ssim
        )


        all_image_results.append({

            "configuration":
                config["name"],

            "wavelet":
                config["wavelet"],

            "level":
                config["level"],

            "method":
                config["method"],

            "image":
                image_id,

            "noisy_psnr":
                noisy_psnr,

            "denoised_psnr":
                denoised_psnr,

            "delta_psnr":
                delta_psnr,

            "noisy_ssim":
                noisy_ssim,

            "denoised_ssim":
                denoised_ssim,

            "delta_ssim":
                delta_ssim,

            "runtime_seconds":
                elapsed
        })


        # Print progress every 25 images
        if (
            count % 25 == 0
            or
            count == len(
                holdout_ids
            )
        ):

            print(
                f"Processed "
                f"{count}/"
                f"{len(holdout_ids)}"
            )


    # ========================================================
    # 13. CALCULATE CONFIGURATION SUMMARY
    # ========================================================

    mean_noisy_psnr = np.mean(
        noisy_psnr_values
    )


    mean_denoised_psnr = np.mean(
        denoised_psnr_values
    )


    mean_delta_psnr = np.mean(
        delta_psnr_values
    )


    mean_noisy_ssim = np.mean(
        noisy_ssim_values
    )


    mean_denoised_ssim = np.mean(
        denoised_ssim_values
    )


    mean_delta_ssim = np.mean(
        delta_ssim_values
    )


    # ========================================================
    # 14. COMPETITION-STYLE SCORE
    # ========================================================

    normalized_psnr = np.clip(

        mean_delta_psnr
        / 15.0,

        0.0,

        1.0
    )


    positive_ssim = max(
        mean_delta_ssim,
        0.0
    )


    score = (

        0.6
        * normalized_psnr

        +

        0.4
        * positive_ssim
    )


    avg_runtime = np.mean(
        runtimes
    )


    summary_results.append({

        "configuration":
            config["name"],

        "wavelet":
            config["wavelet"],

        "level":
            config["level"],

        "method":
            config["method"],

        "mean_noisy_psnr":
            mean_noisy_psnr,

        "mean_denoised_psnr":
            mean_denoised_psnr,

        "mean_delta_psnr":
            mean_delta_psnr,

        "mean_noisy_ssim":
            mean_noisy_ssim,

        "mean_denoised_ssim":
            mean_denoised_ssim,

        "mean_delta_ssim":
            mean_delta_ssim,

        "score":
            score,

        "avg_runtime":
            avg_runtime
    })


    # ========================================================
    # 15. PRINT CONFIGURATION RESULT
    # ========================================================

    print()
    print(
        "Mean Noisy PSNR    : "
        f"{mean_noisy_psnr:.4f} dB"
    )

    print(
        "Mean Denoised PSNR : "
        f"{mean_denoised_psnr:.4f} dB"
    )

    print(
        "Mean Delta PSNR    : "
        f"{mean_delta_psnr:+.4f} dB"
    )

    print()

    print(
        "Mean Noisy SSIM    : "
        f"{mean_noisy_ssim:.6f}"
    )

    print(
        "Mean Denoised SSIM : "
        f"{mean_denoised_ssim:.6f}"
    )

    print(
        "Mean Delta SSIM    : "
        f"{mean_delta_ssim:+.6f}"
    )

    print()

    print(
        "Holdout Score      : "
        f"{score:.8f}"
    )

    print(
        "Runtime/image      : "
        f"{avg_runtime:.3f} s"
    )


# ============================================================
# 16. CREATE DATAFRAMES
# ============================================================

summary_df = pd.DataFrame(
    summary_results
)


image_results_df = pd.DataFrame(
    all_image_results
)


# ============================================================
# 17. SORT BY HOLDOUT SCORE
# ============================================================

summary_df = summary_df.sort_values(
    "score",
    ascending=False
)


# ============================================================
# 18. SAVE RESULTS
# ============================================================

summary_df.to_csv(

    OUTPUT_DIR
    / "holdout_configuration_summary.csv",

    index=False
)


image_results_df.to_csv(

    OUTPUT_DIR
    / "holdout_per_image_metrics.csv",

    index=False
)


# ============================================================
# 19. PRINT FINAL COMPARISON
# ============================================================

print()
print("=" * 105)
print("400-IMAGE HOLDOUT COMPARISON")
print("=" * 105)


print(

    summary_df[
        [
            "configuration",
            "wavelet",
            "level",
            "mean_delta_psnr",
            "mean_delta_ssim",
            "score",
            "avg_runtime"
        ]
    ]
    .to_string(
        index=False
    )
)


print()
print("=" * 105)

print(
    "BEST HOLDOUT CONFIGURATION:"
)

print(
    summary_df.iloc[0][
        "configuration"
    ]
)

print("=" * 105)


print()
print(
    "Results saved to:"
)

print(
    OUTPUT_DIR
)

print()