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
# 2. OUTPUT FOLDER
# ============================================================

OUTPUT_ROOT = (
    SCRIPT_DIR
    / "optimized_wavelet_full_460"
)


DENOISED_DIR = (
    OUTPUT_ROOT
    / "denoised"
)


DENOISED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. IMPORT ORGANIZER DEFECT CORRECTION
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
# 4. FINAL WAVELET SETTINGS
# ============================================================

NUM_IMAGES = 460

WAVELET = "sym4"

WAVELET_LEVELS = 4

METHOD = "BayesShrink"


# ============================================================
# 5. OPTIMIZED WAVELET DENOISING
# ============================================================

def optimized_wavelet_denoise(
    image_rgb
):

    # Convert uint8 [0,255]
    # to float [0,1]

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
    # sym4 Level 4 BayesShrink
    # --------------------------------------------------------

    denoised = denoise_wavelet(
        corrected,
        method=METHOD,
        mode="soft",
        wavelet=WAVELET,
        wavelet_levels=WAVELET_LEVELS,
        channel_axis=-1,
        rescale_sigma=True
    )


    # Keep image inside valid range

    denoised = np.clip(
        denoised,
        0.0,
        1.0
    )


    # Convert back to uint8

    denoised_uint8 = (
        denoised
        * 255.0
    ).round().astype(
        np.uint8
    )


    return denoised_uint8


# ============================================================
# 6. METRIC FUNCTIONS
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
# 7. PROCESS ALL 460 IMAGES
# ============================================================

results = []

total_time = 0.0


for i in range(
    1,
    NUM_IMAGES + 1
):

    image_id = f"{i:03d}"


    noisy_path = (
        NOISY_DIR
        / f"{image_id}_noise.png"
    )


    gt_path = (
        GT_DIR
        / f"{image_id}.png"
    )


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


    # --------------------------------------------------------
    # Load RGB images
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Restore image
    # --------------------------------------------------------

    start = time.perf_counter()


    denoised = optimized_wavelet_denoise(
        noisy
    )


    elapsed = (
        time.perf_counter()
        - start
    )


    total_time += elapsed


    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    output_path = (
        DENOISED_DIR
        / f"{image_id}.png"
    )


    Image.fromarray(
        denoised
    ).save(
        output_path
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    noisy_psnr = calculate_psnr(
        ground_truth,
        noisy
    )


    denoised_psnr = calculate_psnr(
        ground_truth,
        denoised
    )


    delta_psnr = (
        denoised_psnr
        - noisy_psnr
    )


    noisy_ssim = calculate_ssim(
        ground_truth,
        noisy
    )


    denoised_ssim = calculate_ssim(
        ground_truth,
        denoised
    )


    delta_ssim = (
        denoised_ssim
        - noisy_ssim
    )


    results.append({

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


    print(
        f"{image_id} | "
        f"PSNR "
        f"{noisy_psnr:.2f} -> "
        f"{denoised_psnr:.2f} dB | "
        f"Delta {delta_psnr:+.2f} dB | "
        f"SSIM "
        f"{noisy_ssim:.4f} -> "
        f"{denoised_ssim:.4f} | "
        f"{elapsed:.3f} s"
    )


# ============================================================
# 8. SAVE METRICS CSV
# ============================================================

results_df = pd.DataFrame(
    results
)


CSV_PATH = (
    OUTPUT_ROOT
    / "optimized_wavelet_metrics.csv"
)


results_df.to_csv(
    CSV_PATH,
    index=False
)


# ============================================================
# 9. SUMMARY STATISTICS
# ============================================================

mean_noisy_psnr = (
    results_df[
        "noisy_psnr"
    ].mean()
)


mean_denoised_psnr = (
    results_df[
        "denoised_psnr"
    ].mean()
)


mean_delta_psnr = (
    results_df[
        "delta_psnr"
    ].mean()
)


mean_noisy_ssim = (
    results_df[
        "noisy_ssim"
    ].mean()
)


mean_denoised_ssim = (
    results_df[
        "denoised_ssim"
    ].mean()
)


mean_delta_ssim = (
    results_df[
        "delta_ssim"
    ].mean()
)


avg_runtime = (
    total_time
    / len(results_df)
)


# ============================================================
# 10. COMPETITION-STYLE SCORE
# ============================================================

normalized_psnr = np.clip(
    mean_delta_psnr / 15.0,
    0.0,
    1.0
)


positive_ssim = max(
    mean_delta_ssim,
    0.0
)


estimated_score = (
    0.6
    * normalized_psnr
    +
    0.4
    * positive_ssim
)


# ============================================================
# 11. PRINT FINAL RESULTS
# ============================================================

print()
print("=" * 75)

print(
    "OPTIMIZED WAVELET — sym4 LEVEL 4 BAYESSHRINK"
)

print("=" * 75)


print(
    f"Images evaluated       : "
    f"{len(results_df)}"
)

print()


print(
    "Mean Noisy PSNR        : "
    f"{mean_noisy_psnr:.4f} dB"
)

print(
    "Mean Denoised PSNR     : "
    f"{mean_denoised_psnr:.4f} dB"
)

print(
    "Mean Delta PSNR        : "
    f"{mean_delta_psnr:+.4f} dB"
)

print()


print(
    "Mean Noisy SSIM        : "
    f"{mean_noisy_ssim:.6f}"
)

print(
    "Mean Denoised SSIM     : "
    f"{mean_denoised_ssim:.6f}"
)

print(
    "Mean Delta SSIM        : "
    f"{mean_delta_ssim:+.6f}"
)

print()


print(
    "Estimated score        : "
    f"{estimated_score:.8f}"
)


print(
    "Average runtime/image  : "
    f"{avg_runtime:.3f} s"
)

print()


print(
    "Denoised images:"
)

print(
    DENOISED_DIR
)


print()

print(
    "Metrics:"
)

print(
    CSV_PATH
)

print("=" * 75)