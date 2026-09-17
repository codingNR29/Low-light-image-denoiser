from pathlib import Path
import sys
import shutil

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim


# ============================================================
# 1. PROJECT PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Restoration -> My implementation -> Mora SP Cup 2026
PROJECT_ROOT = SCRIPT_DIR.parents[1]


COMPETITION_ROOT = (
    PROJECT_ROOT
    / "Competition repo"
    / "mora_sp_cup_2026"
)


NOISY_SOURCE_DIR = (
    COMPETITION_ROOT
    / "competition_data"
    / "public"
    / "noisy"
)


GT_SOURCE_DIR = (
    COMPETITION_ROOT
    / "competition_data"
    / "public"
    / "ground_truth"
)


# ============================================================
# 2. SAMPLE OUTPUT DIRECTORIES
# ============================================================

SAMPLE_DIR = (
    SCRIPT_DIR
    / "baseline_sample_20"
)

SAMPLE_NOISY_DIR = (
    SAMPLE_DIR
    / "noisy"
)

SAMPLE_GT_DIR = (
    SAMPLE_DIR
    / "ground_truth"
)

SAMPLE_DENOISED_DIR = (
    SAMPLE_DIR
    / "denoised"
)


for folder in [
    SAMPLE_NOISY_DIR,
    SAMPLE_GT_DIR,
    SAMPLE_DENOISED_DIR
]:
    folder.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 3. IMPORT ORGANIZER BASELINE
# ============================================================

ORGANIZER_BASELINE_DIR = (
    SCRIPT_DIR
    / "Organizer_baseline"
)

sys.path.insert(
    0,
    str(ORGANIZER_BASELINE_DIR)
)

ORGANIZER_BASELINE_DIR = SCRIPT_DIR / "Organizer_baseline"
sys.path.insert(0, str(ORGANIZER_BASELINE_DIR))

from denoise import correct_defect_pixels, denoise_nlm

from denoise import (
    correct_defect_pixels,
    denoise_nlm
)


# ============================================================
# 4. SETTINGS
# ============================================================

NUM_IMAGES = 20

NLM_H = 10.0


# ============================================================
# 5. PSNR FUNCTION
# ============================================================

def calculate_psnr(
    ground_truth,
    image
):

    gt = ground_truth.astype(
        np.float64
    )

    pred = image.astype(
        np.float64
    )

    mse = np.mean(
        (gt - pred) ** 2
    )

    if mse == 0:
        return float("inf")

    return (
        10
        * np.log10(
            (255.0 ** 2) / mse
        )
    )


# ============================================================
# 6. SSIM FUNCTION
# ============================================================

def calculate_ssim(
    ground_truth,
    image
):

    return ssim(
        ground_truth,
        image,
        channel_axis=2,
        data_range=255
    )


# ============================================================
# 7. ORGANIZER BASELINE PROCESSING
# ============================================================

def organizer_baseline(
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


    # ------------------------------------
    # Stage 1: defect-pixel correction
    # ------------------------------------

    img_float = correct_defect_pixels(
        img_float,
        threshold=0.25
    )


    # ------------------------------------
    # Stage 2: Non-Local Means
    # ------------------------------------

    img_uint8 = (
        img_float
        * 255
    ).astype(
        np.uint8
    )


    denoised = denoise_nlm(
        img_uint8,
        h=NLM_H,
        h_color=NLM_H
    )


    return denoised


# ============================================================
# 8. PROCESS FIRST 20 IMAGES
# ============================================================

results = []


for i in range(
    1,
    NUM_IMAGES + 1
):

    image_id = f"{i:03d}"


    noisy_source = (
        NOISY_SOURCE_DIR
        / f"{image_id}_noise.png"
    )


    gt_source = (
        GT_SOURCE_DIR
        / f"{image_id}.png"
    )


    if not noisy_source.exists():

        print(
            f"Missing noisy image: "
            f"{noisy_source}"
        )

        continue


    if not gt_source.exists():

        print(
            f"Missing GT image: "
            f"{gt_source}"
        )

        continue


    # ========================================================
    # COPY SAMPLE IMAGES
    # ========================================================

    noisy_sample_path = (
        SAMPLE_NOISY_DIR
        / f"{image_id}_noise.png"
    )


    gt_sample_path = (
        SAMPLE_GT_DIR
        / f"{image_id}.png"
    )


    shutil.copy2(
        noisy_source,
        noisy_sample_path
    )


    shutil.copy2(
        gt_source,
        gt_sample_path
    )


    # ========================================================
    # LOAD IMAGES
    # ========================================================

    noisy = np.array(
        Image.open(
            noisy_source
        ).convert("RGB")
    )


    ground_truth = np.array(
        Image.open(
            gt_source
        ).convert("RGB")
    )


    # ========================================================
    # BASELINE DENOISING
    # ========================================================

    denoised = organizer_baseline(
        noisy
    )


    # Save result
    denoised_path = (
        SAMPLE_DENOISED_DIR
        / f"{image_id}.png"
    )


    Image.fromarray(
        denoised
    ).save(
        denoised_path
    )


    # ========================================================
    # METRICS — ORIGINAL NOISY IMAGE
    # ========================================================

    noisy_psnr = calculate_psnr(
        ground_truth,
        noisy
    )


    noisy_ssim = calculate_ssim(
        ground_truth,
        noisy
    )


    # ========================================================
    # METRICS — DENOISED IMAGE
    # ========================================================

    denoised_psnr = calculate_psnr(
        ground_truth,
        denoised
    )


    denoised_ssim = calculate_ssim(
        ground_truth,
        denoised
    )


    # ========================================================
    # IMPROVEMENT
    # ========================================================

    delta_psnr = (
        denoised_psnr
        - noisy_psnr
    )


    delta_ssim = (
        denoised_ssim
        - noisy_ssim
    )


    # ========================================================
    # STORE RESULTS
    # ========================================================

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
            delta_ssim
    })


    print(
        f"{image_id} | "
        f"PSNR: "
        f"{noisy_psnr:.2f} -> "
        f"{denoised_psnr:.2f} dB | "
        f"Δ = {delta_psnr:+.2f} dB | "
        f"SSIM: "
        f"{noisy_ssim:.4f} -> "
        f"{denoised_ssim:.4f}"
    )


# ============================================================
# 9. SAVE RESULTS CSV
# ============================================================

results_df = pd.DataFrame(
    results
)


csv_path = (
    SAMPLE_DIR
    / "baseline_metrics.csv"
)


results_df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# 10. SUMMARY
# ============================================================

print()
print("=" * 65)
print("ORGANIZER BASELINE — FIRST 20 IMAGES")
print("=" * 65)

print(
    f"Images evaluated : "
    f"{len(results_df)}"
)

print()

print(
    "Mean Noisy PSNR     :",
    f"{results_df['noisy_psnr'].mean():.4f} dB"
)

print(
    "Mean Denoised PSNR  :",
    f"{results_df['denoised_psnr'].mean():.4f} dB"
)

print(
    "Mean Delta PSNR     :",
    f"{results_df['delta_psnr'].mean():+.4f} dB"
)

print()

print(
    "Mean Noisy SSIM     :",
    f"{results_df['noisy_ssim'].mean():.6f}"
)

print(
    "Mean Denoised SSIM  :",
    f"{results_df['denoised_ssim'].mean():.6f}"
)

print(
    "Mean Delta SSIM     :",
    f"{results_df['delta_ssim'].mean():+.6f}"
)

print()
print(
    "Results saved to:"
)

print(
    csv_path
)

print("=" * 65)