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

# Restoration -> My implementation -> Mora SP Cup 2026
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
# 2. OUTPUT PATHS
# ============================================================

OUTPUT_ROOT = (
    SCRIPT_DIR
    / "wavelet_full_460"
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
# 4. SETTINGS
# ============================================================

NUM_IMAGES = 460

WAVELET = "db2"
WAVELET_LEVELS = 3


# ============================================================
# 5. WAVELET DENOISING FUNCTION
# ============================================================

def wavelet_denoise(image_rgb):

    # Convert uint8 [0,255] -> float [0,1]
    img_float = (
        image_rgb.astype(np.float32)
        / 255.0
    )

    # Step 1: organizer's defect-pixel correction
    corrected = correct_defect_pixels(
        img_float,
        threshold=0.25
    )

    # Step 2: wavelet denoising
    denoised = denoise_wavelet(
        corrected,
        method="BayesShrink",
        mode="soft",
        wavelet=WAVELET,
        wavelet_levels=WAVELET_LEVELS,
        channel_axis=-1,
        rescale_sigma=True
    )

    # Clip to valid range
    denoised = np.clip(
        denoised,
        0.0,
        1.0
    )

    # Convert back to uint8
    denoised_uint8 = (
        denoised * 255.0
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

for i in range(1, NUM_IMAGES + 1):

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
        print(f"Missing noisy image: {noisy_path}")
        continue

    if not gt_path.exists():
        print(f"Missing GT image: {gt_path}")
        continue

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    noisy = np.asarray(
        Image.open(noisy_path).convert("RGB")
    )

    ground_truth = np.asarray(
        Image.open(gt_path).convert("RGB")
    )

    # --------------------------------------------------------
    # Denoise
    # --------------------------------------------------------

    start = time.perf_counter()

    denoised = wavelet_denoise(noisy)

    elapsed = time.perf_counter() - start
    total_time += elapsed

    # --------------------------------------------------------
    # Save denoised image
    # --------------------------------------------------------

    output_path = (
        DENOISED_DIR
        / f"{image_id}.png"
    )

    Image.fromarray(denoised).save(output_path)

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

    delta_psnr = denoised_psnr - noisy_psnr

    noisy_ssim = calculate_ssim(
        ground_truth,
        noisy
    )

    denoised_ssim = calculate_ssim(
        ground_truth,
        denoised
    )

    delta_ssim = denoised_ssim - noisy_ssim

    results.append({
        "image": image_id,
        "noisy_psnr": noisy_psnr,
        "denoised_psnr": denoised_psnr,
        "delta_psnr": delta_psnr,
        "noisy_ssim": noisy_ssim,
        "denoised_ssim": denoised_ssim,
        "delta_ssim": delta_ssim,
        "runtime_seconds": elapsed
    })

    print(
        f"{image_id} | "
        f"PSNR {noisy_psnr:.2f} -> {denoised_psnr:.2f} dB | "
        f"Delta {delta_psnr:+.2f} dB | "
        f"SSIM {noisy_ssim:.4f} -> {denoised_ssim:.4f} | "
        f"{elapsed:.3f} s"
    )


# ============================================================
# 8. SAVE CSV
# ============================================================

results_df = pd.DataFrame(results)

CSV_PATH = (
    OUTPUT_ROOT
    / "wavelet_full_metrics.csv"
)

results_df.to_csv(
    CSV_PATH,
    index=False
)


# ============================================================
# 9. SUMMARY
# ============================================================

mean_noisy_psnr = results_df["noisy_psnr"].mean()
mean_denoised_psnr = results_df["denoised_psnr"].mean()
mean_delta_psnr = results_df["delta_psnr"].mean()

mean_noisy_ssim = results_df["noisy_ssim"].mean()
mean_denoised_ssim = results_df["denoised_ssim"].mean()
mean_delta_ssim = results_df["delta_ssim"].mean()

avg_runtime = total_time / len(results_df)


print()
print("=" * 70)
print("WAVELET BAYESSHRINK — FULL 460 IMAGES")
print("=" * 70)

print(f"Images evaluated      : {len(results_df)}")
print()

print(f"Mean Noisy PSNR       : {mean_noisy_psnr:.4f} dB")
print(f"Mean Denoised PSNR    : {mean_denoised_psnr:.4f} dB")
print(f"Mean Delta PSNR       : {mean_delta_psnr:+.4f} dB")
print()

print(f"Mean Noisy SSIM       : {mean_noisy_ssim:.6f}")
print(f"Mean Denoised SSIM    : {mean_denoised_ssim:.6f}")
print(f"Mean Delta SSIM       : {mean_delta_ssim:+.6f}")
print()

print(f"Average runtime/image : {avg_runtime:.3f} s")
print()

print("Results saved to:")
print(CSV_PATH)
print("=" * 70)