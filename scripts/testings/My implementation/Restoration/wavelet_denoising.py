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

#Declating file paths
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

#Creating a folder to store the wavelet results
OUTPUT_ROOT = (
    SCRIPT_DIR
    / "wavelet_sample_20"
)

DENOISED_DIR = (
    OUTPUT_ROOT
    / "denoised"
)

DENOISED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

#Importing the given defect correction
ORGANIZER_DIR = (
    SCRIPT_DIR
    / "Organizer_baseline"
)

sys.path.insert(
    0,
    str(ORGANIZER_DIR)
)

from denoise import correct_defect_pixels

#Setting th experiment parameters
NUM_IMAGES = 20
WAVELET = "db2"
WAVELET_LEVELS = 3

#Creating the wavelet denoising function
def wavelet_denoise(image_rgb):

    img_float = (
        image_rgb.astype(np.float32)
        / 255.0
    )

    #Outlier correction
    corrected = correct_defect_pixels(
        img_float,
        threshold=0.25
    )

    #Wavelet denoising
    denoised = denoise_wavelet(
        corrected,
        method="BayesShrink",
        mode="soft",
        wavelet=WAVELET,
        wavelet_levels=WAVELET_LEVELS,
        channel_axis=-1,
        rescale_sigma=True
    )

    #Making the output valid, it should be between 0 and 1
    denoised = np.clip(
        denoised,
        0.0,
        1.0
    )

    #Converting back to 0 to 255
    denoised_uint8 = (
        denoised * 255.0
    ).round().astype(
        np.uint8
    )

    return denoised_uint8

#PSNR and SSIM functions
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

#Storing the results
results = []

#And time
total_time = 0.0

#Looping through the small 20 image batch
for i in range(
    1,
    NUM_IMAGES + 1
):

    image_id = f"{i:03d}"

    #Constructing paths
    noisy_path = (
        NOISY_DIR
        / f"{image_id}_noise.png"
    )

    gt_path = (
        GT_DIR
        / f"{image_id}.png"
    )

    #Avoiding errors
    if not noisy_path.exists():

        print(
            f"Missing noisy image: {noisy_path}"
        )

        continue

    if not gt_path.exists():

        print(
            f"Missing GT image: {gt_path}"
        )

        continue

    #Laoding images
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

    #Time measuring
    start = time.perf_counter()

    #Denoising
    denoised = wavelet_denoise(
        noisy
    )

    #Stopping the timer
    elapsed = (
        time.perf_counter()
        - start
    )

    total_time += elapsed

    #Saving denoised images
    output_path = (
        DENOISED_DIR
        / f"{image_id}.png"
    )

    Image.fromarray(
        denoised
    ).save(
        output_path
    )

    #Calculating the metrics
    noisy_psnr = calculate_psnr(
        ground_truth,
        noisy
    )

    noisy_ssim = calculate_ssim(
        ground_truth,
        noisy
    )

    denoised_psnr = calculate_psnr(
        ground_truth,
        denoised
    )

    denoised_ssim = calculate_ssim(
        ground_truth,
        denoised
    )

    #Checkning the improvemnents
    delta_psnr = (
        denoised_psnr
        - noisy_psnr
    )

    delta_ssim = (
        denoised_ssim
        - noisy_ssim
    )

    #Saving image results to the memory
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

    #Printing image results
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

#Converting all values
results_df = pd.DataFrame(
    results
)

CSV_PATH = (
    OUTPUT_ROOT
    / "wavelet_metrics.csv"
)

results_df.to_csv(
    CSV_PATH,
    index=False
)

#Averaging
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

#Final summary
print()
print("=" * 70)
print(
    "WAVELET BAYESSHRINK — FIRST 20 IMAGES"
)
print("=" * 70)

print(
    f"Images evaluated : "
    f"{len(results_df)}"
)

print()

print(
    "Mean Noisy PSNR      : "
    f"{mean_noisy_psnr:.4f} dB"
)

print(
    "Mean Denoised PSNR   : "
    f"{mean_denoised_psnr:.4f} dB"
)

print(
    "Mean Delta PSNR      : "
    f"{mean_delta_psnr:+.4f} dB"
)

print()

print(
    "Mean Noisy SSIM      : "
    f"{mean_noisy_ssim:.6f}"
)

print(
    "Mean Denoised SSIM   : "
    f"{mean_denoised_ssim:.6f}"
)

print(
    "Mean Delta SSIM      : "
    f"{mean_delta_ssim:+.6f}"
)

print()

print(
    "Average runtime/image: "
    f"{total_time / len(results_df):.3f} s"
)

print()

print(
    "Results saved to:"
)

print(
    CSV_PATH
)

print("=" * 70)

#Comparing with baseeline
BASELINE_CSV = (
    SCRIPT_DIR
    / "baseline_sample_20"
    / "baseline_metrics.csv"
) 

if BASELINE_CSV.exists():

    baseline_df = pd.read_csv(
        BASELINE_CSV
    )

    baseline_psnr = baseline_df["denoised_psnr"].mean()
    baseline_ssim = baseline_df["denoised_ssim"].mean()

    print()
    print("=" * 70)
    print(
        "COMPARISON WITH ORGANIZER BASELINE"
    )
    print("=" * 70)

    print(
        "Baseline PSNR : "
        f"{baseline_psnr:.4f} dB"
    )

    print(
        "Wavelet PSNR  : "
        f"{mean_denoised_psnr:.4f} dB"
    )

    print(
        "Difference    : "
        f"{mean_denoised_psnr - baseline_psnr:+.4f} dB"
    )

    print()

    print(
        "Baseline SSIM : "
        f"{baseline_ssim:.6f}"
    )

    print(
        "Wavelet SSIM  : "
        f"{mean_denoised_ssim:.6f}"
    )

    print(
        "Difference    : "
        f"{mean_denoised_ssim - baseline_ssim:+.6f}"
    )

    print("=" * 70)