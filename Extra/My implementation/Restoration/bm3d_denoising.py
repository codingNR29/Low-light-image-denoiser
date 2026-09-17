from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from PIL import Image

#Library for new technique
from bm3d import bm3d_rgb

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

#Declaring file paths
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

#Output folder
OUTPUT_ROOT = (
    SCRIPT_DIR
    / "bm3d_sample_20"
)


DENOISED_DIR = (
    OUTPUT_ROOT
    / "denoised"
)


DENOISED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

#Using the organizer's outlier correction
ORGANIZER_DIR = (
    SCRIPT_DIR
    / "Organizer_baseline"
)


sys.path.insert(
    0,
    str(ORGANIZER_DIR)
)


from denoise import correct_defect_pixels

#Initial setting
NUM_IMAGES = 20


BM3D_SIGMA = (
    25.0 / 255.0
)

#BM3D resotration function
def bm3d_denoise(
    image_rgb
):

    # Convert uint8 [0,255]
    # to floating point [0,1]

    img_float = (
        image_rgb.astype(
            np.float32
        )
        / 255.0
    )


    # ---------------------------------------------
    # Stage 1:
    # Strong local outlier correction
    # ---------------------------------------------

    corrected = correct_defect_pixels(
        img_float,
        threshold=0.25
    )


    # ---------------------------------------------
    # Stage 2:
    # BM3D RGB denoising
    # ---------------------------------------------

    denoised = bm3d_rgb(
        corrected,
        BM3D_SIGMA
    )


    # Keep values valid

    denoised = np.clip(
        denoised,
        0.0,
        1.0
    )


    # Back to uint8

    denoised_uint8 = (
        denoised
        * 255.0
    ).round().astype(
        np.uint8
    )


    return denoised_uint8

#Functions for quality metrics
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

#Processing the 20 images
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


    # ---------------------------------------------
    # BM3D
    # ---------------------------------------------

    start = time.perf_counter()


    denoised = bm3d_denoise(
        noisy
    )


    elapsed = (
        time.perf_counter()
        - start
    )


    total_time += elapsed


    # ---------------------------------------------
    # Save output
    # ---------------------------------------------

    output_path = (
        DENOISED_DIR
        / f"{image_id}.png"
    )


    Image.fromarray(
        denoised
    ).save(
        output_path
    )


    # ---------------------------------------------
    # Original metrics
    # ---------------------------------------------

    noisy_psnr = calculate_psnr(
        ground_truth,
        noisy
    )


    noisy_ssim = calculate_ssim(
        ground_truth,
        noisy
    )


    # ---------------------------------------------
    # BM3D metrics
    # ---------------------------------------------

    denoised_psnr = calculate_psnr(
        ground_truth,
        denoised
    )


    denoised_ssim = calculate_ssim(
        ground_truth,
        denoised
    )


    delta_psnr = (
        denoised_psnr
        - noisy_psnr
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
        f"{elapsed:.2f} s"
    )

#Saving results
results_df = pd.DataFrame(
    results
)


CSV_PATH = (
    OUTPUT_ROOT
    / "bm3d_metrics.csv"
)


results_df.to_csv(
    CSV_PATH,
    index=False
)

#Summarizing
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


score = (
    0.6
    * np.clip(
        mean_delta_psnr / 15.0,
        0.0,
        1.0
    )

    +

    0.4
    * max(
        mean_delta_ssim,
        0.0
    )
)


print()
print("=" * 70)
print("BM3D — FIRST 20 IMAGES")
print("=" * 70)


print(
    f"BM3D sigma            : "
    f"{BM3D_SIGMA:.6f} "
    f"({BM3D_SIGMA * 255:.1f}/255)"
)

print()

print(
    f"Mean Noisy PSNR       : "
    f"{mean_noisy_psnr:.4f} dB"
)

print(
    f"Mean Denoised PSNR    : "
    f"{mean_denoised_psnr:.4f} dB"
)

print(
    f"Mean Delta PSNR       : "
    f"{mean_delta_psnr:+.4f} dB"
)

print()

print(
    f"Mean Noisy SSIM       : "
    f"{mean_noisy_ssim:.6f}"
)

print(
    f"Mean Denoised SSIM    : "
    f"{mean_denoised_ssim:.6f}"
)

print(
    f"Mean Delta SSIM       : "
    f"{mean_delta_ssim:+.6f}"
)

print()

print(
    f"Approx. sample score  : "
    f"{score:.8f}"
)

print(
    f"Average runtime/image : "
    f"{total_time / len(results_df):.2f} s"
)

print()

print(
    "Metrics saved to:"
)

print(
    CSV_PATH
)

print("=" * 70)



