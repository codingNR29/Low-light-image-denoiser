from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from PIL import Image

from bm3d import bm3d_rgb

from sklearn.mixture import GaussianMixture

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)


# ============================================================
# 1. PATHS
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


SEVERITY_CSV = (
    PROJECT_ROOT
    / "Noise_characterization"
    / "noise_severity_analysis"
    / "noise_severity_statistics.csv"
)


OUTPUT_DIR = (
    SCRIPT_DIR
    / "bm3d_sigma_sweep"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. IMPORT ORGANIZER DEFECT CORRECTION
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
# 3. BM3D SIGMAS TO TEST
#
# These are expressed in 8-bit image units first.
# We later divide by 255 because BM3D receives [0,1] images.
# ============================================================

SIGMA_VALUES = [
    15,
    20,
    25,
    30,
    35
]


# ============================================================
# 4. LOAD SEVERITY INFORMATION
# ============================================================

severity_df = pd.read_csv(
    SEVERITY_CSV
)


severity_df["image"] = (
    severity_df["image"]
    .astype(int)
)


severity_values = (
    severity_df["severity"]
    .values
    .reshape(-1, 1)
)


# ============================================================
# 5. RECOVER LOW / MEDIUM / HIGH GROUPS
# ============================================================

gmm = GaussianMixture(
    n_components=3,
    random_state=42
)


raw_labels = gmm.fit_predict(
    severity_values
)


centers = (
    gmm.means_
    .flatten()
)


order = np.argsort(
    centers
)


label_map = {

    order[0]:
        "Low",

    order[1]:
        "Medium",

    order[2]:
        "High"
}


severity_df[
    "severity_group"
] = [

    label_map[label]
    for label in raw_labels
]


print()
print("Severity centers:")


for group_name, center in zip(

    [
        "Low",
        "Medium",
        "High"
    ],

    np.sort(
        centers
    )
):

    print(
        f"{group_name:6s}: "
        f"{center:.4f}"
    )


# ============================================================
# 6. SELECT 15 FIXED VALIDATION IMAGES
#
# 5 Low
# 5 Medium
# 5 High
#
# IMPORTANT:
# The same 15 images are used for every sigma.
# ============================================================

RANDOM_SEED = 123

sample_parts = []


for group in [

    "Low",
    "Medium",
    "High"

]:

    group_df = severity_df[

        severity_df[
            "severity_group"
        ]
        == group

    ]


    sampled = group_df.sample(

        n=5,

        random_state=RANDOM_SEED

    )


    sample_parts.append(
        sampled
    )


sample_df = pd.concat(

    sample_parts,

    ignore_index=True

)


sample_df = sample_df.sort_values(
    "image"
)


# Save selected images
sample_df.to_csv(

    OUTPUT_DIR
    / "bm3d_sigma_validation_15.csv",

    index=False

)


print()
print("=" * 70)

print(
    "BM3D SIGMA VALIDATION SET"
)

print("=" * 70)


print(
    sample_df[
        [
            "image",
            "severity",
            "severity_group"
        ]
    ].to_string(
        index=False
    )
)


print()
print(
    "Group counts:"
)

print(
    sample_df[
        "severity_group"
    ].value_counts()
)


# ============================================================
# 7. METRIC FUNCTIONS
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
# 8. PREPARE INPUT IMAGE
#
# Defect correction does not depend on BM3D sigma,
# so we only need to do it once per image.
# ============================================================

def prepare_image(
    image_rgb
):

    img_float = (

        image_rgb.astype(
            np.float32
        )

        / 255.0

    )


    corrected = correct_defect_pixels(

        img_float,

        threshold=0.25

    )


    return corrected


# ============================================================
# 9. BM3D FUNCTION
# ============================================================

def run_bm3d(
    corrected_image,
    sigma_8bit
):

    # Convert sigma from 0-255 scale
    # to normalized 0-1 scale

    sigma_normalized = (

        sigma_8bit
        / 255.0

    )


    denoised = bm3d_rgb(

        corrected_image,

        sigma_normalized

    )


    denoised = np.clip(

        denoised,

        0.0,

        1.0

    )


    denoised_uint8 = (

        denoised
        * 255.0

    ).round().astype(
        np.uint8
    )


    return denoised_uint8


# ============================================================
# 10. PRELOAD ALL 15 IMAGES
#
# Avoid repeated disk reads.
# ============================================================

validation_images = []


for _, row in sample_df.iterrows():

    image_number = int(
        row["image"]
    )


    image_id = (
        f"{image_number:03d}"
    )


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


    # Defect correction done once here

    corrected = prepare_image(
        noisy
    )


    noisy_psnr = calculate_psnr(

        ground_truth,

        noisy

    )


    noisy_ssim = calculate_ssim(

        ground_truth,

        noisy

    )


    validation_images.append({

        "image":
            image_id,

        "severity":
            row["severity"],

        "severity_group":
            row["severity_group"],

        "noisy":
            noisy,

        "corrected":
            corrected,

        "ground_truth":
            ground_truth,

        "noisy_psnr":
            noisy_psnr,

        "noisy_ssim":
            noisy_ssim

    })


# ============================================================
# 11. RESULT STORAGE
# ============================================================

summary_results = []

per_image_results = []


# ============================================================
# 12. SIGMA SWEEP
# ============================================================

for sigma_index, sigma_8bit in enumerate(

    SIGMA_VALUES,

    start=1

):

    print()
    print("=" * 75)

    print(
        f"Sigma test "
        f"{sigma_index}/"
        f"{len(SIGMA_VALUES)}"
    )

    print(
        f"Sigma = "
        f"{sigma_8bit}/255 "
        f"= "
        f"{sigma_8bit / 255.0:.6f}"
    )

    print("=" * 75)


    delta_psnr_values = []

    delta_ssim_values = []

    denoised_psnr_values = []

    denoised_ssim_values = []

    runtimes = []


    # Store results by severity group

    group_delta_psnr = {

        "Low": [],

        "Medium": [],

        "High": []

    }


    group_delta_ssim = {

        "Low": [],

        "Medium": [],

        "High": []

    }


    # ========================================================
    # 13. PROCESS 15 IMAGES
    # ========================================================

    for count, item in enumerate(

        validation_images,

        start=1

    ):

        print(
            f"Processing "
            f"{item['image']} "
            f"({count}/15) ...",
            end=" ",
            flush=True
        )


        start = time.perf_counter()


        denoised = run_bm3d(

            item[
                "corrected"
            ],

            sigma_8bit

        )


        elapsed = (

            time.perf_counter()

            - start

        )


        runtimes.append(
            elapsed
        )


        # -----------------------------------------------
        # Metrics
        # -----------------------------------------------

        denoised_psnr = calculate_psnr(

            item[
                "ground_truth"
            ],

            denoised

        )


        denoised_ssim = calculate_ssim(

            item[
                "ground_truth"
            ],

            denoised

        )


        delta_psnr = (

            denoised_psnr

            - item[
                "noisy_psnr"
            ]

        )


        delta_ssim = (

            denoised_ssim

            - item[
                "noisy_ssim"
            ]

        )


        delta_psnr_values.append(
            delta_psnr
        )


        delta_ssim_values.append(
            delta_ssim
        )


        denoised_psnr_values.append(
            denoised_psnr
        )


        denoised_ssim_values.append(
            denoised_ssim
        )


        group = item[
            "severity_group"
        ]


        group_delta_psnr[
            group
        ].append(
            delta_psnr
        )


        group_delta_ssim[
            group
        ].append(
            delta_ssim
        )


        per_image_results.append({

            "sigma_8bit":
                sigma_8bit,

            "sigma_normalized":
                sigma_8bit / 255.0,

            "image":
                item["image"],

            "severity":
                item["severity"],

            "severity_group":
                group,

            "noisy_psnr":
                item["noisy_psnr"],

            "denoised_psnr":
                denoised_psnr,

            "delta_psnr":
                delta_psnr,

            "noisy_ssim":
                item["noisy_ssim"],

            "denoised_ssim":
                denoised_ssim,

            "delta_ssim":
                delta_ssim,

            "runtime_seconds":
                elapsed

        })


        print(
            f"ΔPSNR "
            f"{delta_psnr:+.2f} dB | "
            f"ΔSSIM "
            f"{delta_ssim:+.4f} | "
            f"{elapsed:.1f} s"
        )


    # ========================================================
    # 14. SIGMA SUMMARY
    # ========================================================

    mean_delta_psnr = np.mean(
        delta_psnr_values
    )


    mean_delta_ssim = np.mean(
        delta_ssim_values
    )


    mean_denoised_psnr = np.mean(
        denoised_psnr_values
    )


    mean_denoised_ssim = np.mean(
        denoised_ssim_values
    )


    # Competition-style score

    score = (

        0.6

        * np.clip(

            mean_delta_psnr
            / 15.0,

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


    summary_results.append({

        "sigma_8bit":
            sigma_8bit,

        "sigma_normalized":
            sigma_8bit / 255.0,

        "mean_denoised_psnr":
            mean_denoised_psnr,

        "mean_delta_psnr":
            mean_delta_psnr,

        "mean_denoised_ssim":
            mean_denoised_ssim,

        "mean_delta_ssim":
            mean_delta_ssim,

        "score":
            score,

        "avg_runtime":
            np.mean(
                runtimes
            ),

        "low_delta_psnr":
            np.mean(
                group_delta_psnr[
                    "Low"
                ]
            ),

        "medium_delta_psnr":
            np.mean(
                group_delta_psnr[
                    "Medium"
                ]
            ),

        "high_delta_psnr":
            np.mean(
                group_delta_psnr[
                    "High"
                ]
            ),

        "low_delta_ssim":
            np.mean(
                group_delta_ssim[
                    "Low"
                ]
            ),

        "medium_delta_ssim":
            np.mean(
                group_delta_ssim[
                    "Medium"
                ]
            ),

        "high_delta_ssim":
            np.mean(
                group_delta_ssim[
                    "High"
                ]
            )

    })


    print()
    print(
        f"Sigma {sigma_8bit} summary:"
    )


    print(
        f"Mean ΔPSNR : "
        f"{mean_delta_psnr:+.4f} dB"
    )


    print(
        f"Mean ΔSSIM : "
        f"{mean_delta_ssim:+.6f}"
    )


    print(
        f"Score      : "
        f"{score:.8f}"
    )


    print(
        f"Runtime    : "
        f"{np.mean(runtimes):.2f} s/image"
    )


# ============================================================
# 15. CREATE DATAFRAMES
# ============================================================

summary_df = pd.DataFrame(
    summary_results
)


per_image_df = pd.DataFrame(
    per_image_results
)


# ============================================================
# 16. SORT BEST SIGMA FIRST
# ============================================================

summary_df = summary_df.sort_values(

    "score",

    ascending=False

)


# ============================================================
# 17. SAVE RESULTS
# ============================================================

summary_df.to_csv(

    OUTPUT_DIR
    / "bm3d_sigma_summary.csv",

    index=False

)


per_image_df.to_csv(

    OUTPUT_DIR
    / "bm3d_sigma_per_image.csv",

    index=False

)


# ============================================================
# 18. FINAL TABLE
# ============================================================

print()
print("=" * 105)

print(
    "BM3D SIGMA SWEEP RESULTS"
)

print("=" * 105)


print(

    summary_df[
        [
            "sigma_8bit",
            "mean_denoised_psnr",
            "mean_delta_psnr",
            "mean_denoised_ssim",
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
    "BEST SIGMA:"
)

print(

    summary_df.iloc[0][
        "sigma_8bit"
    ]

)


print()
print(
    "Results saved to:"
)

print(
    OUTPUT_DIR
)

print("=" * 105)