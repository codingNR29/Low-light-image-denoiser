from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from PIL import Image

from sklearn.mixture import GaussianMixture

from skimage.restoration import denoise_wavelet
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

#Setting paths
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
    / "wavelet_parameter_sweep"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

#Importing organizers outlier correction
ORGANIZER_DIR = (
    SCRIPT_DIR
    / "Organizer_baseline"
)

sys.path.insert(
    0,
    str(ORGANIZER_DIR)
)


from denoise import correct_defect_pixels

#Parameter grid
WAVELETS = [
    "haar",
    "db2",
    "db4",
    "sym4"
]


LEVELS = [
    2,
    3,
    4
]


METHODS = [
    "BayesShrink",
    "VisuShrink"
]

#Loading severity informationn
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

#Getting severity groups
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
    order[0]: "Low",
    order[1]: "Medium",
    order[2]: "High"
}


severity_df["severity_group"] = [
    label_map[label]
    for label in raw_labels
]


print("Severity centers:")

for name, center in zip(
    [
        "Low",
        "Medium",
        "High"
    ],
    np.sort(centers)
):

    print(
        f"{name:6s}: {center:.4f}"
    )

#Creating 60 images for three severity levels
RANDOM_SEED = 42

validation_parts = []


for group in [
    "Low",
    "Medium",
    "High"
]:

    group_df = severity_df[
        severity_df["severity_group"]
        == group
    ]


    sampled = group_df.sample(
        n=20,
        random_state=RANDOM_SEED
    )


    validation_parts.append(
        sampled
    )


validation_df = pd.concat(
    validation_parts,
    ignore_index=True
)


validation_df = validation_df.sort_values(
    "image"
)


validation_df.to_csv(
    OUTPUT_DIR
    / "validation_60_images.csv",
    index=False
)


print()
print(
    "Validation images:",
    len(validation_df)
)

print(
    validation_df[
        "severity_group"
    ].value_counts()
)

#Image metrics
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

#Wavelet denoising functions
def run_wavelet(
    image_rgb,
    wavelet_name,
    level,
    method
):

    # Convert image to float [0,1]
    img_float = (
        image_rgb.astype(
            np.float32
        )
        / 255.0
    )


    # ------------------------------------
    # Organizer's strong-outlier repair
    # ------------------------------------

    corrected = correct_defect_pixels(
        img_float,
        threshold=0.25
    )


    # ------------------------------------
    # Wavelet denoising
    # ------------------------------------

    denoised = denoise_wavelet(
        corrected,
        method=method,
        mode="soft",
        wavelet=wavelet_name,
        wavelet_levels=level,
        channel_axis=-1,
        rescale_sigma=True
    )


    denoised = np.clip(
        denoised,
        0.0,
        1.0
    )


    return (
        denoised
        * 255.0
    ).round().astype(
        np.uint8
    )

#Preloading the 60 images
validation_images = []


for _, row in validation_df.iterrows():

    image_number = int(
        row["image"]
    )

    image_id = f"{image_number:03d}"


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

        "severity_group":
            row["severity_group"],

        "severity":
            row["severity"],

        "noisy":
            noisy,

        "ground_truth":
            ground_truth,

        "noisy_psnr":
            noisy_psnr,

        "noisy_ssim":
            noisy_ssim
    })

#Running the parameter sweep
sweep_results = []

total_configs = (
    len(WAVELETS)
    *
    len(LEVELS)
    *
    len(METHODS)
)


config_number = 0


for wavelet_name in WAVELETS:

    for level in LEVELS:

        for method in METHODS:

            config_number += 1


            print()
            print("=" * 70)

            print(
                f"Configuration "
                f"{config_number}/{total_configs}"
            )

            print(
                f"Wavelet : {wavelet_name}"
            )

            print(
                f"Level   : {level}"
            )

            print(
                f"Method  : {method}"
            )

            print("=" * 70)


            delta_psnr_values = []
            delta_ssim_values = []

            runtimes = []


            # Severity-group storage
            group_psnr = {
                "Low": [],
                "Medium": [],
                "High": []
            }

            group_ssim = {
                "Low": [],
                "Medium": [],
                "High": []
            }


            for item in validation_images:

                start = time.perf_counter()


                denoised = run_wavelet(
                    item["noisy"],
                    wavelet_name,
                    level,
                    method
                )


                elapsed = (
                    time.perf_counter()
                    - start
                )


                runtimes.append(
                    elapsed
                )


                denoised_psnr = calculate_psnr(
                    item["ground_truth"],
                    denoised
                )


                denoised_ssim = calculate_ssim(
                    item["ground_truth"],
                    denoised
                )


                delta_psnr = (
                    denoised_psnr
                    - item["noisy_psnr"]
                )


                delta_ssim = (
                    denoised_ssim
                    - item["noisy_ssim"]
                )


                delta_psnr_values.append(
                    delta_psnr
                )

                delta_ssim_values.append(
                    delta_ssim
                )


                group = item[
                    "severity_group"
                ]


                group_psnr[
                    group
                ].append(
                    delta_psnr
                )


                group_ssim[
                    group
                ].append(
                    delta_ssim
                )


            # =================================================
            # 11. CONFIGURATION SUMMARY
            # =================================================

            mean_delta_psnr = np.mean(
                delta_psnr_values
            )


            mean_delta_ssim = np.mean(
                delta_ssim_values
            )


            # Competition-style score
            normalized_psnr = np.clip(
                mean_delta_psnr / 15.0,
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


            result = {

                "wavelet":
                    wavelet_name,

                "level":
                    level,

                "method":
                    method,

                "mean_delta_psnr":
                    mean_delta_psnr,

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
                        group_psnr["Low"]
                    ),

                "medium_delta_psnr":
                    np.mean(
                        group_psnr["Medium"]
                    ),

                "high_delta_psnr":
                    np.mean(
                        group_psnr["High"]
                    ),

                "low_delta_ssim":
                    np.mean(
                        group_ssim["Low"]
                    ),

                "medium_delta_ssim":
                    np.mean(
                        group_ssim["Medium"]
                    ),

                "high_delta_ssim":
                    np.mean(
                        group_ssim["High"]
                    )
            }


            sweep_results.append(
                result
            )


            print(
                "Mean Delta PSNR : "
                f"{mean_delta_psnr:+.4f} dB"
            )

            print(
                "Mean Delta SSIM : "
                f"{mean_delta_ssim:+.6f}"
            )

            print(
                "Validation Score: "
                f"{score:.8f}"
            )

            print(
                "Runtime/image   : "
                f"{np.mean(runtimes):.3f} s"
            )

#Saving the results
results_df = pd.DataFrame(
    sweep_results
)


results_df = results_df.sort_values(
    "score",
    ascending=False
)


results_df.to_csv(
    OUTPUT_DIR
    / "wavelet_parameter_sweep_results.csv",
    index=False
)

#Printing top configurations
print()
print("=" * 90)
print("TOP WAVELET CONFIGURATIONS")
print("=" * 90)

print(
    results_df[
        [
            "wavelet",
            "level",
            "method",
            "mean_delta_psnr",
            "mean_delta_ssim",
            "score",
            "avg_runtime"
        ]
    ]
    .head(10)
    .to_string(
        index=False
    )
)

#Printing current db-2 level 3 results
current_config = results_df[

    (results_df["wavelet"] == "db2")

    &

    (results_df["level"] == 3)

    &

    (results_df["method"] == "BayesShrink")
]


print()
print("=" * 90)
print("CURRENT METHOD: db2 + Level 3 + BayesShrink")
print("=" * 90)

print(
    current_config[
        [
            "mean_delta_psnr",
            "mean_delta_ssim",
            "score"
        ]
    ]
    .to_string(
        index=False
    )
)


print()
print("=" * 90)
print("PARAMETER SWEEP COMPLETE")
print("=" * 90)

print(
    "Results saved to:"
)

print(
    OUTPUT_DIR
    / "wavelet_parameter_sweep_results.csv"
)

print("=" * 90)


