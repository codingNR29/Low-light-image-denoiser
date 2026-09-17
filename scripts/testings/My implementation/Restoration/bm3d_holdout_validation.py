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


#Getting current script directory
SCRIPT_DIR = Path(
    __file__
).resolve().parent


#Getting main project directory
PROJECT_ROOT = (
    SCRIPT_DIR.parents[1]
)


#Competition repository location
COMPETITION_ROOT = (

    PROJECT_ROOT

    / "Competition repo"

    / "mora_sp_cup_2026"

)


#Noisy image directory
NOISY_DIR = (

    COMPETITION_ROOT

    / "competition_data"

    / "public"

    / "noisy"

)


#Ground truth image directory
GT_DIR = (

    COMPETITION_ROOT

    / "competition_data"

    / "public"

    / "ground_truth"

)


#Noise severity CSV
SEVERITY_CSV = (

    PROJECT_ROOT

    / "Noise_characterization"

    / "noise_severity_analysis"

    / "noise_severity_statistics.csv"

)


#Previous 15 images used for sigma tuning
TUNING_CSV = (

    SCRIPT_DIR

    / "bm3d_sigma_sweep"

    / "bm3d_sigma_validation_15.csv"

)


#Creating output directory
OUTPUT_DIR = (

    SCRIPT_DIR

    / "bm3d_holdout_validation"

)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


#Importing organizer defect pixel correction
ORGANIZER_DIR = (

    SCRIPT_DIR

    / "Organizer_baseline"

)


sys.path.insert(
    0,
    str(ORGANIZER_DIR)
)


from denoise import correct_defect_pixels


#Sigma values selected for holdout validation
SIGMA_VALUES = [

    25,

    30,

    35

]


#Loading severity information
severity_df = pd.read_csv(
    SEVERITY_CSV
)


#Making sure image numbers are integers
severity_df["image"] = (

    severity_df["image"]
    .astype(int)

)


#Preparing severity values for GMM
severity_values = (

    severity_df[
        "severity"
    ]
    .values
    .reshape(-1, 1)

)


#Creating 3 severity groups
gmm = GaussianMixture(

    n_components=3,

    random_state=42

)


#Getting raw severity labels
raw_labels = gmm.fit_predict(
    severity_values
)


#Getting severity group centers
centers = (

    gmm.means_
    .flatten()

)


#Sorting centers from low to high
order = np.argsort(
    centers
)


#Mapping GMM labels to readable severity names
label_map = {

    order[0]:
        "Low",

    order[1]:
        "Medium",

    order[2]:
        "High"

}


#Adding severity group to dataframe
severity_df[
    "severity_group"
] = [

    label_map[label]

    for label in raw_labels

]


#Printing severity centers
print()

print(
    "Severity centers:"
)


for name, center in zip(

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

        f"{name:6s}: "
        f"{center:.4f}"

    )


#Loading previous tuning image list
tuning_df = pd.read_csv(
    TUNING_CSV
)


#Getting IDs already used for sigma tuning
tuning_ids = set(

    tuning_df[
        "image"
    ].astype(int)

)


print()

print(
    "Previous tuning images:",
    len(tuning_ids)
)


#Removing previous tuning images
available_df = severity_df[

    ~severity_df[
        "image"
    ].isin(
        tuning_ids
    )

].copy()


#Selecting a new independent holdout set
#5 Low + 5 Medium + 5 High
RANDOM_SEED = 456

holdout_parts = []


for group in [

    "Low",

    "Medium",

    "High"

]:

    #Getting images belonging to this severity group
    group_df = available_df[

        available_df[
            "severity_group"
        ]
        == group

    ]


    #Randomly selecting 5 images
    sampled = group_df.sample(

        n=5,

        random_state=RANDOM_SEED

    )


    #Adding selected images
    holdout_parts.append(
        sampled
    )


#Combining all severity groups
holdout_df = pd.concat(

    holdout_parts,

    ignore_index=True

)


#Sorting by image number
holdout_df = holdout_df.sort_values(
    "image"
)


#Saving holdout image list
holdout_df.to_csv(

    OUTPUT_DIR
    / "bm3d_holdout_15_images.csv",

    index=False

)


#Checking that tuning and holdout images do not overlap
holdout_ids = set(

    holdout_df[
        "image"
    ].astype(int)

)


overlap = (

    tuning_ids

    &

    holdout_ids

)


print()

print(
    "=" * 70
)

print(
    "BM3D HOLDOUT SET"
)

print(
    "=" * 70
)


print(

    holdout_df[
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
    "Tuning / holdout overlap:",
    len(overlap)
)


if len(overlap) == 0:

    print(
        "No leakage detected."
    )

else:

    print(
        "WARNING: Holdout contains tuning images."
    )


#PSNR calculation
def calculate_psnr(
    ground_truth,
    image
):

    return peak_signal_noise_ratio(

        ground_truth,

        image,

        data_range=255

    )


#SSIM calculation
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


#Preparing image before BM3D
def prepare_image(
    image_rgb
):

    #Converting image from [0,255] to [0,1]
    image_float = (

        image_rgb.astype(
            np.float32
        )

        / 255.0

    )


    #Correcting strong defect pixels
    corrected = correct_defect_pixels(

        image_float,

        threshold=0.25

    )


    return corrected


#Running BM3D
def run_bm3d(
    corrected_image,
    sigma_8bit
):

    #Converting sigma from 8 bit scale to normalized scale
    sigma_normalized = (

        sigma_8bit

        / 255.0

    )


    #Applying RGB BM3D
    denoised = bm3d_rgb(

        corrected_image,

        sigma_normalized

    )


    #Clipping result to valid image range
    denoised = np.clip(

        denoised,

        0.0,

        1.0

    )


    #Converting result back to uint8
    denoised_uint8 = (

        denoised

        * 255.0

    ).round().astype(
        np.uint8
    )


    return denoised_uint8


#Storage for prepared holdout images
holdout_images = []


#Loading all 15 holdout images
for _, row in holdout_df.iterrows():

    #Getting image number
    image_number = int(
        row["image"]
    )


    #Creating three digit image ID
    image_id = (
        f"{image_number:03d}"
    )


    #Creating noisy image path
    noisy_path = (

        NOISY_DIR

        / f"{image_id}_noise.png"

    )


    #Creating ground truth path
    gt_path = (

        GT_DIR

        / f"{image_id}.png"

    )


    #Loading noisy RGB image
    noisy = np.asarray(

        Image.open(
            noisy_path
        ).convert("RGB")

    )


    #Loading ground truth RGB image
    ground_truth = np.asarray(

        Image.open(
            gt_path
        ).convert("RGB")

    )


    #Running defect correction only once
    corrected = prepare_image(
        noisy
    )


    #Calculating original noisy PSNR
    noisy_psnr = calculate_psnr(

        ground_truth,

        noisy

    )


    #Calculating original noisy SSIM
    noisy_ssim = calculate_ssim(

        ground_truth,

        noisy

    )


    #Saving prepared image information
    holdout_images.append({

        "image":
            image_id,

        "severity":
            row["severity"],

        "severity_group":
            row["severity_group"],

        "corrected":
            corrected,

        "ground_truth":
            ground_truth,

        "noisy_psnr":
            noisy_psnr,

        "noisy_ssim":
            noisy_ssim

    })


#Storage for summary results
summary_results = []


#Storage for individual image results
per_image_results = []


#Testing each sigma
for sigma_number, sigma_8bit in enumerate(

    SIGMA_VALUES,

    start=1

):

    print()

    print(
        "=" * 75
    )


    print(

        f"Sigma test "
        f"{sigma_number}/"
        f"{len(SIGMA_VALUES)}"

    )


    print(

        f"Sigma = "
        f"{sigma_8bit}/255"

    )


    print(
        "=" * 75
    )


    #Storage for sigma results
    delta_psnr_values = []

    delta_ssim_values = []

    denoised_psnr_values = []

    denoised_ssim_values = []

    runtimes = []


    #Storage for results from each severity group
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


    #Processing all 15 holdout images
    for count, item in enumerate(

        holdout_images,

        start=1

    ):

        print(

            f"Processing "
            f"{item['image']} "
            f"({count}/15) ...",

            end=" ",

            flush=True

        )


        #Starting runtime measurement
        start = time.perf_counter()


        #Running BM3D
        denoised = run_bm3d(

            item[
                "corrected"
            ],

            sigma_8bit

        )


        #Calculating runtime
        elapsed = (

            time.perf_counter()

            - start

        )


        #Saving runtime
        runtimes.append(
            elapsed
        )


        #Calculating denoised PSNR
        denoised_psnr = calculate_psnr(

            item[
                "ground_truth"
            ],

            denoised

        )


        #Calculating denoised SSIM
        denoised_ssim = calculate_ssim(

            item[
                "ground_truth"
            ],

            denoised

        )


        #Calculating PSNR improvement
        delta_psnr = (

            denoised_psnr

            - item[
                "noisy_psnr"
            ]

        )


        #Calculating SSIM improvement
        delta_ssim = (

            denoised_ssim

            - item[
                "noisy_ssim"
            ]

        )


        #Saving metrics
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


        #Getting severity group
        group = item[
            "severity_group"
        ]


        #Saving group PSNR result
        group_psnr[
            group
        ].append(
            delta_psnr
        )


        #Saving group SSIM result
        group_ssim[
            group
        ].append(
            delta_ssim
        )


        #Saving individual image result
        per_image_results.append({

            "sigma_8bit":
                sigma_8bit,

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

            f"Delta PSNR "
            f"{delta_psnr:+.2f} dB | "

            f"Delta SSIM "
            f"{delta_ssim:+.4f} | "

            f"{elapsed:.1f} s"

        )


    #Calculating mean PSNR improvement
    mean_delta_psnr = np.mean(
        delta_psnr_values
    )


    #Calculating mean SSIM improvement
    mean_delta_ssim = np.mean(
        delta_ssim_values
    )


    #Calculating competition style score
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


    #Saving sigma summary
    summary_results.append({

        "sigma_8bit":
            sigma_8bit,

        "mean_denoised_psnr":
            np.mean(
                denoised_psnr_values
            ),

        "mean_delta_psnr":
            mean_delta_psnr,

        "mean_denoised_ssim":
            np.mean(
                denoised_ssim_values
            ),

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

    })


    print()

    print(

        f"Sigma {sigma_8bit} | "

        f"Mean Delta PSNR "
        f"{mean_delta_psnr:+.4f} dB | "

        f"Mean Delta SSIM "
        f"{mean_delta_ssim:+.6f} | "

        f"Score "
        f"{score:.8f}"

    )


#Creating summary dataframe
summary_df = pd.DataFrame(
    summary_results
)


#Sorting best score first
summary_df = summary_df.sort_values(

    "score",

    ascending=False

)


#Creating per image dataframe
per_image_df = pd.DataFrame(
    per_image_results
)


#Saving summary CSV
summary_df.to_csv(

    OUTPUT_DIR
    / "bm3d_holdout_summary.csv",

    index=False

)


#Saving per image CSV
per_image_df.to_csv(

    OUTPUT_DIR
    / "bm3d_holdout_per_image.csv",

    index=False

)


#Printing final comparison
print()

print(
    "=" * 105
)

print(
    "BM3D INDEPENDENT HOLDOUT RESULTS"
)

print(
    "=" * 105
)


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
    ].to_string(
        index=False
    )

)


print()

print(
    "=" * 105
)


print(
    "BEST HOLDOUT SIGMA:"
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


print(
    "=" * 105
)