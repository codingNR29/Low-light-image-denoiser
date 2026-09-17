from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from PIL import Image

import bm3d

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


#Creating output directory
OUTPUT_DIR = (

    SCRIPT_DIR

    / "bm3d_stage_comparison"

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


#Final BM3D sigma selected from tuning and holdout experiments
BM3D_SIGMA_8BIT = 30


#Converting sigma to normalized [0,1] scale
BM3D_SIGMA = (

    BM3D_SIGMA_8BIT

    / 255.0

)


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


#Getting raw group labels
raw_labels = gmm.fit_predict(
    severity_values
)


#Getting severity centers
centers = (

    gmm.means_
    .flatten()

)


#Sorting severity centers
order = np.argsort(
    centers
)


#Mapping groups to readable labels
label_map = {

    order[0]:
        "Low",

    order[1]:
        "Medium",

    order[2]:
        "High"

}


#Adding severity group column
severity_df[
    "severity_group"
] = [

    label_map[label]

    for label in raw_labels

]


#Selecting representative test images
#5 Low + 5 Medium + 5 High
RANDOM_SEED = 789

sample_parts = []


for group in [

    "Low",

    "Medium",

    "High"

]:

    #Getting images from this severity group
    group_df = severity_df[

        severity_df[
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
    sample_parts.append(
        sampled
    )


#Combining selected images
sample_df = pd.concat(

    sample_parts,

    ignore_index=True

)


#Sorting by image number
sample_df = sample_df.sort_values(
    "image"
)


#Saving selected image list
sample_df.to_csv(

    OUTPUT_DIR
    / "bm3d_stage_test_15.csv",

    index=False

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


#Preparing one image before BM3D
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


#Running BM3D with selected stage
def run_bm3d(
    corrected_image,
    stage
):

    #Running multichannel BM3D
    #Using bm3d() instead of bm3d_rgb()
    #because bm3d() allows us to select the processing stage
    denoised = bm3d.bm3d(

        corrected_image,

        sigma_psd=BM3D_SIGMA,

        stage_arg=stage

    )


    #Clipping output to valid image range
    denoised = np.clip(

        denoised,

        0.0,

        1.0

    )


    #Converting back to uint8
    denoised_uint8 = (

        denoised

        * 255.0

    ).round().astype(
        np.uint8
    )


    return denoised_uint8


#Defining stages to compare
STAGES = [

    {

        "name":
            "Hard_Thresholding_Only",

        "stage":
            bm3d.BM3DStages.HARD_THRESHOLDING

    },

    {

        "name":
            "Full_BM3D",

        "stage":
            bm3d.BM3DStages.ALL_STAGES

    }

]


#Storage for summary results
summary_results = []


#Storage for per image results
per_image_results = []


#Testing each BM3D stage
for stage_info in STAGES:

    print()

    print(
        "=" * 80
    )


    print(

        "Testing stage:",

        stage_info[
            "name"
        ]

    )


    print(
        "=" * 80
    )


    #Storage for metrics
    delta_psnr_values = []

    delta_ssim_values = []

    denoised_psnr_values = []

    denoised_ssim_values = []

    runtimes = []


    #Processing selected 15 images
    for count, row in enumerate(

        sample_df.itertuples(),

        start=1

    ):

        #Getting image ID
        image_id = (

            f"{int(row.image):03d}"

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


        #Loading noisy image
        noisy = np.asarray(

            Image.open(
                noisy_path
            ).convert("RGB")

        )


        #Loading ground truth image
        ground_truth = np.asarray(

            Image.open(
                gt_path
            ).convert("RGB")

        )


        #Preparing noisy image
        corrected = prepare_image(
            noisy
        )


        #Calculating noisy PSNR
        noisy_psnr = calculate_psnr(

            ground_truth,

            noisy

        )


        #Calculating noisy SSIM
        noisy_ssim = calculate_ssim(

            ground_truth,

            noisy

        )


        print(

            f"Processing "
            f"{image_id} "
            f"({count}/15) ...",

            end=" ",

            flush=True

        )


        #Starting runtime measurement
        start = time.perf_counter()


        #Running BM3D stage
        denoised = run_bm3d(

            corrected,

            stage_info[
                "stage"
            ]

        )


        #Calculating runtime
        elapsed = (

            time.perf_counter()

            - start

        )


        #Calculating denoised PSNR
        denoised_psnr = calculate_psnr(

            ground_truth,

            denoised

        )


        #Calculating denoised SSIM
        denoised_ssim = calculate_ssim(

            ground_truth,

            denoised

        )


        #Calculating PSNR improvement
        delta_psnr = (

            denoised_psnr

            - noisy_psnr

        )


        #Calculating SSIM improvement
        delta_ssim = (

            denoised_ssim

            - noisy_ssim

        )


        #Saving results
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


        runtimes.append(
            elapsed
        )


        #Saving per image information
        per_image_results.append({

            "stage":
                stage_info["name"],

            "image":
                image_id,

            "severity":
                row.severity,

            "severity_group":
                row.severity_group,

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


    #Saving stage summary
    summary_results.append({

        "stage":
            stage_info["name"],

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
            )

    })


#Creating summary dataframe
summary_df = pd.DataFrame(
    summary_results
)


#Sorting by score
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
    / "bm3d_stage_summary.csv",

    index=False

)


#Saving per image CSV
per_image_df.to_csv(

    OUTPUT_DIR
    / "bm3d_stage_per_image.csv",

    index=False

)


#Printing final comparison
print()

print(
    "=" * 100
)

print(
    "BM3D STAGE COMPARISON"
)

print(
    "=" * 100
)


print(

    summary_df[
        [
            "stage",
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
    "=" * 100
)