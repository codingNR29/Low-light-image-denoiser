from pathlib import Path
import json

import numpy as np
import pandas as pd

from PIL import Image

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)


#Getting current script directory
SCRIPT_DIR = Path(
    __file__
).resolve().parent


#Getting project root
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


#Robust DnCNN outputs for 368 training images
DNCNN_DIR = (

    SCRIPT_DIR

    / "robust_dncnn_train_368"

    / "denoised"

)


#Optimized wavelet outputs for all 460 images
WAVELET_DIR = (

    PROJECT_ROOT

    / "My implementation"

    / "Restoration"

    / "optimized_wavelet_full_460"

    / "denoised"

)


#Predicted severity information for training images
SEVERITY_PREDICTIONS = (

    SCRIPT_DIR

    / "noise_severity_estimator"

    / "train_severity_predictions.csv"

)


#Creating output directory
OUTPUT_DIR = (

    SCRIPT_DIR

    / "adaptive_fusion_training"

)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


#DnCNN fusion weights to test
#1.00 means pure Robust DnCNN
#0.50 means equal DnCNN and Wavelet contribution
ALPHA_VALUES = np.arange(

    0.50,

    1.001,

    0.05

)


#Loading predicted training severity information
severity_df = pd.read_csv(
    SEVERITY_PREDICTIONS
)


#Making image numbers integers
severity_df["image"] = (

    severity_df["image"]
    .astype(int)

)

#Loading severity estimator metadata
METADATA_PATH = (

    SCRIPT_DIR

    / "noise_severity_estimator"

    / "severity_estimator_metadata.json"

)


#Reading saved severity thresholds
with open(
    METADATA_PATH,
    "r"
) as file:

    severity_metadata = json.load(
        file
    )


#Getting Low to Medium threshold
threshold_low_medium = (

    severity_metadata[
        "threshold_low_medium"
    ]

)


#Getting Medium to High threshold
threshold_medium_high = (

    severity_metadata[
        "threshold_medium_high"
    ]

)


#Converting predicted severity into severity group
def severity_to_group(
    severity
):

    #Low severity
    if severity < threshold_low_medium:

        return "Low"


    #Medium severity
    elif severity < threshold_medium_high:

        return "Medium"


    #High severity
    else:

        return "High"


#Creating predicted severity group for training images
severity_df[
    "predicted_group"
] = [

    severity_to_group(
        value
    )

    for value in severity_df[
        "predicted_severity"
    ]

]


print()

print(
    "=" * 80
)

print(
    "ADAPTIVE FUSION WEIGHT TRAINING"
)

print(
    "=" * 80
)


print(
    "Training images:",
    len(severity_df)
)


print()

print(
    "Predicted severity distribution:"
)


print(
    severity_df[
        "predicted_group"
    ].value_counts()
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


#Creating hybrid prediction
def create_hybrid(
    dncnn,
    wavelet,
    alpha
):

    #Converting DnCNN image to float
    dncnn_float = dncnn.astype(
        np.float32
    )


    #Converting Wavelet image to float
    wavelet_float = wavelet.astype(
        np.float32
    )


    #Applying weighted fusion
    hybrid = (

        alpha
        * dncnn_float

        +

        (1.0 - alpha)
        * wavelet_float

    )


    #Keeping values inside valid image range
    hybrid = np.clip(

        hybrid,

        0.0,

        255.0

    )


    #Converting back to uint8
    hybrid = hybrid.round().astype(
        np.uint8
    )


    return hybrid


#Storage for loaded images
training_images = []


#Loading all 368 training images
for count, row in enumerate(

    severity_df.itertuples(),

    start=1

):

    #Getting image number
    image_number = int(
        row.image
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


    #Creating Robust DnCNN output path
    dncnn_path = (

        DNCNN_DIR

        / f"{image_id}.png"

    )


    #Creating Wavelet output path
    wavelet_path = (

        WAVELET_DIR

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


    #Loading Robust DnCNN output
    dncnn = np.asarray(

        Image.open(
            dncnn_path
        ).convert("RGB")

    )


    #Loading Wavelet output
    wavelet = np.asarray(

        Image.open(
            wavelet_path
        ).convert("RGB")

    )


    #Calculating original noisy metrics
    noisy_psnr = calculate_psnr(

        ground_truth,

        noisy

    )


    noisy_ssim = calculate_ssim(

        ground_truth,

        noisy

    )


    #Saving image information
    training_images.append({

        "image":
            image_id,

        "predicted_severity":
            row.predicted_severity,

        "predicted_group":
            row.predicted_group,

        "ground_truth":
            ground_truth,

        "dncnn":
            dncnn,

        "wavelet":
            wavelet,

        "noisy_psnr":
            noisy_psnr,

        "noisy_ssim":
            noisy_ssim

    })


    #Printing loading progress
    if (

        count % 50 == 0

        or

        count == len(
            severity_df
        )

    ):

        print(

            f"Loaded "
            f"{count}/"
            f"{len(severity_df)}"

        )


#Storage for all fusion results
all_results = []


#Storage for best alpha from each severity group
best_weights = {}


#Testing each severity group separately
for group in [

    "Low",

    "Medium",

    "High"

]:

    print()

    print(
        "=" * 80
    )


    print(

        "OPTIMIZING GROUP:",

        group

    )


    print(
        "=" * 80
    )


    #Selecting images predicted to belong to this group
    group_images = [

        item

        for item in training_images

        if item[
            "predicted_group"
        ] == group

    ]


    print(

        "Images in group:",

        len(group_images)

    )


    #Testing every fusion weight
    for alpha in ALPHA_VALUES:

        delta_psnr_values = []

        delta_ssim_values = []


        #Testing alpha on all images in this severity group
        for item in group_images:

            #Creating hybrid output
            hybrid = create_hybrid(

                item[
                    "dncnn"
                ],

                item[
                    "wavelet"
                ],

                alpha

            )


            #Calculating hybrid PSNR
            hybrid_psnr = calculate_psnr(

                item[
                    "ground_truth"
                ],

                hybrid

            )


            #Calculating hybrid SSIM
            hybrid_ssim = calculate_ssim(

                item[
                    "ground_truth"
                ],

                hybrid

            )


            #Calculating PSNR improvement
            delta_psnr = (

                hybrid_psnr

                -

                item[
                    "noisy_psnr"
                ]

            )


            #Calculating SSIM improvement
            delta_ssim = (

                hybrid_ssim

                -

                item[
                    "noisy_ssim"
                ]

            )


            delta_psnr_values.append(
                delta_psnr
            )


            delta_ssim_values.append(
                delta_ssim
            )


        #Calculating mean improvement
        mean_delta_psnr = np.mean(
            delta_psnr_values
        )


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


        #Saving alpha result
        all_results.append({

            "severity_group":
                group,

            "alpha_dncnn":
                float(alpha),

            "wavelet_weight":
                float(
                    1.0 - alpha
                ),

            "mean_delta_psnr":
                mean_delta_psnr,

            "mean_delta_ssim":
                mean_delta_ssim,

            "score":
                score

        })


        print(

            f"Alpha "
            f"{alpha:.2f} | "

            f"Delta PSNR "
            f"{mean_delta_psnr:+.4f} | "

            f"Delta SSIM "
            f"{mean_delta_ssim:+.6f} | "

            f"Score "
            f"{score:.8f}"

        )


#Creating result dataframe
results_df = pd.DataFrame(
    all_results
)


#Finding best alpha from each severity group
for group in [

    "Low",

    "Medium",

    "High"

]:

    #Getting only this group's results
    group_results = results_df[

        results_df[
            "severity_group"
        ]
        == group

    ]


    #Finding row with maximum score
    best_row = group_results.loc[

        group_results[
            "score"
        ].idxmax()

    ]


    #Saving best DnCNN weight
    best_weights[
        group
    ] = float(

        best_row[
            "alpha_dncnn"
        ]

    )


#Saving all tested fusion results
results_df.to_csv(

    OUTPUT_DIR

    / "adaptive_fusion_weight_results.csv",

    index=False

)


#Saving best severity dependent weights
with open(

    OUTPUT_DIR

    / "adaptive_fusion_weights.json",

    "w"

) as file:

    json.dump(

        best_weights,

        file,

        indent=4

    )


#Printing final selected weights
print()

print(
    "=" * 80
)

print(
    "SELECTED ADAPTIVE FUSION WEIGHTS"
)

print(
    "=" * 80
)


for group in [

    "Low",

    "Medium",

    "High"

]:

    alpha = best_weights[
        group
    ]


    print()

    print(
        group
    )


    print(

        f"DnCNN weight  : "
        f"{alpha:.2f}"

    )


    print(

        f"Wavelet weight: "
        f"{1.0 - alpha:.2f}"

    )


print()

print(
    "Weights saved to:"
)


print(

    OUTPUT_DIR

    / "adaptive_fusion_weights.json"

)


print(
    "=" * 80
)