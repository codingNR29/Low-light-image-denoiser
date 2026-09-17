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


#Validation image CSV
VAL_CSV = (

    SCRIPT_DIR

    / "dataset_split"

    / "validation_images.csv"

)


'''#Previously generated DnCNN validation outputs
DNCNN_DIR = (

    SCRIPT_DIR

    / "dncnn_validation_92"

    / "denoised"

)'''
DNCNN_DIR = (

    SCRIPT_DIR

    / "robust_dncnn_validation_92"

    / "denoised"

)

'''#Creating hybrid output directory
OUTPUT_DIR = (

    SCRIPT_DIR

    / "hybrid_fusion_validation"

)'''
#Creating robust hybrid output directory
OUTPUT_DIR = (

    SCRIPT_DIR

    / "robust_hybrid_fusion_validation"

)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


#Creating directory for wavelet validation outputs
WAVELET_DIR = (

    OUTPUT_DIR

    / "wavelet_denoised"

)


WAVELET_DIR.mkdir(
    parents=True,
    exist_ok=True
)


#Creating directory for best hybrid outputs
BEST_HYBRID_DIR = (

    OUTPUT_DIR

    / "best_hybrid_denoised"

)


BEST_HYBRID_DIR.mkdir(
    parents=True,
    exist_ok=True
)


#Importing organizer defect pixel correction
ORGANIZER_DIR = (

    PROJECT_ROOT

    / "My implementation"

    / "Restoration"

    / "Organizer_baseline"

)


sys.path.insert(
    0,
    str(ORGANIZER_DIR)
)


from denoise import correct_defect_pixels


#Fusion weights
#Alpha represents contribution from DnCNN
ALPHA_VALUES = [

    0.00,

    0.25,

    0.50,

    0.75,

    1.00

]


#Loading validation image list
validation_df = pd.read_csv(
    VAL_CSV
)


print(
    "Validation images:",
    len(validation_df)
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


#Optimized wavelet denoising
def wavelet_denoise(
    noisy_image
):

    #Converting image from [0,255] to [0,1]
    image_float = (

        noisy_image.astype(
            np.float32
        )

        / 255.0

    )


    #Correcting strong defect pixels
    corrected = correct_defect_pixels(

        image_float,

        threshold=0.25

    )


    #Applying optimized wavelet denoising
    #Using the configuration selected from our parameter study
    denoised = denoise_wavelet(

        corrected,

        method="BayesShrink",

        mode="soft",

        wavelet="sym4",

        wavelet_levels=4,

        channel_axis=-1,

        rescale_sigma=True

    )


    #Clipping image to valid range
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


#Creating hybrid image
def create_hybrid(
    dncnn_image,
    wavelet_image,
    alpha
):

    #Converting images to float before fusion
    dncnn_float = dncnn_image.astype(
        np.float32
    )


    wavelet_float = wavelet_image.astype(
        np.float32
    )


    #Combining DnCNN and Wavelet outputs
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


    #Converting hybrid image back to uint8
    hybrid = hybrid.round().astype(
        np.uint8
    )


    return hybrid


#Storage for all loaded validation information
validation_images = []


#Generating wavelet outputs for all validation images
print()

print(
    "=" * 75
)

print(
    "GENERATING WAVELET OUTPUTS"
)

print(
    "=" * 75
)


for count, row in enumerate(

    validation_df.itertuples(),

    start=1

):

    #Getting image number
    image_number = int(
        row.image
    )


    #Creating 3 digit image ID
    image_id = (
        f"{image_number:03d}"
    )


    #Creating noisy image path
    noisy_path = (

        NOISY_DIR

        / f"{image_id}_noise.png"

    )


    #Creating ground truth image path
    gt_path = (

        GT_DIR

        / f"{image_id}.png"

    )


    #Creating DnCNN output path
    dncnn_path = (

        DNCNN_DIR

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


    #Loading previously generated DnCNN output
    dncnn_output = np.asarray(

        Image.open(
            dncnn_path
        ).convert("RGB")

    )


    #Running optimized wavelet method
    start = time.perf_counter()


    wavelet_output = wavelet_denoise(
        noisy
    )


    wavelet_time = (

        time.perf_counter()

        - start

    )


    #Saving wavelet output
    Image.fromarray(
        wavelet_output
    ).save(

        WAVELET_DIR

        / f"{image_id}.png"

    )


    #Calculating noisy metrics once
    noisy_psnr = calculate_psnr(

        ground_truth,

        noisy

    )


    noisy_ssim = calculate_ssim(

        ground_truth,

        noisy

    )


    #Saving everything needed for fusion testing
    validation_images.append({

        "image":
            image_id,

        "noisy":
            noisy,

        "ground_truth":
            ground_truth,

        "dncnn":
            dncnn_output,

        "wavelet":
            wavelet_output,

        "noisy_psnr":
            noisy_psnr,

        "noisy_ssim":
            noisy_ssim,

        "wavelet_runtime":
            wavelet_time

    })


    print(

        f"{count:02d}/"
        f"{len(validation_df)} | "

        f"Image {image_id} | "

        f"Wavelet "
        f"{wavelet_time:.3f} s"

    )


#Storage for fusion summary
summary_results = []


#Storage for individual image metrics
per_image_results = []


#Testing every fusion weight
print()

print(
    "=" * 75
)

print(
    "TESTING HYBRID FUSION WEIGHTS"
)

print(
    "=" * 75
)


for alpha in ALPHA_VALUES:

    print()

    print(
        f"Testing alpha = "
        f"{alpha:.2f}"
    )


    #Storage for this alpha
    denoised_psnr_values = []

    delta_psnr_values = []

    denoised_ssim_values = []

    delta_ssim_values = []


    #Testing hybrid on all validation images
    for item in validation_images:

        #Creating hybrid prediction
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
        denoised_psnr = calculate_psnr(

            item[
                "ground_truth"
            ],

            hybrid

        )


        #Calculating hybrid SSIM
        denoised_ssim = calculate_ssim(

            item[
                "ground_truth"
            ],

            hybrid

        )


        #Calculating PSNR improvement
        delta_psnr = (

            denoised_psnr

            -

            item[
                "noisy_psnr"
            ]

        )


        #Calculating SSIM improvement
        delta_ssim = (

            denoised_ssim

            -

            item[
                "noisy_ssim"
            ]

        )


        #Saving metrics
        denoised_psnr_values.append(
            denoised_psnr
        )


        delta_psnr_values.append(
            delta_psnr
        )


        denoised_ssim_values.append(
            denoised_ssim
        )


        delta_ssim_values.append(
            delta_ssim
        )


        #Saving individual image result
        per_image_results.append({

            "alpha":
                alpha,

            "image":
                item["image"],

            "denoised_psnr":
                denoised_psnr,

            "delta_psnr":
                delta_psnr,

            "denoised_ssim":
                denoised_ssim,

            "delta_ssim":
                delta_ssim

        })


    #Calculating mean metrics
    mean_denoised_psnr = np.mean(
        denoised_psnr_values
    )


    mean_delta_psnr = np.mean(
        delta_psnr_values
    )


    mean_denoised_ssim = np.mean(
        denoised_ssim_values
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


    #Saving summary for this alpha
    summary_results.append({

        "alpha_dncnn":
            alpha,

        "wavelet_weight":
            1.0 - alpha,

        "mean_denoised_psnr":
            mean_denoised_psnr,

        "mean_delta_psnr":
            mean_delta_psnr,

        "mean_denoised_ssim":
            mean_denoised_ssim,

        "mean_delta_ssim":
            mean_delta_ssim,

        "score":
            score

    })


    print(

        f"Alpha {alpha:.2f} | "

        f"Delta PSNR "
        f"{mean_delta_psnr:+.4f} dB | "

        f"Delta SSIM "
        f"{mean_delta_ssim:+.6f} | "

        f"Score "
        f"{score:.8f}"

    )


#Creating summary dataframe
summary_df = pd.DataFrame(
    summary_results
)


#Sorting best fusion first
summary_df = summary_df.sort_values(

    "score",

    ascending=False

)


#Creating per image dataframe
per_image_df = pd.DataFrame(
    per_image_results
)


#Saving fusion summary
summary_df.to_csv(

    OUTPUT_DIR

    / "hybrid_fusion_summary.csv",

    index=False

)


#Saving individual image results
per_image_df.to_csv(

    OUTPUT_DIR

    / "hybrid_fusion_per_image.csv",

    index=False

)


#Getting best alpha
best_alpha = float(

    summary_df.iloc[0][
        "alpha_dncnn"
    ]

)


#Saving outputs from the best hybrid
for item in validation_images:

    #Creating best hybrid image
    best_hybrid = create_hybrid(

        item[
            "dncnn"
        ],

        item[
            "wavelet"
        ],

        best_alpha

    )


    #Saving best hybrid image
    Image.fromarray(
        best_hybrid
    ).save(

        BEST_HYBRID_DIR

        / f"{item['image']}.png"

    )


#Printing final comparison
print()

print(
    "=" * 105
)

print(
    "HYBRID FUSION VALIDATION RESULTS"
)

print(
    "=" * 105
)


print(

    summary_df[
        [
            "alpha_dncnn",
            "wavelet_weight",
            "mean_denoised_psnr",
            "mean_delta_psnr",
            "mean_denoised_ssim",
            "mean_delta_ssim",
            "score"
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
    "BEST HYBRID ALPHA:"
)


print(
    best_alpha
)


print()

print(
    "DnCNN contribution:",
    best_alpha
)


print(
    "Wavelet contribution:",
    1.0 - best_alpha
)


print()

print(
    "Best hybrid images saved to:"
)


print(
    BEST_HYBRID_DIR
)


print(
    "=" * 105
)