from pathlib import Path

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


#Robust DnCNN validation outputs
DNCNN_DIR = (

    SCRIPT_DIR

    / "robust_dncnn_validation_92"

    / "denoised"

)


#Wavelet outputs
WAVELET_DIR = (

    PROJECT_ROOT

    / "My implementation"

    / "Restoration"

    / "optimized_wavelet_full_460"

    / "denoised"

)


#Validation severity predictions
SEVERITY_CSV = (

    SCRIPT_DIR

    / "noise_severity_estimator"

    / "validation_severity_predictions.csv"

)


#Creating output directory
OUTPUT_DIR = (

    SCRIPT_DIR

    / "adaptive_fusion_validation"

)


DENOISED_DIR = (

    OUTPUT_DIR

    / "denoised"

)


DENOISED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


#Frozen fusion weights learned from training images
ADAPTIVE_WEIGHTS = {

    "Low": 0.85,

    "Medium": 0.90,

    "High": 0.90

}


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


#Creating adaptive hybrid
def create_hybrid(
    dncnn,
    wavelet,
    alpha
):

    #Converting images to float
    dncnn_float = dncnn.astype(
        np.float32
    )


    wavelet_float = wavelet.astype(
        np.float32
    )


    #Applying adaptive fusion
    hybrid = (

        alpha
        * dncnn_float

        +

        (1.0 - alpha)
        * wavelet_float

    )


    #Keeping valid image range
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


#Loading validation severity predictions
severity_df = pd.read_csv(
    SEVERITY_CSV
)


#Making image numbers integers
severity_df["image"] = (

    severity_df["image"]
    .astype(int)

)


#Storage for results
result_rows = []


print()

print(
    "=" * 80
)

print(
    "ADAPTIVE FUSION VALIDATION"
)

print(
    "=" * 80
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


print()


#Processing all validation images
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


    #Getting predicted severity group
    severity_group = (
        row.predicted_group
    )


    #Getting corresponding DnCNN weight
    alpha = ADAPTIVE_WEIGHTS[
        severity_group
    ]


    #Creating image paths
    noisy_path = (

        NOISY_DIR

        / f"{image_id}_noise.png"

    )


    gt_path = (

        GT_DIR

        / f"{image_id}.png"

    )


    dncnn_path = (

        DNCNN_DIR

        / f"{image_id}.png"

    )


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


    #Loading robust DnCNN output
    dncnn = np.asarray(

        Image.open(
            dncnn_path
        ).convert("RGB")

    )


    #Loading wavelet output
    wavelet = np.asarray(

        Image.open(
            wavelet_path
        ).convert("RGB")

    )


    #Creating adaptive hybrid
    hybrid = create_hybrid(

        dncnn,

        wavelet,

        alpha

    )


    #Saving adaptive output
    Image.fromarray(
        hybrid
    ).save(

        DENOISED_DIR

        / f"{image_id}.png"

    )


    #Calculating noisy PSNR
    noisy_psnr = calculate_psnr(

        ground_truth,

        noisy

    )


    #Calculating adaptive PSNR
    denoised_psnr = calculate_psnr(

        ground_truth,

        hybrid

    )


    #Calculating noisy SSIM
    noisy_ssim = calculate_ssim(

        ground_truth,

        noisy

    )


    #Calculating adaptive SSIM
    denoised_ssim = calculate_ssim(

        ground_truth,

        hybrid

    )


    #Calculating improvements
    delta_psnr = (

        denoised_psnr

        -

        noisy_psnr

    )


    delta_ssim = (

        denoised_ssim

        -

        noisy_ssim

    )


    #Saving image result
    result_rows.append({

        "image":
            image_id,

        "severity_group":
            severity_group,

        "alpha_dncnn":
            alpha,

        "wavelet_weight":
            1.0 - alpha,

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


    #Printing progress
    if (

        count % 20 == 0

        or

        count == len(
            severity_df
        )

    ):

        print(

            f"Processed "
            f"{count}/"
            f"{len(severity_df)}"

        )


#Creating results dataframe
results_df = pd.DataFrame(
    result_rows
)


#Calculating mean results
mean_noisy_psnr = results_df[
    "noisy_psnr"
].mean()


mean_denoised_psnr = results_df[
    "denoised_psnr"
].mean()


mean_delta_psnr = results_df[
    "delta_psnr"
].mean()


mean_noisy_ssim = results_df[
    "noisy_ssim"
].mean()


mean_denoised_ssim = results_df[
    "denoised_ssim"
].mean()


mean_delta_ssim = results_df[
    "delta_ssim"
].mean()


#Calculating competition score
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


#Saving per image results
results_df.to_csv(

    OUTPUT_DIR

    / "adaptive_fusion_validation_metrics.csv",

    index=False

)


#Printing final results
print()

print(
    "=" * 80
)

print(
    "ADAPTIVE FUSION VALIDATION RESULTS"
)

print(
    "=" * 80
)


print()

print(

    f"Images evaluated        : "
    f"{len(results_df)}"

)


print()

print(

    f"Mean Noisy PSNR         : "
    f"{mean_noisy_psnr:.4f} dB"

)


print(

    f"Mean Denoised PSNR      : "
    f"{mean_denoised_psnr:.4f} dB"

)


print(

    f"Mean Delta PSNR         : "
    f"{mean_delta_psnr:+.4f} dB"

)


print()

print(

    f"Mean Noisy SSIM         : "
    f"{mean_noisy_ssim:.6f}"

)


print(

    f"Mean Denoised SSIM      : "
    f"{mean_denoised_ssim:.6f}"

)


print(

    f"Mean Delta SSIM         : "
    f"{mean_delta_ssim:+.6f}"

)


print()

print(

    f"Validation score        : "
    f"{score:.8f}"

)


print()

print(
    "Current robust DnCNN score: 0.51709573"
)


print()


if score > 0.51709573:

    print(
        "ADAPTIVE FUSION BEATS ROBUST DNCNN!"
    )

else:

    print(
        "ROBUST DNCNN REMAINS THE BEST METHOD."
    )


print()

print(
    "Adaptive outputs saved to:"
)


print(
    DENOISED_DIR
)


print(
    "=" * 80
)