from pathlib import Path
import time

import numpy as np
import pandas as pd

from PIL import Image

import torch

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

from dncnn_model import DnCNN


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


#Best trained DnCNN model
MODEL_PATH = (

    SCRIPT_DIR

    / "dncnn_training"

    / "best_dncnn.pth"

)


#Creating evaluation output directory
OUTPUT_DIR = (

    SCRIPT_DIR

    / "dncnn_validation_92"

)


#Creating directory for denoised validation images
DENOISED_DIR = (

    OUTPUT_DIR

    / "denoised"

)


DENOISED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


#Selecting GPU if available
device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)


print(
    "Using device:",
    device
)


print(
    "Loading model from:"
)

print(
    MODEL_PATH
)


#Creating same DnCNN architecture used during training
model = DnCNN(

    input_channels=3,

    output_channels=3,

    num_features=64,

    num_layers=17

)


#Loading trained model weights
state_dict = torch.load(

    MODEL_PATH,

    map_location=device

)


#Putting trained weights into model
model.load_state_dict(
    state_dict
)


#Moving model to GPU or CPU
model = model.to(
    device
)


#Very important during evaluation
#This changes BatchNorm layers from training behaviour
#to evaluation behaviour
model.eval()


#Loading validation image list
validation_df = pd.read_csv(
    VAL_CSV
)


print()

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


#DnCNN full image inference
def denoise_image(
    noisy_image
):

    #Converting uint8 image from [0,255] to [0,1]
    noisy_float = (

        noisy_image.astype(
            np.float32
        )

        / 255.0

    )


    #Changing image format from
    #H x W x C
    #to
    #C x H x W
    noisy_chw = np.transpose(

        noisy_float,

        (2, 0, 1)

    )


    #Converting NumPy array to PyTorch tensor
    noisy_tensor = torch.from_numpy(

        noisy_chw.copy()

    ).float()


    #Adding batch dimension
    #3 x H x W becomes 1 x 3 x H x W
    noisy_tensor = noisy_tensor.unsqueeze(
        0
    )


    #Moving image to GPU
    noisy_tensor = noisy_tensor.to(
        device
    )


    #Disabling gradient calculations during inference
    with torch.inference_mode():

        #Predicting noise residual
        predicted_residual = model(
            noisy_tensor
        )


        #Reconstructing clean image
        predicted_clean = (

            noisy_tensor

            -

            predicted_residual

        )


        #Keeping values inside valid [0,1] range
        predicted_clean = torch.clamp(

            predicted_clean,

            0.0,

            1.0

        )


    #Removing batch dimension
    predicted_clean = predicted_clean.squeeze(
        0
    )


    #Moving result from GPU to CPU
    predicted_clean = predicted_clean.cpu().numpy()


    #Changing image format from
    #C x H x W
    #back to
    #H x W x C
    predicted_clean = np.transpose(

        predicted_clean,

        (1, 2, 0)

    )


    #Converting [0,1] back to [0,255]
    predicted_clean = (

        predicted_clean

        * 255.0

    ).round().astype(
        np.uint8
    )


    return predicted_clean


#Storage for evaluation results
results = []


#Tracking total runtime
total_runtime = 0.0


#Processing all validation images
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


    #Calculating original noisy metrics
    noisy_psnr = calculate_psnr(

        ground_truth,

        noisy

    )


    noisy_ssim = calculate_ssim(

        ground_truth,

        noisy

    )


    #Starting inference timer
    start = time.perf_counter()


    #Running DnCNN
    denoised = denoise_image(
        noisy
    )


    #Calculating inference runtime
    elapsed = (

        time.perf_counter()

        - start

    )


    total_runtime += elapsed


    #Calculating denoised metrics
    denoised_psnr = calculate_psnr(

        ground_truth,

        denoised

    )


    denoised_ssim = calculate_ssim(

        ground_truth,

        denoised

    )


    #Calculating PSNR improvement
    delta_psnr = (

        denoised_psnr

        -

        noisy_psnr

    )


    #Calculating SSIM improvement
    delta_ssim = (

        denoised_ssim

        -

        noisy_ssim

    )


    #Saving denoised validation image
    Image.fromarray(
        denoised
    ).save(

        DENOISED_DIR

        / f"{image_id}.png"

    )


    #Saving image metrics
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


    #Printing progress
    print(

        f"{count:02d}/"
        f"{len(validation_df)} | "

        f"Image {image_id} | "

        f"PSNR "
        f"{noisy_psnr:.2f} -> "
        f"{denoised_psnr:.2f} | "

        f"Delta "
        f"{delta_psnr:+.2f} dB | "

        f"SSIM "
        f"{noisy_ssim:.4f} -> "
        f"{denoised_ssim:.4f} | "

        f"{elapsed:.3f} s"

    )


#Creating results dataframe
results_df = pd.DataFrame(
    results
)


#Saving individual image metrics
results_df.to_csv(

    OUTPUT_DIR

    / "dncnn_validation_metrics.csv",

    index=False

)


#Calculating mean noisy PSNR
mean_noisy_psnr = results_df[
    "noisy_psnr"
].mean()


#Calculating mean denoised PSNR
mean_denoised_psnr = results_df[
    "denoised_psnr"
].mean()


#Calculating mean PSNR improvement
mean_delta_psnr = results_df[
    "delta_psnr"
].mean()


#Calculating mean noisy SSIM
mean_noisy_ssim = results_df[
    "noisy_ssim"
].mean()


#Calculating mean denoised SSIM
mean_denoised_ssim = results_df[
    "denoised_ssim"
].mean()


#Calculating mean SSIM improvement
mean_delta_ssim = results_df[
    "delta_ssim"
].mean()


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


#Calculating average inference runtime
avg_runtime = (

    total_runtime

    / len(results_df)

)


#Printing final evaluation results
print()

print(
    "=" * 75
)

print(
    "DNCNN VALIDATION RESULTS"
)

print(
    "=" * 75
)


print(

    f"Images evaluated       : "
    f"{len(results_df)}"

)


print()


print(

    f"Mean Noisy PSNR        : "
    f"{mean_noisy_psnr:.4f} dB"

)


print(

    f"Mean Denoised PSNR     : "
    f"{mean_denoised_psnr:.4f} dB"

)


print(

    f"Mean Delta PSNR        : "
    f"{mean_delta_psnr:+.4f} dB"

)


print()


print(

    f"Mean Noisy SSIM        : "
    f"{mean_noisy_ssim:.6f}"

)


print(

    f"Mean Denoised SSIM     : "
    f"{mean_denoised_ssim:.6f}"

)


print(

    f"Mean Delta SSIM        : "
    f"{mean_delta_ssim:+.6f}"

)


print()


print(

    f"Validation score       : "
    f"{score:.8f}"

)


print(

    f"Average runtime/image  : "
    f"{avg_runtime:.3f} s"

)


print()

print(
    "Denoised validation images saved to:"
)

print(
    DENOISED_DIR
)


print()

print(
    "Metrics saved to:"
)

print(

    OUTPUT_DIR

    / "dncnn_validation_metrics.csv"

)


print(
    "=" * 75
)