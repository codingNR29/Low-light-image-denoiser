from pathlib import Path
import time

import numpy as np
import pandas as pd

from PIL import Image

import torch

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


#Training image CSV
TRAIN_CSV = (

    SCRIPT_DIR

    / "dataset_split"

    / "train_images.csv"

)


#Best robust DnCNN model
MODEL_PATH = (

    SCRIPT_DIR

    / "dncnn_robust_finetuning"

    / "best_robust_dncnn.pth"

)


#Creating output directory
OUTPUT_DIR = (

    SCRIPT_DIR

    / "robust_dncnn_train_368"

)


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


#Creating DnCNN architecture
model = DnCNN(

    input_channels=3,

    output_channels=3,

    num_features=64,

    num_layers=17

)


#Loading robust model weights
state_dict = torch.load(

    MODEL_PATH,

    map_location=device

)


#Loading weights into model
model.load_state_dict(
    state_dict
)


#Moving model to GPU
model = model.to(
    device
)


#Setting model to evaluation mode
model.eval()


#Loading training image list
train_df = pd.read_csv(
    TRAIN_CSV
)


print(
    "Training images:",
    len(train_df)
)


#Full image DnCNN inference
def denoise_image(
    noisy_image
):

    #Converting image from [0,255] to [0,1]
    noisy_float = (

        noisy_image.astype(
            np.float32
        )

        / 255.0

    )


    #Changing H x W x C to C x H x W
    noisy_chw = np.transpose(

        noisy_float,

        (2, 0, 1)

    )


    #Converting NumPy array to PyTorch tensor
    noisy_tensor = torch.from_numpy(

        noisy_chw.copy()

    ).float()


    #Adding batch dimension
    noisy_tensor = noisy_tensor.unsqueeze(
        0
    )


    #Moving image to GPU
    noisy_tensor = noisy_tensor.to(
        device
    )


    #Running inference without gradient calculation
    with torch.inference_mode():

        #Predicting residual
        predicted_residual = model(
            noisy_tensor
        )


        #Reconstructing clean image
        predicted_clean = (

            noisy_tensor

            -

            predicted_residual

        )


        #Keeping output inside valid range
        predicted_clean = torch.clamp(

            predicted_clean,

            0.0,

            1.0

        )


    #Removing batch dimension
    predicted_clean = predicted_clean.squeeze(
        0
    )


    #Moving result back to CPU
    predicted_clean = predicted_clean.cpu().numpy()


    #Changing C x H x W back to H x W x C
    predicted_clean = np.transpose(

        predicted_clean,

        (1, 2, 0)

    )


    #Converting back to uint8
    predicted_clean = (

        predicted_clean

        * 255.0

    ).round().astype(
        np.uint8
    )


    return predicted_clean


#Storage for runtime information
runtime_rows = []


#Processing all 368 training images
for count, row in enumerate(

    train_df.itertuples(),

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


    #Loading noisy image
    noisy = np.asarray(

        Image.open(
            noisy_path
        ).convert("RGB")

    )


    #Starting inference timer
    start = time.perf_counter()


    #Running robust DnCNN
    denoised = denoise_image(
        noisy
    )


    #Calculating runtime
    elapsed = (

        time.perf_counter()

        - start

    )


    #Saving robust DnCNN output
    Image.fromarray(
        denoised
    ).save(

        DENOISED_DIR

        / f"{image_id}.png"

    )


    #Saving runtime
    runtime_rows.append({

        "image":
            image_id,

        "runtime_seconds":
            elapsed

    })


    #Printing progress
    if (

        count % 25 == 0

        or

        count == len(
            train_df
        )

    ):

        print(

            f"Processed "
            f"{count}/"
            f"{len(train_df)}"

        )


#Creating runtime dataframe
runtime_df = pd.DataFrame(
    runtime_rows
)


#Saving runtime information
runtime_df.to_csv(

    OUTPUT_DIR

    / "robust_dncnn_train_runtime.csv",

    index=False

)


#Printing final summary
print()

print(
    "=" * 75
)

print(
    "ROBUST DNCNN TRAINING OUTPUT GENERATION COMPLETE"
)

print(
    "=" * 75
)


print(

    "Images processed:",
    len(runtime_df)

)


print(

    "Average runtime/image:",
    f"{runtime_df['runtime_seconds'].mean():.3f} s"

)


print()

print(
    "Outputs saved to:"
)


print(
    DENOISED_DIR
)


print(
    "=" * 75
)