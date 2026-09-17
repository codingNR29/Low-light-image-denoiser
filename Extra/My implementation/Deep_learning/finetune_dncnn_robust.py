from pathlib import Path
import time

import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

from dataset import DenoisingPatchDataset
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


#Training image CSV
TRAIN_CSV = (

    SCRIPT_DIR

    / "dataset_split"

    / "train_images.csv"

)


#Validation image CSV
VAL_CSV = (

    SCRIPT_DIR

    / "dataset_split"

    / "validation_images.csv"

)


#Best DnCNN checkpoint from original MSE training
ORIGINAL_MODEL_PATH = (

    SCRIPT_DIR

    / "dncnn_training"

    / "best_dncnn.pth"

)


#Creating fine tuning output directory
OUTPUT_DIR = (

    SCRIPT_DIR

    / "dncnn_robust_finetuning"

)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


#Fine tuning settings
PATCH_SIZE = 64

PATCHES_PER_IMAGE = 8

BATCH_SIZE = 8

NUM_EPOCHS = 3

LEARNING_RATE = 0.00005

CHARBONNIER_EPSILON = 1e-3


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


#Charbonnier loss
class CharbonnierLoss(nn.Module):

    def __init__(
        self,
        epsilon=1e-3
    ):

        super().__init__()

        self.epsilon = epsilon


    #Calculating robust residual loss
    def forward(
        self,
        prediction,
        target
    ):

        #Calculating prediction error
        difference = (

            prediction

            -

            target

        )


        #Applying Charbonnier penalty
        loss = torch.sqrt(

            difference ** 2

            +

            self.epsilon ** 2

        )


        #Returning mean loss
        return loss.mean()


#Keeping BatchNorm running statistics fixed
def freeze_batchnorm_statistics(
    module
):

    #Checking whether the layer is BatchNorm
    if isinstance(
        module,
        nn.BatchNorm2d
    ):

        #Using previously learned running mean and variance
        module.eval()


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


#Creating training dataset
train_dataset = DenoisingPatchDataset(

    csv_path=TRAIN_CSV,

    noisy_dir=NOISY_DIR,

    gt_dir=GT_DIR,

    patch_size=PATCH_SIZE,

    patches_per_image=PATCHES_PER_IMAGE,

    training=True

)


#Creating training loader
train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0,

    pin_memory=True

)


#Loading validation image list
validation_df = pd.read_csv(
    VAL_CSV
)


print(
    "Fine tuning patches:",
    len(train_dataset)
)


print(
    "Training batches:",
    len(train_loader)
)


print(
    "Full validation images:",
    len(validation_df)
)


#Creating same DnCNN architecture
model = DnCNN(

    input_channels=3,

    output_channels=3,

    num_features=64,

    num_layers=17

)


#Loading best original DnCNN weights
state_dict = torch.load(

    ORIGINAL_MODEL_PATH,

    map_location=device

)


model.load_state_dict(
    state_dict
)


#Moving model to GPU
model = model.to(
    device
)


print()

print(
    "Loaded original best DnCNN:"
)

print(
    ORIGINAL_MODEL_PATH
)


#Creating Charbonnier loss
criterion = CharbonnierLoss(

    epsilon=CHARBONNIER_EPSILON

)


#Using small learning rate because this is fine tuning
optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE

)


#Full image DnCNN inference
def denoise_full_image(
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


    #Converting NumPy image to PyTorch tensor
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


    #Running inference without gradients
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


        #Keeping values inside valid range
        predicted_clean = torch.clamp(

            predicted_clean,

            0.0,

            1.0

        )


    #Removing batch dimension
    predicted_clean = predicted_clean.squeeze(
        0
    )


    #Moving image back to CPU
    predicted_clean = predicted_clean.cpu().numpy()


    #Changing C x H x W back to H x W x C
    predicted_clean = np.transpose(

        predicted_clean,

        (1, 2, 0)

    )


    #Converting image back to uint8
    predicted_clean = (

        predicted_clean

        * 255.0

    ).round().astype(
        np.uint8
    )


    return predicted_clean


#Evaluating model using all 92 full validation images
def evaluate_full_validation():

    #Setting model to evaluation mode
    model.eval()


    delta_psnr_values = []

    delta_ssim_values = []


    #Processing every validation image
    for row in validation_df.itertuples():

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


        #Running DnCNN
        denoised = denoise_full_image(
            noisy
        )


        #Calculating noisy PSNR
        noisy_psnr = calculate_psnr(

            ground_truth,

            noisy

        )


        #Calculating denoised PSNR
        denoised_psnr = calculate_psnr(

            ground_truth,

            denoised

        )


        #Calculating noisy SSIM
        noisy_ssim = calculate_ssim(

            ground_truth,

            noisy

        )


        #Calculating denoised SSIM
        denoised_ssim = calculate_ssim(

            ground_truth,

            denoised

        )


        #Saving PSNR improvement
        delta_psnr_values.append(

            denoised_psnr

            -

            noisy_psnr

        )


        #Saving SSIM improvement
        delta_ssim_values.append(

            denoised_ssim

            -

            noisy_ssim

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


    return (
        mean_delta_psnr,
        mean_delta_ssim,
        score
    )


#Evaluating original checkpoint before fine tuning
print()

print(
    "=" * 75
)

print(
    "ORIGINAL DNCNN FULL VALIDATION"
)

print(
    "=" * 75
)


original_delta_psnr, original_delta_ssim, original_score = (

    evaluate_full_validation()

)


print(

    f"Mean Delta PSNR : "
    f"{original_delta_psnr:+.4f} dB"

)


print(

    f"Mean Delta SSIM : "
    f"{original_delta_ssim:+.6f}"

)


print(

    f"Score           : "
    f"{original_score:.8f}"

)


#Starting best score from original model
best_score = original_score


#Saving original checkpoint as current best robust checkpoint
torch.save(

    model.state_dict(),

    OUTPUT_DIR

    / "best_robust_dncnn.pth"

)


#Storage for training history
history = []


#Fine tuning loop
for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    #Starting epoch timer
    epoch_start = time.perf_counter()


    #Setting model to training mode
    model.train()


    #Freezing BatchNorm running statistics
    model.apply(
        freeze_batchnorm_statistics
    )


    #Tracking training loss
    running_loss = 0.0


    #Processing training batches
    for batch_number, batch in enumerate(

        train_loader,

        start=1

    ):

        #Moving noisy patches to GPU
        noisy = batch[
            "noisy"
        ].to(

            device,

            non_blocking=True

        )


        #Moving true residuals to GPU
        true_residual = batch[
            "residual"
        ].to(

            device,

            non_blocking=True

        )


        #Clearing previous gradients
        optimizer.zero_grad()


        #Predicting residual
        predicted_residual = model(
            noisy
        )


        #Calculating robust Charbonnier loss
        loss = criterion(

            predicted_residual,

            true_residual

        )


        #Calculating gradients
        loss.backward()


        #Clipping very large gradients
        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=1.0

        )


        #Updating model weights
        optimizer.step()


        #Adding training loss
        running_loss += (
            loss.item()
        )


        #Printing progress every 100 batches
        if batch_number % 100 == 0:

            print(

                f"Epoch {epoch:02d} | "

                f"Batch "
                f"{batch_number}/"
                f"{len(train_loader)} | "

                f"Charbonnier loss "
                f"{loss.item():.6f}"

            )


    #Calculating mean training loss
    mean_train_loss = (

        running_loss

        / len(train_loader)

    )


    #Evaluating full validation set
    mean_delta_psnr, mean_delta_ssim, score = (

        evaluate_full_validation()

    )


    #Calculating epoch runtime
    epoch_time = (

        time.perf_counter()

        - epoch_start

    )


    print()

    print(
        "=" * 75
    )


    print(

        f"ROBUST FINE TUNING EPOCH "
        f"{epoch}/"
        f"{NUM_EPOCHS}"

    )


    print(

        f"Training Charbonnier loss : "
        f"{mean_train_loss:.8f}"

    )


    print(

        f"Validation Delta PSNR      : "
        f"{mean_delta_psnr:+.4f} dB"

    )


    print(

        f"Validation Delta SSIM      : "
        f"{mean_delta_ssim:+.6f}"

    )


    print(

        f"Validation score           : "
        f"{score:.8f}"

    )


    print(

        f"Epoch time                 : "
        f"{epoch_time:.2f} seconds"

    )


    print(
        "=" * 75
    )


    #Saving training history
    history.append({

        "epoch":
            epoch,

        "train_charbonnier_loss":
            mean_train_loss,

        "delta_psnr":
            mean_delta_psnr,

        "delta_ssim":
            mean_delta_ssim,

        "score":
            score,

        "epoch_time":
            epoch_time

    })


    #Checking whether this model improved validation score
    if score > best_score:

        #Updating best score
        best_score = score


        #Saving improved model
        torch.save(

            model.state_dict(),

            OUTPUT_DIR

            / "best_robust_dncnn.pth"

        )


        print(
            "New best robust DnCNN saved."
        )


#Saving fine tuning history
history_df = pd.DataFrame(
    history
)


history_df.to_csv(

    OUTPUT_DIR

    / "robust_finetuning_history.csv",

    index=False

)


print()

print(
    "=" * 75
)

print(
    "ROBUST DNCNN FINE TUNING COMPLETE"
)

print(
    "=" * 75
)


print(

    "Original validation score :",
    f"{original_score:.8f}"

)


print(

    "Best robust score         :",
    f"{best_score:.8f}"

)


print()

print(
    "Best robust model:"
)


print(

    OUTPUT_DIR

    / "best_robust_dncnn.pth"

)


print(
    "=" * 75
)