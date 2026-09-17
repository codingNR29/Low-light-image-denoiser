from pathlib import Path
import time

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

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


#Competition repository
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


#Training CSV
TRAIN_CSV = (

    SCRIPT_DIR

    / "dataset_split"

    / "train_images.csv"

)


#Validation CSV
VAL_CSV = (

    SCRIPT_DIR

    / "dataset_split"

    / "validation_images.csv"

)


#Creating output directory
OUTPUT_DIR = (

    SCRIPT_DIR

    / "dncnn_training"

)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


#Training settings
PATCH_SIZE = 64

PATCHES_PER_IMAGE = 16

BATCH_SIZE = 8

NUM_EPOCHS = 20

LEARNING_RATE = 0.001


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


#Creating training dataset
train_dataset = DenoisingPatchDataset(

    csv_path=TRAIN_CSV,

    noisy_dir=NOISY_DIR,

    gt_dir=GT_DIR,

    patch_size=PATCH_SIZE,

    patches_per_image=PATCHES_PER_IMAGE,

    training=True

)


#Creating validation dataset
val_dataset = DenoisingPatchDataset(

    csv_path=VAL_CSV,

    noisy_dir=NOISY_DIR,

    gt_dir=GT_DIR,

    patch_size=PATCH_SIZE,

    patches_per_image=1,

    training=False

)


#Creating training data loader
train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0,

    pin_memory=True

)


#Creating validation data loader
val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=True

)


print(
    "Training patches:",
    len(train_dataset)
)


print(
    "Validation patches:",
    len(val_dataset)
)


print(
    "Training batches:",
    len(train_loader)
)


print(
    "Validation batches:",
    len(val_loader)
)


#Creating DnCNN model
model = DnCNN(

    input_channels=3,

    output_channels=3,

    num_features=64,

    num_layers=17

)


#Moving model to GPU
model = model.to(
    device
)


#Using Mean Squared Error loss
criterion = nn.MSELoss()


#Using Adam optimizer
optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE

)


#Tracking best validation loss
best_val_loss = float(
    "inf"
)


#Training loop
for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    #Starting epoch timer
    epoch_start = time.perf_counter()


    #Setting model to training mode
    model.train()


    #Tracking total training loss
    running_train_loss = 0.0


    #Going through training batches
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


        #Clearing gradients from previous batch
        optimizer.zero_grad()


        #Predicting residual noise
        predicted_residual = model(
            noisy
        )


        #Calculating residual prediction error
        loss = criterion(

            predicted_residual,

            true_residual

        )


        #Calculating gradients
        loss.backward()


        #Updating model parameters
        optimizer.step()


        #Adding batch loss
        running_train_loss += (
            loss.item()
        )


        #Printing progress every 100 batches
        if batch_number % 100 == 0:

            print(

                f"Epoch {epoch:02d} | "

                f"Batch "
                f"{batch_number}/"
                f"{len(train_loader)} | "

                f"Loss "
                f"{loss.item():.6f}"

            )


    #Calculating average training loss
    mean_train_loss = (

        running_train_loss

        / len(train_loader)

    )


    #Setting model to evaluation mode
    model.eval()


    #Tracking validation loss
    running_val_loss = 0.0


    #Disabling gradient calculation for validation
    with torch.no_grad():

        #Going through validation batches
        for batch in val_loader:

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


            #Predicting residual
            predicted_residual = model(
                noisy
            )


            #Calculating validation loss
            val_loss = criterion(

                predicted_residual,

                true_residual

            )


            #Adding validation loss
            running_val_loss += (
                val_loss.item()
            )


    #Calculating mean validation loss
    mean_val_loss = (

        running_val_loss

        / len(val_loader)

    )


    #Calculating epoch time
    epoch_time = (

        time.perf_counter()

        - epoch_start

    )


    print()
    print(
        "=" * 70
    )

    print(

        f"Epoch "
        f"{epoch:02d}/"
        f"{NUM_EPOCHS}"

    )


    print(

        f"Training loss   : "
        f"{mean_train_loss:.8f}"

    )


    print(

        f"Validation loss : "
        f"{mean_val_loss:.8f}"

    )


    print(

        f"Epoch time      : "
        f"{epoch_time:.2f} seconds"

    )


    print(
        "=" * 70
    )


    #Saving best model
    if mean_val_loss < best_val_loss:

        #Updating best validation loss
        best_val_loss = (
            mean_val_loss
        )


        #Saving model checkpoint
        torch.save(

            model.state_dict(),

            OUTPUT_DIR
            / "best_dncnn.pth"

        )


        print(
            "Best model updated and saved."
        )


    #Saving latest model after every epoch
    torch.save(

        model.state_dict(),

        OUTPUT_DIR
        / "latest_dncnn.pth"

    )


#Training finished
print()
print(
    "=" * 70
)

print(
    "DNCNN TRAINING COMPLETE"
)

print(
    "=" * 70
)


print(

    "Best validation loss:",

    best_val_loss

)


print(

    "Best model saved to:",

    OUTPUT_DIR
    / "best_dncnn.pth"

)