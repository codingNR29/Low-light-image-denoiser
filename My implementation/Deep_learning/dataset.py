from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image

import torch
from torch.utils.data import Dataset


#Dataset class
class DenoisingPatchDataset(Dataset):

    def __init__(
        self,
        csv_path,
        noisy_dir,
        gt_dir,
        patch_size=64,
        patches_per_image=16,
        training=True
    ):

        #Loading image IDs from split CSV
        self.df = pd.read_csv(
            csv_path
        )


        #Saving noisy image directory
        self.noisy_dir = Path(
            noisy_dir
        )


        #Saving ground truth image directory
        self.gt_dir = Path(
            gt_dir
        )


        #Size of one training patch
        self.patch_size = patch_size


        #Number of random patches generated from each image per epoch
        self.patches_per_image = (
            patches_per_image
        )


        #Whether data augmentation should be applied
        self.training = training


    #Dataset length
    def __len__(
        self
    ):

        return (
            len(self.df)
            *
            self.patches_per_image
        )


    #Loading one image pair
    def load_pair(
        self,
        image_number
    ):

        #Converting image number to 3 digit format
        image_id = (
            f"{int(image_number):03d}"
        )


        #Noisy image path
        noisy_path = (

            self.noisy_dir
            / f"{image_id}_noise.png"

        )


        #Ground truth image path
        gt_path = (

            self.gt_dir
            / f"{image_id}.png"

        )


        #Loading noisy image as RGB float32
        noisy = np.asarray(

            Image.open(
                noisy_path
            ).convert("RGB"),

            dtype=np.float32

        )


        #Loading ground truth image as RGB float32
        clean = np.asarray(

            Image.open(
                gt_path
            ).convert("RGB"),

            dtype=np.float32

        )


        #Normalizing noisy image from [0,255] to [0,1]
        noisy = (
            noisy
            / 255.0
        )


        #Normalizing clean image from [0,255] to [0,1]
        clean = (
            clean
            / 255.0
        )


        return (
            noisy,
            clean
        )


    #Random patch extraction
    def random_crop(
        self,
        noisy,
        clean
    ):

        #Getting image height and width
        height, width, _ = (
            noisy.shape
        )


        #Maximum possible starting y coordinate
        max_y = (
            height
            - self.patch_size
        )


        #Maximum possible starting x coordinate
        max_x = (
            width
            - self.patch_size
        )


        #Selecting random y coordinate
        y = np.random.randint(
            0,
            max_y + 1
        )


        #Selecting random x coordinate
        x = np.random.randint(
            0,
            max_x + 1
        )


        #Extracting noisy patch
        noisy_patch = noisy[

            y:
            y + self.patch_size,

            x:
            x + self.patch_size,

            :

        ]


        #Extracting clean patch from the same location
        clean_patch = clean[

            y:
            y + self.patch_size,

            x:
            x + self.patch_size,

            :

        ]


        return (
            noisy_patch,
            clean_patch
        )

    #Center patch extraction for validation
    def center_crop(
        self,
        noisy,
        clean
    ):

        #Getting image height and width
        height, width, _ = (
            noisy.shape
        )


        #Finding center starting y coordinate
        y = (
            height
            - self.patch_size
        ) // 2


        #Finding center starting x coordinate
        x = (
            width
            - self.patch_size
        ) // 2


        #Extracting noisy center patch
        noisy_patch = noisy[

            y:
            y + self.patch_size,

            x:
            x + self.patch_size,

            :

        ]


        #Extracting clean center patch
        clean_patch = clean[

            y:
            y + self.patch_size,

            x:
            x + self.patch_size,

            :

        ]


        return (
            noisy_patch,
            clean_patch
        )

    #Data augmentation
    def augment(
        self,
        noisy,
        clean
    ):

        #Horizontal flip
        if np.random.rand() < 0.5:

            noisy = np.flip(
                noisy,
                axis=1
            )


            clean = np.flip(
                clean,
                axis=1
            )


        #Vertical flip
        if np.random.rand() < 0.5:

            noisy = np.flip(
                noisy,
                axis=0
            )


            clean = np.flip(
                clean,
                axis=0
            )


        #Random 90 degree rotation
        k = np.random.randint(
            0,
            4
        )


        noisy = np.rot90(
            noisy,
            k
        )


        clean = np.rot90(
            clean,
            k
        )


        #Making copies because flip and rotation can create
        #arrays with negative memory strides
        return (
            noisy.copy(),
            clean.copy()
        )


    #Getting one training example
    def __getitem__(
        self,
        index
    ):

        #Determining which original image this patch belongs to
        image_index = (

            index

            // self.patches_per_image

        )


        #Getting image number from CSV
        image_number = (

            self.df.iloc[
                image_index
            ]["image"]

        )


        #Loading noisy and clean image pair
        noisy, clean = (
            self.load_pair(
                image_number
            )
        )


        #Using random patches during training
        if self.training:

            noisy_patch, clean_patch = (
                self.random_crop(
                    noisy,
                    clean
                )
            )


        #Using fixed center patches during validation
        else:

            noisy_patch, clean_patch = (
                self.center_crop(
                    noisy,
                    clean
                )
            )


        #Applying data augmentation only during training
        if self.training:

            noisy_patch, clean_patch = (
                self.augment(
                    noisy_patch,
                    clean_patch
                )
            )


        #Creating residual target
        #Residual = Noisy - Clean
        residual_patch = (

            noisy_patch
            -
            clean_patch

        )


        #Changing noisy patch from H x W x C
        #to C x H x W for PyTorch
        noisy_patch = np.transpose(

            noisy_patch,

            (2, 0, 1)

        )


        #Changing clean patch from H x W x C
        #to C x H x W for PyTorch
        clean_patch = np.transpose(

            clean_patch,

            (2, 0, 1)

        )


        #Changing residual patch from H x W x C
        #to C x H x W for PyTorch
        residual_patch = np.transpose(

            residual_patch,

            (2, 0, 1)

        )


        #Converting noisy patch into PyTorch tensor
        noisy_tensor = torch.from_numpy(

            noisy_patch.copy()

        ).float()


        #Converting clean patch into PyTorch tensor
        clean_tensor = torch.from_numpy(

            clean_patch.copy()

        ).float()


        #Converting residual patch into PyTorch tensor
        residual_tensor = torch.from_numpy(

            residual_patch.copy()

        ).float()


        #Returning one training sample
        return {

            "noisy":
                noisy_tensor,

            "clean":
                clean_tensor,

            "residual":
                residual_tensor,

            "image_id":
                f"{int(image_number):03d}"

        }


#Testing dataset
if __name__ == "__main__":

    #Getting location of this Python file
    SCRIPT_DIR = Path(
        __file__
    ).resolve().parent


    #Getting main Mora SP Cup 2026 project folder
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


    #Training split CSV
    TRAIN_CSV = (

        SCRIPT_DIR

        / "dataset_split"

        / "train_images.csv"

    )


    #Creating dataset
    dataset = DenoisingPatchDataset(

        csv_path=TRAIN_CSV,

        noisy_dir=NOISY_DIR,

        gt_dir=GT_DIR,

        patch_size=64,

        patches_per_image=16,

        training=True

    )


    #Printing total number of training patches per epoch
    print(
        "Dataset length:",
        len(dataset)
    )


    #Getting the first sample
    sample = dataset[0]


    #Printing image ID
    print(
        "Image ID:",
        sample["image_id"]
    )


    #Printing noisy tensor shape
    print(
        "Noisy shape:",
        sample["noisy"].shape
    )


    #Printing clean tensor shape
    print(
        "Clean shape:",
        sample["clean"].shape
    )


    #Printing residual tensor shape
    print(
        "Residual shape:",
        sample["residual"].shape
    )


    #Checking normalized noisy image range
    print(
        "Noisy min/max:",
        sample["noisy"].min().item(),
        sample["noisy"].max().item()
    )


    #Checking residual range
    print(
        "Residual min/max:",
        sample["residual"].min().item(),
        sample["residual"].max().item()
    )