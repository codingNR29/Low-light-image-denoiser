from pathlib import Path
import shutil
import pandas as pd


#Getting project root
PROJECT_ROOT = Path(
    __file__
).resolve().parent


#Finding validation split CSV
validation_csv_files = list(

    PROJECT_ROOT.rglob(
        "validation_images.csv"
    )

)


if len(validation_csv_files) == 0:

    raise FileNotFoundError(
        "validation_images.csv could not be found"
    )


VALIDATION_CSV = validation_csv_files[0]


#Public noisy images
NOISY_SOURCE = (

    PROJECT_ROOT

    / "Competition repo"

    / "mora_sp_cup_2026"

    / "competition_data"

    / "public"

    / "noisy"

)


#Public ground truth images
GT_SOURCE = (

    PROJECT_ROOT

    / "Competition repo"

    / "mora_sp_cup_2026"

    / "competition_data"

    / "public"

    / "ground_truth"

)


#Final outputs generated using denoise.py
DENOISED_SOURCE = (

    PROJECT_ROOT

    / "final_test"

    / "public_460"

)


#Creating validation folders
VALIDATION_ROOT = (

    PROJECT_ROOT

    / "final_test"

    / "validation_92"

)


NOISY_DESTINATION = (

    VALIDATION_ROOT

    / "noisy"

)


GT_DESTINATION = (

    VALIDATION_ROOT

    / "ground_truth"

)


DENOISED_DESTINATION = (

    VALIDATION_ROOT

    / "denoised"

)


#Creating folders
NOISY_DESTINATION.mkdir(
    parents=True,
    exist_ok=True
)


GT_DESTINATION.mkdir(
    parents=True,
    exist_ok=True
)


DENOISED_DESTINATION.mkdir(
    parents=True,
    exist_ok=True
)


#Loading validation image IDs
validation_df = pd.read_csv(
    VALIDATION_CSV
)


print(
    "Validation CSV:"
)

print(
    VALIDATION_CSV
)


print()

print(
    "Validation images:",
    len(validation_df)
)


#Copying 92 validation images
for count, row in enumerate(

    validation_df.itertuples(),

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


    #Copying noisy image
    shutil.copy2(

        NOISY_SOURCE
        / f"{image_id}_noise.png",

        NOISY_DESTINATION
        / f"{image_id}_noise.png"

    )


    #Copying ground truth image
    shutil.copy2(

        GT_SOURCE
        / f"{image_id}.png",

        GT_DESTINATION
        / f"{image_id}.png"

    )


    #Copying final denoised result
    shutil.copy2(

        DENOISED_SOURCE
        / f"{image_id}.png",

        DENOISED_DESTINATION
        / f"{image_id}.png"

    )


    #Printing progress
    if (

        count % 20 == 0

        or

        count == len(
            validation_df
        )

    ):

        print(

            f"Copied "
            f"{count}/"
            f"{len(validation_df)}"

        )


print()

print(
    "=" * 70
)

print(
    "VALIDATION 92 DATASET READY"
)

print(
    "=" * 70
)


print(
    "Noisy folder:"
)

print(
    NOISY_DESTINATION
)


print()

print(
    "Ground truth folder:"
)

print(
    GT_DESTINATION
)


print()

print(
    "Denoised folder:"
)

print(
    DENOISED_DESTINATION
)