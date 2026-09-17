from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

#Selecting paths
SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parents[1]


SEVERITY_CSV = (
    PROJECT_ROOT
    / "Noise_characterization"
    / "noise_severity_analysis"
    / "noise_severity_statistics.csv"
)


OUTPUT_DIR = (
    SCRIPT_DIR
    / "dataset_split"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

#Loading severity data
df = pd.read_csv(
    SEVERITY_CSV
)


df["image"] = (
    df["image"]
    .astype(int)
)


print(
    "Total images:",
    len(df)
)

#Recreating low, medium, high severity groups
severity_values = (
    df["severity"]
    .values
    .reshape(-1, 1)
)


gmm = GaussianMixture(
    n_components=3,
    random_state=42
)


raw_labels = gmm.fit_predict(
    severity_values
)


centers = (
    gmm.means_
    .flatten()
)


order = np.argsort(
    centers
)


label_map = {
    order[0]: "Low",
    order[1]: "Medium",
    order[2]: "High"
}


df["severity_group"] = [
    label_map[label]
    for label in raw_labels
]


print()
print("Severity centers:")

for name, center in zip(
    [
        "Low",
        "Medium",
        "High"
    ],
    np.sort(centers)
):

    print(
        f"{name:6s}: "
        f"{center:.4f}"
    )


print()
print("Full dataset distribution:")

print(
    df[
        "severity_group"
    ].value_counts()
)

#Train validation split
train_df, val_df = train_test_split(

    df,

    test_size=0.20,

    random_state=42,

    stratify=df[
        "severity_group"
    ]

)

#Sorting image ID s
train_df = train_df.sort_values(
    "image"
).reset_index(
    drop=True
)


val_df = val_df.sort_values(
    "image"
).reset_index(
    drop=True
)

#Adding split labels
train_df[
    "split"
] = "train"


val_df[
    "split"
] = "validation"


full_split_df = pd.concat(

    [
        train_df,
        val_df
    ],

    ignore_index=True

)


full_split_df = full_split_df.sort_values(
    "image"
).reset_index(
    drop=True
)

#Savind the CSV files
train_df.to_csv(

    OUTPUT_DIR
    / "train_images.csv",

    index=False

)


val_df.to_csv(

    OUTPUT_DIR
    / "validation_images.csv",

    index=False

)


full_split_df.to_csv(

    OUTPUT_DIR
    / "full_dataset_split.csv",

    index=False

)

#Checking for overlaps
train_ids = set(
    train_df[
        "image"
    ]
)


val_ids = set(
    val_df[
        "image"
    ]
)


overlap = (
    train_ids
    &
    val_ids
)

#Priting summary
print()
print("=" * 70)

print(
    "DEEP LEARNING DATASET SPLIT"
)

print("=" * 70)


print(
    f"Training images   : "
    f"{len(train_df)}"
)


print(
    f"Validation images : "
    f"{len(val_df)}"
)


print(
    f"Total             : "
    f"{len(train_df) + len(val_df)}"
)


print()

print(
    "TRAINING SEVERITY DISTRIBUTION"
)

print(
    train_df[
        "severity_group"
    ].value_counts()
)


print()

print(
    "VALIDATION SEVERITY DISTRIBUTION"
)

print(
    val_df[
        "severity_group"
    ].value_counts()
)


print()

print(
    "Train/validation overlap:",
    len(overlap)
)


if len(overlap) == 0:

    print(
        "✓ No image leakage detected."
    )

else:

    print(
        "WARNING: Images exist in both sets!"
    )


print()

print(
    "Files saved to:"
)

print(
    OUTPUT_DIR
)

print("=" * 70)
