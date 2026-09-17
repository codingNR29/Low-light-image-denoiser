from pathlib import Path
import json

import numpy as np
import pandas as pd

from PIL import Image

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.mixture import GaussianMixture

import joblib


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


#Previously generated optimized wavelet outputs
WAVELET_DIR = (

    PROJECT_ROOT

    / "My implementation"

    / "Restoration"

    / "optimized_wavelet_full_460"

    / "denoised"

)


#True severity values obtained during noise characterization
SEVERITY_CSV = (

    PROJECT_ROOT

    / "Noise_characterization"

    / "noise_severity_analysis"

    / "noise_severity_statistics.csv"

)


#Training split
TRAIN_CSV = (

    SCRIPT_DIR

    / "dataset_split"

    / "train_images.csv"

)


#Validation split
VAL_CSV = (

    SCRIPT_DIR

    / "dataset_split"

    / "validation_images.csv"

)


#Creating output directory
OUTPUT_DIR = (

    SCRIPT_DIR

    / "noise_severity_estimator"

)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


#Sampling every fourth pixel
#This makes the calculation much faster
#while still using many thousands of pixels per image
SAMPLE_STRIDE = 4


#Feature names used by the regression model
FEATURE_NAMES = [

    "robust_scale",

    "mean_abs_z",

    "p90_abs_z",

    "p95_abs_z",

    "positive_tail_fraction",

    "negative_tail_fraction"

]


#Signal dependent bias model
def calculate_bias(
    intensity,
    channel
):

    #Red channel bias model
    if channel == 0:

        return (

            -7.41e-06
            * intensity ** 3

            +

            0.002352
            * intensity ** 2

            -

            0.2294
            * intensity

            +

            8.7506

        )


    #Green channel bias model
    elif channel == 1:

        return (

            -6.47e-06
            * intensity ** 3

            +

            0.002031
            * intensity ** 2

            -

            0.2015
            * intensity

            +

            8.0011

        )


    #Blue channel bias model
    else:

        return (

            -7.49e-06
            * intensity ** 3

            +

            0.002372
            * intensity ** 2

            -

            0.2316
            * intensity

            +

            8.7703

        )


#Signal dependent variance model
def calculate_variance(
    intensity,
    channel
):

    #Red channel variance model
    if channel == 0:

        variance = (

            -0.00029456
            * intensity ** 3

            +

            0.063619
            * intensity ** 2

            +

            4.0171
            * intensity

            +

            249.0226

        )


    #Green channel variance model
    elif channel == 1:

        variance = (

            -0.00030771
            * intensity ** 3

            +

            0.071134
            * intensity ** 2

            +

            2.9367
            * intensity

            +

            223.3979

        )


    #Blue channel variance model
    else:

        variance = (

            -0.00030599
            * intensity ** 3

            +

            0.066128
            * intensity ** 2

            +

            4.0396
            * intensity

            +

            239.4319

        )


    #Preventing very small or invalid variance values
    variance = np.maximum(

        variance,

        1.0

    )


    return variance


#Extracting noise severity features from one noisy image
def extract_noise_features(
    noisy,
    wavelet
):

    #Sampling pixels to make calculations faster
    noisy_sample = noisy[
        ::SAMPLE_STRIDE,
        ::SAMPLE_STRIDE,
        :
    ].astype(
        np.float32
    )


    #Using wavelet output as approximate clean signal
    wavelet_sample = wavelet[
        ::SAMPLE_STRIDE,
        ::SAMPLE_STRIDE,
        :
    ].astype(
        np.float32
    )


    #Approximate residual
    residual = (

        noisy_sample

        -

        wavelet_sample

    )


    #Storage for normalized residual channels
    normalized_channels = []


    #Storage for robust scale from each RGB channel
    channel_scales = []


    #Processing R, G and B independently
    for channel in range(
        3
    ):

        #Using wavelet estimate as approximate clean intensity
        intensity = wavelet_sample[
            :,
            :,
            channel
        ]


        #Getting approximate residual for this channel
        channel_residual = residual[
            :,
            :,
            channel
        ]


        #Calculating expected signal dependent bias
        bias = calculate_bias(

            intensity,

            channel

        )


        #Calculating expected signal dependent variance
        variance = calculate_variance(

            intensity,

            channel

        )


        #Converting variance into standard deviation
        sigma = np.sqrt(
            variance
        )


        #Normalizing the residual
        normalized = (

            channel_residual

            -

            bias

        ) / sigma


        #Saving normalized residual
        normalized_channels.append(
            normalized
        )


        #Calculating channel median
        channel_median = np.median(
            normalized
        )


        #Calculating Median Absolute Deviation
        mad = np.median(

            np.abs(

                normalized

                -

                channel_median

            )

        )


        #Converting MAD into robust standard deviation estimate
        robust_scale = (

            mad

            / 0.67448975

        )


        channel_scales.append(
            robust_scale
        )


    #Combining normalized RGB residuals
    normalized_all = np.stack(

        normalized_channels,

        axis=-1

    )


    #Flattening all normalized residual values
    normalized_flat = (
        normalized_all
        .reshape(-1)
    )


    #Absolute normalized residual
    abs_normalized = np.abs(
        normalized_flat
    )


    #Median robust scale across RGB
    robust_scale = np.median(
        channel_scales
    )


    #Mean absolute normalized residual
    mean_abs_z = np.mean(
        abs_normalized
    )


    #90th percentile
    p90_abs_z = np.percentile(

        abs_normalized,

        90

    )


    #95th percentile
    p95_abs_z = np.percentile(

        abs_normalized,

        95

    )


    #Fraction of strong positive outliers
    positive_tail_fraction = np.mean(

        normalized_flat
        > 3.0

    )


    #Fraction of strong negative outliers
    negative_tail_fraction = np.mean(

        normalized_flat
        < -3.0

    )


    #Returning feature dictionary
    return {

        "robust_scale":
            robust_scale,

        "mean_abs_z":
            mean_abs_z,

        "p90_abs_z":
            p90_abs_z,

        "p95_abs_z":
            p95_abs_z,

        "positive_tail_fraction":
            positive_tail_fraction,

        "negative_tail_fraction":
            negative_tail_fraction

    }


#Loading true severity information
severity_df = pd.read_csv(
    SEVERITY_CSV
)


#Converting image number to integer
severity_df["image"] = (

    severity_df["image"]
    .astype(int)

)


#Loading train and validation image lists
train_df = pd.read_csv(
    TRAIN_CSV
)


val_df = pd.read_csv(
    VAL_CSV
)


train_ids = set(

    train_df[
        "image"
    ].astype(int)

)


val_ids = set(

    val_df[
        "image"
    ].astype(int)

)


#Storage for extracted features
feature_rows = []


print()

print(
    "=" * 75
)

print(
    "EXTRACTING NOISE-SEVERITY FEATURES"
)

print(
    "=" * 75
)


#Processing all 460 public images
for count, row in enumerate(

    severity_df.itertuples(),

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


    #Creating optimized wavelet image path
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


    #Loading wavelet image
    wavelet = np.asarray(

        Image.open(
            wavelet_path
        ).convert("RGB")

    )


    #Extracting noisy-only features
    features = extract_noise_features(

        noisy,

        wavelet

    )


    #Determining dataset split
    if image_number in train_ids:

        split = "train"

    elif image_number in val_ids:

        split = "validation"

    else:

        split = "unknown"


    #Saving feature information
    feature_rows.append({

        "image":
            image_number,

        "split":
            split,

        "true_severity":
            row.severity,

        **features

    })


    #Printing progress
    if (

        count % 25 == 0

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


#Creating feature dataframe
features_df = pd.DataFrame(
    feature_rows
)


#Saving extracted features
features_df.to_csv(

    OUTPUT_DIR

    / "noise_severity_features.csv",

    index=False

)


#Separating training rows
train_features_df = features_df[

    features_df[
        "split"
    ]
    == "train"

].copy()


#Separating validation rows
val_features_df = features_df[

    features_df[
        "split"
    ]
    == "validation"

].copy()


#Creating training feature matrix
X_train = train_features_df[
    FEATURE_NAMES
].values


#Creating training severity targets
y_train = train_features_df[
    "true_severity"
].values


#Creating validation feature matrix
X_val = val_features_df[
    FEATURE_NAMES
].values


#Creating validation severity targets
y_val = val_features_df[
    "true_severity"
].values


#Creating regression pipeline
#StandardScaler normalizes features
#Ridge learns the relationship to severity
model = Pipeline(

    [

        (
            "scaler",

            StandardScaler()
        ),

        (
            "ridge",

            Ridge(
                alpha=1.0
            )
        )

    ]

)


#Training severity estimator using only 368 training images
model.fit(

    X_train,

    y_train

)


#Predicting training severity
train_prediction = model.predict(
    X_train
)


#Predicting validation severity
val_prediction = model.predict(
    X_val
)


#Saving predictions
train_features_df[
    "predicted_severity"
] = train_prediction


val_features_df[
    "predicted_severity"
] = val_prediction


#Calculating training correlation
train_correlation = np.corrcoef(

    y_train,

    train_prediction

)[0, 1]


#Calculating validation correlation
val_correlation = np.corrcoef(

    y_val,

    val_prediction

)[0, 1]


#Calculating validation MAE
val_mae = mean_absolute_error(

    y_val,

    val_prediction

)


#Calculating validation RMSE
val_rmse = np.sqrt(

    mean_squared_error(

        y_val,

        val_prediction

    )

)


#Calculating validation R squared
val_r2 = r2_score(

    y_val,

    val_prediction

)


#Creating GMM severity groups using TRAINING true severity only
gmm = GaussianMixture(

    n_components=3,

    random_state=42

)


gmm.fit(

    y_train.reshape(
        -1,
        1
    )

)


#Getting severity centers
centers = np.sort(

    gmm.means_
    .flatten()

)


#Creating simple boundaries between centers
threshold_low_medium = (

    centers[0]

    +

    centers[1]

) / 2.0


threshold_medium_high = (

    centers[1]

    +

    centers[2]

) / 2.0


#Converting continuous severity into group label
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


#Creating validation true groups
val_features_df[
    "true_group"
] = [

    severity_to_group(
        value
    )

    for value in y_val

]


#Creating predicted groups
val_features_df[
    "predicted_group"
] = [

    severity_to_group(
        value
    )

    for value in val_prediction

]


#Calculating group prediction accuracy
group_accuracy = np.mean(

    val_features_df[
        "true_group"
    ].values

    ==

    val_features_df[
        "predicted_group"
    ].values

)


#Saving train predictions
train_features_df.to_csv(

    OUTPUT_DIR

    / "train_severity_predictions.csv",

    index=False

)


#Saving validation predictions
val_features_df.to_csv(

    OUTPUT_DIR

    / "validation_severity_predictions.csv",

    index=False

)


#Saving trained estimator
joblib.dump(

    model,

    OUTPUT_DIR

    / "severity_estimator.joblib"

)


#Saving estimator metadata
metadata = {

    "feature_names":
        FEATURE_NAMES,

    "sample_stride":
        SAMPLE_STRIDE,

    "severity_centers":
        centers.tolist(),

    "threshold_low_medium":
        float(
            threshold_low_medium
        ),

    "threshold_medium_high":
        float(
            threshold_medium_high
        ),

    "train_correlation":
        float(
            train_correlation
        ),

    "validation_correlation":
        float(
            val_correlation
        ),

    "validation_mae":
        float(
            val_mae
        ),

    "validation_rmse":
        float(
            val_rmse
        ),

    "validation_r2":
        float(
            val_r2
        ),

    "validation_group_accuracy":
        float(
            group_accuracy
        )

}


with open(

    OUTPUT_DIR

    / "severity_estimator_metadata.json",

    "w"

) as file:

    json.dump(

        metadata,

        file,

        indent=4

    )


#Printing final results
print()

print(
    "=" * 80
)

print(
    "NOISE-SEVERITY ESTIMATOR RESULTS"
)

print(
    "=" * 80
)


print()

print(
    "Training images:",
    len(train_features_df)
)


print(
    "Validation images:",
    len(val_features_df)
)


print()

print(
    "Severity centers:"
)


print(

    f"Low    : "
    f"{centers[0]:.4f}"

)


print(

    f"Medium : "
    f"{centers[1]:.4f}"

)


print(

    f"High   : "
    f"{centers[2]:.4f}"

)


print()

print(

    f"Low/Medium threshold : "
    f"{threshold_low_medium:.4f}"

)


print(

    f"Medium/High threshold: "
    f"{threshold_medium_high:.4f}"

)


print()

print(

    f"Training correlation   : "
    f"{train_correlation:.4f}"

)


print(

    f"Validation correlation : "
    f"{val_correlation:.4f}"

)


print(

    f"Validation MAE         : "
    f"{val_mae:.4f}"

)


print(

    f"Validation RMSE        : "
    f"{val_rmse:.4f}"

)


print(

    f"Validation R2          : "
    f"{val_r2:.4f}"

)


print(

    f"Severity group accuracy: "
    f"{group_accuracy * 100:.2f}%"

)


print()

print(
    "Results saved to:"
)


print(
    OUTPUT_DIR
)


print(
    "=" * 80
)