from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis, norm


# ============================================================
# 1. PATHS
# ============================================================

GT_DIR = Path(
    r"F:\Mora SP Cup 2026\Competition repo\mora_sp_cup_2026"
    r"\competition_data\public\ground_truth"
)

NOISY_DIR = Path(
    r"F:\Mora SP Cup 2026\Competition repo\mora_sp_cup_2026"
    r"\competition_data\public\noisy"
)

OUTPUT_DIR = Path("standardized_residual_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. FITTED BIAS MODEL COEFFICIENTS
#
# Bias:
# mu(x) = a*x^3 + b*x^2 + c*x + d
# ============================================================

BIAS_COEFFS = {

    "R": (
        -0.00000741,
         0.002352,
        -0.2294,
         8.7506
    ),

    "G": (
        -0.00000647,
         0.002031,
        -0.2015,
         8.0011
    ),

    "B": (
        -0.00000749,
         0.002372,
        -0.2316,
         8.7703
    )
}


# ============================================================
# 3. FITTED VARIANCE MODEL COEFFICIENTS
#
# Variance:
# sigma^2(x) = a*x^3 + b*x^2 + c*x + d
# ============================================================

VARIANCE_COEFFS = {

    "R": (
        -0.00029456,
         0.063619,
         4.0171,
         249.0226
    ),

    "G": (
        -0.00030771,
         0.071134,
         2.9367,
         223.3979
    ),

    "B": (
        -0.00030599,
         0.066128,
         4.0396,
         239.4319
    )
}


# ============================================================
# 4. MODEL FUNCTIONS
# ============================================================

def cubic_model(x, coefficients):

    # Important:
    # convert before x^2 and x^3 to avoid uint8 overflow
    x = x.astype(np.float64)

    a, b, c, d = coefficients

    return (
        a * x**3
        + b * x**2
        + c * x
        + d
    )


def calculate_bias(x, channel):

    return cubic_model(
        x,
        BIAS_COEFFS[channel]
    )


def calculate_variance(x, channel):

    variance = cubic_model(
        x,
        VARIANCE_COEFFS[channel]
    )

    # Safety:
    # variance must always be positive
    variance = np.maximum(
        variance,
        1e-6
    )

    return variance


# ============================================================
# 5. STANDARDIZE ONE CHANNEL
#
# z = (r - mu(x)) / sigma(x)
# ============================================================

def standardize_residual(
    gt_channel,
    residual_channel,
    channel
):

    # Predicted bias
    mu = calculate_bias(
        gt_channel,
        channel
    )

    # Predicted variance
    variance = calculate_variance(
        gt_channel,
        channel
    )

    # Standard deviation
    sigma = np.sqrt(
        variance
    )

    # Standardized residual
    z = (
        residual_channel
        - mu
    ) / sigma

    return z


# ============================================================
# 6. PROCESS ONE IMAGE PAIR
# ============================================================

def analyse_pair(
    gt_path,
    noisy_path,
    rng,
    sample_size=2000
):

    gt = cv.imread(
        str(gt_path)
    )

    noisy = cv.imread(
        str(noisy_path)
    )

    if gt is None or noisy is None:

        raise ValueError(
            f"Could not load "
            f"{gt_path.name} or "
            f"{noisy_path.name}"
        )


    # BGR -> RGB
    gt = cv.cvtColor(
        gt,
        cv.COLOR_BGR2RGB
    )

    noisy = cv.cvtColor(
        noisy,
        cv.COLOR_BGR2RGB
    )


    # Float conversion
    gt_f = gt.astype(
        np.float64
    )

    noisy_f = noisy.astype(
        np.float64
    )


    # Raw residual
    residual = (
        noisy_f
        - gt_f
    )


    row = {
        "image": gt_path.stem
    }


    channel_names = [
        "R",
        "G",
        "B"
    ]


    # Store sampled standardized pixels
    samples = {}


    for c, channel in enumerate(
        channel_names
    ):

        gt_channel = gt_f[
            :, :, c
        ]

        residual_channel = residual[
            :, :, c
        ]


        # ------------------------------------
        # Standardize residual
        # ------------------------------------

        z = standardize_residual(
            gt_channel,
            residual_channel,
            channel
        )


        z_flat = z.ravel()


        # ------------------------------------
        # Statistics for this image
        # ------------------------------------

        row[
            f"{channel}_z_mean"
        ] = np.mean(z_flat)

        row[
            f"{channel}_z_std"
        ] = np.std(z_flat)

        row[
            f"{channel}_z_skew"
        ] = skew(z_flat)

        row[
            f"{channel}_z_kurtosis"
        ] = kurtosis(z_flat)


        # ------------------------------------
        # Random sample for global histogram
        # ------------------------------------

        n_samples = min(
            sample_size,
            z_flat.size
        )

        indices = rng.choice(
            z_flat.size,
            size=n_samples,
            replace=False
        )

        samples[channel] = (
            z_flat[indices]
            .astype(np.float32)
        )


    return row, samples


# ============================================================
# 7. PROCESS ALL 460 IMAGES
# ============================================================

rows = []

# Random samples collected from all images
global_samples = {
    "R": [],
    "G": [],
    "B": []
}


# Fixed seed so results are reproducible
rng = np.random.default_rng(
    seed=42
)


for i in range(
    1,
    461
):

    image_id = f"{i:03d}"


    gt_path = (
        GT_DIR
        / f"{image_id}.png"
    )

    noisy_path = (
        NOISY_DIR
        / f"{image_id}_noise.png"
    )


    row, samples = analyse_pair(
        gt_path,
        noisy_path,
        rng
    )


    rows.append(row)


    for channel in [
        "R",
        "G",
        "B"
    ]:

        global_samples[
            channel
        ].append(
            samples[channel]
        )


    print(
        f"Processed "
        f"{image_id}/460"
    )


# ============================================================
# 8. SAVE PER-IMAGE STATISTICS
# ============================================================

statistics_df = pd.DataFrame(
    rows
)


statistics_path = (
    OUTPUT_DIR
    / "standardized_residual_statistics.csv"
)


statistics_df.to_csv(
    statistics_path,
    index=False
)


print()
print(
    "Saved:",
    statistics_path
)


# ============================================================
# 9. COMBINE RANDOM PIXEL SAMPLES
# ============================================================

for channel in [
    "R",
    "G",
    "B"
]:

    global_samples[channel] = (
        np.concatenate(
            global_samples[channel]
        )
    )


# ============================================================
# 10. DATASET-WIDE STANDARDIZED STATISTICS
# ============================================================

summary_rows = []


for channel in [
    "R",
    "G",
    "B"
]:

    z = global_samples[channel]


    mean_value = np.mean(z)
    std_value = np.std(z)
    skew_value = skew(z)
    kurt_value = kurtosis(z)


    summary_rows.append({

        "channel":
            channel,

        "mean":
            mean_value,

        "std":
            std_value,

        "skewness":
            skew_value,

        "kurtosis":
            kurt_value
    })


    print()
    print(
        f"{channel} channel"
    )

    print(
        "Mean      :",
        mean_value
    )

    print(
        "Std       :",
        std_value
    )

    print(
        "Skewness  :",
        skew_value
    )

    print(
        "Kurtosis  :",
        kurt_value
    )


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    OUTPUT_DIR
    / "dataset_standardized_summary.csv",
    index=False
)


# ============================================================
# 11. SAVE SAMPLED STANDARDIZED RESIDUALS
#
# Useful later without reprocessing images
# ============================================================

np.savez_compressed(

    OUTPUT_DIR
    / "standardized_residual_samples.npz",

    R=global_samples["R"],

    G=global_samples["G"],

    B=global_samples["B"]
)


# ============================================================
# 12. DATASET-WIDE HISTOGRAMS
# ============================================================

x_axis = np.linspace(
    -5,
    5,
    500
)


for channel in [
    "R",
    "G",
    "B"
]:

    z = global_samples[
        channel
    ]


    plt.figure(
        figsize=(9, 5)
    )


    plt.hist(
        z,
        bins=120,
        density=True,
        alpha=0.7,
        label=(
            f"{channel} standardized residual"
        )
    )


    # Ideal standard Gaussian
    plt.plot(
        x_axis,
        norm.pdf(
            x_axis,
            0,
            1
        ),
        label="Standard Normal N(0,1)"
    )


    plt.xlabel(
        "Standardized Residual"
    )

    plt.ylabel(
        "Density"
    )

    plt.title(
        f"Dataset-Wide Standardized "
        f"Residual - {channel} Channel"
    )


    plt.legend()

    plt.grid()

    plt.tight_layout()


    plt.savefig(
        OUTPUT_DIR
        / (
            f"{channel}_standardized_"
            f"residual_histogram.png"
        ),
        dpi=300
    )


    plt.show()


# ============================================================
# 13. STANDARDIZED STD ACROSS IMAGES
#
# This helps detect remaining image-level severity.
# ============================================================

plt.figure(
    figsize=(12, 6)
)


for channel in [
    "R",
    "G",
    "B"
]:

    plt.plot(
        statistics_df[
            f"{channel}_z_std"
        ].values,

        label=channel
    )


plt.axhline(
    1,
    linestyle="--",
    label="Ideal standardized std = 1"
)


plt.xlabel(
    "Image Index"
)

plt.ylabel(
    "Standardized Residual Std"
)

plt.title(
    "Standardized Noise Strength "
    "Across 460 Images"
)

plt.legend()

plt.grid()

plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "standardized_std_across_images.png",
    dpi=300
)


plt.show()