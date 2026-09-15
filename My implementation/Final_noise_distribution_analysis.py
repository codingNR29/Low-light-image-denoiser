from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis, norm, probplot


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

SEVERITY_CSV = Path(
    r"noise_severity_analysis"
    r"\noise_severity_statistics.csv"
)

OUTPUT_DIR = Path(
    "final_noise_distribution_analysis"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# 2. BIAS MODEL COEFFICIENTS
#
# mu(x) = ax^3 + bx^2 + cx + d
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
# 3. VARIANCE MODEL COEFFICIENTS
#
# sigma^2(x) = ax^3 + bx^2 + cx + d
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


CHANNELS = [
    "R",
    "G",
    "B"
]


# ============================================================
# 4. CUBIC MODEL
# ============================================================

def cubic_model(x, coefficients):

    x = x.astype(
        np.float64
    )

    a, b, c, d = coefficients

    return (
        a * x**3
        + b * x**2
        + c * x
        + d
    )


# ============================================================
# 5. CALCULATE EPSILON
#
# epsilon =
#
#       residual - bias(x)
# --------------------------------
# severity * sqrt(variance(x))
#
# ============================================================

def calculate_epsilon(
    gt,
    noisy,
    severity
):

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


    epsilon = np.zeros_like(
        residual,
        dtype=np.float64
    )


    for c, channel in enumerate(
        CHANNELS
    ):

        x = gt_f[
            :, :, c
        ]

        r = residual[
            :, :, c
        ]


        # ----------------------------------
        # Intensity-dependent bias
        # ----------------------------------

        mu = cubic_model(
            x,
            BIAS_COEFFS[channel]
        )


        # ----------------------------------
        # Intensity-dependent variance
        # ----------------------------------

        variance = cubic_model(
            x,
            VARIANCE_COEFFS[channel]
        )


        # Prevent invalid sqrt
        variance = np.maximum(
            variance,
            1e-6
        )


        sigma = np.sqrt(
            variance
        )


        # ----------------------------------
        # First standardization
        # ----------------------------------

        z = (
            r - mu
        ) / sigma


        # ----------------------------------
        # Remove image-level severity
        # ----------------------------------

        epsilon[
            :, :, c
        ] = z / severity


    return epsilon


# ============================================================
# 6. LOAD SEVERITY VALUES
# ============================================================

severity_df = pd.read_csv(
    SEVERITY_CSV
)


# Convert image ID to integer
severity_df["image"] = (
    severity_df["image"]
    .astype(int)
)


severity_map = dict(
    zip(
        severity_df["image"],
        severity_df["severity"]
    )
)


# ============================================================
# 7. SETTINGS
# ============================================================

# Random pixels taken from each image.
# 5000 x 460 ≈ 2.3 million samples/channel.
SAMPLE_SIZE_PER_IMAGE = 5000


rng = np.random.default_rng(
    seed=42
)


global_samples = {

    "R": [],
    "G": [],
    "B": []
}


per_image_rows = []


# ============================================================
# 8. PROCESS ALL 460 IMAGES
# ============================================================

for i in range(
    1,
    461
):

    image_id = f"{i:03d}"


    gt_path = (
        GT_DIR
        /
        f"{image_id}.png"
    )

    noisy_path = (
        NOISY_DIR
        /
        f"{image_id}_noise.png"
    )


    gt = cv.imread(
        str(gt_path)
    )

    noisy = cv.imread(
        str(noisy_path)
    )


    if gt is None or noisy is None:

        print(
            f"Skipping {image_id}"
        )

        continue


    # OpenCV loads BGR
    # Convert to RGB
    gt = cv.cvtColor(
        gt,
        cv.COLOR_BGR2RGB
    )

    noisy = cv.cvtColor(
        noisy,
        cv.COLOR_BGR2RGB
    )


    severity = severity_map[
        i
    ]


    # ----------------------------------------------
    # Final normalized random component epsilon
    # ----------------------------------------------

    epsilon = calculate_epsilon(
        gt,
        noisy,
        severity
    )


    row = {

        "image":
            image_id,

        "severity":
            severity
    }


    # ========================================================
    # CHANNEL-WISE ANALYSIS
    # ========================================================

    for c, channel in enumerate(
        CHANNELS
    ):

        eps_channel = epsilon[
            :, :, c
        ]


        eps_flat = (
            eps_channel
            .ravel()
        )


        # ----------------------------------
        # Per-image statistics
        # ----------------------------------

        row[
            f"{channel}_mean"
        ] = np.mean(
            eps_flat
        )

        row[
            f"{channel}_std"
        ] = np.std(
            eps_flat
        )

        row[
            f"{channel}_skew"
        ] = skew(
            eps_flat
        )

        row[
            f"{channel}_kurtosis"
        ] = kurtosis(
            eps_flat
        )


        # ----------------------------------
        # Random sample for global analysis
        # ----------------------------------

        n_samples = min(
            SAMPLE_SIZE_PER_IMAGE,
            eps_flat.size
        )


        indices = rng.choice(
            eps_flat.size,
            size=n_samples,
            replace=False
        )


        samples = eps_flat[
            indices
        ]


        global_samples[
            channel
        ].append(
            samples.astype(
                np.float32
            )
        )


    per_image_rows.append(
        row
    )


    print(
        f"Processed {image_id}/460"
    )


# ============================================================
# 9. SAVE PER-IMAGE RESULTS
# ============================================================

per_image_df = pd.DataFrame(
    per_image_rows
)


per_image_df.to_csv(

    OUTPUT_DIR
    /
    "final_normalized_statistics_per_image.csv",

    index=False
)


# ============================================================
# 10. COMBINE GLOBAL SAMPLES
# ============================================================

for channel in CHANNELS:

    global_samples[
        channel
    ] = np.concatenate(
        global_samples[
            channel
        ]
    )


# ============================================================
# 11. FINAL DATASET-WIDE STATISTICS
# ============================================================

summary_rows = []


print()
print("=" * 65)
print("FINAL NORMALIZED RANDOM COMPONENT")
print("=" * 65)


for channel in CHANNELS:

    eps = global_samples[
        channel
    ]


    mean_value = np.mean(
        eps
    )

    std_value = np.std(
        eps
    )

    skew_value = skew(
        eps
    )

    kurt_value = kurtosis(
        eps
    )


    summary_rows.append({

        "channel":
            channel,

        "mean":
            mean_value,

        "std":
            std_value,

        "skewness":
            skew_value,

        "excess_kurtosis":
            kurt_value
    })


    print()
    print(
        f"{channel} channel"
    )

    print(
        "Mean            :",
        mean_value
    )

    print(
        "Std             :",
        std_value
    )

    print(
        "Skewness        :",
        skew_value
    )

    print(
        "Excess Kurtosis :",
        kurt_value
    )


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(

    OUTPUT_DIR
    /
    "final_noise_distribution_summary.csv",

    index=False
)


# ============================================================
# 12. FINAL HISTOGRAMS
# ============================================================

x_axis = np.linspace(
    -5,
    5,
    600
)


for channel in CHANNELS:

    eps = global_samples[
        channel
    ]


    plt.figure(
        figsize=(9, 6)
    )


    # Actual normalized residual
    plt.hist(
        eps,
        bins=150,
        density=True,
        alpha=0.7,
        label=(
            f"{channel} normalized residual"
        )
    )


    # Ideal Gaussian
    plt.plot(
        x_axis,
        norm.pdf(
            x_axis,
            loc=0,
            scale=1
        ),
        linewidth=2,
        label="Standard Normal N(0,1)"
    )


    plt.xlabel(
        "Normalized Random Component ε"
    )

    plt.ylabel(
        "Density"
    )

    plt.title(
        f"Final Noise Distribution - {channel} Channel"
    )


    plt.xlim(
        -5,
        5
    )

    plt.legend()

    plt.grid()

    plt.tight_layout()


    plt.savefig(

        OUTPUT_DIR
        /
        f"{channel}_final_noise_distribution.png",

        dpi=300
    )


    plt.show()


# ============================================================
# 13. Q-Q PLOTS
#
# If Gaussian:
# points should approximately follow a straight line.
# ============================================================

for channel in CHANNELS:

    eps = global_samples[
        channel
    ]


    # Limit Q-Q sample size
    # because millions of points are unnecessary
    QQ_SAMPLE_SIZE = min(
        50000,
        eps.size
    )


    indices = rng.choice(
        eps.size,
        size=QQ_SAMPLE_SIZE,
        replace=False
    )


    qq_sample = eps[
        indices
    ]


    plt.figure(
        figsize=(7, 7)
    )


    probplot(
        qq_sample,
        dist="norm",
        plot=plt
    )


    plt.title(
        f"Normal Q-Q Plot - {channel} Channel"
    )


    plt.grid()

    plt.tight_layout()


    plt.savefig(

        OUTPUT_DIR
        /
        f"{channel}_normal_QQ_plot.png",

        dpi=300
    )


    plt.show()


# ============================================================
# 14. CHECK PER-IMAGE STD AFTER SEVERITY REMOVAL
# ============================================================

plt.figure(
    figsize=(12, 6)
)


for channel in CHANNELS:

    plt.plot(

        per_image_df[
            f"{channel}_std"
        ],

        label=channel
    )


plt.axhline(
    1,
    linestyle="--",
    label="Ideal std = 1"
)


plt.xlabel(
    "Image Index"
)

plt.ylabel(
    "Std of ε"
)

plt.title(
    "Noise Strength After Removing Image Severity"
)


plt.legend()

plt.grid()

plt.tight_layout()


plt.savefig(

    OUTPUT_DIR
    /
    "final_std_across_images.png",

    dpi=300
)


plt.show()


# ============================================================
# 15. SAVE GLOBAL SAMPLES
# ============================================================

np.savez_compressed(

    OUTPUT_DIR
    /
    "final_normalized_noise_samples.npz",

    R=global_samples["R"],

    G=global_samples["G"],

    B=global_samples["B"]
)


print()
print("=" * 65)
print("FINAL NOISE ANALYSIS COMPLETE")
print("=" * 65)