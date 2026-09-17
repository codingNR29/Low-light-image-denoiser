from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    "residual_correlation_analysis"
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

def cubic_model(x, coeffs):

    x = x.astype(
        np.float64
    )

    a, b, c, d = coeffs

    return (
        a * x**3
        + b * x**2
        + c * x
        + d
    )


# ============================================================
# 5. PEARSON CORRELATION
# ============================================================

def pearson_corr(a, b):

    a = np.asarray(
        a,
        dtype=np.float64
    )

    b = np.asarray(
        b,
        dtype=np.float64
    )

    a = a - np.mean(a)
    b = b - np.mean(b)

    denominator = np.sqrt(
        np.sum(a**2)
        *
        np.sum(b**2)
    )

    if denominator < 1e-12:
        return np.nan

    return (
        np.sum(a * b)
        /
        denominator
    )


# ============================================================
# 6. CALCULATE FINAL NORMALIZED RANDOM COMPONENT
#
# epsilon =
#
# residual - bias(x)
# ------------------
# sigma(x) * severity
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


        # Predicted bias
        mu = cubic_model(
            x,
            BIAS_COEFFS[channel]
        )


        # Predicted variance
        variance = cubic_model(
            x,
            VARIANCE_COEFFS[channel]
        )

        variance = np.maximum(
            variance,
            1e-6
        )


        sigma = np.sqrt(
            variance
        )


        # Standardized residual
        z = (
            r - mu
        ) / sigma


        # Remove image-level severity
        epsilon[
            :, :, c
        ] = z / severity


    return epsilon


# ============================================================
# 7. SAMPLE HORIZONTAL SPATIAL CORRELATION
# ============================================================

def horizontal_correlation(
    epsilon,
    lag,
    sample_size,
    rng
):

    h, w, _ = epsilon.shape

    n = min(
        sample_size,
        h * (w - lag)
    )


    ys = rng.integers(
        0,
        h,
        size=n
    )

    xs = rng.integers(
        0,
        w - lag,
        size=n
    )


    pixel_1 = epsilon[
        ys,
        xs,
        :
    ]

    pixel_2 = epsilon[
        ys,
        xs + lag,
        :
    ]


    correlations = {}


    for c, channel in enumerate(
        CHANNELS
    ):

        correlations[channel] = (
            pearson_corr(
                pixel_1[:, c],
                pixel_2[:, c]
            )
        )


    return correlations


# ============================================================
# 8. SAMPLE VERTICAL SPATIAL CORRELATION
# ============================================================

def vertical_correlation(
    epsilon,
    lag,
    sample_size,
    rng
):

    h, w, _ = epsilon.shape

    n = min(
        sample_size,
        (h - lag) * w
    )


    ys = rng.integers(
        0,
        h - lag,
        size=n
    )

    xs = rng.integers(
        0,
        w,
        size=n
    )


    pixel_1 = epsilon[
        ys,
        xs,
        :
    ]

    pixel_2 = epsilon[
        ys + lag,
        xs,
        :
    ]


    correlations = {}


    for c, channel in enumerate(
        CHANNELS
    ):

        correlations[channel] = (
            pearson_corr(
                pixel_1[:, c],
                pixel_2[:, c]
            )
        )


    return correlations


# ============================================================
# 9. RGB CHANNEL CORRELATION
# ============================================================

def rgb_correlation(
    epsilon,
    sample_size,
    rng
):

    h, w, _ = epsilon.shape


    n = min(
        sample_size,
        h * w
    )


    ys = rng.integers(
        0,
        h,
        size=n
    )

    xs = rng.integers(
        0,
        w,
        size=n
    )


    pixels = epsilon[
        ys,
        xs,
        :
    ]


    R = pixels[:, 0]
    G = pixels[:, 1]
    B = pixels[:, 2]


    return {

        "RG":
            pearson_corr(R, G),

        "RB":
            pearson_corr(R, B),

        "GB":
            pearson_corr(G, B)
    }


# ============================================================
# 10. LOAD SEVERITY VALUES
# ============================================================

severity_df = pd.read_csv(
    SEVERITY_CSV
)


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
# 11. ANALYSIS SETTINGS
# ============================================================

MAX_LAG = 10

SPATIAL_SAMPLE_SIZE = 5000

RGB_SAMPLE_SIZE = 20000


rng = np.random.default_rng(
    seed=42
)


spatial_rows = []
rgb_rows = []


# ============================================================
# 12. PROCESS ALL 460 IMAGES
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


    epsilon = calculate_epsilon(
        gt,
        noisy,
        severity
    )


    # --------------------------------------------------------
    # RGB correlation
    # --------------------------------------------------------

    rgb_corr = rgb_correlation(
        epsilon,
        RGB_SAMPLE_SIZE,
        rng
    )


    rgb_rows.append({

        "image":
            image_id,

        "severity":
            severity,

        "RG_corr":
            rgb_corr["RG"],

        "RB_corr":
            rgb_corr["RB"],

        "GB_corr":
            rgb_corr["GB"]
    })


    # --------------------------------------------------------
    # Spatial correlation for lags 1 to 10
    # --------------------------------------------------------

    for lag in range(
        1,
        MAX_LAG + 1
    ):

        horizontal = (
            horizontal_correlation(
                epsilon,
                lag,
                SPATIAL_SAMPLE_SIZE,
                rng
            )
        )


        vertical = (
            vertical_correlation(
                epsilon,
                lag,
                SPATIAL_SAMPLE_SIZE,
                rng
            )
        )


        spatial_rows.append({

            "image":
                image_id,

            "lag":
                lag,

            "R_horizontal":
                horizontal["R"],

            "G_horizontal":
                horizontal["G"],

            "B_horizontal":
                horizontal["B"],

            "R_vertical":
                vertical["R"],

            "G_vertical":
                vertical["G"],

            "B_vertical":
                vertical["B"]
        })


    print(
        f"Processed {image_id}/460"
    )


# ============================================================
# 13. SAVE RAW RESULTS
# ============================================================

spatial_df = pd.DataFrame(
    spatial_rows
)

rgb_df = pd.DataFrame(
    rgb_rows
)


spatial_df.to_csv(

    OUTPUT_DIR
    /
    "spatial_correlation_per_image.csv",

    index=False
)


rgb_df.to_csv(

    OUTPUT_DIR
    /
    "rgb_residual_correlation.csv",

    index=False
)


# ============================================================
# 14. SPATIAL CORRELATION SUMMARY
# ============================================================

spatial_summary = (

    spatial_df
    .groupby("lag")
    .mean(
        numeric_only=True
    )

)


spatial_summary.to_csv(

    OUTPUT_DIR
    /
    "spatial_correlation_summary.csv"

)


print()
print("=" * 60)
print("SPATIAL CORRELATION SUMMARY - LAG 1")
print("=" * 60)


lag1 = spatial_summary.loc[1]


for channel in CHANNELS:

    h_corr = lag1[
        f"{channel}_horizontal"
    ]

    v_corr = lag1[
        f"{channel}_vertical"
    ]


    print()

    print(
        f"{channel} channel"
    )

    print(
        "Horizontal :",
        h_corr
    )

    print(
        "Vertical   :",
        v_corr
    )

    print(
        "Average    :",
        (h_corr + v_corr) / 2
    )


# ============================================================
# 15. RGB CORRELATION SUMMARY
# ============================================================

print()
print("=" * 60)
print("RGB RESIDUAL CORRELATION")
print("=" * 60)


for pair in [
    "RG_corr",
    "RB_corr",
    "GB_corr"
]:

    print()

    print(
        pair,
        "Mean   :",
        rgb_df[pair].mean()
    )

    print(
        pair,
        "Median :",
        rgb_df[pair].median()
    )

    print(
        pair,
        "Std    :",
        rgb_df[pair].std()
    )


# ============================================================
# 16. PLOT SPATIAL AUTOCORRELATION VS LAG
# ============================================================

plt.figure(
    figsize=(9, 6)
)


for channel in CHANNELS:

    average_corr = (

        spatial_summary[
            f"{channel}_horizontal"
        ]

        +

        spatial_summary[
            f"{channel}_vertical"
        ]

    ) / 2


    plt.plot(

        spatial_summary.index,

        average_corr,

        marker="o",

        label=channel
    )


plt.axhline(
    0,
    linestyle="--"
)


plt.xlabel(
    "Pixel Lag"
)

plt.ylabel(
    "Pearson Correlation"
)

plt.title(
    "Spatial Correlation of Normalized Residual"
)

plt.legend()

plt.grid()

plt.tight_layout()


plt.savefig(

    OUTPUT_DIR
    /
    "spatial_correlation_vs_lag.png",

    dpi=300
)


plt.show()


# ============================================================
# 17. RGB CORRELATION MATRIX
# ============================================================

RG = rgb_df[
    "RG_corr"
].mean()

RB = rgb_df[
    "RB_corr"
].mean()

GB = rgb_df[
    "GB_corr"
].mean()


rgb_matrix = np.array([

    [1.0, RG,  RB],

    [RG,  1.0, GB],

    [RB,  GB,  1.0]

])


plt.figure(
    figsize=(6, 5)
)


plt.imshow(
    rgb_matrix,
    vmin=-1,
    vmax=1
)


plt.colorbar(
    label="Pearson Correlation"
)


plt.xticks(
    [0, 1, 2],
    ["R", "G", "B"]
)

plt.yticks(
    [0, 1, 2],
    ["R", "G", "B"]
)


for i in range(3):

    for j in range(3):

        plt.text(

            j,
            i,

            f"{rgb_matrix[i, j]:.3f}",

            ha="center",
            va="center"
        )


plt.title(
    "RGB Correlation of Normalized Residual"
)


plt.tight_layout()


plt.savefig(

    OUTPUT_DIR
    /
    "rgb_residual_correlation_matrix.png",

    dpi=300
)


plt.show()