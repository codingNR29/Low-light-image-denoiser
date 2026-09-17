from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Getting current script directory
SCRIPT_DIR = Path(
    __file__
).resolve().parent


#Getting testing directory
TESTING_DIR = (
    SCRIPT_DIR.parents[1]
)


#Getting intensity statistics CSV
CSV_PATH = (

    TESTING_DIR

    / "Noise_characterization"

    / "noise_analysis"

    / "intensity_statistics.csv"

)


#Setting output directory
OUTPUT_DIR = (

    TESTING_DIR

    / "Noise_characterization"

    / "noise_model_analysis"

    / "report_figures"

)


#Creating output directory
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print(
    "Reading intensity statistics from:"
)

print(
    CSV_PATH
)


print()

print(
    "Saving report figures to:"
)

print(
    OUTPUT_DIR
)


#Loading intensity statistics
df = pd.read_csv(
    CSV_PATH
)


#Calculating residual variance
df[
    "residual_variance"
] = (

    df[
        "residual_std"
    ] ** 2

)


#RGB channels
channels = [
    "R",
    "G",
    "B"
]


#Colors corresponding to RGB channels
channel_colors = {

    "R": "red",

    "G": "green",

    "B": "blue"

}


#Function to fit cubic model
def fit_cubic(
    x,
    y
):

    #Fitting cubic polynomial
    coefficients = np.polyfit(

        x,

        y,

        deg=3

    )


    return coefficients


#Function to calculate R squared
def calculate_r2(
    y_true,
    y_pred
):

    #Calculating residual sum of squares
    ss_res = np.sum(

        (
            y_true

            -

            y_pred

        ) ** 2

    )


    #Calculating total sum of squares
    ss_tot = np.sum(

        (
            y_true

            -

            np.mean(
                y_true
            )

        ) ** 2

    )


    #Calculating R squared
    r2 = (

        1

        -

        ss_res
        / ss_tot

    )


    return r2


# ============================================================
# Residual variance vs intensity
# ============================================================

plt.figure(
    figsize=(8.5, 5.2)
)


for channel in channels:

    #Selecting current channel
    channel_df = df[

        df[
            "channel"
        ]

        == channel

    ].copy()


    #Sorting according to intensity
    channel_df = channel_df.sort_values(

        "intensity_center"

    )


    #Getting intensity values
    x = channel_df[

        "intensity_center"

    ].values


    #Getting measured residual variance
    y = channel_df[

        "residual_variance"

    ].values


    #Fitting cubic model
    coefficients = fit_cubic(

        x,

        y

    )


    #Creating smooth intensity values
    x_fit = np.linspace(

        x.min(),

        x.max(),

        500

    )


    #Calculating smooth fitted curve
    y_fit = np.polyval(

        coefficients,

        x_fit

    )


    #Calculating prediction at measured points
    y_pred = np.polyval(

        coefficients,

        x

    )


    #Calculating R squared
    r2 = calculate_r2(

        y,

        y_pred

    )


    #Plotting measured values
    plt.scatter(

        x,

        y,

        color=channel_colors[
            channel
        ],

        s=20,

        alpha=0.55

    )


    #Plotting fitted cubic curve
    plt.plot(

        x_fit,

        y_fit,

        color=channel_colors[
            channel
        ],

        linewidth=2.2,

        label=(
            f"{channel} cubic fit "
            f"($R^2$={r2:.3f})"
        )

    )


    #Printing fitted model
    a, b, c, d = coefficients


    print()

    print(
        f"{channel} variance model:"
    )


    print(

        f"v(x) = "
        f"{a:.8f}x^3 "
        f"+ {b:.6f}x^2 "
        f"+ {c:.4f}x "
        f"+ {d:.4f}"

    )


    print(
        f"R² = {r2:.6f}"
    )


#Setting axis labels
plt.xlabel(
    "Ground Truth Intensity"
)


plt.ylabel(
    "Residual Variance"
)


#Setting title
plt.title(
    "Signal-Dependent Residual Variance"
)


#Adding legend
plt.legend(
    frameon=True
)


#Adding grid
plt.grid(
    alpha=0.25
)


#Making layout compact
plt.tight_layout()


#Saving PNG version
plt.savefig(

    OUTPUT_DIR

    / "variance_vs_intensity.png",

    dpi=300,

    bbox_inches="tight"

)


#Saving PDF version for LaTeX report
plt.savefig(

    OUTPUT_DIR

    / "variance_vs_intensity.pdf",

    bbox_inches="tight"

)


#Showing figure
plt.show()


# ============================================================
# Residual mean / bias vs intensity
# ============================================================

plt.figure(
    figsize=(8.5, 5.2)
)


for channel in channels:

    #Selecting current channel
    channel_df = df[

        df[
            "channel"
        ]

        == channel

    ].copy()


    #Sorting according to intensity
    channel_df = channel_df.sort_values(

        "intensity_center"

    )


    #Getting intensity values
    x = channel_df[

        "intensity_center"

    ].values


    #Getting measured residual mean
    y = channel_df[

        "residual_mean"

    ].values


    #Fitting cubic model
    coefficients = fit_cubic(

        x,

        y

    )


    #Creating smooth intensity values
    x_fit = np.linspace(

        x.min(),

        x.max(),

        500

    )


    #Calculating smooth fitted curve
    y_fit = np.polyval(

        coefficients,

        x_fit

    )


    #Calculating prediction at measured points
    y_pred = np.polyval(

        coefficients,

        x

    )


    #Calculating R squared
    r2 = calculate_r2(

        y,

        y_pred

    )


    #Plotting measured bias values
    plt.scatter(

        x,

        y,

        color=channel_colors[
            channel
        ],

        s=20,

        alpha=0.55

    )


    #Plotting fitted cubic curve
    plt.plot(

        x_fit,

        y_fit,

        color=channel_colors[
            channel
        ],

        linewidth=2.2,

        label=(
            f"{channel} cubic fit "
            f"($R^2$={r2:.3f})"
        )

    )


    #Printing fitted model
    a, b, c, d = coefficients


    print()

    print(
        f"{channel} bias model:"
    )


    print(

        f"b(x) = "
        f"{a:.8f}x^3 "
        f"+ {b:.6f}x^2 "
        f"+ {c:.4f}x "
        f"+ {d:.4f}"

    )


    print(
        f"R² = {r2:.6f}"
    )


#Showing zero bias line
plt.axhline(

    y=0,

    color="black",

    linestyle="--",

    linewidth=1.1,

    alpha=0.7

)


#Setting labels
plt.xlabel(
    "Ground Truth Intensity"
)


plt.ylabel(
    "Mean Residual"
)


#Setting title
plt.title(
    "Intensity-Dependent Residual Bias"
)


#Adding legend
plt.legend(
    frameon=True
)


#Adding grid
plt.grid(
    alpha=0.25
)


#Making layout compact
plt.tight_layout()


#Saving PNG version
plt.savefig(

    OUTPUT_DIR

    / "bias_vs_intensity.png",

    dpi=300,

    bbox_inches="tight"

)


#Saving PDF version for LaTeX report
plt.savefig(

    OUTPUT_DIR

    / "bias_vs_intensity.pdf",

    bbox_inches="tight"

)


#Showing figure
plt.show()


print()

print(
    "=" * 70
)

print(
    "REPORT FIGURES CREATED"
)

print(
    "=" * 70
)


print(
    OUTPUT_DIR
)