from pathlib import Path
import torch
import pandas as pd
import matplotlib.pyplot as plt

#Importing the intensity statistics from noise analysis folder
CSV_PATH = Path(__file__).resolve().parents[1] / "noise_analysis" / "intensity_statistics.csv"

#Setting the output path
OUTPUT_DIR = Path("noise_model_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH)

'''
#Just to confirm the content
print(df.head())
print(df["channel"].unique())'''

#Creating a new column to store the variance values
df["residual_variance"] = df["residual_std"] ** 2

'''
#Checking it
print(
    df[
        [
            "channel",
            "intensity_center",
            "residual_std",
            "residual_variance"
        ]
    ].head()
)
'''

#Plotting the data points for the variance
'''
#Plotting for R G B
plt.figure(figsize=(10, 6))

channels = ["R", "G", "B"]

for channel in channels:

    channel_df = df[
        df["channel"] == channel
    ].copy()

    channel_df = channel_df.sort_values(
        "intensity_center"
    )

    if channel == 'R':
        plt.scatter(
            channel_df["intensity_center"],
            channel_df["residual_variance"],
            label=channel,
            color = 'red'
        )
    elif channel == 'G':
        plt.scatter(
            channel_df["intensity_center"],
            channel_df["residual_variance"],
            label=channel,
            color = 'green'
        )
    else:
        plt.scatter(
            channel_df["intensity_center"],
            channel_df["residual_variance"],
            label=channel,
            color = 'blue'
        )
        
        

plt.xlabel("Ground Truth Intensity")
plt.ylabel("Residual Variance")
plt.title("Residual Variance vs Signal Intensity")

plt.legend()
plt.grid()

plt.savefig(
    OUTPUT_DIR / "variance_vs_intensity.png",
    dpi=300
)

plt.show()

#Connected version
plt.figure(figsize=(10, 6))

for channel in channels:

    channel_df = df[
        df["channel"] == channel
    ].copy()

    channel_df = channel_df.sort_values(
        "intensity_center"
    )

    if channel == 'R':
        plt.plot(
            channel_df["intensity_center"],
            channel_df["residual_variance"],
            marker="o",
            label=channel,
            color = 'red'
        )
    elif channel == 'G':
        plt.plot(
            channel_df["intensity_center"],
            channel_df["residual_variance"],
            marker="o",
            label=channel,
            color = 'green'
        )
    else:
        plt.plot(
            channel_df["intensity_center"],
            channel_df["residual_variance"],
            marker="o",
            label=channel,
            color = 'blue'
        )

plt.xlabel("Ground Truth Intensity")
plt.ylabel("Residual Variance")
plt.title("Residual Variance Trend vs Signal Intensity")

plt.legend()
plt.grid()

plt.savefig(
    OUTPUT_DIR / "variance_vs_intensity_trend.png",
    dpi=300
)

plt.show()
'''

#Defining functions to fit the data set
def fit_linear(x, y):

    x = torch.tensor(
        x,
        dtype=torch.float32
    )

    y = torch.tensor(
        y,
        dtype=torch.float32
    )

    # Design matrix:
    # [x, 1]
    X = torch.stack(
        [
            x,
            torch.ones_like(x)
        ],
        dim=1
    )

    # Solve least squares
    solution = torch.linalg.lstsq(
        X,
        y
    ).solution

    a = solution[0]
    b = solution[1]

    return a.item(), b.item()

def fit_quadratic(x, y):

    x = torch.tensor(
        x,
        dtype=torch.float32
    )

    y = torch.tensor(
        y,
        dtype=torch.float32
    )

    # Design matrix:
    # [x^2, x, 1]
    X = torch.stack(
        [
            x ** 2,
            x,
            torch.ones_like(x)
        ],
        dim=1
    )

    solution = torch.linalg.lstsq(
        X,
        y
    ).solution

    a = solution[0]
    b = solution[1]
    c = solution[2]

    return (
        a.item(),
        b.item(),
        c.item()
    )

def fit_cubic(x, y):

    x = torch.tensor(
        x,
        dtype=torch.float64
    )

    y = torch.tensor(
        y,
        dtype=torch.float64
    )

    # Design matrix:
    # [x^3, x^2, x, 1]
    X = torch.stack(
        [
            x ** 3,
            x ** 2,
            x,
            torch.ones_like(x)
        ],
        dim=1
    )

    solution = torch.linalg.lstsq(
        X,
        y
    ).solution

    a = solution[0]
    b = solution[1]
    c = solution[2]
    d = solution[3]

    return (
        a.item(),
        b.item(),
        c.item(),
        d.item()
    )


#Checking for red only
red_df = df[
    df["channel"] == "R"
].copy()

red_df = red_df.sort_values(
    "intensity_center"
)

x = red_df[
    "intensity_center"
].values

y = red_df[
    "residual_variance"
].values

#Fitting the models
a_lin, b_lin = fit_linear(
    x,
    y
)

a_quad, b_quad, c_quad = fit_quadratic(
    x,
    y
)

a_cubic, b_cubic, c_cubic, d_cubic = fit_cubic(
    x,
    y
)

#Printing the equations
print("Linear model:")
print(
    f"y = {a_lin:.4f}x + {b_lin:.4f}"
)

print()

print("Quadratic model:")
print(
    f"y = {a_quad:.6f}x^2 "
    f"+ {b_quad:.4f}x "
    f"+ {c_quad:.4f}"
)

print()

print("Cubic model:")
print(
    f"y = {a_cubic:.6f}x^3 "
    f"+ {b_cubic:.4f}x^2 "
    f"+ {c_cubic:.4f}x"
    f"+ {d_cubic:.4f}"
)

#Generating predicted values
y_linear = (
    a_lin * x
    + b_lin
)

y_quadratic = (
    a_quad * x ** 2
    + b_quad * x
    + c_quad
)

y_cubic = (
    a_cubic * x ** 3
    + b_cubic * x ** 2
    + c_cubic * x
    + d_cubic
)
    

#Plotting for red
'''
plt.figure(figsize=(10, 6))

plt.scatter(
    x,
    y,
    label="Measured data"
)

plt.plot(
    x,
    y_linear,
    label="Linear fit"
)

plt.plot(
    x,
    y_quadratic,
    label="Quadratic fit"
)

plt.plot(
    x,
    y_cubic,
    label="Cubic fit"
)

plt.xlabel("Ground Truth Intensity")
plt.ylabel("Residual Variance")

plt.title(
    "Linear vs Quadratic vs Cubic Fit - Red Channel"
)

plt.legend()
plt.grid()

plt.show()
'''

#Function to caluclate R^2
def calculate_r2(
    y_true,
    y_pred
):

    y_true = torch.tensor(
        y_true,
        dtype=torch.float32
    )

    y_pred = torch.tensor(
        y_pred,
        dtype=torch.float32
    )

    ss_res = torch.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = torch.sum(
        (
            y_true
            - torch.mean(y_true)
        ) ** 2
    )

    r2 = 1 - ss_res / ss_tot

    return r2.item()

#Calculating values
r2_linear = calculate_r2(
    y,
    y_linear
)

r2_quadratic = calculate_r2(
    y,
    y_quadratic
)

r2_cubic = calculate_r2(
    y,
    y_cubic
)

print(
    "Linear R²:",
    r2_linear
)

print(
    "Quadratic R²:",
    r2_quadratic
)

print(
    "Cubic R²:",
    r2_cubic
)

#Fitting for all three channels
channels = ["R", "G", "B"]

for channel in channels:

    channel_df = df[
        df["channel"] == channel
    ].copy()

    channel_df = channel_df.sort_values(
        "intensity_center"
    )

    x = channel_df[
        "intensity_center"
    ].values

    y = channel_df[
        "residual_variance"
    ].values

    a, b, c, d = fit_cubic(x, y)

    y_pred = (
        a * x**3
        + b * x**2
        + c * x
        + d
    )

    r2 = calculate_r2(
        y,
        y_pred
    )

    print()
    print(f"{channel} channel")
    print(
        f"Variance = {a:.8f}x^3 "
        f"+ {b:.6f}x^2 "
        f"+ {c:.4f}x "
        f"+ {d:.4f}"
    )
    print(f"R² = {r2:.6f}")

###################################################################################################
#Now looking at mean
'''
plt.figure(figsize=(10, 6))

channels = ["R", "G", "B"]

for channel in channels:

    channel_df = df[
        df["channel"] == channel
    ].copy()

    channel_df = channel_df.sort_values(
        "intensity_center"
    )
    if channel == 'R':
        plt.plot(
            channel_df["intensity_center"],
            channel_df["residual_mean"],
            color='red',
            marker="o",
            label=channel
        )
    elif channel == 'G':
        plt.plot(
            channel_df["intensity_center"],
            channel_df["residual_mean"],
            color='green',
            marker="o",
            label=channel
        )
    else:
        plt.plot(
            channel_df["intensity_center"],
            channel_df["residual_mean"],
            color='blue',
            marker="o",
            label=channel
        )

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Ground Truth Intensity")
plt.ylabel("Mean Residual")
plt.title("Residual Bias vs Signal Intensity")

plt.legend()
plt.grid()
plt.tight_layout()

plt.show()
'''

#Testing for red channel
'''
red_df = df[
    df["channel"] == "R"
].copy()

red_df = red_df.sort_values(
    "intensity_center"
)

x = red_df[
    "intensity_center"
].values

y = red_df[
    "residual_mean"
].values

a, b, c, d = fit_cubic(
    x,
    y
)

y_cubic = (
    a * x**3
    + b * x**2
    + c * x
    + d
)

r2 = calculate_r2(
    y,
    y_cubic
)

print("Bias cubic model:")

print(
    f"Mean residual = {a:.8f}x^3 "
    f"+ {b:.6f}x^2 "
    f"+ {c:.4f}x "
    f"+ {d:.4f}"
)

print(f"R² = {r2:.6f}")

plt.figure(figsize=(10, 6))

plt.scatter(
    x,
    y,
    label="Measured bias"
)

plt.plot(
    x,
    y_cubic,
    label=f"Cubic fit (R²={r2:.3f})"
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Ground Truth Intensity")
plt.ylabel("Mean Residual")

plt.title(
    "Cubic Bias Model - Red Channel"
)

plt.legend()
plt.grid()

plt.show()
'''

#Doing this for the whole channels
channels = ["R", "G", "B"]

for channel in channels:

    channel_df = df[
        df["channel"] == channel
    ].copy()

    channel_df = channel_df.sort_values(
        "intensity_center"
    )

    x = channel_df[
        "intensity_center"
    ].values

    y = channel_df[
        "residual_mean"
    ].values

    a, b, c, d = fit_cubic(
        x,
        y
    )

    y_pred = (
        a * x**3
        + b * x**2
        + c * x
        + d
    )

    r2 = calculate_r2(
        y,
        y_pred
    )

    print()
    print(f"{channel} channel")

    print(
        f"Bias = {a:.8f}x^3 "
        f"+ {b:.6f}x^2 "
        f"+ {c:.4f}x "
        f"+ {d:.4f}"
    )

    print(f"R² = {r2:.6f}")

##############################################################################################################
#Standardizing the residual
