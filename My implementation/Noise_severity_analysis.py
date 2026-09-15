from pathlib import Path
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Point to the previous stage's output relative to this script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "standardized_residual_analysis" / "standardized_residual_statistics.csv"

OUTPUT_DIR = Path("noise_severity_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

#Loading and checking
df = pd.read_csv(CSV_PATH)

'''print(df.head())
print()
print(df.columns)'''

#Calculating severity
std_columns = [
    "R_z_std",
    "G_z_std",
    "B_z_std"
]

df["severity_mean"] = df[
    std_columns
].mean(axis=1)

df["severity_median"] = df[
    std_columns
].median(axis=1)

df["severity"] = df["severity_median"]

#Checking
'''print(
    df[
        [
            "image",
            "R_z_std",
            "G_z_std",
            "B_z_std",
            "severity"
        ]
    ].head(20)
)'''

#Checking the correlation of severity of three channels
correlation = df[
    [
        "R_z_std",
        "G_z_std",
        "B_z_std"
    ]
].corr()

'''print()
print("Channel severity correlation:")
print(correlation)'''

#Plotting
plt.figure(figsize=(12, 6))

plt.plot(
    df["R_z_std"],
    label="R",
    color='red'
)

plt.plot(
    df["G_z_std"],
    label="G",
    color='green'
)

plt.plot(
    df["B_z_std"],
    label="B",
    color='blue'
)

plt.xlabel("Image Index")
plt.ylabel("Standardized Residual Std")

plt.title(
    "Channel-wise Noise Severity Across Images"
)

plt.legend()
plt.grid()
plt.tight_layout()

plt.show()

#Single severity score
plt.figure(figsize=(12, 6))

plt.plot(
    df["severity"],
    marker=".",
    markersize=4
)

plt.axhline(
    1,
    linestyle="--",
    label="Dataset-average scale"
)

plt.xlabel("Image Index")
plt.ylabel("Severity $s_i$")

plt.title(
    "Estimated Image-Level Noise Severity"
)

plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "severity_vs_image.png",
    dpi=300
)

plt.show()

#Severity histogram
plt.figure(figsize=(9, 6))

plt.hist(
    df["severity"],
    bins=30,
    edgecolor="black"
)

plt.xlabel("Severity $s_i$")
plt.ylabel("Number of Images")

plt.title(
    "Distribution of Image-Level Noise Severity"
)

plt.grid(axis="y")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "severity_histogram.png",
    dpi=300
)

plt.show()

#Sorted severity plot
severity_sorted = np.sort(
    df["severity"].values
)

plt.figure(figsize=(10, 6))

plt.plot(
    severity_sorted,
    marker="."
)

plt.xlabel("Images Sorted by Severity")
plt.ylabel("Severity $s_i$")

plt.title(
    "Sorted Image-Level Noise Severity"
)

plt.grid()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "sorted_severity.png",
    dpi=300
)

plt.show()

#Calculating descriptive satatistics
severity = df["severity"]

print()
print("Severity statistics")

print(
    "Mean   :",
    severity.mean()
)

print(
    "Median :",
    severity.median()
)

print(
    "Std    :",
    severity.std()
)

print(
    "Min    :",
    severity.min()
)

print(
    "Max    :",
    severity.max()
)

#Calculating percentiles
percentiles = severity.quantile(
    [
        0.10,
        0.25,
        0.50,
        0.75,
        0.90
    ]
)

print()
print("Severity percentiles:")
print(percentiles)

#Saving the dataset
df.to_csv(
    OUTPUT_DIR / "noise_severity_statistics.csv",
    index=False
)

#Gaussian model comparison using Baysian Informatino Criterion
severity_values = df["severity"].values.reshape(-1, 1)

bic_scores = []

for n_components in range(1, 6):

    gmm = GaussianMixture(
        n_components=n_components,
        random_state=42
    )

    gmm.fit(severity_values)

    bic = gmm.bic(severity_values)

    bic_scores.append(bic)

    print(
        f"{n_components} components -> BIC = {bic:.2f}"
    )

#Plotting
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 6),
    bic_scores,
    marker="o"
)

plt.xlabel("Number of Severity Groups")
plt.ylabel("BIC")

plt.title(
    "Choosing Number of Noise Severity Groups"
)

plt.grid()
plt.show()

#Finding the three groups
gmm = GaussianMixture(
    n_components=3,
    random_state=42
)

labels = gmm.fit_predict(
    severity_values
)

centers = gmm.means_.flatten()

print("Raw severity centers:")
print(centers)

#Sorting the centers
sorted_centers = np.sort(centers)

print()
print("Sorted severity centers:")

print(
    "Low severity    :",
    sorted_centers[0]
)

print(
    "Medium severity :",
    sorted_centers[1]
)

print(
    "High severity   :",
    sorted_centers[2]
)

#Printing proptions
weights = gmm.weights_

order = np.argsort(
    gmm.means_.flatten()
)

sorted_weights = weights[order]

print()
print("Group proportions:")

print(
    "Low    :",
    sorted_weights[0]
)

print(
    "Medium :",
    sorted_weights[1]
)

print(
    "High   :",
    sorted_weights[2]
)