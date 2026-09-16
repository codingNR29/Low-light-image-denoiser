from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture

#Setting paths
SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parents[1]

#Results from wavelet
WAVELET_CSV = (
    SCRIPT_DIR
    / "wavelet_full_460"
    / "wavelet_full_metrics.csv"
)

#Results from severity analysis
SEVERITY_CSV = (
    PROJECT_ROOT
    / "Noise_characterization"
    / "noise_severity_analysis"
    / "noise_severity_statistics.csv"
)

#Creating output path
OUTPUT_DIR = (
    SCRIPT_DIR
    / "wavelet_severity_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

#Loading CSV files
wavelet_df = pd.read_csv(
    WAVELET_CSV
)

severity_df = pd.read_csv(
    SEVERITY_CSV
)

#Checking
'''print("Wavelet columns:")
print(wavelet_df.columns)

print()

print("Severity columns:")
print(severity_df.columns)'''

#Conveting image ID from 001 to 1
wavelet_df["image"] = (
    wavelet_df["image"]
    .astype(int)
)

severity_df["image"] = (
    severity_df["image"]
    .astype(int)
)

#Merging the analysis
df = pd.merge(
    wavelet_df,
    severity_df[
        [
            "image",
            "severity"
        ]
    ],
    on="image",
    how="inner"
)

#Checking
'''print()
print("Merged dataset:")
print(df.head())

print()
print(
    "Images merged:",
    len(df)
)'''

#Getting three severity groups
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

#Sorting them using the centers
centers = (
    gmm.means_
    .flatten()
)

order = np.argsort(
    centers
)

#Mapping low, medium, high
label_map = {
    order[0]: "Low",
    order[1]: "Medium",
    order[2]: "High"
}

#And assigning
df["severity_group"] = [
    label_map[label]
    for label in raw_labels
]

#Printing the centers
'''sorted_centers = np.sort(
    centers
)

print()
print("Severity centers:")

print(
    "Low    :",
    sorted_centers[0]
)

print(
    "Medium :",
    sorted_centers[1]
)

print(
    "High   :",
    sorted_centers[2]
)'''

#Couting the number of changes in each group
group_counts = (
    df["severity_group"]
    .value_counts()
    .reindex(
        [
            "Low",
            "Medium",
            "High"
        ]
    )
)

'''print()
print("Images per severity group:")
print(group_counts)'''

#Calculating wavelet performance for each group
group_summary = (
    df
    .groupby(
        "severity_group"
    )
    .agg({

        "severity":
            "mean",

        "noisy_psnr":
            "mean",

        "denoised_psnr":
            "mean",

        "delta_psnr":
            "mean",

        "noisy_ssim":
            "mean",

        "denoised_ssim":
            "mean",

        "delta_ssim":
            "mean"
    })
)

#Putting the gorups in logical order
group_summary = (
    group_summary
    .reindex(
        [
            "Low",
            "Medium",
            "High"
        ]
    )
)

#Printing the vlaues
'''print()
print("=" * 70)

print(
    "WAVELET PERFORMANCE BY NOISE SEVERITY"
)

print("=" * 70)

print(
    group_summary
)'''

#Priting with a more comprehensive manner
for group in [
    "Low",
    "Medium",
    "High"
]:

    row = group_summary.loc[
        group
    ]

    print()
    print(
        f"{group.upper()} SEVERITY"
    )

    print(
        "Mean severity      :",
        f"{row['severity']:.4f}"
    )

    print(
        "Noisy PSNR         :",
        f"{row['noisy_psnr']:.4f} dB"
    )

    print(
        "Denoised PSNR      :",
        f"{row['denoised_psnr']:.4f} dB"
    )

    print(
        "Delta PSNR         :",
        f"{row['delta_psnr']:+.4f} dB"
    )

    print(
        "Noisy SSIM         :",
        f"{row['noisy_ssim']:.6f}"
    )

    print(
        "Denoised SSIM      :",
        f"{row['denoised_ssim']:.6f}"
    )

    print(
        "Delta SSIM         :",
        f"{row['delta_ssim']:+.6f}"
    )

#Correlation with severity
psnr_correlation = (
    df[
        [
            "severity",
            "delta_psnr"
        ]
    ]
    .corr()
    .iloc[0, 1]
)

ssim_correlation = (
    df[
        [
            "severity",
            "delta_ssim"
        ]
    ]
    .corr()
    .iloc[0, 1]
)

#Printing the valeus
print()
print("=" * 70)
print("SEVERITY VS IMPROVEMENT")
print("=" * 70)

print(
    "Severity vs Delta PSNR correlation:",
    psnr_correlation
)

print(
    "Severity vs Delta SSIM correlation:",
    ssim_correlation
)

#Plotting severity vs PSNR
plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    df["severity"],
    df["delta_psnr"],
    alpha=0.6
)

plt.xlabel(
    "Image Noise Severity $s_i$"
)

plt.ylabel(
    "Wavelet ΔPSNR (dB)"
)

plt.title(
    "Wavelet Improvement vs Noise Severity"
)

plt.grid()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "severity_vs_delta_psnr.png",
    dpi=300
)

plt.show()

#Plotting severity vs SSIM
plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    df["severity"],
    df["delta_ssim"],
    alpha=0.6
)

plt.xlabel(
    "Image Noise Severity $s_i$"
)

plt.ylabel(
    "Wavelet ΔSSIM"
)

plt.title(
    "Wavelet SSIM Improvement vs Noise Severity"
)

plt.grid()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "severity_vs_delta_ssim.png",
    dpi=300
)

plt.show()

#Boxplotting delta PSNR
groups = [
    df[
        df["severity_group"]
        == "Low"
    ]["delta_psnr"],

    df[
        df["severity_group"]
        == "Medium"
    ]["delta_psnr"],

    df[
        df["severity_group"]
        == "High"
    ]["delta_psnr"]
]

plt.figure(
    figsize=(8, 6)
)

plt.boxplot(
    groups,
    tick_labels=[
        "Low",
        "Medium",
        "High"
    ]
)

plt.ylabel(
    "ΔPSNR (dB)"
)

plt.xlabel(
    "Noise Severity Group"
)

plt.title(
    "Wavelet PSNR Improvement by Severity"
)

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "delta_psnr_by_severity.png",
    dpi=300
)

plt.show()

#Boxplotting for SSIM
groups_ssim = [
    df[
        df["severity_group"]
        == "Low"
    ]["delta_ssim"],

    df[
        df["severity_group"]
        == "Medium"
    ]["delta_ssim"],

    df[
        df["severity_group"]
        == "High"
    ]["delta_ssim"]
]

plt.figure(
    figsize=(8, 6)
)

plt.boxplot(
    groups_ssim,
    tick_labels=[
        "Low",
        "Medium",
        "High"
    ]
)

plt.ylabel(
    "ΔSSIM"
)

plt.xlabel(
    "Noise Severity Group"
)

plt.title(
    "Wavelet SSIM Improvement by Severity"
)

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "delta_ssim_by_severity.png",
    dpi=300
)

plt.show()

#Savind everything
df.to_csv(
    OUTPUT_DIR
    / "wavelet_severity_merged.csv",
    index=False
)

group_summary.to_csv(
    OUTPUT_DIR
    / "wavelet_severity_group_summary.csv"
)

print()
print("=" * 70)

print(
    "WAVELET SEVERITY ANALYSIS COMPLETE"
)

print("=" * 70)

print(
    "Results saved to:"
)

print(
    OUTPUT_DIR
)