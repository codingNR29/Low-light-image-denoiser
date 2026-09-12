from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

#Setting paths
GT_DIR = Path(
    r"F:\\Mora SP Cup 2026\\Competition repo\\mora_sp_cup_2026"
    r"\\competition_data\\public\\ground_truth"
)

NOISY_DIR = Path(
    r"F:\\Mora SP Cup 2026\\Competition repo\\mora_sp_cup_2026"
    r"\\competition_data\\public\\noisy"
)

OUTPUT_DIR = Path("noise_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

#Defining intensitiy bins
BIN_SIZE = 16

bin_edges = np.arange(0, 257, BIN_SIZE)

N_BINS = len(bin_edges) - 1
N_CHANNELS = 3

#Pair-wise analysis function
def analyse_pair(gt_path, noisy_path):

    gt = cv.imread(str(gt_path))
    noisy = cv.imread(str(noisy_path))

    if gt is None or noisy is None:
        raise ValueError(
            f"Could not load {gt_path.name} or {noisy_path.name}"
        )

    # BGR -> RGB
    gt = cv.cvtColor(gt, cv.COLOR_BGR2RGB)
    noisy = cv.cvtColor(noisy, cv.COLOR_BGR2RGB)

    gt_f = gt.astype(np.float32)
    noisy_f = noisy.astype(np.float32)

    residual = noisy_f - gt_f

    # ---------------------------------------
    # Per-image statistics
    # ---------------------------------------

    row = {
        "image": gt_path.stem
    }

    names = ["R", "G", "B"]

    for c, name in enumerate(names):

        r = residual[:, :, c].ravel()

        row[f"{name}_mean"] = np.mean(r)
        row[f"{name}_std"] = np.std(r)
        row[f"{name}_skew"] = skew(r)
        row[f"{name}_kurtosis"] = kurtosis(r)

    row["overall_mean"] = np.mean(residual)
    row["overall_std"] = np.std(residual)

    row["psnr"] = peak_signal_noise_ratio(
        gt,
        noisy,
        data_range=255
    )

    row["ssim"] = structural_similarity(
        gt,
        noisy,
        channel_axis=2,
        data_range=255
    )

    # ---------------------------------------
    # Dataset-level intensity accumulators
    # ---------------------------------------

    counts = np.zeros(
        (N_CHANNELS, N_BINS),
        dtype=np.int64
    )

    sums = np.zeros(
        (N_CHANNELS, N_BINS),
        dtype=np.float64
    )

    sums_sq = np.zeros(
        (N_CHANNELS, N_BINS),
        dtype=np.float64
    )

    for c in range(3):

        gt_channel = gt[:, :, c]
        r_channel = residual[:, :, c]

        # Convert intensities to bin numbers:
        # 0-15 -> 0
        # 16-31 -> 1
        # ...
        indices = np.minimum(
            gt_channel // BIN_SIZE,
            N_BINS - 1
        )

        for b in range(N_BINS):

            values = r_channel[indices == b]

            if values.size == 0:
                continue

            counts[c, b] += values.size
            sums[c, b] += values.sum()
            sums_sq[c, b] += np.square(values).sum()

    return row, counts, sums, sums_sq

#Looping through the image batch
rows = []

global_counts = np.zeros(
    (N_CHANNELS, N_BINS),
    dtype=np.int64
)

global_sums = np.zeros(
    (N_CHANNELS, N_BINS),
    dtype=np.float64
)

global_sums_sq = np.zeros(
    (N_CHANNELS, N_BINS),
    dtype=np.float64
)


for i in range(1, 461):

    image_id = f"{i:03d}"

    gt_path = GT_DIR / f"{image_id}.png"
    noisy_path = NOISY_DIR / f"{image_id}_noise.png"

    row, counts, sums, sums_sq = analyse_pair(
        gt_path,
        noisy_path
    )

    rows.append(row)

    global_counts += counts
    global_sums += sums
    global_sums_sq += sums_sq

    print(f"Processed {image_id}/460")

#Saving per image dataset
image_df = pd.DataFrame(rows)

image_df.to_csv(
    OUTPUT_DIR / "image_statistics.csv",
    index=False
)

#Caluclating whole dataset intensity statistics
intensity_rows = []

channel_names = ["R", "G", "B"]

for c, channel in enumerate(channel_names):

    for b in range(N_BINS):

        n = global_counts[c, b]

        if n == 0:
            continue

        mean = global_sums[c, b] / n

        variance = (
            global_sums_sq[c, b] / n
            - mean ** 2
        )

        variance = max(variance, 0)

        std = np.sqrt(variance)

        intensity_rows.append({

            "channel": channel,

            "intensity_low":
                int(bin_edges[b]),

            "intensity_high":
                int(bin_edges[b + 1] - 1),

            "intensity_center":
                float(
                    (
                        bin_edges[b]
                        + bin_edges[b + 1] - 1
                    ) / 2
                ),

            "pixel_count":
                int(n),

            "residual_mean":
                mean,

            "residual_std":
                std
        })

#Saving the whole dataset intensity statistics
intensity_df = pd.DataFrame(
    intensity_rows
)

intensity_df.to_csv(
    OUTPUT_DIR / "intensity_statistics.csv",
    index=False
)