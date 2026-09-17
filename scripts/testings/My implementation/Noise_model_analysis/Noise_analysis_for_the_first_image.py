import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy as sc
from scipy.stats import skew, kurtosis

def calculate_noise(gt_im, noisy_im):

    #Converting to RGB
    gt_im_rgb = cv.cvtColor(gt_im, cv.COLOR_BGR2RGB)
    noisy_im_rgb = cv.cvtColor(noisy_im, cv.COLOR_BGR2RGB)

    gt_f = gt_im_rgb.astype(np.float32)
    noisy_f = noisy_im_rgb.astype(np.float32)

    #Extracting noise
    residual = noisy_f - gt_f

    #Avoiding the negatives
    residual_vis = np.clip(residual + 128, 0, 255).astype(np.uint8)

    return gt_im_rgb, noisy_im_rgb, residual, residual_vis

#Testing on a single image
im1_gt = cv.imread('F:\\Mora SP Cup 2026\\Competition repo\\mora_sp_cup_2026\\competition_data\\public\\ground_truth\\001.png')
im1_noisy = cv.imread('F:\\Mora SP Cup 2026\\Competition repo\\mora_sp_cup_2026\\competition_data\\public\\noisy\\001_noise.png')

im1_gt_rgb, im1_noisy_rgb, residual, residual_vis = calculate_noise(im1_gt, im1_noisy)

#Displaying the noise
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(im1_gt_rgb)
plt.title("Ground Truth")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(im1_noisy_rgb)
plt.title("Noisy")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(residual_vis)
plt.title("Residual: Noisy - GT")
plt.axis("off")

plt.show()

#Caluclating general statisitics
print("Overall residual mean:", np.mean(residual))
print("Overall residual std :", np.std(residual))

print()

print("R mean:", np.mean(residual[:,:,0]))
print("G mean:", np.mean(residual[:,:,1]))
print("B mean:", np.mean(residual[:,:,2]))

print()

print("R std:", np.std(residual[:,:,0]))
print("G std:", np.std(residual[:,:,1]))
print("B std:", np.std(residual[:,:,2]))

#Noise histogram for three channels
plt.figure(figsize=(10,5))

plt.hist(residual[:,:,0].flatten(), bins=100,
         alpha=0.5, label="R")

plt.hist(residual[:,:,1].flatten(), bins=100,
         alpha=0.5, label="G")

plt.hist(residual[:,:,2].flatten(), bins=100,
         alpha=0.5, label="B")

plt.xlabel("Residual value")
plt.ylabel("Frequency")
plt.title("RGB Residual Histograms - Image 001")
plt.legend()

plt.show()

#Channel wise part noise analysis
def intensity_noise_analysis(gt_rgb, residual, channel):

    gt_channel = gt_rgb[:, :, channel].astype(np.float32)
    res_channel = residual[:, :, channel]

    bin_edges = np.arange(0, 257, 16)

    bin_centers = []
    means = []
    stds = []

    for low, high in zip(bin_edges[:-1], bin_edges[1:]):

        mask = (gt_channel >= low) & (gt_channel < high)

        values = res_channel[mask]

        if len(values) > 0:
            bin_centers.append((low + high) / 2)
            means.append(np.mean(values))
            stds.append(np.std(values))

    return np.array(bin_centers), np.array(means), np.array(stds)

channels = ["R", "G", "B"]

plt.figure(figsize=(9,5))

for c in range(3):

    intensity, mean_noise, std_noise = intensity_noise_analysis(
        im1_gt_rgb, residual, c
    )

    plt.plot(
        intensity,
        std_noise,
        marker="o",
        label=channels[c]
    )

plt.xlabel("Ground Truth Intensity")
plt.ylabel("Residual Standard Deviation")
plt.title("Noise Strength vs Signal Intensity - Image 001")

plt.legend()
plt.grid()

plt.show()

#mean residual vs intensity
plt.figure(figsize=(9,5))

for c in range(3):

    intensity, mean_noise, std_noise = intensity_noise_analysis(
        im1_gt_rgb, residual, c
    )

    plt.plot(
        intensity,
        mean_noise,
        marker="o",
        label=channels[c]
    )

plt.axhline(0, linestyle="--")

plt.xlabel("Ground Truth Intensity")
plt.ylabel("Mean Residual")
plt.title("Residual Bias vs Signal Intensity - Image 001")

plt.legend()
plt.grid()

plt.show()

#More statistics
for c, name in enumerate(channels):

    data = residual[:,:,c].flatten()

    print(name)
    print("Mean     :", np.mean(data))
    print("Std      :", np.std(data))
    print("Skewness :", skew(data))
    print("Kurtosis :", kurtosis(data))
    print()