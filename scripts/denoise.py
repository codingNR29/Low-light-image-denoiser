"""
Final Mora SP Cup 2026 denoising pipeline

Pipeline:
    Noisy image
        -> Robust Charbonnier fine-tuned DnCNN
        -> Optimized Sym4 Wavelet denoising
        -> Noise severity estimation
        -> Severity-aware adaptive fusion
        -> Final denoised image

Required command:
    python scripts/denoise.py --noise_dir <input> --denoised_dir <output>

Example:
    python scripts/denoise.py ^
        --noise_dir competition_data/submissions/noisy ^
        --denoised_dir competition_data/submissions/denoised
"""

import argparse
import time

from pathlib import Path

import cv2
import joblib
import numpy as np

from PIL import Image

import torch

from skimage.restoration import denoise_wavelet

from dncnn_model import DnCNN


#Getting current script directory
SCRIPT_DIR = Path(
    __file__
).resolve().parent


#Valid image extensions
VALID_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg"
}


#Suffix used by noisy competition images
NOISE_SUFFIX = "_noise"


#Robust DnCNN model path
DNCNN_MODEL_PATH = (

    SCRIPT_DIR

    / "best_robust_dncnn.pth"

)


#Noise severity estimator path
SEVERITY_MODEL_PATH = (

    SCRIPT_DIR

    / "severity_estimator.joblib"

)


#Severity thresholds learned from training data
LOW_MEDIUM_THRESHOLD = 0.6843

MEDIUM_HIGH_THRESHOLD = 1.0091


#Adaptive fusion weights learned from training data
ADAPTIVE_WEIGHTS = {

    "Low": 0.85,

    "Medium": 0.90,

    "High": 0.90

}


#Wavelet settings found from parameter sweep
WAVELET_NAME = "sym4"

WAVELET_LEVEL = 4

WAVELET_METHOD = "BayesShrink"

WAVELET_MODE = "soft"


#Pixel sampling stride used by severity estimator
SAMPLE_STRIDE = 4


#Feature order used when training severity estimator
FEATURE_NAMES = [

    "robust_scale",

    "mean_abs_z",

    "p90_abs_z",

    "p95_abs_z",

    "positive_tail_fraction",

    "negative_tail_fraction"

]


#Selecting inference device
def select_device(
    requested_device
):

    #Automatically using GPU when available
    if requested_device == "auto":

        if torch.cuda.is_available():

            return torch.device(
                "cuda"
            )

        return torch.device(
            "cpu"
        )


    #Forcing CUDA
    if requested_device == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA was requested but no CUDA GPU is available."
            )

        return torch.device(
            "cuda"
        )


    #Forcing CPU
    return torch.device(
        "cpu"
    )


#Removing _noise from output filename
def strip_noise_suffix(
    stem
):

    if stem.lower().endswith(
        NOISE_SUFFIX
    ):

        return stem[
            :-len(NOISE_SUFFIX)
        ]


    return stem


#Correcting strong local defective pixels
def correct_defect_pixels(
    image_float,
    threshold=0.25,
    kernel_size=3
):

    #Creating output copy
    corrected = image_float.copy()


    #Processing RGB channels separately
    for channel in range(
        image_float.shape[2]
    ):

        #Converting channel to uint8 for median filtering
        channel_uint8 = (

            image_float[
                :,
                :,
                channel
            ]

            * 255.0

        ).astype(
            np.uint8
        )


        #Calculating local median
        median_uint8 = cv2.medianBlur(

            channel_uint8,

            kernel_size

        )


        #Returning median to [0,1]
        median = (

            median_uint8.astype(
                np.float32
            )

            / 255.0

        )


        #Calculating local deviation
        deviation = np.abs(

            image_float[
                :,
                :,
                channel
            ]

            -

            median

        )


        #Detecting strong outliers
        outlier_mask = (

            deviation

            > threshold

        )


        #Replacing strong outliers by local median
        corrected[
            :,
            :,
            channel
        ] = np.where(

            outlier_mask,

            median,

            image_float[
                :,
                :,
                channel
            ]

        )


    return corrected


#Optimized classical wavelet denoising
def run_wavelet(
    noisy_uint8
):

    #Converting image into [0,1]
    noisy_float = (

        noisy_uint8.astype(
            np.float32
        )

        / 255.0

    )


    #Correcting strong defective pixels
    corrected = correct_defect_pixels(

        noisy_float,

        threshold=0.25,

        kernel_size=3

    )


    #Applying optimized wavelet denoising
    wavelet_float = denoise_wavelet(

        corrected,

        wavelet=WAVELET_NAME,

        wavelet_levels=WAVELET_LEVEL,

        method=WAVELET_METHOD,

        mode=WAVELET_MODE,

        convert2ycbcr=False,

        rescale_sigma=True,

        channel_axis=-1

    )


    #Keeping valid range
    wavelet_float = np.clip(

        wavelet_float,

        0.0,

        1.0

    )


    #Converting back to uint8
    wavelet_uint8 = (

        wavelet_float

        * 255.0

    ).round().astype(
        np.uint8
    )


    return wavelet_uint8


#Loading Robust DnCNN
def load_dncnn(
    device
):

    #Creating same architecture used during training
    model = DnCNN(

        input_channels=3,

        output_channels=3,

        num_features=64,

        num_layers=17

    )


    #Loading trained model weights
    try:

        state_dict = torch.load(

            DNCNN_MODEL_PATH,

            map_location=device,

            weights_only=True

        )


    #Fallback for older PyTorch versions
    except TypeError:

        state_dict = torch.load(

            DNCNN_MODEL_PATH,

            map_location=device

        )


    #Supporting checkpoints that contain model_state_dict
    if (

        isinstance(
            state_dict,
            dict
        )

        and

        "model_state_dict"
        in state_dict

    ):

        state_dict = state_dict[
            "model_state_dict"
        ]


    #Loading parameters
    model.load_state_dict(
        state_dict
    )


    #Moving model to selected device
    model = model.to(
        device
    )


    #Setting inference mode
    model.eval()


    return model


#Running Robust DnCNN
def run_dncnn(
    noisy_uint8,
    model,
    device
):

    #Converting image to [0,1]
    noisy_float = (

        noisy_uint8.astype(
            np.float32
        )

        / 255.0

    )


    #Changing H x W x C into C x H x W
    noisy_chw = np.transpose(

        noisy_float,

        (2, 0, 1)

    )


    #Creating PyTorch tensor
    noisy_tensor = torch.from_numpy(

        noisy_chw.copy()

    ).float()


    #Adding batch dimension
    noisy_tensor = noisy_tensor.unsqueeze(
        0
    )


    #Moving image to inference device
    noisy_tensor = noisy_tensor.to(
        device
    )


    #Running model without gradient calculation
    with torch.inference_mode():

        #Predicting residual noise
        predicted_residual = model(
            noisy_tensor
        )


        #Residual learning reconstruction
        predicted_clean = (

            noisy_tensor

            -

            predicted_residual

        )


        #Keeping values inside valid range
        predicted_clean = torch.clamp(

            predicted_clean,

            0.0,

            1.0

        )


    #Removing batch dimension
    predicted_clean = predicted_clean.squeeze(
        0
    )


    #Moving prediction back to CPU
    predicted_clean = predicted_clean.cpu().numpy()


    #Changing C x H x W back to H x W x C
    predicted_clean = np.transpose(

        predicted_clean,

        (1, 2, 0)

    )


    #Converting to uint8
    predicted_clean = (

        predicted_clean

        * 255.0

    ).round().astype(
        np.uint8
    )


    return predicted_clean


#Signal dependent bias model
def calculate_bias(
    intensity,
    channel
):

    #Red channel
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


    #Green channel
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


    #Blue channel
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

    #Red channel
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


    #Green channel
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


    #Blue channel
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


    #Preventing invalid variance
    variance = np.maximum(

        variance,

        1.0

    )


    return variance


#Extracting noisy-only severity features
def extract_noise_features(
    noisy,
    wavelet
):

    #Sampling pixels for faster calculation
    noisy_sample = noisy[

        ::SAMPLE_STRIDE,

        ::SAMPLE_STRIDE,

        :

    ].astype(
        np.float32
    )


    #Using wavelet result as approximate clean image
    wavelet_sample = wavelet[

        ::SAMPLE_STRIDE,

        ::SAMPLE_STRIDE,

        :

    ].astype(
        np.float32
    )


    #Approximate noise residual
    residual = (

        noisy_sample

        -

        wavelet_sample

    )


    #Storage for normalized channels
    normalized_channels = []


    #Storage for channel robust scales
    channel_scales = []


    #Processing RGB channels
    for channel in range(
        3
    ):

        #Approximate clean intensity
        intensity = wavelet_sample[
            :,
            :,
            channel
        ]


        #Approximate channel residual
        channel_residual = residual[
            :,
            :,
            channel
        ]


        #Expected intensity dependent bias
        bias = calculate_bias(

            intensity,

            channel

        )


        #Expected signal dependent variance
        variance = calculate_variance(

            intensity,

            channel

        )


        #Expected standard deviation
        sigma = np.sqrt(
            variance
        )


        #Normalizing approximate residual
        normalized = (

            channel_residual

            -

            bias

        ) / sigma


        normalized_channels.append(
            normalized
        )


        #Calculating median
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


        #Converting MAD into robust scale
        robust_scale = (

            mad

            / 0.67448975

        )


        channel_scales.append(
            robust_scale
        )


    #Combining RGB normalized residuals
    normalized_all = np.stack(

        normalized_channels,

        axis=-1

    )


    #Flattening residual values
    normalized_flat = normalized_all.reshape(
        -1
    )


    #Absolute normalized residuals
    abs_normalized = np.abs(
        normalized_flat
    )


    #Calculating final features
    robust_scale = np.median(
        channel_scales
    )


    mean_abs_z = np.mean(
        abs_normalized
    )


    p90_abs_z = np.percentile(

        abs_normalized,

        90

    )


    p95_abs_z = np.percentile(

        abs_normalized,

        95

    )


    positive_tail_fraction = np.mean(

        normalized_flat

        > 3.0

    )


    negative_tail_fraction = np.mean(

        normalized_flat

        < -3.0

    )


    #Returning features in same order used for training
    features = np.array(

        [[

            robust_scale,

            mean_abs_z,

            p90_abs_z,

            p95_abs_z,

            positive_tail_fraction,

            negative_tail_fraction

        ]],

        dtype=np.float64

    )


    return features


#Converting predicted severity into Low Medium or High
def severity_to_group(
    severity
):

    #Low severity
    if severity < LOW_MEDIUM_THRESHOLD:

        return "Low"


    #Medium severity
    elif severity < MEDIUM_HIGH_THRESHOLD:

        return "Medium"


    #High severity
    else:

        return "High"


#Creating final adaptive fusion
def adaptive_fusion(
    dncnn,
    wavelet,
    severity_group
):

    #Getting DnCNN weight for this severity
    alpha = ADAPTIVE_WEIGHTS[
        severity_group
    ]


    #Converting images to float
    dncnn_float = dncnn.astype(
        np.float32
    )


    wavelet_float = wavelet.astype(
        np.float32
    )


    #Applying severity-aware fusion
    fused = (

        alpha
        * dncnn_float

        +

        (1.0 - alpha)
        * wavelet_float

    )


    #Keeping valid image range
    fused = np.clip(

        fused,

        0.0,

        255.0

    )


    #Returning uint8 image
    fused = fused.round().astype(
        np.uint8
    )


    return (
        fused,
        alpha
    )


#Processing one image
def process_image(
    noisy,
    dncnn_model,
    severity_model,
    device
):

    #Running robust DnCNN branch
    dncnn_output = run_dncnn(

        noisy,

        dncnn_model,

        device

    )


    #Running optimized Wavelet branch
    wavelet_output = run_wavelet(
        noisy
    )


    #Extracting noisy-only severity features
    features = extract_noise_features(

        noisy,

        wavelet_output

    )


    #Predicting continuous noise severity
    predicted_severity = float(

        severity_model.predict(
            features
        )[0]

    )


    #Converting severity into group
    severity_group = severity_to_group(

        predicted_severity

    )


    #Applying adaptive fusion
    final_output, alpha = adaptive_fusion(

        dncnn_output,

        wavelet_output,

        severity_group

    )


    return (
        final_output,
        predicted_severity,
        severity_group,
        alpha
    )


def main():

    #Creating command line parser
    parser = argparse.ArgumentParser(

        description=(
            "Noise-severity-aware Robust DnCNN + Wavelet denoising"
        )

    )


    #Required competition input directory
    parser.add_argument(

        "--noise_dir",

        "--input_dir",

        dest="noise_dir",

        required=True,

        type=Path,

        help="Directory containing noisy input images"

    )


    #Required competition output directory
    parser.add_argument(

        "--denoised_dir",

        "--output_dir",

        dest="denoised_dir",

        required=True,

        type=Path,

        help="Directory where denoised images will be saved"

    )


    #Optional device selection
    parser.add_argument(

        "--device",

        choices=[
            "auto",
            "cuda",
            "cpu"
        ],

        default="auto",

        help="Inference device. Default automatically uses CUDA when available."

    )


    args = parser.parse_args()


    #Checking required files
    if not DNCNN_MODEL_PATH.exists():

        raise FileNotFoundError(

            f"DnCNN model not found: "
            f"{DNCNN_MODEL_PATH}"

        )


    if not SEVERITY_MODEL_PATH.exists():

        raise FileNotFoundError(

            f"Severity estimator not found: "
            f"{SEVERITY_MODEL_PATH}"

        )


    if not args.noise_dir.exists():

        raise FileNotFoundError(

            f"Noise directory not found: "
            f"{args.noise_dir}"

        )


    #Creating output directory
    args.denoised_dir.mkdir(

        parents=True,

        exist_ok=True

    )


    #Selecting device
    device = select_device(
        args.device
    )


    print()

    print(
        "=" * 80
    )

    print(
        "MORA SP CUP 2026 FINAL DENOISING PIPELINE"
    )

    print(
        "=" * 80
    )


    print(
        "Device:",
        device
    )


    print(
        "Method: Noise-Severity-Aware Robust DnCNN + Wavelet Fusion"
    )


    print()


    #Loading DnCNN once
    print(
        "Loading Robust DnCNN..."
    )


    dncnn_model = load_dncnn(
        device
    )


    #Loading severity estimator once
    print(
        "Loading noise severity estimator..."
    )


    severity_model = joblib.load(
        SEVERITY_MODEL_PATH
    )


    #Finding input images
    image_paths = sorted(

        [

            path

            for path
            in args.noise_dir.iterdir()

            if (

                path.is_file()

                and

                path.suffix.lower()
                in VALID_EXTENSIONS

            )

        ]

    )


    #Checking input folder
    if len(
        image_paths
    ) == 0:

        raise RuntimeError(

            f"No valid images found in "
            f"{args.noise_dir}"

        )


    print(
        f"Images found: {len(image_paths)}"
    )


    print()


    #Starting total timer
    total_start = time.perf_counter()


    #Storage for processing times
    processing_times = []


    #Processing every noisy image
    for index, image_path in enumerate(

        image_paths,

        start=1

    ):

        #Loading RGB image
        noisy = np.asarray(

            Image.open(
                image_path
            ).convert("RGB")

        )


        #Starting image timer
        image_start = time.perf_counter()


        #Running complete pipeline
        (
            final_output,
            predicted_severity,
            severity_group,
            alpha

        ) = process_image(

            noisy,

            dncnn_model,

            severity_model,

            device

        )


        #Calculating image runtime
        image_time = (

            time.perf_counter()

            -

            image_start

        )


        processing_times.append(
            image_time
        )


        #Removing _noise from filename
        output_id = strip_noise_suffix(

            image_path.stem

        )


        #Creating required PNG output path
        output_path = (

            args.denoised_dir

            / f"{output_id}.png"

        )


        #Saving final denoised image
        Image.fromarray(

            final_output,

            mode="RGB"

        ).save(

            output_path

        )


        #Printing image information
        print(

            f"[{index:02d}/{len(image_paths):02d}] "

            f"{image_path.name} -> "

            f"{output_path.name} | "

            f"severity={predicted_severity:.4f} | "

            f"group={severity_group} | "

            f"DnCNN={alpha:.2f} | "

            f"Wavelet={1.0 - alpha:.2f} | "

            f"{image_time:.3f}s"

        )


    #Calculating total runtime
    total_time = (

        time.perf_counter()

        -

        total_start

    )


    #Calculating average runtime
    average_time = np.mean(
        processing_times
    )


    print()

    print(
        "=" * 80
    )

    print(
        "DENOISING COMPLETE"
    )

    print(
        "=" * 80
    )


    print(

        f"Images processed     : "
        f"{len(image_paths)}"

    )


    print(

        f"Average runtime/image: "
        f"{average_time:.3f} s"

    )


    print(

        f"Total runtime        : "
        f"{total_time:.2f} s"

    )


    print(

        f"Outputs saved to     : "
        f"{args.denoised_dir}"

    )


    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()