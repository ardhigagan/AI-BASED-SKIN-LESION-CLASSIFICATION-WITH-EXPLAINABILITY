from pathlib import Path

import cv2
import numpy as np


# ===========================================================================
# INPUT / OUTPUT FOLDERS
# ===========================================================================

# Folder that contains the images to process
INPUT_FOLDER = Path("Testing_data")

# Folder where processed images are saved (same file names as the input)
OUTPUT_FOLDER = Path("Testing_data_preprocessed")

# If True, images inside sub-folders of INPUT_FOLDER are processed too and
# the same sub-folder structure is created inside OUTPUT_FOLDER.
INCLUDE_SUBFOLDERS = True

# If True, skip images that already exist in OUTPUT_FOLDER (lets you resume)
SKIP_EXISTING = False

# Debug files (hair mask + hair-removed-only image) are saved for every
# image, in these folders, with the same file names as the inputs, so you
# can inspect what the hair detector is doing.
SAVE_DEBUG = True
DEBUG_MASK_FOLDER = Path("Testing_data_debug_masks")
DEBUG_HAIR_REMOVED_FOLDER = Path("Testing_data_debug_hair_removed")


# ===========================================================================
# SUPPORTED IMAGE EXTENSIONS
# ===========================================================================

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


# ===========================================================================
# CONFIGURATION
# ===========================================================================

CLAHE_CLIP_LIMIT = 1.5
CLAHE_TILE_GRID = (8, 8)
CLAHE_STRENGTH = 0.5

SELECTED_DENOISER = "none"

MEDIAN_KERNEL = 3

GAUSSIAN_KERNEL = 3
GAUSSIAN_SIGMA = 0

MEAN_KERNEL = 3

BILATERAL_DIAMETER = 5
BILATERAL_SIGMA_COLOR = 50
BILATERAL_SIGMA_SPACE = 50


# ===========================================================================
# HAIR DETECTION
# ===========================================================================
# Technique: black-hat morphology -> threshold -> component filtering ->
# dilation -> cv2.inpaint (TELEA) -> median smoothing.
#
# The black-hat uses a THIN LINE kernel at many angles (instead of one big
# round kernel), so thin dark hairs respond strongly while wide dark blobs
# (lesion texture) do not. Detection is repeated in several passes, each
# pass inpainting before the next one detects.

# Each pass = (line_length_fraction, min_component_length_fraction,
#              seed_threshold, dilation_kernel)
#
#   Pass 1: sharp, in-focus hairs
#   Pass 2: wide / blurry out-of-focus hairs (longer line kernel)
#   Pass 3: leftovers (strict, so it can't eat lesion texture)
#
# TUNING:
#   * hairs still left       -> lower seed_thr (e.g. 14 -> 10),
#                               lower min_component_length_fraction
#   * lesion looks smeared   -> raise seed_thr, lower dilation kernel
#   * ghost outlines of hair -> raise dilation kernel
HAIR_PASSES = [
    (0.022, 0.05, 14, 7),
    (0.050, 0.08, 10, 9),
    (0.022, 0.10, 20, 7),
]

# Number of orientations of the thin line kernel (more = catches curved
# hairs better, slower)
HAIR_N_ANGLES = 16

# Hysteresis thresholding: weak pixels are only kept if their connected
# component also contains at least one strong "seed" pixel.
HAIR_WEAK_THRESHOLD = 9

# A pixel must beat the LOCAL AVERAGE black-hat response by this factor.
# Lesion texture has a high local average (so it is suppressed); an
# isolated hair stands far above it.
HAIR_WEAK_RATIO = 1.5
HAIR_SEED_RATIO = 2.0
HAIR_LOCAL_WINDOW_FRACTION = 0.06

# Last-resort safety net: if the mask covers more than this fraction of the
# image, stop adding passes (something is badly wrong).
HAIR_MAX_COVERAGE = 0.65


# ===========================================================================
# HAIR REMOVAL
# ===========================================================================

INPAINT_RADIUS = 3
HAIR_SMOOTHING_KERNEL = 3


# ===========================================================================
# OPTIONAL SHARPENING
# ===========================================================================

USE_FINAL_SHARPEN = False

FINAL_SHARPEN_SIGMA = 1.0
FINAL_SHARPEN_AMOUNT = 0.25


# ===========================================================================
# WHITE STREAK REMOVAL (runs right after hair removal)
# ===========================================================================
# Removes the bright, glassy streaks that are left behind after hair removal.
# Technique: oriented WHITE top-hat (thin bright lines at many angles) ->
# hysteresis threshold -> component filtering -> dilation -> smooth
# multi-scale fill from nearby clean pixels (NOT cv2.inpaint, which drags
# neighbouring white streaks into the hole and gives a blotchy mosaic).
#
# TUNING:
#   * streaks still left     -> lower WHITE_SEED_THR / WHITE_WEAK_THR,
#                               or set WHITE_PASSES = 3
#   * lesion looks too smooth -> raise WHITE_SEED_THR / WHITE_WEAK_THR,
#                               or set WHITE_PASSES = 1
#   * wide streaks remain    -> raise WHITE_LINE_LENGTH (must be larger
#                               than the widest streak, in pixels)

USE_WHITE_STREAK_REMOVAL = True

WHITE_PASSES = 2
WHITE_LINE_LENGTH = 31
WHITE_N_ANGLES = 24
WHITE_SEED_THR = 22
WHITE_WEAK_THR = 12
WHITE_SEED_RATIO = 2.0
WHITE_WEAK_RATIO = 1.5
WHITE_LOCAL_WINDOW = 61
WHITE_MIN_STREAK_DIAG = 40
WHITE_DILATION = 7
WHITE_FEATHER_SIGMA = 3
WHITE_FILL_SIGMAS = (3, 6, 12, 24, 48)


# ===========================================================================
# IMAGE READING / SAVING
# ===========================================================================

def read_bgr(path: Path) -> np.ndarray:

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {path}"
        )

    return image


def save_image(
    path: Path,
    image: np.ndarray
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not cv2.imwrite(
        str(path),
        image
    ):
        raise IOError(
            f"Could not save image: {path}"
        )


# ===========================================================================
# UTILITY FUNCTIONS
# ===========================================================================

def odd_kernel(value: int) -> int:

    value = max(
        1,
        int(value)
    )

    return (
        value
        if value % 2
        else value + 1
    )


def normalize_to_uint8(
    channel: np.ndarray
) -> np.ndarray:

    channel = channel.astype(
        np.float32
    )

    mn = float(channel.min())
    mx = float(channel.max())

    if mx - mn < 1e-8:
        return np.zeros_like(
            channel,
            dtype=np.uint8
        )

    return (
        (
            (channel - mn)
            / (mx - mn)
            * 255.0
        )
        .clip(0, 255)
        .astype(np.uint8)
    )


def bgr_to_rgb(
    image_bgr: np.ndarray
) -> np.ndarray:

    return cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )


# ===========================================================================
# STEP 1: HAIR DETECTION + REMOVAL
# ===========================================================================

def line_kernel(
    length: int,
    angle_deg: float
) -> np.ndarray:

    # A 1-pixel-thick line structuring element at the given angle.

    k = np.zeros(
        (length, length),
        np.uint8
    )

    c = length // 2

    a = np.deg2rad(angle_deg)

    dx = np.cos(a) * c
    dy = np.sin(a) * c

    cv2.line(
        k,
        (
            int(round(c - dx)),
            int(round(c - dy))
        ),
        (
            int(round(c + dx)),
            int(round(c + dy))
        ),
        1,
        1
    )

    return k


def oriented_blackhat(
    gray: np.ndarray,
    length: int
) -> np.ndarray:

    # Black-hat with a THIN LINE kernel at many angles, max over angles.
    # Thin dark lines (hair) respond; wide dark blobs (lesion) do not.

    pad = length

    padded = cv2.copyMakeBorder(
        gray,
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_REFLECT_101
    )

    out = np.zeros_like(padded)

    for i in range(HAIR_N_ANGLES):

        kern = line_kernel(
            length,
            i * 180.0 / HAIR_N_ANGLES
        )

        out = np.maximum(
            out,
            cv2.morphologyEx(
                padded,
                cv2.MORPH_BLACKHAT,
                kern
            )
        )

    h, w = gray.shape

    return out[
        pad:pad + h,
        pad:pad + w
    ]


def find_hair_mask(
    image_bgr: np.ndarray,
    len_frac: float = 0.022,
    min_len_frac: float = 0.05,
    seed_thr: float = 14,
    dilation: int = 7
) -> np.ndarray:

    h, w = image_bgr.shape[:2]

    short = min(h, w)

    gray = cv2.GaussianBlur(
        cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2GRAY
        ),
        (3, 3),
        0
    )

    blackhat = oriented_blackhat(
        gray,
        odd_kernel(
            max(
                11,
                round(len_frac * short)
            )
        )
    ).astype(np.float32)

    # Local average response: lesion texture has a high local average,
    # an isolated hair stands far above it.
    win = odd_kernel(
        round(
            HAIR_LOCAL_WINDOW_FRACTION * short
        )
    )

    local = cv2.blur(
        blackhat,
        (win, win)
    )

    seeds = (
        (blackhat > seed_thr)
        & (blackhat > HAIR_SEED_RATIO * local)
    ).astype(np.uint8)

    weak = (
        (blackhat > HAIR_WEAK_THRESHOLD)
        & (blackhat > HAIR_WEAK_RATIO * local)
    ).astype(np.uint8)

    # Hysteresis: keep weak components that contain at least one seed

    n, labels = cv2.connectedComponents(
        weak,
        connectivity=8
    )

    good = np.zeros(
        n,
        dtype=bool
    )

    good[
        np.unique(
            labels[seeds > 0]
        )
    ] = True

    good[0] = False

    mask = (
        good[labels]
        .astype(np.uint8)
    ) * 255

    # Drop short specks (texture, dust)

    n, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    diag = np.hypot(
        stats[:, cv2.CC_STAT_WIDTH],
        stats[:, cv2.CC_STAT_HEIGHT]
    )

    keep = diag >= min_len_frac * short

    keep[0] = False

    mask = (
        keep[labels]
        .astype(np.uint8)
    ) * 255

    d = odd_kernel(dilation)

    return cv2.dilate(
        mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (d, d)
        )
    )


def remove_hair(
    image_bgr: np.ndarray
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    bool
]:

    current = image_bgr.copy()

    total = np.zeros(
        image_bgr.shape[:2],
        np.uint8
    )

    k = odd_kernel(
        HAIR_SMOOTHING_KERNEL
    )

    for len_frac, min_len, seed_thr, dil in HAIR_PASSES:

        mask = find_hair_mask(
            current,
            len_frac,
            min_len,
            seed_thr,
            dil
        )

        if not np.any(mask):
            continue

        # Safety net
        if (
            np.count_nonzero(total | mask)
            / mask.size
            > HAIR_MAX_COVERAGE
        ):
            break

        total |= mask

        inpainted = cv2.medianBlur(
            cv2.inpaint(
                current,
                mask,
                INPAINT_RADIUS,
                cv2.INPAINT_TELEA
            ),
            k
        )

        current[mask > 0] = inpainted[mask > 0]

    coverage = (
        float(np.count_nonzero(total))
        / float(total.size)
    )

    # Return: (image, mask, coverage, skipped)
    return (
        current,
        total,
        coverage,
        False
    )


# ===========================================================================
# STEP 2: WHITE STREAK REMOVAL
# ===========================================================================

def oriented_tophat(
    gray: np.ndarray,
    length: int
) -> np.ndarray:

    # White top-hat with a THIN LINE kernel at many angles, max over angles.
    # Thin bright lines (glassy streaks) respond; wide bright areas do not.

    pad = length

    padded = cv2.copyMakeBorder(
        gray,
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_REFLECT_101
    )

    out = np.zeros_like(padded)

    for i in range(WHITE_N_ANGLES):

        kern = line_kernel(
            length,
            i * 180.0 / WHITE_N_ANGLES
        )

        out = np.maximum(
            out,
            cv2.morphologyEx(
                padded,
                cv2.MORPH_TOPHAT,
                kern
            )
        )

    h, w = gray.shape

    return out[
        pad:pad + h,
        pad:pad + w
    ]


def find_white_mask(
    image_bgr: np.ndarray
) -> np.ndarray:

    gray = cv2.GaussianBlur(
        cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2GRAY
        ),
        (3, 3),
        0
    )

    tophat = oriented_tophat(
        gray,
        WHITE_LINE_LENGTH
    ).astype(np.float32)

    local = cv2.blur(
        tophat,
        (WHITE_LOCAL_WINDOW, WHITE_LOCAL_WINDOW)
    )

    seeds = (
        (tophat > WHITE_SEED_THR)
        & (tophat > WHITE_SEED_RATIO * local)
    ).astype(np.uint8)

    weak = (
        (tophat > WHITE_WEAK_THR)
        & (tophat > WHITE_WEAK_RATIO * local)
    ).astype(np.uint8)

    # Hysteresis: keep weak components that contain at least one seed

    n, labels = cv2.connectedComponents(
        weak,
        connectivity=8
    )

    good = np.zeros(
        n,
        dtype=bool
    )

    good[
        np.unique(
            labels[seeds > 0]
        )
    ] = True

    good[0] = False

    mask = (
        good[labels]
        .astype(np.uint8)
    ) * 255

    # Drop short specks

    n, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    diag = np.hypot(
        stats[:, cv2.CC_STAT_WIDTH],
        stats[:, cv2.CC_STAT_HEIGHT]
    )

    keep = diag >= WHITE_MIN_STREAK_DIAG

    keep[0] = False

    mask = (
        keep[labels]
        .astype(np.uint8)
    ) * 255

    d = odd_kernel(WHITE_DILATION)

    return cv2.dilate(
        mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (d, d)
        )
    )


def smooth_fill(
    image_bgr: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:

    # Fill masked pixels with a weighted average of nearby VALID pixels,
    # using the smallest blur scale that has enough clean support.

    valid = (mask == 0).astype(np.float32)

    f = image_bgr.astype(np.float32)

    out = f.copy()

    remaining = mask > 0

    num = None
    den = None

    for s in WHITE_FILL_SIGMAS:

        k = odd_kernel(int(s * 6))

        num = cv2.GaussianBlur(
            f * valid[..., None],
            (k, k),
            s
        )

        den = cv2.GaussianBlur(
            valid,
            (k, k),
            s
        )

        ok = remaining & (den > 0.25)

        out[ok] = num[ok] / den[ok][:, None]

        remaining &= ~ok

    if remaining.any():

        out[remaining] = (
            num[remaining]
            / np.maximum(den[remaining], 1e-3)[:, None]
        )

    return np.clip(
        out,
        0,
        255
    ).astype(np.uint8)


def remove_white_streaks(
    image_bgr: np.ndarray
) -> tuple[
    np.ndarray,
    np.ndarray,
    float
]:

    current = image_bgr.copy()

    total = np.zeros(
        image_bgr.shape[:2],
        np.uint8
    )

    for _ in range(WHITE_PASSES):

        mask = find_white_mask(
            current
        )

        if not np.any(mask):
            break

        total |= mask

        fill = smooth_fill(
            current,
            mask
        )

        alpha = cv2.GaussianBlur(
            mask.astype(np.float32) / 255.0,
            (0, 0),
            WHITE_FEATHER_SIGMA
        )

        alpha = np.clip(
            alpha * 1.5,
            0,
            1
        )[..., None]

        current = (
            fill * alpha
            + current * (1 - alpha)
        ).astype(np.uint8)

    coverage = (
        float(np.count_nonzero(total))
        / float(total.size)
    )

    return (
        current,
        total,
        coverage
    )


# ===========================================================================
# STEP 3: DENOISING
# ===========================================================================

def denoise_median(
    image_bgr: np.ndarray
) -> np.ndarray:

    return cv2.medianBlur(
        image_bgr,
        odd_kernel(MEDIAN_KERNEL)
    )


def denoise_gaussian(
    image_bgr: np.ndarray
) -> np.ndarray:

    k = odd_kernel(
        GAUSSIAN_KERNEL
    )

    return cv2.GaussianBlur(
        image_bgr,
        (k, k),
        GAUSSIAN_SIGMA
    )


def denoise_mean(
    image_bgr: np.ndarray
) -> np.ndarray:

    k = odd_kernel(
        MEAN_KERNEL
    )

    return cv2.blur(
        image_bgr,
        (k, k)
    )


def denoise_bilateral(
    image_bgr: np.ndarray
) -> np.ndarray:

    return cv2.bilateralFilter(
        image_bgr,
        BILATERAL_DIAMETER,
        BILATERAL_SIGMA_COLOR,
        BILATERAL_SIGMA_SPACE
    )


def denoise_none(
    image_bgr: np.ndarray
) -> np.ndarray:

    return image_bgr.copy()


DENOISERS = {
    "none": denoise_none,
    "median": denoise_median,
    "gaussian": denoise_gaussian,
    "mean": denoise_mean,
    "bilateral": denoise_bilateral
}


# ===========================================================================
# STEP 4: CONTRAST ENHANCEMENT
# ===========================================================================

def rgb_to_lab(
    image_bgr: np.ndarray
) -> np.ndarray:

    return cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2LAB
    )


def clahe_lab(
    image_bgr: np.ndarray
) -> np.ndarray:

    lab = rgb_to_lab(
        image_bgr
    )

    l, a, b = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID
    )

    enhanced_l = clahe.apply(
        l
    )

    strength = float(
        np.clip(
            CLAHE_STRENGTH,
            0.0,
            1.0
        )
    )

    if strength < 1.0:

        enhanced_l = cv2.addWeighted(
            enhanced_l,
            strength,
            l,
            1.0 - strength,
            0
        )

    return cv2.cvtColor(
        cv2.merge(
            (
                enhanced_l,
                a,
                b
            )
        ),
        cv2.COLOR_LAB2BGR
    )


# ===========================================================================
# STEP 5: OPTIONAL SHARPENING
# ===========================================================================

def final_sharpen(
    image_bgr: np.ndarray
) -> np.ndarray:

    blurred = cv2.GaussianBlur(
        image_bgr,
        (0, 0),
        FINAL_SHARPEN_SIGMA
    )

    result = cv2.addWeighted(
        image_bgr,
        1.0 + FINAL_SHARPEN_AMOUNT,
        blurred,
        -FINAL_SHARPEN_AMOUNT,
        0
    )

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


# ===========================================================================
# MAIN PREPROCESSING PIPELINE (one image)
# ===========================================================================
# Steps: hair removal -> white streak removal -> denoising -> CLAHE ->
# optional sharpening.
# (No resizing: the output keeps the original image size.)

def preprocess_image(
    original_bgr: np.ndarray,
    debug_mask_path: Path = None,
    debug_hair_removed_path: Path = None
) -> np.ndarray:

    # STEP 1: HAIR REMOVAL

    hair_removed, hair_mask, hair_coverage, hair_skipped = (
        remove_hair(
            original_bgr
        )
    )

    # --- DEBUG OUTPUT: shows exactly what the hair detector is doing ---
    print()
    print("---- HAIR DETECTION DEBUG ----")
    print(f"Hair coverage      : {hair_coverage * 100:.2f}% of image")
    print(f"Hair removal skipped (mask unreliable): {hair_skipped}")
    if hair_coverage == 0.0:
        print("No hair-like structures were detected at all.")
    print("-------------------------------")
    print()

    if SAVE_DEBUG:
        if debug_mask_path is not None:
            save_image(debug_mask_path, hair_mask)
        if debug_hair_removed_path is not None:
            save_image(debug_hair_removed_path, hair_removed)
    # --- END DEBUG OUTPUT ---

    # STEP 2: WHITE STREAK REMOVAL

    if USE_WHITE_STREAK_REMOVAL:

        streak_free, white_mask, white_coverage = (
            remove_white_streaks(
                hair_removed
            )
        )

        print("---- WHITE STREAK DEBUG ----")
        print(f"White streak coverage: {white_coverage * 100:.2f}% of image")
        print("----------------------------")
        print()

    else:

        streak_free = hair_removed

    # STEP 3: DENOISING

    denoised = DENOISERS[
        SELECTED_DENOISER
    ](
        streak_free
    )

    # STEP 4: CLAHE

    contrast = clahe_lab(
        denoised
    )

    # STEP 5: OPTIONAL SHARPENING

    if USE_FINAL_SHARPEN:

        final = final_sharpen(
            contrast
        )

    else:

        final = contrast

    return final


# ===========================================================================
# FOLDER HELPERS
# ===========================================================================

def list_images(
    folder: Path
) -> list[Path]:

    iterator = (
        folder.rglob("*")
        if INCLUDE_SUBFOLDERS
        else folder.glob("*")
    )

    files = [
        p for p in iterator
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(files)


def is_inside(
    child: Path,
    parent: Path
) -> bool:

    try:
        child.resolve().relative_to(
            parent.resolve()
        )
        return True

    except ValueError:
        return False


# ===========================================================================
# MAIN — WHOLE FOLDER
# ===========================================================================

def main():

    if not INPUT_FOLDER.exists() or not INPUT_FOLDER.is_dir():

        raise FileNotFoundError(
            f"Input folder not found: "
            f"{INPUT_FOLDER.resolve()}"
        )

    if is_inside(OUTPUT_FOLDER, INPUT_FOLDER):

        raise ValueError(
            "OUTPUT_FOLDER must not be inside INPUT_FOLDER."
        )

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    images = list_images(
        INPUT_FOLDER
    )

    print("========================================")
    print("FOLDER IMAGE PREPROCESSING")
    print("========================================")
    print(f"Input folder : {INPUT_FOLDER.resolve()}")
    print(f"Output folder: {OUTPUT_FOLDER.resolve()}")
    print(f"Images found : {len(images)}")
    print(f"White streaks: {'on' if USE_WHITE_STREAK_REMOVAL else 'off'}")
    print(f"Denoiser     : {SELECTED_DENOISER}")
    print(
        f"CLAHE        : {CLAHE_CLIP_LIMIT}"
        f" | strength={CLAHE_STRENGTH}"
    )
    print("========================================")
    print()

    if not images:
        print("No supported images found. Nothing to do.")
        return

    done = 0
    skipped = 0
    failed = []

    for idx, img_path in enumerate(images, start=1):

        rel = img_path.relative_to(
            INPUT_FOLDER
        )

        out_path = OUTPUT_FOLDER / rel

        prefix = f"[{idx}/{len(images)}] {rel}"

        if SKIP_EXISTING and out_path.exists():

            print(f"{prefix} -> skipped (already exists)")

            skipped += 1

            continue

        try:

            original = read_bgr(
                img_path
            )

            print("Image loaded successfully.")

            final = preprocess_image(
                original,
                (DEBUG_MASK_FOLDER / rel).with_suffix(".png"),
                DEBUG_HAIR_REMOVED_FOLDER / rel
            )

            save_image(
                out_path,
                final
            )

            print(f"{prefix} -> done")

            done += 1

        except Exception as e:

            print(f"{prefix} -> FAILED: {e}")

            failed.append(
                (rel, str(e))
            )

    print()
    print("========================================")
    print("PROCESSING COMPLETED")
    print("========================================")
    print(f"Processed: {done}")
    print(f"Skipped  : {skipped}")
    print(f"Failed   : {len(failed)}")

    for rel, err in failed:
        print(f"  - {rel}: {err}")

    print(f"Saved to : {OUTPUT_FOLDER.resolve()}")
    print("========================================")


# ===========================================================================
# RUN
# ===========================================================================

if __name__ == "__main__":
    main()