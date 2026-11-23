"""
Shared palmprint engine for enrollment and identification.

Keeps one definition of: ROI extraction, embedding computation, and the
enrolled-identity database, so the Enrollment and Identification features
always speak the same feature space.
"""
import io
import json
import os
import sqlite3
from datetime import datetime

import cv2
import numpy as np
import torch

from preprocess.preprocessor import (
    enhance_contrast_clahe,
    extract_roi_centroid,
    normalize_roi,
    reduce_noise,
    resize_roi,
    segment_palm,
)
from self_supervised.model import PalmprintEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODER_PATH = os.path.join(BASE_DIR, 'output', 'model', 'palmprint_encoder.pth')
DB_PATH = os.path.join(BASE_DIR, 'data', 'palm_users.db')
ROI_DIR = os.path.join(BASE_DIR, 'data', 'enrolled_rois')
# The full frame each enrolment came from. Keeping it means a change to the
# extraction geometry can be applied by rebuilding templates, instead of asking
# everybody to enrol again -- which this project has already had to do twice.
RAW_DIR = os.path.join(BASE_DIR, 'data', 'enrolled_frames')
GALLERY_DIR = os.path.join(BASE_DIR, 'dataset', 'preprocessed_images')
EMB_PATH = os.path.join(BASE_DIR, 'output', 'embeddings', 'all_embeddings.npy')
NAMES_PATH = os.path.join(BASE_DIR, 'output', 'embeddings', 'image_names.npy')

CAMERA_SETTINGS_PATH = os.path.join(BASE_DIR, 'data', 'camera_settings.json')

TARGET_SIZE = (138, 138)      # encoder input size
PALM_STORE_SIZE = (256, 256)  # full-palm image kept in the database
PALM_MARGIN = 0.08            # breathing room around the detected hand
# ROI side as a multiple of the inscribed-circle radius. Sizing from the palm's
# own radius is what makes the crop scale-invariant; the factor decides how much
# of the palm is kept. 1.6 held only an inscribed patch (~60% of the hand), so
# the stored image was not a full palmprint. 2.6 spans the whole palm.
PALM_ROI_RADIUS_FACTOR = 2.6
# Fraction of the crop that must be hand. The crop is masked, so the remainder
# is black padding; too little means the palm is cut off or not flat.
MIN_PALM_COVERAGE = 0.55
# How much redder than blue a pixel must be to count as skin (colour images).
SKIN_RED_MARGIN = 10
EMBEDDING_DIM = 512  # ResNet-18 trunk output, matching output/embeddings
# Swept on the real enrolled palms (24 re-captures x 2 people, plus 1500
# unenrolled strangers drawn from the dataset gallery):
#
#   thresh | genuine FRR | enrolled-impostor accepts | stranger accepts
#   0.70   |   0.0%      | 0 (19 blocked by the      | 70/3000
#   0.75   |   0.0%      | 0  relative check)        | 22/3000
#   0.80   |  12.5%      | 0                         | 14/3000
#   0.84   |  16.7%      | 0                         |  2/3000
#   0.88   |  16.7%      | 0                         |  0/3000
#   0.95   |  29.2%      | 0                         |  0/3000
#
# 0.88 is the lowest threshold with zero stranger acceptances, and raising it
# further only costs genuine accepts. Re-calibrate if the pipeline changes.
DEFAULT_THRESHOLD = 0.88
# The top candidate must beat the runner-up by this much to be accepted.
DEFAULT_MARGIN = 0.06
# Crease energy (variance of the Laplacian) a ROI must show to be a real palm.
# Measured on the extracted ROI: real palms 0.0037-0.0130 (the low end being a
# soft, low-contrast capture), while flat walls and smooth gradients sit at
# 0.0000-0.0001. 0.0012 is ~10x above the flat cases and ~3x below the weakest
# genuine palm, so it rejects featureless frames without rejecting real ones.
MIN_PALM_DETAIL = 0.0012

# Extraction methods. Embeddings are only comparable within the same method, so
# the method a sample was enrolled with is recorded alongside it.
METHOD_FULL_PALM = "full-palm-v2"    # whole palmprint, background masked out
METHOD_DATASET = "dataset-style"     # threshold + centroid crop (Kaggle TIFFs)
# Superseded extractions. Their geometry differs, so their embeddings are not
# comparable with current captures and must never take part in matching.
METHOD_LEGACY_CROP = "webcam/photo"  # fixed centre crop
METHOD_LEGACY_INSCRIBED = "full-palm"  # inscribed patch, only ~60% of the palm
STALE_METHODS = (METHOD_LEGACY_CROP, METHOD_LEGACY_INSCRIBED)

LEFT, RIGHT = "Left Hand", "Right Hand"
REQUIRED_HANDS = (LEFT, RIGHT)  # a person must enroll both before they can log in


# ===================== 0. CAMERA ORIENTATION =====================
# Laptop webcams vary: some stream upside-down, some mirrored, some both. The
# correction is a stored setting rather than a hardcoded flip, and it is applied
# to every frame before detection so preview and processing never disagree.
DEFAULT_CAMERA_SETTINGS = {"index": 0, "rotation": 0, "mirror": True}

_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def load_camera_settings():
    settings = dict(DEFAULT_CAMERA_SETTINGS)
    if os.path.exists(CAMERA_SETTINGS_PATH):
        try:
            with open(CAMERA_SETTINGS_PATH, "r", encoding="utf-8") as f:
                stored = json.load(f)
            if isinstance(stored, dict):
                settings.update({k: stored[k] for k in settings if k in stored})
        except (OSError, ValueError):
            pass
    settings["rotation"] = int(settings["rotation"]) % 360
    if settings["rotation"] not in (0, 90, 180, 270):
        settings["rotation"] = 0
    settings["mirror"] = bool(settings["mirror"])
    settings["index"] = int(settings["index"])
    return settings


def save_camera_settings(settings):
    os.makedirs(os.path.dirname(CAMERA_SETTINGS_PATH), exist_ok=True)
    with open(CAMERA_SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


def apply_orientation(frame, settings=None):
    """Corrects a raw camera frame: rotate first, then mirror."""
    if frame is None:
        return None
    settings = settings or load_camera_settings()
    rot = _ROTATIONS.get(int(settings.get("rotation", 0)) % 360)
    if rot is not None:
        frame = cv2.rotate(frame, rot)
    if settings.get("mirror", False):
        frame = cv2.flip(frame, 1)
    return frame


# ===================== 1. MODEL =====================
def load_encoder():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = PalmprintEncoder().encoder
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
    encoder.eval()
    encoder.to(device)
    return encoder, device


# ===================== 2. ROI EXTRACTION =====================
def roi_dataset_style(bgr):
    """Threshold + centroid ROI, identical to preprocess/preprocessor.py.

    This is the path the 11,996 dataset embeddings were built with, so dataset
    images (dark background TIFFs) must go through exactly this.
    """
    # NOTE: RGB2GRAY on a BGR array is intentional -- it reproduces the exact
    # channel weighting used when the dataset embeddings were generated.
    gray = cv2.cvtColor(bgr, cv2.COLOR_RGB2GRAY)
    enhanced = reduce_noise(enhance_contrast_clahe(gray))
    _, contours = segment_palm(enhanced)
    roi = extract_roi_centroid(enhanced, contours)
    if roi is None or roi.size == 0:
        return None
    return normalize_roi(resize_roi(roi))


def is_greyscale(bgr, tolerance=6):
    """True when the three channels are effectively identical."""
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        return True
    b, g, r = cv2.split(bgr)
    return (int(cv2.absdiff(b, g).max()) <= tolerance
            and int(cv2.absdiff(g, r).max()) <= tolerance)


def classify_hand_side(contour, palm_cx, palm_cy, mirrored=True):
    """Left vs right hand from thumb spread around the palm centroid.

    The label depends on whether the view is mirrored: in a mirrored (selfie)
    view a left palm's thumb extends to the left of frame, and un-mirroring
    swaps that. This is a hint only -- enrollment records the hand the operator
    explicitly selects.
    """
    pts = contour[:, 0, :]
    left_pts = pts[pts[:, 0] < palm_cx]
    right_pts = pts[pts[:, 0] > palm_cx]

    left_lower = left_pts[left_pts[:, 1] > (palm_cy - 20)] if len(left_pts) else []
    right_lower = right_pts[right_pts[:, 1] > (palm_cy - 20)] if len(right_pts) else []

    left_spread = (palm_cx - np.min(left_lower[:, 0])) if len(left_lower) else 0
    right_spread = (np.max(right_lower[:, 0]) - palm_cx) if len(right_lower) else 0

    if left_spread > right_spread + 14:
        return LEFT if mirrored else RIGHT
    if right_spread > left_spread + 14:
        return RIGHT if mirrored else LEFT
    return "Palm"


def detect_palm(bgr, mirrored=True):
    """Auto-detect a human palm in a natural photo or webcam frame.

    Dual color-space skin segmentation (HSV + YCrCb) so it holds up under
    varied lighting. Returns (ok, message, roi_norm, roi_rgb, hand_side).
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    if brightness < 20:
        return False, "Too dark - increase lighting", None, None, "Palm"
    if brightness > 250:
        return False, "Severe glare - adjust angle", None, None, "Palm"

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 18, 30]), np.array([35, 255, 255])),
        cv2.inRange(ycrcb, np.array([0, 120, 60]), np.array([255, 190, 145])),
    )
    # The YCrCb range alone accepts neutral greys, so a wall or a sheet of paper
    # can pass as skin. On a colour image, demand actual skin chroma. A genuinely
    # greyscale image carries no chroma to judge, so it is exempted and relies on
    # the texture gate below instead.
    saturation = hsv[:, :, 1]
    if not is_greyscale(bgr):
        mask = cv2.bitwise_and(mask, (saturation >= 15).astype(np.uint8) * 255)
        # Skin is reddish: measured on real palms the red channel runs 31-47
        # above blue (95% of pixels clear +10), while a grey-blue office wall
        # sits 4 BELOW. Without this, walls and paper pass as skin.
        blue, _, red = cv2.split(bgr.astype(np.int16))
        mask = cv2.bitwise_and(
            mask, (red > blue + SKIN_RED_MARGIN).astype(np.uint8) * 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Deliberately loose: a hand held further back is still a valid capture. The
    # real quality gate is the palm radius measured below, not how much of the
    # frame the hand happens to fill.
    if np.count_nonzero(mask) / (h * w) < 0.03:
        return False, "No palm detected - show your open palm", None, None, "Palm"

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, "No palm detected - show your open palm", None, None, "Palm"

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 5000:
        return False, "Palm too small - bring your hand closer", None, None, "Palm"

    m = cv2.moments(largest)
    if m["m00"] > 0:
        cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
    else:
        cx, cy = w // 2, h // 2

    # --- Register the palm on its own geometry, not on the frame ---------
    # The largest circle that fits inside the hand sits in the palm centre and
    # its radius scales with the palm. Cropping relative to that makes the ROI
    # independent of where the hand sits and how close it is to the camera --
    # which is what lets two captures of one palm actually line up. Fingers are
    # excluded: they are pose, not identity.
    hand_mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(hand_mask, [largest], -1, 255, cv2.FILLED)

    # A one-pixel zero border is essential: if the hand fills the frame the mask
    # has no background pixel, and distanceTransform then returns FLT_MAX
    # everywhere with its maximum at (0,0) -- a degenerate result that silently
    # destroys the crop. The border bounds distances by the frame edge instead.
    dist = cv2.distanceTransform(
        cv2.copyMakeBorder(hand_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0),
        cv2.DIST_L2, 5)
    _, radius, _, centre = cv2.minMaxLoc(dist)
    pcx, pcy = int(centre[0]) - 1, int(centre[1]) - 1
    if not np.isfinite(radius):
        return False, "Could not locate the palm centre", None, None, "Palm"

    if radius < 24:
        return False, "Palm too small - bring your hand closer", None, None, "Palm"

    # Size the crop from the palm's own radius -- that is what makes the ROI
    # scale-invariant -- but large enough to hold the WHOLE palmprint rather
    # than an inscribed patch of it.
    side = int(radius * PALM_ROI_RADIUS_FACTOR)

    # Everything outside the hand goes black, so enlarging the crop cannot let
    # the wall or the person's face into the template. Padding keeps the square
    # centred on the palm instead of sliding off-centre near a frame edge.
    hand_only = cv2.bitwise_and(bgr, bgr, mask=hand_mask)
    pad = side
    padded = cv2.copyMakeBorder(hand_only, pad, pad, pad, pad,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
    padded_mask = cv2.copyMakeBorder(hand_mask, pad, pad, pad, pad,
                                     cv2.BORDER_CONSTANT, value=0)
    half = side // 2
    x1, y1 = pcx + pad - half, pcy + pad - half
    palm_bgr = padded[y1:y1 + side, x1:x1 + side]
    crop_mask = padded_mask[y1:y1 + side, x1:x1 + side]
    if palm_bgr.size == 0 or min(palm_bgr.shape[:2]) < 30:
        return False, "Could not crop a usable palm region", None, None, "Palm"

    # The crop must be mostly palm. Too little means the hand is cut off by the
    # frame or is not presented flat.
    coverage = float(np.count_nonzero(crop_mask)) / float(side * side)
    if coverage < MIN_PALM_COVERAGE:
        return (False, "Show your whole open palm, flat towards the camera",
                None, None, "Palm")

    palm_store = cv2.resize(palm_bgr, PALM_STORE_SIZE, interpolation=cv2.INTER_AREA)
    roi_rgb = cv2.cvtColor(palm_store, cv2.COLOR_BGR2RGB)
    roi_norm = palm_roi_to_input(palm_bgr)

    # Identity lives in the crease pattern. A flat region -- a wall, a blurred
    # or badly lit frame -- has none, and enrolling it would store a template
    # that matches anything equally badly.
    #
    # Measure strictly INSIDE the silhouette: masking the background creates a
    # hard black edge, and that edge alone is enough Laplacian energy to pass
    # the gate, which let a plain wall through as a palm.
    small_mask = cv2.resize(crop_mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    interior = cv2.erode(small_mask,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    laplacian = cv2.Laplacian(roi_norm, cv2.CV_32F)
    inside = laplacian[interior > 0]
    detail = float(inside.var()) if inside.size > 100 else 0.0
    if detail < MIN_PALM_DETAIL:
        return (False, "Palm lines not visible - hold steady, improve lighting",
                None, None, "Palm")

    hand = classify_hand_side(largest, cx, cy, mirrored=mirrored)
    return True, hand + " detected (palm registered)", roi_norm, roi_rgb, hand


# ===================== PALM LINE VISUALISATION =====================
# Minimum black-hat response for a pixel to count as crease rather than noise.
MIN_CREASE_RESPONSE = 8
# Distinct colours for the principal lines, longest first (RGB).
PRINCIPAL_LINE_COLOURS = [(239, 68, 68), (52, 211, 153), (250, 204, 21),
                          (96, 165, 250), (232, 121, 249)]


def detect_palm_lines(palm_rgb, max_principal=3, min_length=40):
    """Find the crease pattern in a palm ROI and draw it.

    Returns (overlay_rgb, creases_rgb, lines). `lines` is a list of dicts with
    'length' and 'mask', longest first. This is a visualisation of the texture
    the matcher relies on -- if nothing is drawn, the capture carries no usable
    crease detail and will not match well.
    """
    if palm_rgb is None or palm_rgb.size == 0:
        return None, None, []

    work = cv2.resize(palm_rgb, PALM_STORE_SIZE, interpolation=cv2.INTER_AREA) \
        if palm_rgb.shape[:2] != PALM_STORE_SIZE else palm_rgb.copy()
    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)

    # Flatten illumination so shading gradients are not mistaken for lines.
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=work.shape[0] / 8.0)
    flat = cv2.divide(gray, background, scale=128)
    enhanced = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(flat)

    # Creases are thin DARK valleys: black-hat keeps structures narrower than
    # the kernel and discards everything broader.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.GaussianBlur(blackhat, (3, 3), 0)

    # Otsu always splits an image in two, even one with no creases at all, so a
    # smooth gradient would yield phantom "lines". Measured black-hat response:
    # real palms threshold at 16-23 (99th pct 43-66), while a gradient lands at
    # 1 and a heavily blurred frame at 2. A floor of 8 separates them.
    otsu, _ = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary = cv2.threshold(blackhat, max(otsu, MIN_CREASE_RESPONSE), 255,
                              cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Rank connected components by how long and elongated they are, so the
    # principal lines win over speckle and skin pores.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    lines = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        span = float(np.hypot(bw, bh))
        if area < 25 or span < min_length:
            continue
        mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        length = cv2.arcLength(contours[0], True) / 2.0 if contours else span
        lines.append({"length": float(length), "span": span, "area": int(area),
                      "mask": mask})
    lines.sort(key=lambda d: d["length"], reverse=True)

    # All creases, on black -- the matcher's eye view.
    creases = np.zeros_like(work)
    creases[binary > 0] = (34, 211, 238)

    # Overlay: every crease faint, the principal lines bold and coloured.
    overlay = work.copy()
    tint = overlay.copy()
    tint[binary > 0] = (34, 211, 238)
    overlay = cv2.addWeighted(tint, 0.45, overlay, 0.55, 0)

    for rank, line in enumerate(lines[:max_principal]):
        colour = PRINCIPAL_LINE_COLOURS[rank % len(PRINCIPAL_LINE_COLOURS)]
        thick = cv2.dilate(line["mask"],
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        overlay[thick > 0] = colour

    return overlay, creases, lines


def palm_roi_to_input(palm_bgr):
    """Palm crop -> 138x138 encoder input, with the creases emphasised.

    Illumination is flattened before CLAHE so that room brightness and skin
    tone stop dominating the embedding; what should drive it is the crease
    pattern, which is the part that differs between people.
    """
    gray = cv2.cvtColor(cv2.resize(palm_bgr, TARGET_SIZE, interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2GRAY)
    # Divide out low-frequency shading (a flat-field correction).
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=TARGET_SIZE[0] / 8.0)
    flat = cv2.divide(gray, background, scale=128)
    return normalize_roi(enhance_contrast_clahe(flat))


def extract_roi(bgr, mirrored=True):
    """Palm ROI from any source.

    Tries live-photo skin detection first, then falls back to the dataset-style
    threshold pipeline so Kaggle TIFFs still work. Returns
    (roi_norm, roi_rgb, hand, source, message); roi_norm is None on failure.
    """
    # Route on what the image IS, rather than letting one pipeline silently catch
    # the other's failures. The dataset pipeline is only valid for dataset-style
    # imagery (a bright palm on a near-black field); applied to an ordinary photo
    # it returns a frame-centred crop of whatever happened to be in the middle --
    # a template that looks real but carries no identity, which is exactly what
    # produces false accepts.
    if looks_like_dataset_image(bgr):
        roi_norm = roi_dataset_style(bgr)
        if roi_norm is None:
            return None, None, "Palm", None, "Dataset-style ROI extraction failed"
        roi_rgb = cv2.cvtColor((roi_norm * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        return (roi_norm, roi_rgb, "Palm", METHOD_DATASET,
                "Palm ROI extracted (dataset pipeline)")

    ok, msg, roi_norm, roi_rgb, hand = detect_palm(bgr, mirrored=mirrored)
    if ok:
        return roi_norm, roi_rgb, hand, METHOD_FULL_PALM, msg
    return None, None, "Palm", None, msg


def looks_like_dataset_image(bgr):
    """True for Kaggle-style captures: a bright palm isolated on a dark field."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dark_fraction = float(np.count_nonzero(gray < 40)) / gray.size
    _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    foreground = float(np.count_nonzero(binary)) / binary.size
    return dark_fraction > 0.35 and 0.05 < foreground < 0.75


def decode_image(data):
    """Bytes (uploaded file or camera snapshot) -> BGR array."""
    arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:  # TIFF variants that imdecode rejects
        try:
            import tifffile
            img = tifffile.imread(io.BytesIO(data))
            img = img.astype(np.uint8)
            bgr = (cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2
                   else cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR))
        except Exception:
            return None
    return bgr


# ===================== 3. EMBEDDINGS =====================
def embed(roi_norm, encoder, device):
    """138x138 float ROI in [0,1] -> L2-normalized 512-D feature vector."""
    tensor = torch.from_numpy(roi_norm[None]).float().repeat(1, 3, 1, 1).to(device)
    with torch.no_grad():
        vec = encoder(tensor).cpu().numpy().flatten()
    return vec / (np.linalg.norm(vec) + 1e-8)


def blob_to_vec(blob, strict=False):
    """Decodes an embedding BLOB to an L2-normalized vector.

    Samples are written as float32, but older rows may use float64, so pick the
    dtype whose length lands on EMBEDDING_DIM. Returns None when `strict` and
    the bytes do not decode to a usable vector -- the legacy `users` table in
    this project holds blobs that are not embeddings at all.
    """
    vec = None
    for dtype in (np.float32, np.float64):
        if len(blob) == EMBEDDING_DIM * np.dtype(dtype).itemsize:
            vec = np.frombuffer(blob, dtype=dtype).astype(np.float32)
            break
    if vec is None:
        if strict:
            return None
        vec = np.frombuffer(blob, dtype=np.float32)  # unknown layout: best effort

    norm = float(np.linalg.norm(vec.astype(np.float64)))
    if strict and (not np.isfinite(norm) or norm < 1e-6 or not np.all(np.isfinite(vec))):
        return None
    return vec / (norm + 1e-8)


# ===================== 4. DATABASE =====================
def connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(encoder=None, device=None):
    """Creates the enrollment schema and migrates any legacy `users` rows.

    Legacy feature blobs are only trusted when they decode to a valid
    EMBEDDING_DIM vector; otherwise the embedding is recomputed from the stored
    ROI image (when an encoder is supplied). Returns a list of migration notes.
    """
    notes = []
    conn = connect()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS persons (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT NOT NULL UNIQUE,
        person_code TEXT,
        email       TEXT,
        phone       TEXT,
        department  TEXT,
        notes       TEXT,
        created_at  TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS samples (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id  INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
        hand       TEXT,
        source     TEXT,
        roi_path   TEXT,
        raw_path   TEXT,
        embedding  BLOB NOT NULL,
        dim        INTEGER NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS identify_log (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        ts         TEXT NOT NULL,
        matched    TEXT,
        score      REAL,
        threshold  REAL,
        decision   TEXT,
        source     TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_samples_person ON samples(person_id);
    """)

    # CREATE TABLE IF NOT EXISTS leaves an existing table alone, so columns
    # added later have to be migrated in explicitly.
    existing = {r[1] for r in conn.execute("PRAGMA table_info(samples)")}
    if "raw_path" not in existing:
        conn.execute("ALTER TABLE samples ADD COLUMN raw_path TEXT")
        notes.append("added samples.raw_path (original frames kept from now on)")

    # Migrate the legacy single-sample `users` table, if present.
    legacy = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='users'").fetchone()
    if legacy:
        for row in conn.execute(
                "SELECT name, enrolled_at, roi_path, features FROM users").fetchall():
            if conn.execute("SELECT id FROM persons WHERE name = ?",
                            (row["name"],)).fetchone():
                continue

            vec = blob_to_vec(row["features"], strict=True)
            source = "legacy"
            if vec is None:
                # Legacy blob is not a usable embedding; rebuild it from the ROI.
                roi_path = os.path.join(BASE_DIR, (row["roi_path"] or "").replace("\\", os.sep))
                if encoder is not None and row["roi_path"] and os.path.exists(roi_path):
                    bgr = cv2.imread(roi_path)
                    roi_norm, _, _, _, _ = extract_roi(bgr) if bgr is not None \
                        else (None, None, None, None, None)
                    if roi_norm is not None:
                        vec = embed(roi_norm, encoder, device)
                        source = "legacy (re-encoded from ROI)"
                        notes.append("%s: embedding rebuilt from %s"
                                     % (row["name"], os.path.basename(roi_path)))
                if vec is None:
                    notes.append("%s: legacy blob unusable and ROI could not be "
                                 "re-encoded - please re-enroll" % row["name"])
                    continue

            cur = conn.execute(
                "INSERT INTO persons (name, notes, created_at) VALUES (?, ?, ?)",
                (row["name"], "migrated from legacy users table", row["enrolled_at"]))
            conn.execute(
                "INSERT INTO samples (person_id, hand, source, roi_path, embedding, dim, created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (cur.lastrowid, "Palm", source, row["roi_path"],
                 vec.astype(np.float32).tobytes(), len(vec), row["enrolled_at"]))
    conn.commit()
    conn.close()
    return notes


def upsert_person(details):
    """Creates or updates a person record; returns person_id.

    On update, only fields with a supplied value are overwritten -- enrolling an
    extra palm sample with a partly-blank form must not erase details already on
    file.
    """
    conn = connect()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = conn.execute("SELECT id FROM persons WHERE name = ?", (details["name"],)).fetchone()
    if row:
        pid = row["id"]
        conn.execute("""
            UPDATE persons SET
                person_code = COALESCE(?, person_code),
                email       = COALESCE(?, email),
                phone       = COALESCE(?, phone),
                department  = COALESCE(?, department),
                notes       = COALESCE(?, notes)
            WHERE id = ?""",
            (details.get("person_code") or None, details.get("email") or None,
             details.get("phone") or None, details.get("department") or None,
             details.get("notes") or None, pid))
    else:
        cur = conn.execute(
            "INSERT INTO persons (name, person_code, email, phone, department, notes, created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (details["name"], details.get("person_code"), details.get("email"),
             details.get("phone"), details.get("department"), details.get("notes"), now))
        pid = cur.lastrowid
    conn.commit()
    conn.close()
    return pid


def add_sample(person_id, name, embedding, roi_norm, roi_rgb, hand, source,
               append_to_gallery=True, raw_bgr=None):
    """Stores one palm sample.

    ROI image goes to disk, embedding into SQLite, and (optionally) the ROI is
    appended to the dataset gallery so the dataset-wide viewer in app.py can
    surface this person too.
    """
    os.makedirs(ROI_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    slug = "".join(c if c.isalnum() else "_" for c in name).strip("_").lower() or "person"
    roi_path = os.path.join(ROI_DIR, slug + "_" + stamp + ".png")
    cv2.imwrite(roi_path, cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR))

    # Keep the untouched frame so templates can be rebuilt if the extraction
    # geometry ever changes again.
    raw_path = None
    if raw_bgr is not None:
        os.makedirs(RAW_DIR, exist_ok=True)
        raw_path = os.path.join(RAW_DIR, slug + "_" + stamp + ".png")
        cv2.imwrite(raw_path, raw_bgr)

    conn = connect()
    conn.execute(
        "INSERT INTO samples (person_id, hand, source, roi_path, raw_path,"
        " embedding, dim, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (person_id, hand, source, roi_path, raw_path,
         embedding.astype(np.float32).tobytes(), len(embedding),
         datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

    gallery_name = None
    if append_to_gallery:
        gallery_name = append_to_dataset_gallery(
            "enrolled_" + slug + "_" + stamp, roi_norm, embedding)
    return roi_path, gallery_name


def append_to_dataset_gallery(gallery_name, roi_norm, embedding):
    """Appends this ROI + embedding to dataset/preprocessed_images and
    output/embeddings, so the enrolled palm joins the searchable gallery."""
    os.makedirs(GALLERY_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(EMB_PATH), exist_ok=True)
    np.save(os.path.join(GALLERY_DIR, gallery_name + ".npy"), roi_norm)

    if os.path.exists(EMB_PATH):
        emb = np.load(EMB_PATH)
    else:
        emb = np.zeros((0, len(embedding)), np.float32)
    names = np.load(NAMES_PATH).tolist() if os.path.exists(NAMES_PATH) else []

    if emb.shape[0] and emb.shape[1] != len(embedding):
        return None  # dimension mismatch: leave the gallery untouched

    dtype = emb.dtype if emb.size else np.float32
    emb = np.vstack([emb, embedding.astype(dtype)[None]])
    names.append(gallery_name)
    np.save(EMB_PATH, emb)
    np.save(NAMES_PATH, np.array(names))
    return gallery_name


def load_templates():
    """All enrolled samples grouped per person, ready for matching."""
    conn = connect()
    rows = conn.execute("""
        SELECT p.id AS pid, p.name, p.person_code, p.email, p.phone, p.department,
               p.created_at AS enrolled_at,
               s.id AS sid, s.hand, s.source, s.roi_path, s.embedding
        FROM persons p JOIN samples s ON s.person_id = p.id
        ORDER BY p.name, s.id
    """).fetchall()
    conn.close()

    people = {}
    for r in rows:
        person = people.setdefault(r["name"], {
            "pid": r["pid"], "name": r["name"], "person_code": r["person_code"],
            "email": r["email"], "phone": r["phone"], "department": r["department"],
            "enrolled_at": r["enrolled_at"], "samples": [], "stale_count": 0,
        })
        # Superseded extraction: the stored template is not comparable with a
        # full-palm query, so it must not contribute a score.
        if is_stale_source(r["source"]):
            person["stale_count"] += 1
            continue
        person["samples"].append({
            "sid": r["sid"], "hand": r["hand"], "source": r["source"],
            "roi_path": r["roi_path"],
            "embedding": blob_to_vec(r["embedding"]),
        })

    for person in people.values():
        person.update(enrollment_status([s["hand"] for s in person["samples"]]))
    return people


def is_stale_source(source):
    """True for samples captured with any superseded extraction geometry."""
    return source in STALE_METHODS


def enrollment_status(hands):
    """Dual-palm policy: a person is only login-ready with both palms on file."""
    present = {h for h in hands if h in REQUIRED_HANDS}
    missing = [h for h in REQUIRED_HANDS if h not in present]
    return {
        "has_left": LEFT in present,
        "has_right": RIGHT in present,
        "missing_hands": missing,
        "complete": not missing,
    }


def list_persons():
    conn = connect()
    rows = conn.execute("""
        SELECT p.*,
               COUNT(s.id) AS sample_count,
               SUM(CASE WHEN s.source IN (%(stale)s) THEN 1 ELSE 0 END) AS stale_count,
               SUM(CASE WHEN s.hand = ? AND s.source NOT IN (%(stale)s)
                        THEN 1 ELSE 0 END) AS left_count,
               SUM(CASE WHEN s.hand = ? AND s.source NOT IN (%(stale)s)
                        THEN 1 ELSE 0 END) AS right_count
        FROM persons p LEFT JOIN samples s ON s.person_id = p.id
        GROUP BY p.id ORDER BY p.name
    """ % {"stale": ",".join("?" * len(STALE_METHODS))},
        tuple(STALE_METHODS) + (LEFT,) + tuple(STALE_METHODS)
        + (RIGHT,) + tuple(STALE_METHODS)).fetchall()
    conn.close()

    people = []
    for r in rows:
        p = dict(r)
        p["left_count"] = p["left_count"] or 0
        p["right_count"] = p["right_count"] or 0
        p["stale_count"] = p["stale_count"] or 0
        hands = [LEFT] * p["left_count"] + [RIGHT] * p["right_count"]
        p.update(enrollment_status(hands))
        people.append(p)
    return people


def person_samples(name):
    conn = connect()
    rows = conn.execute("""
        SELECT s.* FROM samples s JOIN persons p ON p.id = s.person_id
        WHERE p.name = ? ORDER BY s.id
    """, (name,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def reencode_samples(encoder, device, name=None):
    """Rebuild stored embeddings from their ROI images using the current pipeline.

    Templates enrolled before a pipeline change are not comparable with new
    captures. Where the ROI image is still on disk the template can be rebuilt
    in place, which beats making people enrol again. Returns (updated, skipped).
    """
    conn = connect()
    if name:
        rows = conn.execute("""
            SELECT s.id, s.roi_path, s.raw_path, s.source FROM samples s
            JOIN persons p ON p.id = s.person_id WHERE p.name = ?""", (name,)).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, roi_path, raw_path, source FROM samples").fetchall()

    updated, skipped = 0, []
    for r in rows:
        # Prefer the original frame: re-cropping an already-cropped ROI cannot
        # recover palm outside it, and only pads the image, which corrupts the
        # template rather than rebuilding it.
        path = r["raw_path"] if (r["raw_path"] and os.path.exists(r["raw_path"])) \
            else r["roi_path"]
        if not path or not os.path.exists(path):
            skipped.append((r["id"], "no stored image to rebuild from"))
            continue
        if not r["raw_path"]:
            skipped.append((r["id"], "only a cropped ROI is stored, not the "
                                     "original frame - re-enrol this palm"))
            continue
        bgr = cv2.imread(path)
        if bgr is None:
            skipped.append((r["id"], "ROI image unreadable"))
            continue
        roi_norm, _, _, source, msg = extract_roi(bgr)
        if roi_norm is None:
            skipped.append((r["id"], msg))
            continue
        vec = embed(roi_norm, encoder, device)
        conn.execute("UPDATE samples SET embedding = ?, dim = ?, source = ? WHERE id = ?",
                     (vec.astype(np.float32).tobytes(), len(vec), source, r["id"]))
        updated += 1
    conn.commit()
    conn.close()
    return updated, skipped


def set_sample_hand(sample_id, hand):
    """Relabels a stored sample's hand.

    Samples enrolled before the dual-palm policy (or whose side auto-detection
    was undecided) land as "Palm"; this assigns them without re-enrolling.
    """
    conn = connect()
    conn.execute("UPDATE samples SET hand = ? WHERE id = ?", (hand, sample_id))
    conn.commit()
    conn.close()


def delete_person(name):
    conn = connect()
    rows = conn.execute("""
        SELECT s.roi_path FROM samples s JOIN persons p ON p.id = s.person_id
        WHERE p.name = ?""", (name,)).fetchall()
    conn.execute("DELETE FROM persons WHERE name = ?", (name,))
    conn.commit()
    conn.close()
    for r in rows:
        if r["roi_path"] and os.path.exists(r["roi_path"]):
            try:
                os.remove(r["roi_path"])
            except OSError:
                pass


def log_identification(matched, score, threshold, decision, source):
    conn = connect()
    conn.execute(
        "INSERT INTO identify_log (ts, matched, score, threshold, decision, source)"
        " VALUES (?,?,?,?,?,?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), matched, score, threshold,
         decision, source))
    conn.commit()
    conn.close()


def recent_identifications(limit=25):
    conn = connect()
    rows = conn.execute(
        "SELECT * FROM identify_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ===================== 5. MATCHING =====================
def identify(query_emb, people, threshold=DEFAULT_THRESHOLD, top_k=5,
             require_both_palms=True, min_margin=DEFAULT_MARGIN):
    """Ranks every enrolled person against the query palm.

    Each person is scored by their best-matching sample across BOTH hands, so
    either palm can be presented to log in. Under the dual-palm policy only
    people with left and right on file can be granted a match; anyone still
    mid-enrollment is reported separately as `pending` so the UI can say why
    a strong score was not accepted.

    Returns (ranked_list, matched_person_or_None, pending_list).
    """
    scored = []
    for name, person in people.items():
        scores = [float(np.dot(query_emb, s["embedding"])) for s in person["samples"]]
        if not scores:
            continue
        best = int(np.argmax(scores))
        scored.append({
            "name": name,
            "score": scores[best],
            "mean_score": float(np.mean(scores)),
            "hand": person["samples"][best]["hand"],
            "roi_path": person["samples"][best]["roi_path"],
            "sample_count": len(scores),
            "person_code": person["person_code"],
            "email": person["email"],
            "phone": person["phone"],
            "department": person["department"],
            "enrolled_at": person["enrolled_at"],
            "complete": person.get("complete", True),
            "missing_hands": person.get("missing_hands", []),
            "has_left": person.get("has_left", False),
            "has_right": person.get("has_right", False),
        })
    scored.sort(key=lambda r: r["score"], reverse=True)

    eligible = [r for r in scored if r["complete"]] if require_both_palms else scored
    pending = [r for r in scored
               if not r["complete"] and r["score"] >= threshold] if require_both_palms else []

    ranked = eligible[:top_k]

    # Threshold alone is not enough. If the runner-up is nearly as close, the
    # top score is not evidence of identity -- that is exactly how one person
    # gets recognised as another. Require clear separation, or refuse.
    match, margin, reason = None, None, None
    if ranked and ranked[0]["score"] >= threshold:
        if len(eligible) > 1:
            margin = eligible[0]["score"] - eligible[1]["score"]
            if margin >= min_margin:
                match = ranked[0]
            else:
                reason = ("ambiguous: %s scored %.2f%% but %s is within %.2f points "
                          "(needs %.2f)" % (eligible[0]["name"], eligible[0]["score"] * 100,
                                            eligible[1]["name"], margin, min_margin))
        else:
            margin = float("inf")
            match = ranked[0]

    return ranked, match, pending, {"margin": margin, "rejected_reason": reason}


def verify(query_emb, people, claimed, threshold=DEFAULT_THRESHOLD,
           min_margin=DEFAULT_MARGIN, use_cohort=True):
    """1:1 authentication: does this palm belong to the claimed person?

    The verdict is only ever GRANTED-as-`claimed` or DENIED -- no other identity
    can be returned in its place.

    It is not enough to ask "did the claimed person score above a threshold?".
    Measured on real palms, a genuine re-capture can fall to 0.77 while a
    different person reaches 0.86, so the two overlap and NO fixed threshold
    separates them. What does separate them, cleanly, is the comparison: a palm
    scores ~0.98 against its true owner and at most ~0.86 against anybody else.

    So the other enrolled templates are used as a reference, never as
    candidates: if some other enrolled palm explains this capture better than
    the claim does, the claim is refused. That is what stops one person's palm
    from authenticating as another. Set use_cohort=False for threshold-only
    behaviour (less secure, and it is the mode that produced false accepts).
    """
    person = people.get(claimed, {})
    scores = [(s["hand"], float(np.dot(query_emb, s["embedding"])))
              for s in person.get("samples", [])]
    if not scores:
        return {"accepted": False, "score": None, "per_sample": [], "runner_up": None,
                "margin": None, "reason": "no usable templates on file for this person"}

    best_hand, best = max(scores, key=lambda hs: hs[1])

    runner_up, margin = None, None
    if use_cohort:
        others = []
        for name, other in people.items():
            if name == claimed or not other.get("samples"):
                continue
            others.append((name, max(float(np.dot(query_emb, s["embedding"]))
                                     for s in other["samples"])))
        if others:
            runner_up = max(others, key=lambda t: t[1])
            margin = best - runner_up[1]

    if best < threshold:
        reason = ("score %.2f%% is below the %.2f%% threshold"
                  % (best * 100, threshold * 100))
    elif margin is not None and margin < min_margin:
        reason = ("another enrolled palm matches this capture at least as well "
                  "(%.2f%% vs %.2f%%), so this claim cannot be trusted"
                  % (runner_up[1] * 100, best * 100))
    else:
        reason = None

    return {
        "accepted": reason is None,
        "score": best,
        "hand": best_hand,
        "mean": float(np.mean([s for _, s in scores])),
        "per_sample": scores,
        "runner_up": runner_up,
        "margin": margin,
        "reason": reason,
    }


def nearest_match(query_emb, people, top_k=5, require_both_palms=True):
    """Rank everyone by similarity, with no accept/reject decision applied."""
    ranked = []
    for name, person in people.items():
        scores = [float(np.dot(query_emb, s["embedding"])) for s in person["samples"]]
        if not scores:
            continue
        if require_both_palms and not person.get("complete", True):
            continue
        best = int(np.argmax(scores))
        ranked.append({
            "name": name, "score": scores[best],
            "mean_score": float(np.mean(scores)),
            "hand": person["samples"][best]["hand"],
            "roi_path": person["samples"][best]["roi_path"],
            "sample_count": len(scores),
            "person_code": person["person_code"], "email": person["email"],
            "phone": person["phone"], "department": person["department"],
            "enrolled_at": person["enrolled_at"],
        })
    ranked.sort(key=lambda r: r["score"], reverse=True)
    return ranked[:top_k]


def search_dataset(query_emb, top_k=5):
    """Nearest neighbours in the unlabeled dataset gallery (plus any enrolled
    ROIs that were appended to it)."""
    if not (os.path.exists(EMB_PATH) and os.path.exists(NAMES_PATH)):
        return []
    emb = np.load(EMB_PATH)
    names = np.load(NAMES_PATH).tolist()
    if emb.size == 0 or emb.shape[1] != len(query_emb):
        return []
    sims = (emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)) @ query_emb
    idx = np.argsort(sims)[::-1][:top_k]
    return [{"name": names[i], "score": float(sims[i])} for i in idx]
