import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

from depth_anything_v2.dpt import DepthAnythingV2


# ============================================================
# CONFIG
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


MODEL_CONFIGS = {

    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [
            48,
            96,
            192,
            384,
        ],
    },

    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [
            96,
            192,
            384,
            768,
        ],
    },

    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [
            256,
            512,
            1024,
            1024,
        ],
    },
}


# ============================================================
# Unicode-safe image IO
# ============================================================

def read_image_unicode(path):

    data = np.fromfile(
        str(path),
        dtype=np.uint8
    )

    if data.size == 0:
        return None

    return cv2.imdecode(
        data,
        cv2.IMREAD_COLOR
    )


def write_image_unicode(path, image):

    path = Path(path)

    success, encoded = cv2.imencode(
        path.suffix,
        image
    )

    if not success:

        raise IOError(
            f"Cannot encode image: {path}"
        )

    encoded.tofile(
        str(path)
    )


# ============================================================
# Find images
# ============================================================

def collect_images(input_dir):

    input_dir = Path(input_dir)

    if not input_dir.exists():

        raise FileNotFoundError(
            f"Input directory does not exist:\n"
            f"{input_dir}"
        )

    files = []

    for path in input_dir.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        ):

            files.append(path)

    return sorted(files)


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser(
    description=
    "Build RGB + Relative Depth dataset "
    "using Depth Anything V2"
)


parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Input image directory"
)


parser.add_argument(
    "--output",
    type=str,
    default="dataset",
    help="Dataset output directory"
)


parser.add_argument(
    "--encoder",
    type=str,
    default="vitb",
    choices=[
        "vits",
        "vitb",
        "vitl",
    ]
)


parser.add_argument(
    "--input-size",
    type=int,
    default=518
)


parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Rebuild existing scenes"
)


args = parser.parse_args()


# ============================================================
# Device
# ============================================================

if torch.cuda.is_available():

    DEVICE = "cuda"

elif (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):

    DEVICE = "mps"

else:

    DEVICE = "cpu"


print()
print(
    "=========================================="
)

print(
    "BUILD STREET DEPTH DATASET"
)

print(
    "=========================================="
)

print()

print(
    "Torch:",
    torch.__version__
)

print(
    "Device:",
    DEVICE
)

if DEVICE == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# Locate checkpoint
# ============================================================

SCRIPT_DIR = Path(
    __file__
).resolve().parent


checkpoint = (

    SCRIPT_DIR
    / "checkpoints"
    / f"depth_anything_v2_{args.encoder}.pth"

)


if not checkpoint.exists():

    raise FileNotFoundError(
        f"Checkpoint not found:\n"
        f"{checkpoint}"
    )


print()
print(
    "Checkpoint:",
    checkpoint
)


# ============================================================
# Load model
# ============================================================

print()
print(
    "Loading Depth Anything V2..."
)


model = DepthAnythingV2(
    **MODEL_CONFIGS[
        args.encoder
    ]
)


state_dict = torch.load(
    checkpoint,
    map_location="cpu"
)


model.load_state_dict(
    state_dict
)


model = model.to(
    DEVICE
).eval()


print(
    "Model ready."
)


# ============================================================
# Input images
# ============================================================

images = collect_images(
    args.input
)


if len(images) == 0:

    raise RuntimeError(
        "No supported images found."
    )


print()
print(
    f"Found {len(images)} images."
)


# ============================================================
# Dataset directories
# ============================================================

dataset_root = Path(
    args.output
)


scenes_root = (

    dataset_root
    / "scenes"

)


scenes_root.mkdir(
    parents=True,
    exist_ok=True
)


manifest_path = (

    dataset_root
    / "manifest.csv"

)


# ============================================================
# Manifest
# ============================================================

manifest_rows = []


# ============================================================
# Process
# ============================================================

for index, image_path in enumerate(
    images,
    start=1
):

    scene_id = (
        f"scene_{index:06d}"
    )


    scene_dir = (
        scenes_root
        / scene_id
    )


    print()
    print(
        "=========================================="
    )

    print(
        f"[{index}/{len(images)}] "
        f"{scene_id}"
    )

    print(
        image_path
    )


    # --------------------------------------------------------
    # Skip existing
    # --------------------------------------------------------

    if (
        scene_dir.exists()
        and not args.overwrite
    ):

        print(
            "Already exists -> skipped"
        )

        continue


    scene_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # Read image
    # ========================================================

    image = read_image_unicode(
        image_path
    )


    if image is None:

        print(
            "WARNING: Cannot read image."
        )

        continue


    height, width = (
        image.shape[:2]
    )


    print(
        f"Resolution: "
        f"{width} x {height}"
    )


    # ========================================================
    # Depth inference
    # ========================================================

    print(
        "Depth inference..."
    )


    raw_depth = model.infer_image(
        image,
        args.input_size
    )


    raw_depth = np.asarray(
        raw_depth,
        dtype=np.float32
    )


    # Safety
    if raw_depth.shape != (
        height,
        width
    ):

        raw_depth = cv2.resize(

            raw_depth,

            (
                width,
                height
            ),

            interpolation=
            cv2.INTER_LINEAR
        )


    # ========================================================
    # Validate depth
    # ========================================================

    finite_mask = np.isfinite(
        raw_depth
    )


    finite_ratio = float(
        finite_mask.mean()
    )


    if finite_ratio < 0.99:

        print(
            "WARNING: "
            f"finite depth ratio = "
            f"{finite_ratio:.4f}"
        )


    valid_depth = raw_depth[
        finite_mask
    ]


    raw_min = float(
        valid_depth.min()
    )


    raw_max = float(
        valid_depth.max()
    )


    raw_mean = float(
        valid_depth.mean()
    )


    raw_std = float(
        valid_depth.std()
    )


    # ========================================================
    # Normalize relative depth
    #
    # Depth Anything relative output:
    #
    # 0 -> farther
    # 1 -> nearer
    #
    # This is NOT metric distance.
    # ========================================================

    depth_range = (
        raw_max
        - raw_min
    )


    if depth_range < 1e-12:

        print(
            "WARNING: "
            "Depth range too small."
        )

        continue


    depth_norm = (

        raw_depth
        - raw_min

    ) / depth_range


    depth_norm = np.clip(
        depth_norm,
        0.0,
        1.0
    ).astype(
        np.float32
    )


    # ========================================================
    # Save RGB
    #
    # Standardize training image format.
    # ========================================================

    rgb_path = (
        scene_dir
        / "rgb.jpg"
    )


    write_image_unicode(
        rgb_path,
        image
    )


    # ========================================================
    # Save raw relative depth
    # ========================================================

    raw_depth_path = (
        scene_dir
        / "depth_raw.npy"
    )


    np.save(
        raw_depth_path,
        raw_depth
    )


    # ========================================================
    # Save normalized depth
    # ========================================================

    norm_depth_path = (
        scene_dir
        / "depth_norm.npy"
    )


    np.save(
        norm_depth_path,
        depth_norm
    )


    # ========================================================
    # Save human-readable depth preview
    #
    # white = closer
    # black = farther
    # ========================================================

    preview = (

        depth_norm
        * 255.0

    ).astype(
        np.uint8
    )


    preview_path = (
        scene_dir
        / "depth_preview.png"
    )


    write_image_unicode(
        preview_path,
        preview
    )


    # ========================================================
    # Metadata
    # ========================================================

    metadata = {

        "scene_id":
            scene_id,

        "source_path":
            str(image_path),

        "source_filename":
            image_path.name,

        "width":
            width,

        "height":
            height,

        "encoder":
            args.encoder,

        "input_size":
            args.input_size,

        "device":
            DEVICE,

        "depth_type":
            "relative",

        "depth_direction":
            "larger_is_nearer",

        "metric_depth":
            False,

        "raw_depth": {

            "min":
                raw_min,

            "max":
                raw_max,

            "mean":
                raw_mean,

            "std":
                raw_std,

            "finite_ratio":
                finite_ratio,
        },

        "normalized_depth": {

            "min":
                0.0,

            "max":
                1.0,

            "black":
                "farther",

            "white":
                "nearer",
        },
    }


    metadata_path = (
        scene_dir
        / "metadata.json"
    )


    with open(

        metadata_path,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            metadata,

            file,

            indent=4,

            ensure_ascii=False

        )


    # ========================================================
    # Manifest row
    # ========================================================

    manifest_rows.append({

        "scene_id":
            scene_id,

        "source":
            str(image_path),

        "rgb":
            str(
                rgb_path.relative_to(
                    dataset_root
                )
            ),

        "depth_raw":
            str(
                raw_depth_path.relative_to(
                    dataset_root
                )
            ),

        "depth_norm":
            str(
                norm_depth_path.relative_to(
                    dataset_root
                )
            ),

        "preview":
            str(
                preview_path.relative_to(
                    dataset_root
                )
            ),

        "metadata":
            str(
                metadata_path.relative_to(
                    dataset_root
                )
            ),

        "width":
            width,

        "height":
            height,

        "depth_min":
            raw_min,

        "depth_max":
            raw_max,

        "depth_mean":
            raw_mean,

        "depth_std":
            raw_std,
    })


    print(
        "Saved."
    )


# ============================================================
# Write manifest.csv
# ============================================================

if len(manifest_rows) > 0:

    fieldnames = list(
        manifest_rows[0].keys()
    )


    with open(

        manifest_path,

        "w",

        newline="",

        encoding="utf-8-sig"

    ) as csv_file:

        writer = csv.DictWriter(

            csv_file,

            fieldnames=
            fieldnames

        )


        writer.writeheader()


        writer.writerows(
            manifest_rows
        )


# ============================================================
# Dataset metadata
# ============================================================

dataset_metadata = {

    "dataset_type":
        "RGB + Relative Depth",

    "depth_model":
        "Depth Anything V2",

    "encoder":
        args.encoder,

    "input_size":
        args.input_size,

    "device":
        DEVICE,

    "number_of_input_images":
        len(images),

    "number_of_processed_images":
        len(manifest_rows),

    "depth_definition": {

        "raw":
            "Depth Anything V2 relative prediction",

        "normalized":
            "per-image min-max normalized relative depth",

        "normalized_0":
            "farther",

        "normalized_1":
            "nearer",

        "metric":
            False,
    },

}


with open(

    dataset_root
    / "dataset.json",

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        dataset_metadata,

        file,

        indent=4,

        ensure_ascii=False

    )


# ============================================================
# Finish
# ============================================================

print()
print(
    "=========================================="
)

print(
    "DATASET BUILD COMPLETE"
)

print(
    "=========================================="
)

print()

print(
    "Dataset:",
    dataset_root.resolve()
)

print(
    "Processed:",
    len(manifest_rows)
)

print(
    "Manifest:",
    manifest_path.resolve()
)