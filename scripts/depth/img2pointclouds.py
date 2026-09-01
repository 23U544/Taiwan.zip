import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from depth_anything_v2.dpt import DepthAnythingV2


# ============================================================
# Settings / constants
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

    path = str(path)

    data = np.fromfile(
        path,
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
            f"Unable to encode image: {path}"
        )

    encoded.tofile(
        str(path)
    )


# ============================================================
# Input collection
# ============================================================

def collect_images(img_path):

    path = Path(img_path)

    if path.is_file():

        if (
            path.suffix.lower()
            not in SUPPORTED_EXTENSIONS
        ):

            raise ValueError(
                f"Unsupported image format: "
                f"{path.suffix}"
            )

        return [path]

    if path.is_dir():

        files = []

        for p in path.rglob("*"):

            if (
                p.is_file()
                and
                p.suffix.lower()
                in SUPPORTED_EXTENSIONS
            ):

                files.append(p)

        return sorted(files)

    raise FileNotFoundError(
        f"Input path does not exist:\n"
        f"{img_path}"
    )


# ============================================================
# PLY writer
# ============================================================

def write_binary_ply(
    path,
    xyz,
    rgb
):

    count = len(xyz)

    vertices = np.empty(

        count,

        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )

    vertices["x"] = xyz[:, 0]
    vertices["y"] = xyz[:, 1]
    vertices["z"] = xyz[:, 2]

    vertices["red"] = rgb[:, 0]
    vertices["green"] = rgb[:, 1]
    vertices["blue"] = rgb[:, 2]


    header = f"""ply
format binary_little_endian 1.0
element vertex {count}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""


    with open(
        path,
        "wb"
    ) as f:

        f.write(
            header.encode("ascii")
        )

        vertices.tofile(f)


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser(
    description=
    "Depth Anything V2 -> Rhino-friendly RGB point cloud"
)


parser.add_argument(
    "--img-path",
    type=str,
    required=True
)


parser.add_argument(
    "--outdir",
    type=str,
    default="pointcloud_output"
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


# ------------------------------------------------------------
# Point density
# ------------------------------------------------------------

parser.add_argument(
    "--stride",
    type=int,
    default=4,
    help=
    "Use one point every N pixels"
)


# ------------------------------------------------------------
# Pseudo-distance parameters
# ------------------------------------------------------------

parser.add_argument(
    "--near",
    type=float,
    default=1.0,
    help=
    "Pseudo distance of nearest region"
)


parser.add_argument(
    "--far",
    type=float,
    default=8.0,
    help=
    "Pseudo distance of farthest region"
)


parser.add_argument(
    "--depth-gamma",
    type=float,
    default=1.0,
    help=
    "Controls nonlinear inverse-depth distribution"
)


# ------------------------------------------------------------
# Robust normalization
#
# Removes extreme outliers before conversion.
# ------------------------------------------------------------

parser.add_argument(
    "--clip-low",
    type=float,
    default=1.0
)


parser.add_argument(
    "--clip-high",
    type=float,
    default=99.0
)


# ------------------------------------------------------------
# Camera
# ------------------------------------------------------------

parser.add_argument(
    "--fov",
    type=float,
    default=60.0,
    help=
    "Approx horizontal camera FOV"
)


parser.add_argument(
    "--fx",
    type=float,
    default=None
)


parser.add_argument(
    "--fy",
    type=float,
    default=None
)


parser.add_argument(
    "--cx",
    type=float,
    default=None
)


parser.add_argument(
    "--cy",
    type=float,
    default=None
)


args = parser.parse_args()


# ============================================================
# Validate
# ============================================================

if args.stride < 1:

    raise ValueError(
        "--stride must be >= 1"
    )


if args.near <= 0:

    raise ValueError(
        "--near must be > 0"
    )


if args.far <= args.near:

    raise ValueError(
        "--far must be > --near"
    )


if args.depth_gamma <= 0:

    raise ValueError(
        "--depth-gamma must be > 0"
    )


if not (
    0 <= args.clip_low
    < args.clip_high
    <= 100
):

    raise ValueError(
        "clip-low/high must satisfy "
        "0 <= low < high <= 100"
    )


# ============================================================
# Device
# ============================================================

if torch.cuda.is_available():

    DEVICE = "cuda"

elif (
    hasattr(torch.backends, "mps")
    and
    torch.backends.mps.is_available()
):

    DEVICE = "mps"

else:

    DEVICE = "cpu"


print()
print(
    "=========================================="
)

print(
    "Depth Anything V2 -> Point Cloud"
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
# Paths
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


output_dir = Path(
    args.outdir
)


output_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Load model
# ============================================================

print()
print(
    "Checkpoint:",
    checkpoint
)

print(
    "Loading model..."
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
# Images
# ============================================================

images = collect_images(
    args.img_path
)


if not images:

    raise RuntimeError(
        "No supported images found."
    )


print()
print(
    "Images:",
    len(images)
)


# ============================================================
# Processing loop
# ============================================================

for image_index, image_path in enumerate(
    images,
    start=1
):

    print()
    print(
        "=========================================="
    )

    print(
        f"Processing "
        f"{image_index}/{len(images)}"
    )

    print(
        image_path
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
        f"Image size: "
        f"{width} x {height}"
    )


    # ========================================================
    # AI relative depth
    # ========================================================

    print(
        "Running depth inference..."
    )


    raw_depth = model.infer_image(
        image,
        args.input_size
    )


    raw_depth = np.asarray(
        raw_depth,
        dtype=np.float32
    )


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
    # Save RAW relative/inverse depth
    # ========================================================

    base_name = (
        image_path.stem
    )


    raw_depth_path = (

        output_dir
        / f"{base_name}_raw_depth.npy"

    )


    np.save(
        raw_depth_path,
        raw_depth
    )


    # ========================================================
    # Official-style visualization
    #
    # min = black
    # max = white
    #
    # Relative DA-V2:
    #
    # larger output ~= closer
    # smaller output ~= farther
    # ========================================================

    raw_min = float(
        np.nanmin(raw_depth)
    )

    raw_max = float(
        np.nanmax(raw_depth)
    )


    official_norm = (

        raw_depth
        - raw_min

    ) / (

        raw_max
        - raw_min
        + 1e-12

    )


    official_norm = np.clip(
        official_norm,
        0.0,
        1.0
    )


    official_gray = (

        official_norm
        * 255.0

    ).astype(
        np.uint8
    )


    official_gray_path = (

        output_dir
        / f"{base_name}_depth_official.png"

    )


    write_image_unicode(
        official_gray_path,
        official_gray
    )


    # ========================================================
    # Robust normalization for GEOMETRY
    #
    # Extreme minimum / maximum values can badly stretch
    # point cloud distance.
    #
    # We use percentile clipping:
    #
    # low  -> far
    # high -> near
    # ========================================================

    depth_low = float(
        np.nanpercentile(
            raw_depth,
            args.clip_low
        )
    )


    depth_high = float(
        np.nanpercentile(
            raw_depth,
            args.clip_high
        )
    )


    if (
        depth_high
        - depth_low
        < 1e-12
    ):

        raise RuntimeError(
            "Depth range is too small."
        )


    relative = (

        raw_depth
        - depth_low

    ) / (

        depth_high
        - depth_low

    )


    relative = np.clip(
        relative,
        0.0,
        1.0
    )


    # ========================================================
    # Gamma control
    #
    # relative = 0 -> far
    # relative = 1 -> near
    # ========================================================

    relative_gamma = np.power(
        relative,
        args.depth_gamma
    )


    geometry_depth_path = (

        output_dir
        / f"{base_name}_depth_geometry.png"

    )


    geometry_gray = (

        relative_gamma
        * 255.0

    ).astype(
        np.uint8
    )


    write_image_unicode(
        geometry_depth_path,
        geometry_gray
    )


    # ========================================================
    # INVERSE DEPTH -> PSEUDO DISTANCE
    #
    # Depth Anything relative model is treated as
    # inverse depth / disparity-like output.
    #
    #
    # inv_distance =
    #
    #     1/far
    #       +
    #     relative *
    #     (1/near - 1/far)
    #
    #
    # Therefore:
    #
    # black / 0 -> distance = far
    #
    # white / 1 -> distance = near
    #
    # ========================================================

    inverse_far = (
        1.0
        / args.far
    )


    inverse_near = (
        1.0
        / args.near
    )


    inverse_distance = (

        inverse_far

        +

        relative_gamma
        *
        (
            inverse_near
            - inverse_far
        )

    )


    distance = (

        1.0
        /
        (
            inverse_distance
            + 1e-12
        )

    ).astype(
        np.float32
    )


    # ========================================================
    # Save distance map
    # ========================================================

    distance_path = (

        output_dir
        / f"{base_name}_distance.npy"

    )


    np.save(
        distance_path,
        distance
    )


    # --------------------------------------------------------
    # Distance visualization
    #
    # IMPORTANT:
    #
    # white = NEAR
    # black = FAR
    #
    # So visually it remains consistent with Depth Anything.
    # --------------------------------------------------------

    distance_visual = (

        (
            args.far
            - distance
        )

        /

        (
            args.far
            - args.near
        )

    )


    distance_visual = np.clip(
        distance_visual,
        0.0,
        1.0
    )


    distance_visual = (

        distance_visual
        * 255.0

    ).astype(
        np.uint8
    )


    distance_visual_path = (

        output_dir
        / f"{base_name}_distance_visual.png"

    )


    write_image_unicode(
        distance_visual_path,
        distance_visual
    )


    # ========================================================
    # DEBUG
    # ========================================================

    dark_idx = np.unravel_index(
        np.nanargmin(raw_depth),
        raw_depth.shape
    )


    bright_idx = np.unravel_index(
        np.nanargmax(raw_depth),
        raw_depth.shape
    )


    print()
    print(
        "---------- DEPTH DEBUG ----------"
    )


    print(
        f"Raw min: "
        f"{raw_min:.6f}"
    )


    print(
        f"Raw max: "
        f"{raw_max:.6f}"
    )


    print(
        f"Geometry clip: "
        f"{depth_low:.6f}"
        f" -> "
        f"{depth_high:.6f}"
    )


    print()


    print(
        "Darkest / farthest:"
    )


    print(
        f"  raw = "
        f"{raw_depth[dark_idx]:.6f}"
    )


    print(
        f"  relative = "
        f"{relative[dark_idx]:.6f}"
    )


    print(
        f"  distance = "
        f"{distance[dark_idx]:.6f}"
    )


    print()


    print(
        "Brightest / nearest:"
    )


    print(
        f"  raw = "
        f"{raw_depth[bright_idx]:.6f}"
    )


    print(
        f"  relative = "
        f"{relative[bright_idx]:.6f}"
    )


    print(
        f"  distance = "
        f"{distance[bright_idx]:.6f}"
    )


    print()


    print(
        "Distance range:"
    )


    print(
        f"  "
        f"{float(np.nanmin(distance)):.6f}"
        f" -> "
        f"{float(np.nanmax(distance)):.6f}"
    )


    # ========================================================
    # Camera intrinsics
    # ========================================================

    if args.fx is not None:

        fx = args.fx

    else:

        fov_rad = np.deg2rad(
            args.fov
        )


        fx = (

            0.5
            * width

            /

            np.tan(
                fov_rad
                / 2.0
            )

        )


    if args.fy is not None:

        fy = args.fy

    else:

        fy = fx


    if args.cx is not None:

        cx = args.cx

    else:

        cx = (
            width
            / 2.0
        )


    if args.cy is not None:

        cy = args.cy

    else:

        cy = (
            height
            / 2.0
        )


    print()
    print(
        "---------- CAMERA ----------"
    )


    print(
        f"fx = {fx:.4f}"
    )

    print(
        f"fy = {fy:.4f}"
    )

    print(
        f"cx = {cx:.4f}"
    )

    print(
        f"cy = {cy:.4f}"
    )


    print(
        f"Approx horizontal FOV "
        f"= {args.fov:.2f} deg"
    )


    # ========================================================
    # Pixel grid
    # ========================================================

    stride = args.stride


    v, u = np.mgrid[

        0:height:stride,

        0:width:stride

    ]


    D = distance[

        0:height:stride,

        0:width:stride

    ]


    # ========================================================
    # Camera-space back projection
    #
    # Standard pinhole:
    #
    # Xcam = (u-cx) D / fx
    #
    # Ycam = -(v-cy) D / fy
    #
    # Zcam = D
    #
    # ========================================================

    X_cam = (

        (
            u
            - cx
        )

        *
        D

        /
        fx

    )


    Y_cam = -(

        (
            v
            - cy
        )

        *
        D

        /
        fy

    )


    Z_cam = D


    # ========================================================
    # Rhino-friendly coordinates
    #
    #
    #             +Z
    #              ^
    #              |
    #              |
    #    -X <------O------> +X
    #             Camera
    #              |
    #              |
    #              v
    #             -Y
    #
    #
    # Camera = (0, 0, 0)
    #
    # Camera looks toward -Y
    #
    # image right -> +X
    #
    # image up -> +Z
    #
    # distance -> -Y
    #
    #
    # Near:
    # Y ~= -near
    #
    # Far:
    # Y ~= -far
    #
    # ========================================================

    X_rhino = X_cam

    Y_rhino = Z_cam

    Z_rhino = Y_cam


    # ========================================================
    # RGB
    # ========================================================

    rgb = image[

        0:height:stride,

        0:width:stride,

        ::-1

    ]


    # ========================================================
    # Flatten
    # ========================================================

    xyz = np.column_stack(

        (
            X_rhino.reshape(-1),

            Y_rhino.reshape(-1),

            Z_rhino.reshape(-1),
        )

    ).astype(
        np.float32
    )


    rgb = rgb.reshape(
        -1,
        3
    ).astype(
        np.uint8
    )


    valid = np.isfinite(
        xyz
    ).all(
        axis=1
    )


    xyz = xyz[
        valid
    ]


    rgb = rgb[
        valid
    ]


    # ========================================================
    # PLY
    # ========================================================

    ply_path = (

        output_dir
        / f"{base_name}.ply"

    )


    write_binary_ply(
        ply_path,
        xyz,
        rgb
    )


    # ========================================================
    # Save metadata
    # ========================================================

    metadata = {

        "source":
            str(image_path),

        "encoder":
            args.encoder,

        "device":
            DEVICE,

        "image_width":
            width,

        "image_height":
            height,

        "input_size":
            args.input_size,

        "stride":
            args.stride,

        "depth_model":
            "relative inverse depth",

        "distance_type":
            "pseudo distance",

        "near":
            args.near,

        "far":
            args.far,

        "depth_gamma":
            args.depth_gamma,

        "clip_low_percentile":
            args.clip_low,

        "clip_high_percentile":
            args.clip_high,

        "fx":
            float(fx),

        "fy":
            float(fy),

        "cx":
            float(cx),

        "cy":
            float(cy),

        "fov":
            args.fov,

        "point_count":
            int(len(xyz)),

        "coordinate_system": {

            "origin":
                "camera",

            "X":
                "image right",

            "Y":
                "positive forward / scene depth",

            "Z":
                "image up",

            "camera_forward":
                "+Y",
        },

        "metric":
            False,
    }


    json_path = (

        output_dir
        / f"{base_name}_metadata.json"

    )


    with open(

        json_path,

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            metadata,

            f,

            indent=4,

            ensure_ascii=False

        )


    # ========================================================
    # Finished image
    # ========================================================

    print()
    print(
        "---------- OUTPUT ----------"
    )


    print(
        "Raw depth:"
    )

    print(
        raw_depth_path
    )


    print(
        "Official depth preview:"
    )

    print(
        official_gray_path
    )


    print(
        "Geometry depth preview:"
    )

    print(
        geometry_depth_path
    )


    print(
        "Pseudo distance:"
    )

    print(
        distance_path
    )


    print(
        "Distance preview:"
    )

    print(
        distance_visual_path
    )


    print(
        "Point cloud:"
    )

    print(
        ply_path
    )


    print(
        "Metadata:"
    )

    print(
        json_path
    )


    print(
        f"Points: "
        f"{len(xyz):,}"
    )


print()
print(
    "=========================================="
)

print(
    "ALL DONE"
)

print(
    "=========================================="
)