import argparse
import fnmatch
import shutil
from pathlib import Path


# ============================================================
# DEFAULT GENERATED OUTPUT FOLDER PATTERNS
#
# These are analysis / detector output folders that can be
# safely regenerated from the source scene data.
#
# IMPORTANT:
# The script does NOT delete the core files inside each scene:
#
#   rgb.jpg
#   depth_raw.npy
#   depth_norm.npy
#   depth_preview.png
#   metadata.json
#
# It only removes matching generated subfolders.
# ============================================================

DEFAULT_PATTERNS = [
    "architectural_*",
    "window_v*",
    "semantic_test_output",
]


def matches_any(name: str, patterns: list[str]) -> bool:
    return any(
        fnmatch.fnmatch(name, pattern)
        for pattern in patterns
    )


def collect_scene_dirs(
    root: Path,
    selected_scenes: list[str] | None,
) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(
            f"Scenes root does not exist:\n{root}"
        )

    if selected_scenes:
        scene_dirs = []

        for scene_name in selected_scenes:
            scene_dir = root / scene_name

            if not scene_dir.exists():
                print(
                    f"[WARNING] Scene not found: {scene_dir}"
                )
                continue

            if not scene_dir.is_dir():
                print(
                    f"[WARNING] Not a directory: {scene_dir}"
                )
                continue

            scene_dirs.append(scene_dir)

        return scene_dirs

    return sorted(
        [
            path
            for path in root.iterdir()
            if path.is_dir()
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Clean generated detector / semantic output folders "
            "inside dataset scene folders."
        )
    )

    parser.add_argument(
        "--root",
        type=str,
        default=r"dataset\scenes",
        help=(
            "Root folder containing scene_XXXXXX folders. "
            r'Default: dataset\scenes'
        ),
    )

    parser.add_argument(
        "--scene",
        nargs="*",
        default=None,
        help=(
            "Optional scene names to clean. "
            "Example: --scene scene_000001 scene_000002"
        ),
    )

    parser.add_argument(
        "--patterns",
        nargs="*",
        default=None,
        help=(
            "Optional custom folder patterns. "
            'Example: --patterns "window_v*" "architectural_*"'
        ),
    )

    parser.add_argument(
        "--keep",
        nargs="*",
        default=[],
        help=(
            "Folder names or wildcard patterns to keep. "
            'Example: --keep "window_v6"'
        ),
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Actually delete folders. "
            "Without --apply the script only previews what would be deleted."
        ),
    )

    args = parser.parse_args()

    root = Path(args.root)

    patterns = (
        args.patterns
        if args.patterns
        else DEFAULT_PATTERNS
    )

    keep_patterns = args.keep or []

    scene_dirs = collect_scene_dirs(
        root,
        args.scene,
    )

    print()
    print("=" * 72)
    print("SCENE OUTPUT CLEANER")
    print("=" * 72)

    print()
    print("Root:")
    print(root.resolve())

    print()
    print("Delete patterns:")
    for pattern in patterns:
        print(f"  - {pattern}")

    if keep_patterns:
        print()
        print("Keep patterns:")
        for pattern in keep_patterns:
            print(f"  - {pattern}")

    print()
    print(
        "Mode:",
        "DELETE" if args.apply else "DRY RUN"
    )

    print()

    targets: list[Path] = []

    for scene_dir in scene_dirs:
        scene_targets = []

        for child in scene_dir.iterdir():
            if not child.is_dir():
                continue

            if not matches_any(
                child.name,
                patterns,
            ):
                continue

            if keep_patterns and matches_any(
                child.name,
                keep_patterns,
            ):
                continue

            scene_targets.append(child)
            targets.append(child)

        if scene_targets:
            print(f"[{scene_dir.name}]")

            for target in scene_targets:
                print(
                    f"  {'DELETE' if args.apply else 'WOULD DELETE'} "
                    f"{target.name}"
                )

            print()

    if not targets:
        print("No matching generated folders found.")
        return

    print("-" * 72)
    print(
        f"Matched {len(targets)} generated folder(s)."
    )

    if not args.apply:
        print()
        print(
            "Nothing was deleted."
        )
        print(
            "Run again with --apply when the preview is correct."
        )
        return

    deleted = 0
    failed = 0

    print()
    print("Deleting...")

    for target in targets:
        try:
            shutil.rmtree(target)
            deleted += 1
            print(
                f"  [OK] {target}"
            )

        except Exception as exc:
            failed += 1
            print(
                f"  [FAILED] {target}"
            )
            print(
                f"           {exc}"
            )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)

    print()
    print(f"Deleted: {deleted}")
    print(f"Failed : {failed}")


if __name__ == "__main__":
    main()
