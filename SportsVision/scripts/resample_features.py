"""Resample Baidu features from 1 FPS to 2 FPS using linear interpolation"""

import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path
from tqdm import tqdm
import argparse


def resample_features(input_path, output_path, source_fps=1.0, target_fps=2.0):
    """
    Resample features from source_fps to target_fps using linear interpolation.

    Args:
        input_path: Path to input .npy file (T, D) at source_fps
        output_path: Path to save resampled .npy file
        source_fps: Original frame rate (default: 1.0)
        target_fps: Target frame rate (default: 2.0)

    Returns:
        T_source: Original number of frames
        T_target: Resampled number of frames
    """
    # Load features: shape (T_source, D) where D=8576
    features = np.load(input_path)
    T_source, D = features.shape

    # Calculate video duration and target length
    duration = T_source / source_fps  # in seconds
    T_target = int(duration * target_fps)

    # Create time axes
    source_times = np.arange(T_source) / source_fps  # [0, 1, 2, ...] seconds
    target_times = np.arange(T_target) / target_fps  # [0, 0.5, 1, 1.5, ...] seconds

    # Interpolate each feature dimension
    interpolator = interp1d(
        source_times,
        features,
        axis=0,
        kind='linear',
        fill_value='extrapolate'
    )
    resampled = interpolator(target_times)

    # Save resampled features
    np.save(output_path, resampled.astype(np.float32))

    return T_source, T_target


def resample_single_file(input_path, output_path=None, source_fps=1.0, target_fps=2.0):
    """
    Resample a single feature file.

    Args:
        input_path: Path to input .npy file
        output_path: Path to save output (default: same location with _2fps suffix)
        source_fps: Source frame rate
        target_fps: Target frame rate
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_2fps.npy"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    T_src, T_tgt = resample_features(input_path, output_path, source_fps, target_fps)

    print(f"Resampled: {input_path.name}")
    print(f"  Source: {T_src} frames @ {source_fps} FPS")
    print(f"  Target: {T_tgt} frames @ {target_fps} FPS")
    print(f"  Saved to: {output_path}")

    return output_path


def resample_dataset(input_dir, output_dir, source_fps=1.0, target_fps=2.0):
    """
    Resample all features in a dataset directory.

    Expected structure:
        input_dir/
            league/season/match/
                1_baidu_soccer_embeddings.npy
                2_baidu_soccer_embeddings.npy

    Args:
        input_dir: Directory containing original features
        output_dir: Directory to save resampled features
        source_fps: Source frame rate (default: 1.0)
        target_fps: Target frame rate (default: 2.0)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Find all feature files
    feature_files = list(input_dir.rglob("*_baidu_soccer_embeddings.npy"))

    if not feature_files:
        print(f"No feature files found in {input_dir}")
        print("Looking for files matching: *_baidu_soccer_embeddings.npy")
        return

    print(f"Found {len(feature_files)} feature files")
    print(f"Resampling from {source_fps} FPS to {target_fps} FPS")
    print("-" * 50)

    success_count = 0
    error_count = 0

    for input_path in tqdm(feature_files, desc="Resampling"):
        try:
            # Create corresponding output path
            relative_path = input_path.relative_to(input_dir)
            output_path = output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Resample
            T_src, T_tgt = resample_features(
                input_path, output_path, source_fps, target_fps
            )
            success_count += 1

        except Exception as e:
            print(f"\nError processing {input_path}: {e}")
            error_count += 1
            continue

    print("-" * 50)
    print(f"Completed: {success_count} files resampled successfully")
    if error_count > 0:
        print(f"Errors: {error_count} files failed")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Resample Baidu features from 1 FPS to 2 FPS using linear interpolation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Resample entire dataset
  python resample_features.py --input_dir /path/to/SoccerNet --output_dir ./data/features/baidu_2.0

  # Resample single file
  python resample_features.py --input_file video_features.npy --output_file video_features_2fps.npy

  # Custom frame rates
  python resample_features.py --input_dir ./features --output_dir ./features_4fps --target_fps 4.0
        """
    )

    # Dataset mode
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory containing original features (for batch processing)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save resampled features")

    # Single file mode
    parser.add_argument("--input_file", type=str, default=None,
                        help="Single input .npy file to resample")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output path for single file (default: adds _2fps suffix)")

    # Frame rate options
    parser.add_argument("--source_fps", type=float, default=1.0,
                        help="Source frame rate (default: 1.0)")
    parser.add_argument("--target_fps", type=float, default=2.0,
                        help="Target frame rate (default: 2.0)")

    args = parser.parse_args()

    # Validate arguments
    if args.input_file:
        # Single file mode
        resample_single_file(
            args.input_file,
            args.output_file,
            args.source_fps,
            args.target_fps
        )
    elif args.input_dir and args.output_dir:
        # Dataset mode
        resample_dataset(
            args.input_dir,
            args.output_dir,
            args.source_fps,
            args.target_fps
        )
    else:
        parser.print_help()
        print("\nError: Please provide either --input_dir and --output_dir for batch processing,")
        print("       or --input_file for single file processing.")


if __name__ == "__main__":
    main()
