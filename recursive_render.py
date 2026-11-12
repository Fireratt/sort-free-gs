#!/usr/bin/env python3
"""Render every sort-free dataset below a directory and write a CSV benchmark.

This entry point uses this repository's Scene, GaussianModel and
gaussian_renderer.render APIs. It does not depend on the distributed Grendel
renderer that the previous version of this file imported.

e.g.
 python recursive_render.py --recursive /home/ps/hjf/3dgs/res/training/deep-blending \
--iteration 10000 --resolutions 1.0 0.5 3840x2160 2>&1"""

import argparse
import csv
import os
import statistics
import time
from argparse import Namespace
from pathlib import Path

import torch

from arguments import ModelParams, PipelineParams
from gaussian_renderer import GaussianModel, render
from scene import Scene

DEFAULT_RESOLUTIONS = ["1.0"]
INFERENCE_BATCH_SIZE = 1


def is_dataset(path: Path):
    return (path / "sparse").is_dir() or (path / "transforms_train.json").is_file()


def discover_datasets(dataset_root: Path):
    if is_dataset(dataset_root):
        return [dataset_root]
    return sorted(
        (path for path in dataset_root.iterdir() if path.is_dir() and is_dataset(path)),
        key=lambda path: path.name,
    )


def parse_resolution_specs(specs):
    values = []
    for spec in specs:
        values.extend(value.strip() for value in str(spec).split(","))
    values = [value for value in values if value]
    if not values:
        raise ValueError("At least one resolution must be configured.")
    return values


def resolve_resolution(spec, base_width, base_height):
    normalized = str(spec).lower().replace(" ", "")
    if "x" in normalized:
        width_text, height_text = normalized.split("x", 1)
        width = int(width_text)
        height = int(height_text)
        label = f"{width}x{height}"
    else:
        scale = float(normalized)
        if scale <= 0:
            raise ValueError(f"Resolution scale must be positive: {spec}")
        width = max(1, round(base_width * scale))
        height = max(1, round(base_height * scale))
        label = f"{width}x{height}"
    if width <= 0 or height <= 0:
        raise ValueError(f"Resolution must be positive: {spec}")
    return label, width, height


def load_model_args(source_path: Path, model_path: Path, eval_mode: bool):
    parser = argparse.ArgumentParser(add_help=False)
    # This parser supplies real defaults for older or missing training configs.
    # Sentinel defaults are only appropriate when merging CLI overrides.
    ModelParams(parser)
    PipelineParams(parser)
    args = parser.parse_args([])

    cfg_path = model_path / "cfg_args"
    if cfg_path.is_file():
        cfg_args = eval(cfg_path.read_text(), {"Namespace": Namespace})
        values = vars(args).copy()
        values.update({key: value for key, value in vars(cfg_args).items() if value is not None})
        args = Namespace(**values)
    else:
        print(
            f"[WARNING] Training config not found: {cfg_path}. "
            "Using default model parameters (including sh_degree=3 and "
            "depth_correct=False); confirm these match the trained model."
        )

    args.source_path = str(source_path.resolve())
    args.model_path = str(model_path.resolve())
    args.eval = eval_mode
    args.image_loading_mode = "lazy"
    args.lazy_image_loading = True
    return args


def resolve_iteration(model_path: Path, requested_iteration: int):
    point_cloud_root = model_path / "point_cloud"
    if requested_iteration != -1:
        checkpoint = point_cloud_root / f"iteration_{requested_iteration}" / "point_cloud.ply"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Point cloud not found: {checkpoint}")
        return requested_iteration

    iterations = []
    if point_cloud_root.is_dir():
        for path in point_cloud_root.iterdir():
            if not path.is_dir() or not path.name.startswith("iteration_"):
                continue
            try:
                iteration = int(path.name.removeprefix("iteration_"))
            except ValueError:
                continue
            if (path / "point_cloud.ply").is_file():
                iterations.append(iteration)
    if not iterations:
        raise FileNotFoundError(
            f"No point_cloud/iteration_*/point_cloud.ply found under {model_path}"
        )
    return max(iterations)


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def render_one(view, gaussians, pipeline, background):
    return render(view, gaussians, pipeline, background)["render"]


def benchmark_resolution(
    views,
    gaussians,
    pipeline,
    background,
    resolution_spec,
    warmup_frames,
    repeats,
):
    if not views:
        raise ValueError("No cameras available for rendering.")

    base_width = int(views[0].image_width)
    base_height = int(views[0].image_height)
    for view in views:
        if int(view.image_width) != base_width or int(view.image_height) != base_height:
            raise ValueError("All cameras must have the same resolution.")

    label, width, height = resolve_resolution(resolution_spec, base_width, base_height)
    for view in views:
        view.image_width = width
        view.image_height = height

    warmup_frames = min(max(warmup_frames, 0), len(views))
    for view in views[:warmup_frames]:
        render_one(view, gaussians, pipeline, background)
    synchronize()

    latencies_ms = []
    synchronize()
    start_time = time.perf_counter()
    for _ in range(repeats):
        for view in views:
            synchronize()
            frame_start = time.perf_counter()
            render_one(view, gaussians, pipeline, background)
            synchronize()
            latencies_ms.append((time.perf_counter() - frame_start) * 1000.0)
    elapsed = time.perf_counter() - start_time

    sorted_latencies = sorted(latencies_ms)
    p95_index = min(
        len(sorted_latencies) - 1,
        max(0, int(len(sorted_latencies) * 0.95) - 1),
    )
    frames = len(latencies_ms)
    return {
        "resolution": label,
        "width": width,
        "height": height,
        "frames": frames,
        "seconds": elapsed,
        "fps": frames / elapsed if elapsed > 0 else 0.0,
        "avg_frame_latency_ms": statistics.mean(latencies_ms),
        "p50_frame_latency_ms": statistics.median(latencies_ms),
        "p95_frame_latency_ms": sorted_latencies[p95_index],
        "min_frame_latency_ms": sorted_latencies[0],
        "max_frame_latency_ms": sorted_latencies[-1],
    }


def render_dataset(
    source_path: Path,
    model_path: Path,
    requested_iteration: int,
    resolution_specs,
    split: str,
    eval_mode: bool,
    warmup_frames: int,
    repeats: int,
    camera_resolution=None,
):
    iteration = resolve_iteration(model_path, requested_iteration)
    args = load_model_args(source_path, model_path, eval_mode)
    dataset_args = ModelParams(argparse.ArgumentParser(), sentinel=True).extract(args)
    pipeline_args = PipelineParams(argparse.ArgumentParser()).extract(args)

    with torch.inference_mode():
        gaussians = GaussianModel(dataset_args.sh_degree, dataset_args.depth_correct)
        scene = Scene(dataset_args, gaussians, load_iteration=iteration, shuffle=False,
                      load_images=False, camera_resolution=camera_resolution)

        if split == "test":
            views = scene.getTestCameras()
            actual_split = "test"
            if not views:
                views = scene.getTrainCameras()
                actual_split = "train"
        else:
            views = scene.getTrainCameras()
            actual_split = "train"
        if not views:
            raise ValueError(f"No cameras found for dataset: {source_path}")

        background_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0]
        background = torch.tensor(background_color, dtype=torch.float32, device="cuda")
        results = []
        original_sizes = [(view.image_width, view.image_height) for view in views]
        for spec in resolution_specs:
            for view, (width, height) in zip(views, original_sizes):
                view.image_width, view.image_height = width, height
            result = benchmark_resolution(
                views,
                gaussians,
                pipeline_args,
                background,
                spec,
                warmup_frames,
                repeats,
            )
            results.append(result)
            print(
                f"{source_path.name}: iteration {iteration}, {result['resolution']}, "
                f"{result['frames']} frames, {result['fps']:.4f} FPS, "
                f"avg latency {result['avg_frame_latency_ms']:.3f} ms"
            )
        return iteration, actual_split, results


FIELDNAMES = [
    "dataset", "source_path", "model_path", "status", "error", "iteration",
    "split", "resolution", "width", "height", "frames", "seconds", "fps",
    "avg_frame_latency_ms", "p50_frame_latency_ms", "p95_frame_latency_ms",
    "min_frame_latency_ms", "max_frame_latency_ms", "world_size", "bsz",
]


def result_row(dataset_path, model_path, iteration, split, result):
    return {
        "dataset": dataset_path.name,
        "source_path": str(dataset_path),
        "model_path": str(model_path),
        "status": "ok",
        "error": "",
        "iteration": iteration,
        "split": split,
        "resolution": result["resolution"],
        "width": result["width"],
        "height": result["height"],
        "frames": result["frames"],
        "seconds": f'{result["seconds"]:.6f}',
        "fps": f'{result["fps"]:.6f}',
        "avg_frame_latency_ms": f'{result["avg_frame_latency_ms"]:.6f}',
        "p50_frame_latency_ms": f'{result["p50_frame_latency_ms"]:.6f}',
        "p95_frame_latency_ms": f'{result["p95_frame_latency_ms"]:.6f}',
        "min_frame_latency_ms": f'{result["min_frame_latency_ms"]:.6f}',
        "max_frame_latency_ms": f'{result["max_frame_latency_ms"]:.6f}',
        "world_size": 1,
        "bsz": INFERENCE_BATCH_SIZE,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Render all datasets below a directory and write an FPS CSV."
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("path1", nargs="?", help="dataset root or single dataset")
    parser.add_argument("path2", nargs="?", help="model root or single model")
    parser.add_argument("--csv", dest="csv_path", default=None)
    parser.add_argument(
        "--iteration", type=int, default=-1,
        help="checkpoint iteration; -1 means latest",
    )
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument(
        "--no-eval", action="store_true",
        help="do not create an eval/test split",
    )
    parser.add_argument(
        "--resolutions", nargs="+", default=DEFAULT_RESOLUTIONS,
        help="scale factors or WxH values, e.g. 1.0 0.5 1920x1080",
    )
    parser.add_argument("--warmup-frames", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--camera-resolution", metavar="WIDTHxHEIGHT",
        help="Source image dimensions for camera geometry (required for Blender if w/h are missing; overrides COLMAP dimensions).",
    )
    parser.add_argument("--quiet", action="store_true")
    cli_args = parser.parse_args()

    if cli_args.repeats <= 0:
        parser.error("--repeats must be positive")
    camera_resolution = None
    if cli_args.camera_resolution:
        try:
            spec = cli_args.camera_resolution.lower()
            width, height = (int(part) for part in spec.split("x"))
            if width <= 0 or height <= 0:
                raise ValueError
            camera_resolution = (width, height)
        except ValueError:
            parser.error("--camera-resolution must be WIDTHxHEIGHT with positive integers")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "recursive_render.py requires CUDA, as does sort-free-gs rendering."
        )
    torch.cuda.set_device(0)

    script_dir = Path(__file__).resolve().parent
    default_source = script_dir / "../res/training/tanksandtemples/train"
    source_text = cli_args.path1 or os.environ.get("SOURCE_PATH", str(default_source))
    source_path = Path(source_text).expanduser().resolve()

    if cli_args.recursive:
        if not source_path.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {source_path}")
        dataset_paths = discover_datasets(source_path)
        if not dataset_paths:
            raise FileNotFoundError(f"No datasets found under: {source_path}")
        model_text = cli_args.path2 or os.environ.get("MODEL_PATH", "")
        model_root = (
            Path(model_text).expanduser().resolve() if model_text else source_path
        )
        output_root = model_root
    else:
        if not is_dataset(source_path):
            raise ValueError(f"Unrecognized dataset layout: {source_path}")
        dataset_paths = [source_path]
        model_text = cli_args.path2 or os.environ.get(
            "MODEL_PATH", str(source_path / "trained")
        )
        model_root = Path(model_text).expanduser().resolve()
        output_root = model_root

    resolution_specs = parse_resolution_specs(cli_args.resolutions)
    rows = []

    for dataset_path in dataset_paths:
        if cli_args.recursive and model_root == source_path:
            model_path = dataset_path / "trained"
        elif cli_args.recursive:
            model_path = model_root / dataset_path.name
        else:
            model_path = model_root
        model_path = model_path.resolve()

        print(f"\n=== {dataset_path.name} ===")
        print(f"source: {dataset_path}")
        print(f"model:  {model_path}")
        try:
            iteration, actual_split, results = render_dataset(
                dataset_path,
                model_path,
                cli_args.iteration,
                resolution_specs,
                cli_args.split,
                not cli_args.no_eval,
                cli_args.warmup_frames,
                cli_args.repeats,
                camera_resolution,
            )
            rows.extend(
                result_row(dataset_path, model_path, iteration, actual_split, result)
                for result in results
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"[FAILED] {dataset_path.name}: {error}")
            rows.append({
                "dataset": dataset_path.name,
                "source_path": str(dataset_path),
                "model_path": str(model_path),
                "status": "failed",
                "error": error,
                "iteration": "",
                "split": "",
                "resolution": "",
                "width": "",
                "height": "",
                "frames": "",
                "seconds": "",
                "fps": "",
                "avg_frame_latency_ms": "",
                "p50_frame_latency_ms": "",
                "p95_frame_latency_ms": "",
                "min_frame_latency_ms": "",
                "max_frame_latency_ms": "",
                "world_size": 1,
                "bsz": INFERENCE_BATCH_SIZE,
            })
        finally:
            torch.cuda.empty_cache()

    csv_path = (
        Path(cli_args.csv_path).expanduser().resolve()
        if cli_args.csv_path
        else output_root / "recursive_render_fps.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nFPS CSV written to: {csv_path}")
    return 1 if any(row["status"] == "failed" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
