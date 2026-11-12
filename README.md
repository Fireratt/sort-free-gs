# Sort-free Gaussian Splatting via Weighted Sum Rendering



This repository is a re-implementation of the ICLR'25 paper "Sort-free Gaussian Splatting via Weighted Sum Rendering". [Paper](https://arxiv.org/pdf/2410.18931).

We are honored that this  repository has been recognized and linked as the released  reference code by the author, [Qiqi Hou](https://hqqxyy.github.io/).

Our implementation builds upon the original 3DGS by incorporating global parameters `sigma` and `weight_background`, as well as a set of `sh` parameters for each GS to decode the view-dependent effects `Vi`.

## Performance on MipNeRF360 

### PSNR

|            | bicycle | bonsai | counter | flowers | garden | kitchen | room  | stump | treehill | AVG   |
| ---------- | ------- | ------ | ------- | ------- | ------ | ------- | ----- | ----- | -------- | ----- |
| Paper      | 24.20   | 32.05  | 29.53   | 20.45   | 27.78  | 31.38   | 31.93 | 25.39 | 22.01    | 27.19 |
| Ours       | 23.89   | 33.20  | 29.96   | 20.62   | 27.07  | 31.51   | 32.35 | 24.79 | 21.49    | 27.21 |
| Ours-w/o D | 23.89   | 33.05  | 29.83   | 20.78   | 27.14  | 31.21   | 31.65 | 24.90 | 21.61    | 27.12 |

### SSIM

|            | bicycle | bonsai | counter | flowers | garden | kitchen | room  | stump | treehill | AVG   |
| ---------- | ------- | ------ | ------- | ------- | ------ | ------- | ----- | ----- | -------- | ----- |
| Paper      | 0.744   | 0.938  | 0.909   | 0.580   | 0.872  | 0.923   | 0.925 | 0.728 | 0.614    | 0.804 |
| Ours       | 0.714   | 0.951  | 0.919   | 0.581   | 0.850  | 0.931   | 0.930 | 0.708 | 0.620    | 0.800 |
| Ours-w/o D | 0.714   | 0.948  | 0.918   |	0.582 	| 0.851  |0.931    | 0.929 |0.713  | 0.622    | 0.801 |

### LPIPS

|            | bicycle | bonsai | counter | flowers | garden | kitchen | room  | stump | treehill | AVG   |
| ---------- | ------- | ------ | ------- | ------- | ------ | ------- | ----- | ----- | -------- | ----- |
| Paper      | 0.205   | 0.199  | 0.191   | 0.342   | 0.097  | 0.125   | 0.197 | 0.235 | 0.311    | 0.211 |
| Ours       | 0.265   | 0.144  | 0.176   | 0.329   | 0.128  | 0.117   | 0.190 | 0.275 | 0.347    | 0.219 |
| Ours-w/o D | 0.265   | 0.175  | 0.176   | 0.329   | 0.128  | 0.117   | 0.190 | 0.272 | 0.344    | 0.222 |


`Ours - w/o D` means not adding the `--depth_correct` parameter during training.

We have added a new parameter `--depth_correct` to `train.py`, which controls the effect of Equation 10 in the paper. If this parameter is included, the weight calculation is performed as follows:

$$
w(d_i) = \max\left(0, 1 - \frac{d_i}{\sigma}\right) v_i
$$

which is consistent with the original paper. If the parameter is not included, the weight calculation change to:

$$
w(d_i) = v_i
$$

This modification allows us to experiment with the impact of depth correction on the rendering process and compare the results with the original model.

## Install

This project depends on two CUDA extension submodules: `submodules/simple-knn` and `submodules/diff-gaussian-rasterization`. The included `environment.yml` targets the CUDA 12.4 setup used for this workspace. If your machine uses a different CUDA toolkit, keep `pytorch-cuda` aligned with `nvcc --version`.

1. Clone this repo with submodules:

```shell
git clone --recursive https://github.com/LiYukeee/sort-free-gs.git
cd sort-free-gs
```

If you already cloned without submodules:

```shell
git submodule update --init --recursive
```

2. Recreate and activate the conda environment:

```shell
conda env remove -n sortfree
conda env create --file environment.yml
conda activate sortfree
```

If `sortfree` does not exist yet, skip the remove command.

3. Build the local CUDA extensions:

```shell
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=4
pip install --no-build-isolation submodules/simple-knn
pip install --no-build-isolation submodules/diff-gaussian-rasterization
```

For GPUs other than RTX 3090, change `TORCH_CUDA_ARCH_LIST` or omit it.

4. Verify the installation:

```shell
python -c "import simple_knn; from diff_gaussian_rasterization import GaussianRasterizer; print('extensions ok')"
```

### Docker

For a reproducible CUDA environment, build the included
`Dockerfile.sortfree-runner` from the repository root. The image creates the
`sortfree` Conda environment and compiles both CUDA extensions inside the
container.

```shell
docker build \
  --build-arg TORCH_CUDA_ARCH_LIST="8.6" \
  -f Dockerfile.sortfree-runner \
  -t sortfree-gs:cuda124 .
```

Use the architecture matching the target GPU. For an image supporting several
GPU generations, omit `--build-arg TORCH_CUDA_ARCH_LIST` and use the default
architecture list in the Dockerfile. The host requires a working NVIDIA
driver, Docker, and NVIDIA Container Toolkit. Conda is installed inside the
image; it is not required on the host.

Start a container with the dataset and output directories mounted. Keeping
datasets and checkpoints outside the image avoids rebuilding the image when
the data changes.

```shell
docker run --rm --gpus all --ipc=host -it \
  -v /path/to/datasets:/data \
  -v /path/to/outputs:/workspace/sort-free-gs/output \
  sortfree-gs:cuda124
```

Verify the CUDA extensions inside the container:

```shell
python -c "import torch, simple_knn; from diff_gaussian_rasterization import GaussianRasterizer; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print('extensions ok')"
```

The container starts in `/workspace/sort-free-gs` with the `sortfree` Python
environment enabled through `PATH`, so the normal training and rendering
commands can be used directly:

```shell
python train.py --eval \
  -s /data/<dataset> \
  -m /workspace/sort-free-gs/output/<scene> \
  --depth_correct

./render.sh \
  /data/<dataset> \
  /workspace/sort-free-gs/output/<scene> \
  -1 1.0 3840x2160
```

The arguments after the model path in `render.sh` are, in order:

```text
iteration opacity resolution
```

`-1` selects the latest checkpoint, `1.0` keeps the trained opacity, and
`3840x2160` requests an explicit 4K output size. The resolution can also be a
positive integer to specify a target width, or `-1` to use the default
resolution behavior.

### FPS benchmark without reference images

`recursive_render.py` benchmarks a trained model without reading reference
images, saving rendered images, or computing image-quality metrics. It writes
FPS and frame latency statistics to CSV. The benchmark uses one GPU; run it
with the `sortfree` environment or inside the Docker container.

For a single COLMAP scene, provide the camera files under `sparse/0/` and a
trained model root containing `cfg_args` and
`point_cloud/iteration_<iteration>/point_cloud.ply`. The camera files are
`cameras.bin` and `images.bin`, or their `.txt` equivalents. Neither the
reference images nor the COLMAP initial point cloud are needed:

```shell
python recursive_render.py /data/scene /workspace/sort-free-gs/output/scene \
  --iteration -1 --resolutions 3840x2160 --warmup-frames 2 --repeats 10 \
  --csv /workspace/sort-free-gs/output/scene/fps.csv
```

For a directory of datasets, add `--recursive` and pass the dataset root and
model root. Direct child scenes map to model directories of the same name:

```shell
python recursive_render.py --recursive /data/datasets /workspace/sort-free-gs/output \
  --iteration -1 --resolutions 1920x1080 3840x2160 \
  --csv /workspace/sort-free-gs/output/recursive_render_fps.csv
```

Without a separate model root, recursive mode looks for `trained/` inside each
scene. `--iteration -1` selects the latest saved PLY. The CSV records the
actual split, resolution, frame count, FPS, latency statistics and any
per-scene errors. A failed scene does not prevent later scenes from being
tested, but causes a nonzero exit status.

By default, COLMAP cameras are sorted by image name and indices `0, 8, 16, ...`
form the test split. Blender uses the order in `transforms_test.json`.
`--split train` benchmarks training cameras; if the test split is empty, the
script falls back to training cameras and reports `train` in the CSV. The
default is two warmup frames followed by one traversal of the selected split;
`--repeats` changes the number of timed traversals.

COLMAP camera dimensions come from `cameras.bin` or `cameras.txt`; Blender
requires `w` and `h` in its transform JSON files, either per frame or at the
top level. If Blender metadata lacks them, supply the original image dimensions
with `--camera-resolution WIDTHxHEIGHT`, for example:

```shell
python recursive_render.py /data/blender_scene /workspace/sort-free-gs/output/scene \
  --camera-resolution 800x800 --resolutions 3840x2160 --csv /workspace/sort-free-gs/output/scene/fps.csv
```

`--camera-resolution` also overrides COLMAP dimensions, which is useful when
the training images were resized relative to COLMAP's original camera size.
It describes the source images, not the output: choose dimensions matching the
training images to preserve their camera geometry and base resolution.
`--resolutions` controls output size using explicit `WIDTHxHEIGHT` values or
scale factors relative to the loaded camera resolution. The configured training
`resolution` setting is applied to the source dimensions before these scales.
The selected cameras must have the same base resolution.

## Data

First, create a ``data/`` folder inside the project path by

```
mkdir data
```

The data structure will be organised as follows:

```
data/
├── dataset_name
│   ├── scene1/
│   │   ├── images
│   │   │   ├── IMG_0.jpg
│   │   │   ├── IMG_1.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
│   ├── scene2/
│   │   ├── images
│   │   │   ├── IMG_0.jpg
│   │   │   ├── IMG_1.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
...
```

The MipNeRF360 scenes are provided by the paper author [here](https://jonbarron.info/mipnerf360/).

## Training

In our repository, the parameter settings for `train.py`, `render.py`, and `metrics.py` are identical to those of [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). 

### MipNeRF360

```shell
bash train_mipnerf360.sh
```

### Single Scene
The convenience script defaults to `../res/training/tanksandtemples/train` and writes to `./trained`. You can override both paths:

```shell
./train.sh <path to COLMAP or NeRF Synthetic dataset> <path to output>
./render.sh <dataset_path> <model_path> <iteration> [opacity] [resolution]
```


```shell
python train.py --eval -s <path to COLMAP or NeRF Synthetic dataset> -m <path to output> --depth_correct
# Enable CPU offload to keep Gaussian parameters and Adam state in system memory:
python train.py --eval -s <path to COLMAP or NeRF Synthetic dataset> -m <path to output> --depth_correct --cpu_offload
python render.py --eval -s <path to COLMAP or NeRF Synthetic dataset> -m <path to output root> --iteration <iteration> --depth_correct --store_image
# `-m` must be the training output root containing cfg_args, not point_cloud/iteration_x.
# To render the latest checkpoint, omit --iteration or use --iteration -1.
python metrics.py -m <path to output>
```

`--cpu_offload` keeps Gaussian parameters and Adam optimizer state in system memory, transfers parameters to CUDA for each forward/backward pass, then copies gradients back for the CPU optimizer step. Densification is disabled in this mode. The renderer still requires all Gaussian parameters and gradients for the current pass to fit in GPU memory, so this reduces optimizer-state memory but does not support a model whose Gaussian parameters alone exceed GPU capacity. CPU RAM must hold the parameters and optimizer state.

## LICENSE

Please follow the LICENSE of [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting).
