#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# MODEL_PATH=/home/ps/hjf/3dgs/sort-free-gs/output/matrix_city_sortfree
# SOURCE_PATH=/home/ps/hjf/3dgs/res/matrix_city/small_city/aerial_downsampled/pose/block_all

SOURCE_PATH="${1:-${SOURCE_PATH:-"${SCRIPT_DIR}/../res/training/tanksandtemples/train"}}"
MODEL_PATH="${2:-${MODEL_PATH:-"${SCRIPT_DIR}/trained"}}"
ITERATION="${3:-${ITERATION:--1}}"
OPACITY="${4:-${OPACITY:-1.0}}"
RESOLUTION="${5:-${RESOLUTION:--1}}"

if ! [[ "${OPACITY}" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk -v opacity="${OPACITY}" 'BEGIN { exit !(opacity >= 0 && opacity <= 1) }'; then
  echo "Opacity must be a number in [0, 1]: ${OPACITY}" >&2
  exit 1
fi
if ! [[ "${RESOLUTION}" =~ ^(-1|[1-9][0-9]*|[1-9][0-9]*x[1-9][0-9]*)$ ]]; then
  echo "Resolution must be -1, a positive integer, or WIDTHxHEIGHT: ${RESOLUTION}" >&2
  exit 1
fi

if [[ ! -d "${SOURCE_PATH}" ]]; then
  echo "Dataset path does not exist: ${SOURCE_PATH}" >&2
  echo "Usage: $0 [dataset_path] [model_path] [iteration] [opacity] [resolution]" >&2
  exit 1
fi

if [[ "$(basename "${MODEL_PATH}")" =~ ^iteration_([0-9]+)$ && "$(basename "$(dirname "${MODEL_PATH}")")" == "point_cloud" ]]; then
  ITERATION="${BASH_REMATCH[1]}"
  MODEL_PATH="$(dirname "$(dirname "${MODEL_PATH}")")"
fi

if [[ ! -f "${MODEL_PATH}/cfg_args" ]]; then
  echo "Model path must be the training output root containing cfg_args: ${MODEL_PATH}" >&2
  echo "Do not pass point_cloud/iteration_x as -m; use model root plus --iteration." >&2
  echo "Usage: $0 [dataset_path] [model_path] [iteration] [opacity] [resolution]" >&2
  exit 1
fi

if [[ "${ITERATION}" != "-1" && ! -f "${MODEL_PATH}/point_cloud/iteration_${ITERATION}/point_cloud.ply" ]]; then
  echo "Missing point cloud: ${MODEL_PATH}/point_cloud/iteration_${ITERATION}/point_cloud.ply" >&2
  exit 1
fi

if [[ ! -d "${SOURCE_PATH}/sparse" && ! -f "${SOURCE_PATH}/transforms_train.json" ]]; then
  echo "Unrecognized dataset layout: ${SOURCE_PATH}" >&2
  echo "Expected COLMAP data with sparse/ or Blender data with transforms_train.json." >&2
  exit 1
fi
if [[ "${RESOLUTION}" == *x* ]]; then
  RESOLUTION_ARGS=(--render_resolution "${RESOLUTION}")
else
  RESOLUTION_ARGS=(--resolution "${RESOLUTION}")
fi
# 测试FPS时把这个关掉
# --store_image
conda run --no-capture-output \
-n sortfree python "${SCRIPT_DIR}/render.py" \
--eval -s "${SOURCE_PATH}" -m "${MODEL_PATH}" \
--iteration "${ITERATION}" --depth_correct \
--opacity "${OPACITY}" "${RESOLUTION_ARGS[@]}"
# --store_image
