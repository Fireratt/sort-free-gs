#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ITERATIONS=7000        # 训迭代数

usage() {
    echo "Usage:"
    echo "  $0 <dataset_path> [model_path]"
    echo "  $0 --recursive <dataset_root> [model_root]"
    echo
    echo "Examples:"
    echo "  $0 /path/to/stump"
    echo "  $0 /path/to/stump /path/to/stump/trained"
    echo "  $0 --recursive /path/to/mipnerf360"
    echo "  $0 --recursive /path/to/mipnerf360 /path/to/mipnerf360_trained"
}

train_one() {
    local source_path="$1"
    local model_path="$2"

    echo
    echo "============================================================"
    echo "Training dataset:"
    echo "  Source: ${source_path}"
    echo "  Model:  ${model_path}"
    echo "============================================================"
    echo

    conda run --no-capture-output -n sortfree \
        python "${SCRIPT_DIR}/train.py" \
        --eval \
        -s "${source_path}" \
        -m "${model_path}" \
        --depth_correct \
        --iterations "${ITERATIONS}" \
        --image_loading_mode lazy \
        --quiet 2>&1 | tee "${model_path}/train.log"

    local status=$?

    echo
    echo "[DEBUG] conda run exit code: ${status}"
    echo

    return "${status}"
}
is_dataset() {
    local path="$1"

    [[ -d "${path}/sparse" || -f "${path}/transforms_train.json" ]]
}

# ----------------------------------------------------------------------
# Recursive mode
# ----------------------------------------------------------------------
if [[ "${1:-}" == "--recursive" ]]; then
    DATASET_ROOT="${2:-${SOURCE_PATH:-}}"
    MODEL_ROOT="${3:-${MODEL_PATH:-}}"

    if [[ -z "${DATASET_ROOT}" ]]; then
        echo "Error: dataset root is required." >&2
        usage
        exit 1
    fi

    if [[ ! -d "${DATASET_ROOT}" ]]; then
        echo "Dataset root does not exist: ${DATASET_ROOT}" >&2
        exit 1
    fi

    # Default: each dataset is trained into its own "trained" directory.
    if [[ -z "${MODEL_ROOT}" ]]; then
        MODEL_ROOT="${DATASET_ROOT}"
    fi

    mapfile -t DATASETS < <(
        find "${DATASET_ROOT}" \
            -mindepth 1 \
            -maxdepth 1 \
            -type d \
            -print | sort
    )

    if [[ ${#DATASETS[@]} -eq 0 ]]; then
        echo "No dataset directories found under: ${DATASET_ROOT}" >&2
        exit 1
    fi

TRAINED_COUNT=0
FAILED_DATASETS=()

echo "Discovered dataset directories:"
printf '  %s\n' "${DATASETS[@]}"
echo

for DATASET_PATH in "${DATASETS[@]}"; do
    DATASET_NAME="$(basename "${DATASET_PATH}")"

    if ! is_dataset "${DATASET_PATH}"; then
        echo "Skipping non-dataset directory: ${DATASET_NAME}"
        continue
    fi

    if [[ "${MODEL_ROOT}" == "${DATASET_ROOT}" ]]; then
        CURRENT_MODEL_PATH="${DATASET_PATH}/trained"
    else
        CURRENT_MODEL_PATH="${MODEL_ROOT}/${DATASET_NAME}"
    fi

    echo
    echo "############################################################"
    echo "# Dataset: ${DATASET_NAME}"
    echo "# Source:  ${DATASET_PATH}"
    echo "# Model:   ${CURRENT_MODEL_PATH}"
    echo "############################################################"
    echo

    set +e
    train_one "${DATASET_PATH}" "${CURRENT_MODEL_PATH}"
    STATUS=$?
    set -e

    echo
    echo "[DEBUG] train_one returned: ${STATUS}"
    echo "[DEBUG] dataset: ${DATASET_NAME}"
    echo

    if [[ ${STATUS} -eq 0 ]]; then
        TRAINED_COUNT=$((TRAINED_COUNT + 1))
        echo "[SUCCESS] ${DATASET_NAME}"
    else
        FAILED_DATASETS+=("${DATASET_NAME}")
        echo "[FAILED] ${DATASET_NAME}, exit code=${STATUS}"
    fi

    echo "[DEBUG] Moving to next dataset..."
done
fi

# ----------------------------------------------------------------------
# Normal single-dataset mode
# ----------------------------------------------------------------------
SOURCE_PATH="${1:-${SOURCE_PATH:-"${SCRIPT_DIR}/../res/training/tanksandtemples/train"}}"
MODEL_PATH="${2:-${MODEL_PATH:-"${SOURCE_PATH}/trained"}}"

if [[ ! -d "${SOURCE_PATH}" ]]; then
    echo "Dataset path does not exist: ${SOURCE_PATH}" >&2
    usage
    exit 1
fi

if ! is_dataset "${SOURCE_PATH}"; then
    echo "Unrecognized dataset layout: ${SOURCE_PATH}" >&2
    echo "Expected COLMAP data with sparse/ or Blender data with transforms_train.json." >&2
    exit 1
fi

train_one "${SOURCE_PATH}" "${MODEL_PATH}"