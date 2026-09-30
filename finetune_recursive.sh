#!/usr/bin/env bash
# 事先需要创建软连接：output/3dgs-ply/point_cloud.ply，作为续训的起始点云
set -euo pipefail

# ========== 配置区 ==========
ITERATIONS=20000        # 续训迭代数，通常5k-10k足够收敛新参数
TARGET_ITERATIONS=70000 # 目标模型轮数
# ============================

# ========== 环境变量 → 可选参数映射 ==========
EXTRA_TRAIN_ARGS=()
[[ "${ENABLE_DEPTH_CORRECT:-}" == "1" ]] && EXTRA_TRAIN_ARGS+=(--depth_correct)
[[ "${ENABLE_EVAL:-}"        == "1" ]] && EXTRA_TRAIN_ARGS+=(--eval)
[[ "${ENABLE_QUIET:-}"       == "1" ]] && EXTRA_TRAIN_ARGS+=(--quiet)
# 如需更多开关，在此追加即可，例如：
[[ "${ENABLE_LAZY_LOAD:-}" == "1" ]] && EXTRA_TRAIN_ARGS+=(--image_loading_mode lazy)
[[ "${ENABLE_CPU_OFFLOAD:-}" == "1" ]] && EXTRA_TRAIN_ARGS+=(--cpu_offload)

# ================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_DATE="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${SCRIPT_DIR}/train_${CURRENT_DATE}.log"
echo "Redirecting output to: ${LOG_FILE}" >&2
exec >"${LOG_FILE}" 2>&1

echo "Log file: ${LOG_FILE}"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo

usage() {
    echo "Usage:"
    echo "  $0 <dataset_path> [model_path]"
    echo "  $0 --recursive <dataset_root> [model_root]"
    echo
    echo "Environment variables (set to '1' to enable):"
    echo "  ENABLE_DEPTH_CORRECT   Pass --depth_correct to train.py"
    echo "  ENABLE_EVAL            Pass --eval to train.py"
    echo "  ENABLE_QUIET           Pass --quiet to train.py"
    echo
    echo "Examples:"
    echo "  $0 /path/to/stump"
    echo "  ENABLE_EVAL=1 ENABLE_DEPTH_CORRECT=1 $0 /path/to/stump"
    echo "  $0 --recursive /path/to/mipnerf360"
}

create_symlink() {
    local source_path="$1"
    local target_path="$2"
    local target_dir
    target_dir="$(dirname "${target_path}")"
    mkdir -p "${target_dir}"
    rm -rf "${target_path}"
    ln -s "${source_path}" "${target_path}"
    echo "Created symbolic link:"
    echo "  ${target_path} -> ${source_path}"
}

is_trained() {
    local model_path="$1"
    local checkpoint_path="${model_path}/point_cloud/iteration_${ITERATIONS}"
    [[ -d "${checkpoint_path}" ]]
}

train_one() {
    local source_path="$1"
    local model_path="$2"
    local grendel_ply="${model_path}/point_cloud/iteration_${TARGET_ITERATIONS}/point_cloud.ply"
    local output_dir="${SCRIPT_DIR}/output"
    local grendel_link="${output_dir}/3dgs-ply/point_cloud.ply"

    echo
    echo "============================================================"
    echo "Training dataset:"
    echo "  Source: ${source_path}"
    echo "  Model:  ${model_path}"
    echo "============================================================"
    echo

    if [[ ! -f "${grendel_ply}" ]]; then
        echo "ERROR: Grendel checkpoint PLY does not exist:"
        echo "  ${grendel_ply}" >&2
        return 1
    fi

    echo "Preparing Grendel checkpoint symbolic link..."
    echo "  Source : ${grendel_ply}"
    echo "  Link   : ${grendel_link}"
    create_symlink "${grendel_ply}" "${grendel_link}"

    if [[ ! -L "${grendel_link}" ]]; then
        echo "ERROR: Failed to create symbolic link:" >&2
        echo "  ${grendel_link}" >&2
        return 1
    fi
    echo "Symbolic link ready."
    echo

    echo "🚀 Sort-Free-GS Finetune from Grendel checkpoint"
    echo "   Source PLY : ${grendel_ply}"
    echo "   Link       : ${grendel_link}"
    echo "   Data       : ${source_path}"
    echo "   Output     : ${model_path}"
    echo "   Extra args : ${EXTRA_TRAIN_ARGS[*]:-<none>}"
    echo

    # ✅ 核心改动：移除硬编码标志，改用数组安全展开
    python train.py \
        -s "${source_path}" \
        -m "${model_path}" \
        --iterations "${ITERATIONS}" \
        --save_iterations "${ITERATIONS}" \
        --densify_until_iter 0 \
        --test_iterations 500 1500 2500 \
        "${EXTRA_TRAIN_ARGS[@]}" \
        2>&1 | tee "${model_path}/train.log"

    echo
    echo "✅ Finetune complete. Evaluating..."
    python metrics.py -m "${model_path}"
    local status=$?
    echo "[DEBUG] train_one exit code: ${status}"
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
    if [[ -z "${MODEL_ROOT}" ]]; then
        MODEL_ROOT="${DATASET_ROOT}"
    fi

    mapfile -t DATASETS < <(
        find "${DATASET_ROOT}" -mindepth 1 -maxdepth 1 -type d -print | sort
    )
    if [[ ${#DATASETS[@]} -eq 0 ]]; then
        echo "No dataset directories found under: ${DATASET_ROOT}" >&2
        exit 1
    fi

    TRAINED_COUNT=0
    SKIPPED_COUNT=0
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

        if is_trained "${CURRENT_MODEL_PATH}"; then
            echo "# Status:  ALREADY TRAINED - SKIPPING"
            echo "############################################################"
            SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
            continue
        fi

        set +e
        train_one "${DATASET_PATH}" "${CURRENT_MODEL_PATH}"
        STATUS=$?
        set -e

        if [[ ${STATUS} -eq 0 ]]; then
            TRAINED_COUNT=$((TRAINED_COUNT + 1))
            echo "[SUCCESS] ${DATASET_NAME}"
        else
            FAILED_DATASETS+=("${DATASET_NAME}")
            echo "[FAILED] ${DATASET_NAME}, exit code=${STATUS}"
        fi
    done

    echo
    echo "===== Summary ====="
    echo "Trained: ${TRAINED_COUNT}, Skipped: ${SKIPPED_COUNT}, Failed: ${#FAILED_DATASETS[@]}"
    if [[ ${#FAILED_DATASETS[@]} -gt 0 ]]; then
        echo "Failed datasets: ${FAILED_DATASETS[*]}"
        exit 1
    fi
    exit 0
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