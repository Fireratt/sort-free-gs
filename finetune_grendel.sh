#!/bin/bash
# 事先需要创建软连接：output/3dgs-ply/point_cloud.ply，作为续训的起始点云
set -e

# ========== 配置区 ==========
MATRIX_DATA_PATH="/home/ps/hjf/3dgs/res/matrix_city/small_city/aerial_downsampled/pose/block_all"
MATRIX_OUTPUT_PATH="./output/matrix_city_sortfree"
RUBBLE_DATA_PATH="/home/ps/hjf/3dgs/res/rubble-colmap"
RUBBLE_OUTPUT_PATH="./output/rubble_sortfree"
BUILDING_DATA_PATH="/home/ps/hjf/3dgs/res/training/building"
BUILDING_OUTPUT_PATH="/home/ps/hjf/3dgs/res/training/building/output"
DATA_PATH=${BUILDING_DATA_PATH}
OUTPUT_PATH=${BUILDING_OUTPUT_PATH}
ITERATIONS=20000        # 续训迭代数，通常5k-10k足够收敛新参数
SAVE_INTERVAL=4000
# ============================

echo "🚀 Sort-Free-GS Finetune from Grendel checkpoint"
echo "   Source PLY : $GRENDEL_PLY"
echo "   Data       : $DATA_PATH"
echo "   Output     : $OUTPUT_PATH"

python train.py \
    -s "$DATA_PATH" \
    -m "$OUTPUT_PATH" \
    --iterations "$ITERATIONS" \
    --save_iterations "$SAVE_INTERVAL" "$ITERATIONS" \
    --densify_until_iter 0 \
    --eval \
    --depth_correct \
    --image_loading_mode lazy \
    --test_iterations 2000 4000 6000 8000 10000 \
    --quiet 2>&1 | tee "$OUTPUT_PATH/train.log"

echo "✅ Finetune complete. Evaluating..."
python metrics.py -m "$OUTPUT_PATH"