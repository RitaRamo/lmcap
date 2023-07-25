
IMAGE_FEATURES="features/xm3600_clip_multilingual_large.hdf5"
DATASET="xm3600"
OUTPUT_DIR="categories/$DATASET"
DATASET_SPLITS="../rerank/data/$DATASET/datasets/dataset_splits.json"

args="""
--image_features_filename $IMAGE_FEATURES \
--dataset_splits $DATASET_SPLITS \
--outputs_dir $OUTPUT_DIR \
"""

export PYTHONWARNINGS="ignore"

time python3 src/categories.py $args


