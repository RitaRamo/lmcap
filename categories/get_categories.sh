
IMAGE_FEATURES="features/coco2014_clip_multilingual_large.hdf5"
DATASET="coco2014"
OUTPUT_DIR="categories/$DATASET"
DATASET_SPLITS="../rerank/data/$DATASET/datasets/dataset_splits.json"

args="""
--image_features_filename $IMAGE_FEATURES \
--dataset_splits $DATASET_SPLITS \
--outputs_dir $OUTPUT_DIR \
"""

export PYTHONWARNINGS="ignore"

time python3 src/categories.py $args


