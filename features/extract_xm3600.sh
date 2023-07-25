DATA_DIR="../rerank/data/xm3600"
IMAGES_DIR="/media/rprstorage5/rprstorage/rita/rerank/data/xm3600/images/"
IMAGES_NAMES="$DATA_DIR/datasets/images_names.json"
DATASET_SPLITS="$DATA_DIR/datasets/dataset_splits.json"
FEATURES_DIR="features/xm3600_clip_multilingual_large.hdf5" 


args="""
--images_dir $IMAGES_DIR \
--images_names $IMAGES_NAMES \
--dataset_splits $DATASET_SPLITS \
--image_features_path $FEATURES_DIR \
"""

export PYTHONWARNINGS="ignore"

time python3 src/extract_features.py $args

