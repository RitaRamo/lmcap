DATA_DIR="../rerank/data/coco2014"
IMAGES_DIR="/media/rprstorage5/rprstorage/rita/remote-sensing-images-caption/src/data/COCO/raw_dataset/images/"
IMAGES_NAMES="$DATA_DIR/datasets/images_names.json"
DATASET_SPLITS="$DATA_DIR/datasets/dataset_splits.json"
FEATURES_DIR="features/coco2014_clip_multilingual_large.hdf5" 

args="""
--images_dir $IMAGES_DIR \
--images_names $IMAGES_NAMES \
--dataset_splits $DATASET_SPLITS \
--image_features_path $FEATURES_DIR \
"""

export PYTHONWARNINGS="ignore"

time python3 src/extract_features.py $args

