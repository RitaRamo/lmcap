RETRIEVAL_DIR="datastore/coco2014"
ANNOTATIONS="/media/rprstorage5/rprstorage/rita/rerank/data/coco2014/annotations/captions_train2014.json"

#run for train, val and test
args="""
--annotation_filename $ANNOTATIONS \
--datastore_path $RETRIEVAL_DIR \
--batch_size 10
"""

export PYTHONWARNINGS="ignore"

time python3 src/index_datastore.py $args