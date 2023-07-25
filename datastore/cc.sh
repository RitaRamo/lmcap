DATASET_LANGUAGE="cc_es"
RETRIEVAL_DIR="datastore/$DATASET_LANGUAGE"
ANNOTATIONS="/media/rprstorage5/rprstorage/rita/rerank/data/$DATASET_LANGUAGE/annotations/captions_train2014.json"

#run for train, val and test
args="""
--annotation_filename $ANNOTATIONS \
--datastore_path $RETRIEVAL_DIR \
--batch_size 10
--large_data
"""

export PYTHONWARNINGS="ignore"

time python3 src/index_datastore.py $args