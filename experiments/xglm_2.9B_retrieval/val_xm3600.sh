DATASET="xm3600"
DATASET_SPLITS="data/${DATASET}/datasets/dataset_splits.json"
MODEL_TYPE="2.9B"
MODEL_ABBR="xglm_${MODEL_TYPE}_retrieval"
EXP_DIR="experiments/${MODEL_ABBR}"

SPLIT="val"
TEMPLATE="retrieval"
LanguageArray=("es" "zh" "hi" "en")
KArray=( "4" )
ContextArray=("3" )

for LANGUAGE in ${LanguageArray[*]}; do
    for K in ${KArray[*]}; do
        for CONTEXT in ${ContextArray[*]}; do
            #generate the captions based on:
            args="""
                --dataset $DATASET \
                --dataset_splits $DATASET_SPLITS \
                --output_path $EXP_DIR \
                --split $SPLIT \
                --beam_size 3 \
                --eval_beam_size 3 \
                --template_type $TEMPLATE \
                --language $LANGUAGE \
                --k $K \
                --in_context
                --n_context $CONTEXT
                --batch_size 1
                --xglm_type $MODEL_TYPE
                --retrieve_filename caps_retrieved/xm3600/nn_setup_coco2014.json \
                --support_reference_caps  caps_support/coco2014/reference_caps.json
                --support_retrieved_caps caps_support/coco2014/retrieved_caps.json
            """

            time python3 src/eval.py $args $model_args

            DATA_DIR="data/${DATASET}_${LANGUAGE}"

            echo "DATA_DIR: $LANGUAGE"
            echo "DATA_DIR: $DATA_DIR"
            echo "EXP_DIR: $EXP_DIR"
            RES_FN="$EXP_DIR/outputs/${DATASET}_${TEMPLATE}_${LANGUAGE}_${K}_${CONTEXT}_${SPLIT}.beam_3.json"
            OUT_DIR="$EXP_DIR/results"

            args="""
                --results_fn $RES_FN \
                --output_dir $OUT_DIR \
                --annotations_path ${DATA_DIR}/annotations/captions_${SPLIT}2014_new.json \
            """
            export PYTHONWARNINGS="ignore"

            python3 src/score.py $args
        done
    done
done

yourfilenames=`ls ${EXP_DIR}/outputs/${DATASET}*_3_val.beam_3.top_3.json`
for eachfile in $yourfilenames
do
   #rerank: have the best cap of the beam according to CLIP (the one most sim to the image)
   echo $eachfile
   FEATURES_NAME="features/${DATASET}_clip_multilingual_large.hdf5"

   args="""
    --image_features_filename $FEATURES_NAME \
    --caps_path $eachfile \
    """

    time python3 src/rerank.py $args

done

for LANGUAGE in ${LanguageArray[*]}; do
    for K in ${KArray[*]}; do
        for CONTEXT in ${ContextArray[*]}; do
            #score that reranking
            RES_FN="$EXP_DIR/outputs/rerank_${DATASET}_${TEMPLATE}_${LANGUAGE}_${K}_${CONTEXT}_${SPLIT}.beam_3.top_3.json"
            OUT_DIR="$EXP_DIR/results"

            DATA_DIR="data/xm3600_${LANGUAGE}"

            args="""
                --results_fn $RES_FN \
                --output_dir $OUT_DIR \
                --annotations_path ${DATA_DIR}/annotations/captions_${SPLIT}2014_new.json \
                --language $LANGUAGE
            """
            export PYTHONWARNINGS="ignore"

            python3 src/score.py $args
        done
    done
done

