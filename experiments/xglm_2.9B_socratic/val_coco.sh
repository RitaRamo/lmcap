
DATASET="coco2014"
DATASET_SPLITS="data/${DATASET}/datasets/dataset_splits.json"
CATEGORIES_DIR="categories/${DATASET}"
SUPPORT_CATEGORIES_DIR="categories/${DATASET}"

MODEL_TYPE="2.9B"
MODEL_ABBR="xglm_${MODEL_TYPE}_socratic"
EXP_DIR="experiments/${MODEL_ABBR}"

SPLIT="val"
TEMPLATE="socratic"
LanguageArray=("es")
ContextArray=("3" "5" "4" "6" "7" "8" "9")

for LANGUAGE in ${LanguageArray[*]}; do
    for CONTEXT in ${ContextArray[*]}; do
        #generate the captions based on:
        args="""
            --categories_dir $CATEGORIES_DIR \
            --support_categories_dir $SUPPORT_CATEGORIES_DIR \
            --dataset $DATASET \
            --dataset_splits $DATASET_SPLITS \
            --output_path $EXP_DIR \
            --split $SPLIT \
            --beam_size 3 \
            --eval_beam_size 3 \
            --template_type $TEMPLATE \
            --language $LANGUAGE \
            --in_context
            --n_context $CONTEXT
            --batch_size 1
            --xglm_type $MODEL_TYPE
            --support_reference_caps  caps_support/coco2014/reference_caps.json
        """

        time python3 src/eval.py $args $model_args

        #then compute score of the generated captions
        if [ "$LANGUAGE" = "en" ]; then
            DATA_DIR="data/coco2014"
        else
            DATA_DIR="data/coco_${LANGUAGE}"
        fi

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


yourfilenames=`ls ${EXP_DIR}/outputs/coco2014*_3_val.beam_3.top_3.json`
for eachfile in $yourfilenames
do
   #rerank: have the best cap of the beam according to CLIP (the one most sim to the image)
   echo $eachfile
   FEATURES_NAME="features/coco2014_clip_multilingual_large.hdf5"

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

             if [ "$LANGUAGE" = "en" ]; then
                DATA_DIR="data/coco2014"
            else
                DATA_DIR="data/coco_${LANGUAGE}"
            fi

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



