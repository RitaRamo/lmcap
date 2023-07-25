# LMCap

## Interacting with LMCap

Our model did not require any training, hence you only need to interact with it at inferece time. 

### Data

To evaluate our model, we used the COCO Karpathy splits and the XM3600 dataset. 

You can download the precomputed data from [here](), placing them in the corresponding folders:
- Download `data`, i.e., the COCO and XM3600 annotations and splits from "en,es, zh, hi"
- Download `caps_retrieved`, i.e., the retrieved captions for each dataset
- Donwload `caps_support`, i.e., captions for the support examples for in context learning
- Donwload `categories`, i.e., in case you want to run the socratic baseline

Alternatively, you can get all this data by processing it yourself. For that, following the next steps:
1. Run ```python3 src/extract_features.py $args``` to extract the images features (have an example inside folder `features/` to see how to fill the command-line options $args)
2. Run ```python3 src/index_datastore.py $args``` to create the datastore indexes (have an example inside folder `datastore/` to see how to fill the command-line options $args)
3. Run ```python3 src/retrieve_caps.py $args``` to use the index datastores to retrieve the corresponding captions (have an example inside folder `retrieve_caps/` to see how to fill the command-line options $args)
4. Run ```python3 src/categories.py $args``` to extract the categories for the Socratic baseline (have an example inside folder `categories/` to see how to fill the command-line options $args)


### Inference

Inside folder `experiments.json`, you have it there how you can run the experiments of our paper:
- run ```./experiments/xglm_2.9B_retrieval/val_xm3600.sh``` (for xm3600 results)
- run ```./experiments/xglm_2.9B_retrieval/val_coco.sh``` (for coco results)

You can play around with the languages (LanguageArray), number of k retrieved captions (KArray), the number of support examples (ContextArray), the split ("val" or "test"):

LanguageArray=("es" "zh" "hi" "en")
KArray=( "4" )
ContextArray=("3" )
SPLIT="val"

Alternative, you can run:
- ```python3 src/eval.py $args``` (to generate the corresponding captions).
- ```python3 src/score.py $args``` (to compute the metric scores for the generated captions).
- ```python3 src/rerank.py $args``` (to rerank the generated captions, i.e., the final image–text similarity step to find the
caption that best describes the input image based on the M-CLIP model.)

In all of the above, you would need to fill the command-line options $args according to your preference. Note you have an example of how to fill those $args in the aformentioned `experiments` folder. 

### Paper

If you find our code/data/models or ideas useful in your research, please consider citing the [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Ramos_SmallCap_Lightweight_Image_Captioning_Prompted_With_Retrieval_Augmentation_CVPR_2023_paper.pdf):
```
@inproceedings{ramos-etal-2023-lmcap,
    title = "{LMC}ap: Few-shot Multilingual Image Captioning by Retrieval Augmented Language Model Prompting",
    author = "Ramos, Rita  and
      Martins, Bruno  and
      Elliott, Desmond",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.104",
    pages = "1635--1651",
}
```
