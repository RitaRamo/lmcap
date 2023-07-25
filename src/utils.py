"""General utility functions and variables"""

import os
import os.path
import json

CAPTIONS_FILENAME = "captions_raw.json"
CAPTIONS_DATASTORE_FILENAME = "captions_raw_datastore.json"
CAPTIONS_LARGE_FILENAME = "external_captions_ccs_synthetic_filtered_large.json"
CAPTIONS_IDS_LARGE_FILENAME = "external_captions_ids_ccs_synthetic_filtered_large.json"
IMAGES_NAMES_FILENAME = "images_names.json"
DATASET_SPLITS_FILENAME = "dataset_splits.json"
TRAIN_SPLIT = "train_images"
VALID_SPLIT = "val_images"
TEST_SPLIT = "test_images"
CLIP_DIM=1024


def get_split(args):
    all_ids = json.load(open(args.dataset_splits))

    if args.split == "val":
        split = all_ids[VALID_SPLIT]
    elif args.split == "train":
        split = all_ids[TRAIN_SPLIT]
    else:
        split = all_ids[TEST_SPLIT]

    return split


def get_default_jsons(args):
    images_names = json.load(open(args.images_names))
    split= get_split(args)
    return images_names, split


def get_socratic_categories(categories_dir):
    img_types = json.load(open(os.path.join(categories_dir, "img_type.json"))) 
    objects = json.load(open(os.path.join(categories_dir, "objects.json")))
    places = json.load(open(os.path.join(categories_dir, "places.json")))
    ppls = json.load(open(os.path.join(categories_dir, "ppl.json")))
    return img_types, objects, places, ppls


def save_results(args,generated_captions):
    if args.in_context:
        name = args.dataset + "_" + args.template_type +"_" + args.language +"_" + str(args.k) + "_" + str(args.n_context) + "_"+ args.split
    else:
        name = args.dataset + "_" + args.template_type +"_" + args.language +"_" + str(args.k) + "_" +args.split
    name += ".beam_" + str(args.beam_size)
    
    outputs_dir = os.path.join(args.output_path, "outputs")
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    # Save results file with top caption for each image :
    # (JSON file of image -> caption output by the model)
    results = []
    for coco_id, top_k_captions in generated_captions.items():
        caption = top_k_captions[0]
        results.append({"image_id": int(coco_id), "caption": caption})
    
    results_output_file_name = os.path.join(outputs_dir, name + ".json")
    json.dump(results, open(results_output_file_name, "w"), ensure_ascii=False)

    # Save results file with all generated captions for each image:
    # JSON file of image -> top-k captions output by the model. Used for reranking after
    results = []
    for coco_id, top_k_captions in generated_captions.items():
        captions = top_k_captions 
        results.append({"image_id": int(coco_id), "captions": captions})

    results_output_file_name = os.path.join(outputs_dir, name + ".top_%d" % args.eval_beam_size + ".json")
    json.dump(results, open(results_output_file_name, "w"),ensure_ascii=False)