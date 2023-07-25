import json
import torch
import sys
import argparse
import os
from tqdm import tqdm
import open_clip
import numpy as np
from profanity_filter import ProfanityFilter
import h5py
from extract_features import ClipOpenAIFeaturesDataset
from utils import (
    TRAIN_SPLIT,
    VALID_SPLIT,
    TEST_SPLIT,
    DATASET_SPLITS_FILENAME,
    CLIP_DIM
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PPL_TEXTS= ['no people', 'people']
PPL_SEVERAL_TEXTS=['is one person', 'are two people', 'are three people', 'are several people', 'are many people']
IMG_TYPES=['photo', 'cartoon', 'sketch', 'painting']

def get_text_feats(model, tokenizer, in_text, batch_size=64):
    text_tokens = tokenizer(in_text).to(device)
    text_id = 0
    text_feats = np.zeros((len(in_text), CLIP_DIM), dtype=np.float32)
    while text_id < len(text_tokens):  # Batched inference.
        batch_size = min(len(in_text) - text_id, batch_size)
        text_batch = text_tokens[text_id:text_id+batch_size]
        with torch.no_grad():
            batch_feats = model.encode_text(text_batch).float()
        batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
        batch_feats = np.float32(batch_feats.cpu())
        text_feats[text_id:text_id+batch_size, :] = batch_feats
        text_id += batch_size
    return text_feats

def get_categories_feats(model,tokenizer):
    print("getting places features")
    place_categories = np.loadtxt('categories/categories_places365.txt', dtype=str)
    place_texts = []
    for place in place_categories[:, 0]:
        place = place.split('/')[2:]
        if len(place) > 1:
            place = place[1] + ' ' + place[0]
        else:
            place = place[0]
            place = place.replace('_', ' ')
            place_texts.append(place)
    place_feats = get_text_feats(model, tokenizer, [f'Photo of a {p}.' for p in place_texts])
    

    print("getting object features")
    with open('categories/dictionary_and_semantic_hierarchy.txt') as fid:
        object_categories = fid.readlines()
    object_texts = []
    pf = ProfanityFilter()
    for object_text in object_categories[1:]:
        object_text = object_text.strip()
        object_text = object_text.split('\t')[3]
        safe_list = ''
        for variant in object_text.split(','):
            text = variant.strip()
            if pf.is_clean(text):
                safe_list += f'{text}, '
        safe_list = safe_list[:-2]
        if len(safe_list) > 0:
            object_texts.append(safe_list)
    object_texts = [o for o in list(set(object_texts)) if o not in place_texts]  # Remove redundant categories.
    object_feats = get_text_feats(model, tokenizer,[f'Photo of a {o}.' for o in object_texts])


    print("getting features for image_types")
    img_types_feats = get_text_feats(model, tokenizer,[f'This is a {t}.' for t in IMG_TYPES])

    print("getting features for ppl_texts")
    ppl_feats = get_text_feats(model, tokenizer,[f'There are {p} in this photo.' for p in PPL_TEXTS])

    return place_texts, object_texts,place_feats,object_feats,img_types_feats, ppl_feats


def get_img_feats(imgs_features, coco_id):
    img_feats = imgs_features[coco_id][()]
    return img_feats

def get_nn_text(raw_texts, text_feats, img_feats):
    scores = text_feats @ img_feats.T
    scores = scores.squeeze()
    high_to_low_ids = np.argsort(scores).squeeze()[::-1]
    high_to_low_texts = [raw_texts[i] for i in high_to_low_ids]
    high_to_low_scores = np.sort(scores).squeeze()[::-1]
    return high_to_low_texts, high_to_low_scores

def main(args):

    model, _, preprocess = open_clip.create_model_and_transforms("xlm-roberta-large-ViT-H-14", pretrained='frozen_laion5b_s13b_b90k',device=device)
    tokenizer= open_clip.get_tokenizer('xlm-roberta-large-ViT-H-14')

    model.cuda().eval()

    with open(args.dataset_splits) as f:
        split = json.load(f)

    img_ids=split[VALID_SPLIT]
    img_ids.extend(split[TEST_SPLIT])
    img_ids.extend(split[TRAIN_SPLIT])

    imgs_features = h5py.File(args.image_features_filename, "r")

    place_texts, object_texts, place_feats, object_feats, img_types_feats,ppl_feats = get_categories_feats(model,tokenizer)

    all_img_types={}
    all_ppl_result={}
    all_sorted_places={}
    all_object_list={}

    j=0
    for img_id in tqdm(img_ids):

        # Load image.
        img_feats = get_img_feats(imgs_features, img_id)
        

        # Zero-shot VLM: classify image type.
        sorted_img_types, img_type_scores = get_nn_text(IMG_TYPES, img_types_feats, img_feats)
        img_type = sorted_img_types[0]

        all_img_types[img_id]=img_type

        # Zero-shot VLM: classify number of people.
        sorted_ppl_texts, ppl_scores = get_nn_text(PPL_TEXTS, ppl_feats, img_feats)
        ppl_result = sorted_ppl_texts[0]
        
        if ppl_result == 'people':
            ppl_texts_sev = PPL_SEVERAL_TEXTS
            ppl_feats_sev = get_text_feats(model, tokenizer, [f'There {p} in this photo.' for p in ppl_texts_sev])
            sorted_ppl_texts, ppl_scores = get_nn_text(ppl_texts_sev, ppl_feats_sev, img_feats)
            ppl_result = sorted_ppl_texts[0]
        else:
            ppl_result = f'are {ppl_result}'       
        all_ppl_result[img_id]=ppl_result

        
        # Zero-shot VLM: classify places.
        sorted_places, places_scores = get_nn_text(place_texts, place_feats, img_feats)
        all_sorted_places[img_id]=sorted_places


        # Zero-shot VLM: classify objects.
        obj_topk = 10
        sorted_obj_texts, obj_scores = get_nn_text(object_texts, object_feats, img_feats)
        object_list = ''
        for i in range(obj_topk):
            object_list += f'{sorted_obj_texts[i]}, '
        object_list = object_list[:-2]

        all_object_list[img_id]=object_list

    outputs_dir =args.outputs_dir
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    with open(os.path.join(outputs_dir,"img_type.json"), 'w+') as f:
        json.dump(all_img_types, f, indent=2)

    with open(os.path.join(outputs_dir,"ppl.json"), 'w+') as f:
        json.dump(all_ppl_result, f, indent=2)

    with open(os.path.join(outputs_dir,"places.json"), 'w+') as f:
        json.dump(all_sorted_places, f, indent=2)

    with open(os.path.join(outputs_dir,"objects.json"), 'w+') as f:
        json.dump(all_object_list, f, indent=2)

def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_features_filename",
                        help="Folder where the preprocessed image data is located")
    parser.add_argument("--dataset_splits", help="Json containing the dataset splits")
    parser.add_argument("--outputs_dir", help="Folder where the categories should be located")

    parsed_args = parser.parse_args(args)
    return parsed_args

if __name__ == "__main__":
    args = check_args(sys.argv[1:])
    main(args)