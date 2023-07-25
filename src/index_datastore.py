
from retrieval.dataset import (
    RetrievalDataset
)
from retrieval.indexes import IndexFlat,IndexIVFFlat
import json
import torch
import clip
import sys
import argparse
import os
from collections import defaultdict
import open_clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
 
    with open(args.annotation_filename) as json_file:
        ann = json.load(json_file)  

    id_cap={}
    id_imgids={}
    cap_index=0

    #use training data (train annotations) for the datastore
    for row in ann["annotations"]:
        id_cap[cap_index]=row["caption"].lower().replace(".", "").strip()
        id_imgids[cap_index]=row["image_id"]
        cap_index+=1

   
    if not os.path.exists(args.datastore_path):
        os.makedirs(args.datastore_path)

    datastore_name =args.datastore_path.split("/")[-1]

    with open(args.datastore_path + "/ids_to_caps_"+datastore_name + ".json", 'w+') as f:
        json.dump(id_cap, f, indent=2, ensure_ascii=False)

    with open(args.datastore_path + "/ids_to_imgsids_"+datastore_name+ ".json", 'w+') as f:
        json.dump(id_imgids, f, indent=2, ensure_ascii=False)


    clip_model, _, feature_extractor = open_clip.create_model_and_transforms("xlm-roberta-large-ViT-H-14", pretrained='frozen_laion5b_s13b_b90k', device=device)
    tokenizer= open_clip.get_tokenizer('xlm-roberta-large-ViT-H-14')
    clip_model.eval()
    encoder_output_dim=1024
    
    dataset= RetrievalDataset(feature_extractor=tokenizer, data=id_cap)
    data_loader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    
    index_name =args.datastore_path + "/index_"+ datastore_name
    retrieval_kargs= (encoder_output_dim, data_loader, device, index_name, clip_model)

    if args.large_data: 
        # if data is large use a train index
        image_retrieval = IndexIVFFlat(
            *retrieval_kargs
        )  

    else:
        image_retrieval = IndexFlat(
            *retrieval_kargs
        )  

    image_retrieval.create()

def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--datastore_path", help="Folder where the retrieval index is going to be located")
    parser.add_argument("--annotation_filename", help="Filename containing the annotations")
    parser.add_argument("--large_data", default=False, action="store_true", help="To be activated in case the dataset is too big (e.g., Conceptual Captions)")    
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parsed_args = parser.parse_args(args)
    return parsed_args

if __name__ == "__main__":
    args = check_args(sys.argv[1:])
    main(args)
















