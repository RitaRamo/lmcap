
from retrieval.dataset import RetrievalQueryDataset
from retrieval.clip import IndexFlat,IndexIVFFlat
import json
import torch
import clip
import sys
import argparse
import os
from collections import defaultdict
from utils import CAPTIONS_FILENAME, CLIP_DIM
import open_clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_nearest_caps(img_id,nearest_ids, D, id_cap, id_imgids):
    list_of_similar_caps = []
    a=0
    caps_so_far=[]
    for nn_id in nearest_ids:

        nn_img_id = id_imgids[str(nn_id)]
        if nn_img_id==img_id:
            #to not have captions retrieved from the actual image
            continue

        nearest_cap = id_cap[str(nn_id)]
        if nearest_cap in caps_so_far:
            continue

        caps_so_far.append(nearest_cap)
        distance_cap=D[a]
        list_of_similar_caps.append((nearest_cap,distance_cap))
        a+=1

    return list_of_similar_caps



def main(args):
    
    #get datastore index and datastore jsons (ids of each caption stored and their images)
    datastore_name =args.datastore_dir.split("/")[-1]
    index_name=args.datastore_dir + "/index_"+datastore_name

    with open(args.datastore_dir + "/ids_to_caps_"+datastore_name+".json") as json_file:
        id_cap = json.load(json_file)

    with open(args.datastore_dir + "/ids_to_imgsids_"+datastore_name+".json") as json_file:
        id_imgids = json.load(json_file)  

    data_loader = torch.utils.data.DataLoader(
                        RetrievalQueryDataset(args.dataset_splits,args.image_features_filename),
                        batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    )

    retrieval_kargs= (CLIP_DIM, data_loader, device, index_name)
    if args.large_data:
        image_retrieval = IndexIVFFlat(*retrieval_kargs)  
    else:
        image_retrieval = IndexFlat(*retrieval_kargs)
    print("loading index")
    image_retrieval.load()

    #for each image, get the nearest captions in the datastore and their distances
    img2nearest_caps={}
    img2nearest_distances={}
    for i, (vision_embedding, img_id) in enumerate(data_loader):
        if i%100==0:
            print("i and img index of ImageRetrival", i)
        
        D, nearest_ids=image_retrieval.retrieve(vision_embedding[0].numpy())

        # batch of 1, hence [0]
        img_id = int(img_id[0])
        nearest_ids = nearest_ids[0]
        D = D[0]

        most_sim_caps,sorted_distances=zip(*get_nearest_caps(img_id,nearest_ids, D, id_cap, id_imgids))
        
        img2nearest_caps[str(img_id)] = most_sim_caps                        
        img2nearest_distances[str(img_id)] = [str(distance) for distance in sorted_distances]

    outputs_dir =args.outputs_dir
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    
    with open(os.path.join(outputs_dir,"nn_setup_"+datastore_name +".json"), 'w+') as f:
        json.dump(img2nearest_caps, f, indent=2,ensure_ascii=False)

    with open(os.path.join(outputs_dir,"distances_setup_"+datastore_name+".json"), 'w+') as f:
        json.dump(img2nearest_distances, f, indent=2,ensure_ascii=False)

        

def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", help="Folder where the retrieval index is located")
    parser.add_argument("--datastore_dir", help="Folder where the retrieval index is located")
    parser.add_argument("--image_features_filename", help="Folder where the preprocessed image data is located")
    parser.add_argument("--dataset_splits", help="Json file containing the splits")
    parser.add_argument("--large_data", default=False, action="store_true",
                        help="")    
    parsed_args = parser.parse_args(args)
    return parsed_args

if __name__ == "__main__":
    args = check_args(sys.argv[1:])
    main(args)
