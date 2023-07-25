import json
import torch
import sys
import argparse
import h5py
import os
from PIL import Image
import numpy as np
import clip
import requests
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch.nn as nn
import open_clip
from utils import get_default_jsons
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class ClipOpenAIFeaturesDataset():

    def __init__(self, args, clip_processor):
        super().__init__()

        self.images_names, self.split = get_default_jsons(args)   
        self.clip_processor = clip_processor

        self.images_dir = args.images_dir


    def __getitem__(self, i):
        coco_id = self.split[i]
        image_name=self.images_names[coco_id]
        image_filename= self.images_dir+self.images_names[coco_id]
        img_open = Image.open(image_filename).copy()
        img = np.array(img_open)
        if len(img.shape) ==2 or img.shape[-1]!=3: #convert grey or CMYK to RGB
            img_open = img_open.convert('RGB')
        inputs_features=self.clip_processor(img_open)

        return coco_id, inputs_features

    def __len__(self):
        return len(self.split)

def main(args):
    output_name = args.image_features_path
    clip_encoder, _, clip_processor = open_clip.create_model_and_transforms("xlm-roberta-large-ViT-H-14", pretrained='frozen_laion5b_s13b_b90k',device=device)
    encoder_dim=1024

    clip_encoder.eval()
    with torch.no_grad():
        with h5py.File(args.image_features_path, 'w') as h5py_file:
            #features_h5 = h.create_dataset('features', (123 287, 50, 768))   

            for split in ["train", "val", "test"]:
                args.split=split
                print("currently in split:", split)
                data_loader = torch.utils.data.DataLoader(
                            ClipOpenAIFeaturesDataset(args, clip_processor),
                            batch_size=1, shuffle=False, num_workers=1, pin_memory=True
                )

                for i, (coco_id,inputs_features) in enumerate(data_loader):
                    if i%1000==0:
                        print("i",i)
                    inputs_features=inputs_features.to(device)
                    clip_features=clip_encoder.encode_image(inputs_features)
                    h5py_file.create_dataset(coco_id[0], (1, encoder_dim), data=clip_features.cpu().numpy())


def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", help="Folder where the images are located")
    parser.add_argument("--images_names", help="Json that maps the imgs_ids with their names(ids2names)")
    parser.add_argument("--dataset_splits", help="Json containing the dataset splits")
    parser.add_argument("--image_features_path", help="Folder where the preprocessed image will be located")
    parsed_args = parser.parse_args(args)
    return parsed_args

if __name__ == "__main__":
    args = check_args(sys.argv[1:])
    main(args)