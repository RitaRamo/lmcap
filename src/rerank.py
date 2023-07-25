
import json
import torch
import sys
import argparse
import h5py
import open_clip
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClipMultilingualDataset():

    def __init__(self, feature_extractor, tokenizer, data):
        super().__init__()
        self.tokenizer = tokenizer
        self.captions_text = data
        self.image_features = h5py.File(feature_extractor, "r")

    
    def __getitem__(self, i):
        coco_id = self.captions_text[i]["image_id"]
        image_features = self.image_features[str(coco_id)][()]
        raw_captions = self.captions_text[i]["captions"]
        text_features = self.tokenizer(raw_captions)
        return coco_id, image_features, text_features, raw_captions

    def __len__(self):
        return len(self.captions_text)


def main(args):

    model, _, clip_processor = open_clip.create_model_and_transforms('xlm-roberta-large-ViT-H-14', pretrained='frozen_laion5b_s13b_b90k', device=device)
    tokenizer = open_clip.get_tokenizer('xlm-roberta-large-ViT-H-14')

    caps_path=args.caps_path
    img_caps = json.load(open(caps_path)) 

    data_loader = torch.utils.data.DataLoader(
                ClipMultilingualDataset(args.image_features_filename, tokenizer, img_caps),
                batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    )

    new_captions=[]
    with torch.no_grad():
        for i, (coco_id,image_features, text, raw_captions) in tqdm(enumerate(data_loader)):
            if i%1000==0:
                print("i",i)

            text=text.squeeze(0).to(device)
            text_features = model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            image_features=image_features.to(device)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            best_cap_index= text_probs.argmax().item()
            new_captions.append({"image_id": int(coco_id), "caption": raw_captions[best_cap_index][0]})

    new_path= "/".join(caps_path.split("/")[:-1])+"/rerank_"+caps_path.split("/")[-1]
    json.dump(new_captions, open(new_path, "w"), ensure_ascii=False)


def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_features_filename", help="Folder where the preprocessed image data is located")
    parser.add_argument("--caps_path", help="Name of the of the captions")
    parsed_args = parser.parse_args(args)
    return parsed_args

if __name__ == "__main__":
    args = check_args(sys.argv[1:])
    main(args)
