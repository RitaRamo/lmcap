import os
import json
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import h5py
import torch

from utils import (
    VALID_SPLIT,
    TEST_SPLIT
)


class RetrievalDataset():
    def __init__(self, feature_extractor, data):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.captions_text = data
    
    def __getitem__(self, i):
        coco_id = i
        caption = self.captions_text[coco_id] 
        inputs = self.feature_extractor(caption)
        return inputs, np.array(coco_id, dtype=np.int64)

    def __len__(self):
        return len(self.captions_text)


class RetrievalQueryDataset():
    def __init__(self, dataset_splits, image_features):
        super().__init__()
        
        self.split = json.load(open(dataset_splits))
        inference_splits = self.split[VALID_SPLIT]
        inference_splits.extend(self.split[TEST_SPLIT])
        self.split= inference_splits

        self.image_features = h5py.File(image_features, "r")
    
    def __getitem__(self, i):
        coco_id = self.split[i]

        image_data = self.image_features[coco_id][()]
        image_features = torch.FloatTensor(image_data)
        image_features /= image_features.norm(dim=-1, keepdim=True)
     
        return image_features.detach(), coco_id

    def __len__(self):
        return len(self.split)