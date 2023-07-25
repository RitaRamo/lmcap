import faiss
import numpy as np
import os
import gc
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IndexFlat():
    def __init__(self, dim_examples, train_dataloader_images, device, index_name, clip_model=None):

        self.device=device
        self.index_name = index_name
        self.clip_model=clip_model
        self.dim_examples=dim_examples
        self.train_dataloader_images=train_dataloader_images


    def create(self):
        self.datastore = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim_examples)) 

        print("\nadding input examples to index/datastore")
        for i, (inputs, imgs_indexes) in enumerate(self.train_dataloader_images):

            text_features = self.clip_model.encode_text(inputs.squeeze(1).to(device))        
            caption_embedding = text_features / text_features.norm(dim=-1, keepdim=True)

            self.datastore.add_with_ids(
                caption_embedding.detach().cpu().numpy().astype(np.float32), 
                np.array(imgs_indexes.view(-1), dtype=np.int64)
            )

            if i%100==0:
                print("i and img index of ImageRetrival",i, imgs_indexes)
                print("n of examples", self.datastore.ntotal)

        print("n of examples", self.datastore.ntotal)
        faiss.write_index(self.datastore, self.index_name)

    def load(self):
        print("loading")
        self.datastore = faiss.read_index(self.index_name)

    def retrieve(self, query_img, k=20):
        D, I = self.datastore.search(query_img, k)
        return D, I[:,:k]



class IndexIVFFlat():

    def __init__(self, dim_examples, train_dataloader_images,device, index_name, clip_model=None, nlist = 10000, nprobe=1000, max_to_fit_in_memory=2500000):
        
        self.device=device
        self.index_name = index_name
        self.train_dataloader_images=train_dataloader_images
        self.clip_model=clip_model
        self.dim_examples=dim_examples
        self.nlist=nlist

        self.train_dataloader_images=train_dataloader_images
        self.nprobe=nprobe
        self.max_to_fit_in_memory=max_to_fit_in_memory
        

    def create(self):
        print("starting training")

        quantizer = faiss.IndexFlatIP(self.dim_examples)
        self.datastore = faiss.IndexIVFFlat(quantizer, self.dim_examples, self.nlist, faiss.METRIC_INNER_PRODUCT)     
        self.datastore.nprobe=self.nprobe

        start_training=True
        captions_in_memory=np.ones((self.max_to_fit_in_memory,self.dim_examples), dtype=np.float32)
        captions_ids_in_memory=np.ones((self.max_to_fit_in_memory), dtype=np.int64)
        is_to_add = False
        added_so_far=0
        i=0
        for (inputs, captions_ids)  in self.train_dataloader_images:
            i+=1
            text_features = self.clip_model.encode_text(inputs.squeeze(1).to("cuda"))        
            captions_embedding = text_features / text_features.norm(dim=-1, keepdim=True)
            
            if start_training:
                print("added_so_far",added_so_far)

                batch_size=len(captions_ids)
                captions_in_memory[added_so_far:(added_so_far+batch_size),:] = captions_embedding.detach().cpu().numpy()   
                captions_ids_in_memory[added_so_far:(added_so_far+batch_size)]=np.array(captions_ids.view(-1), dtype=np.int64)
                added_so_far+=batch_size

                if added_so_far>=self.max_to_fit_in_memory:
                    print("training")
                    self.datastore.train(captions_in_memory)
                    self.datastore.add_with_ids(captions_in_memory, captions_ids_in_memory)
                    start_training = False
                    
                    print("saving")
                    faiss.write_index(self.datastore,self.index_name)
                    captions_in_memory=[]
                    captions_ids_in_memory=[]
                    gc.collect()
            
            else:
                print("added_so_far after",added_so_far)
                captions = captions_embedding.detach().cpu().numpy().astype(np.float32)
                captions_ids= np.array(captions_ids.view(-1), dtype=np.int64)
                self.datastore.add_with_ids(captions, captions_ids)
                added_so_far+=batch_size

        faiss.write_index(self.datastore, self.index_name)
        print("n of examples", self.datastore.ntotal)
    
    def load(self):
        print("loading large index")
        self.datastore = faiss.read_index(self.index_name)


    def retrieve(self, query_img, k=20):
        D, I = self.datastore.search(query_img, k)
        return D, I[:,:k]
