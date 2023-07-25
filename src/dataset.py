import json
import torch
from torch.utils.data import Dataset
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from utils import (
    get_socratic_categories,
    get_split
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class GeneralEvalDataset():

    LANGUAGE_PROMPT ={
            "en": "A creative short caption I can generate to describe this image is: ",
            "pt": "A creative short caption I can generate to describe this image in portuguese is: ",
            "zh": "A creative short caption I can generate to describe this image in chinese is: ",
            "de": "A creative short caption I can generate to describe this image in german is: ",
            "es": "A creative short caption I can generate to describe this image in spanish is: ",
            "hi": "A creative short caption I can generate to describe this image in hindi is: ",
            "cs": "A creative short caption I can generate to describe this image in czech is: ",
            "ar": "A creative short caption I can generate to describe this image in arabic is: ",
            "bn": "A creative short caption I can generate to describe this image in bengali is: ",
            "da": "A creative short caption I can generate to describe this image in danish is: ",
            "el": "A creative short caption I can generate to describe this image in greek is: ",
            "fa": "A creative short caption I can generate to describe this image in persian is: ",
            "fi": "A creative short caption I can generate to describe this image in finnish is: ",
            "fil": "A creative short caption I can generate to describe this image in filipino is: ",
            "fr": "A creative short caption I can generate to describe this image in french is: ",
            "he": "A creative short caption I can generate to describe this image in hebrew is: ",
            "hr": "A creative short caption I can generate to describe this image in croatian is: ",
            "hu": "A creative short caption I can generate to describe this image in hungarian is: ",
            "id": "A creative short caption I can generate to describe this image in indonesian is: ",
            "it": "A creative short caption I can generate to describe this image in italian is: ",
            "ja": "A creative short caption I can generate to describe this image in japanese is: ",
            "ko": "A creative short caption I can generate to describe this image in korean is: ",
            "mi": "A creative short caption I can generate to describe this image in maori is: ",
            "nl": "A creative short caption I can generate to describe this image in dutch is: ",
            "no": "A creative short caption I can generate to describe this image in norwegian is: ",
            "pl": "A creative short caption I can generate to describe this image in polish is: ",
            "ro": "A creative short caption I can generate to describe this image in romanian is: ",
            "ru": "A creative short caption I can generate to describe this image in russian is: ",
            "sv": "A creative short caption I can generate to describe this image in swedish is: ",
            "sw": "A creative short caption I can generate to describe this image in swahili is: ",
            "te": "A creative short caption I can generate to describe this image in telugu is: ",
            "th": "A creative short caption I can generate to describe this image in thai is: ",
            "tr":  "A creative short caption I can generate to describe this image in turkish is: ",
            "uk":  "A creative short caption I can generate to describe this image in ukrainian is: ",
            "vi":  "A creative short caption I can generate to describe this image in vietnamese is: "
    }

    COCO_SUPPORT_EXAMPLES=["539984","318556","3967","487050", "235302"]

    def __init__(self, dataset_splits_dir, tokenizer, args=None):
        super().__init__()

        self.language_prompt= self.LANGUAGE_PROMPT
        self.split = get_split(args)
        self.tokenizer = tokenizer
        self.language= args.language
        self.k=args.k
        self.in_context=args.in_context
        self.prefix=""
        
        if self.in_context:
            self.support_refence_text = json.load(open(args.support_reference_caps))
            self.translation_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
            self.translation_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
            self.n_context=args.n_context

            if args.ids:
                self.support_examples_ids=args.ids.split(",")
                print("self.support_examples_ids",self.support_examples_ids)
            else:
                self.support_examples_ids = self.COCO_SUPPORT_EXAMPLES


    def translate_caption(self,english_cap, target_language):
        if target_language in ["mi", "sw", "te"]:

            self.special_lg_examples= {
                "mi":{
                    "a brown horse is grazing grass near a red house": "he hoiho parauri kei te whangai tarutaru i te taha o tetahi whare whero",
                    "a very clean and well decorated empty bathroom": "he kaukau kau tino ma me te whakapaipai pai",
                    "a woman leaning over looking into her phone as she gets ready to take a picture": "he wahine e okioki ana ki te titiro ki tana waea i a ia e reri ana ki te tango pikitia",
                    },
                "sw":
                    {"a brown horse is grazing grass near a red house": "farasi wa kahawia anachunga nyasi karibu na nyumba nyekundu",
                    "a very clean and well decorated empty bathroom":"bafuni safi sana na iliyopambwa vizuri tupu",
                    "a woman leaning over looking into her phone as she gets ready to take a picture": "mwanamke aliyeinama akitazama kwenye simu yake huku akijiandaa kupiga picha",
                    },
                "te":
                    {"a brown horse is grazing grass near a red house": "ఆ ఎర్రటి ఇంటి దగ్గర ఆ గోధుమరంగు గుర్రం గడ్డి మేస్తోంది",
                    "a very clean and well decorated empty bathroom": "చాలా శుభ్రంగా మరియు చక్కగా అలంకరించబడిన ఖాళీ బాత్రూమ్",
                    "a woman leaning over looking into her phone as she gets ready to take a picture": "ఒక స్త్రీ ఫోటో తీయడానికి సిద్ధమవుతున్నప్పుడు ఆమె ఫోన్‌లోకి వంగి చూస్తోంది",
                    },
            }

            return self.special_lg_examples[target_language][english_cap]
    
        self.translation_tokenizer.src_lang = "en"
        encoded_en = self.translation_tokenizer(english_cap, return_tensors="pt")
        generated_tokens = self.translation_model.generate(**encoded_en, forced_bos_token_id=self.translation_tokenizer.get_lang_id(target_language))
        translated_cap=self.translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translated_cap

    def __getitem__(self, i):
        # Convert index depending on the dataset split
        coco_id = self.split[i]
        return coco_id

    def __len__(self):
        return len(self.split)





class SocraticDataset(GeneralEvalDataset):
    
    SOCRATIC_PROMPT = 'I am an intelligent image captioning bot. This image is a {img_type}. There {ppl_result}. I think this photo was taken at a {sorted_places_0}, {sorted_places_1}, or {sorted_places_2}. I think there might be a {object_list} in this {img_type}.'

    def __init__(self, dataset_splits_dir, tokenizer, args=None):
        super().__init__(dataset_splits_dir, tokenizer, args)

        self.prompt = self.SOCRATIC_PROMPT
        self.img_types, self.objects, self.places, self.ppls = get_socratic_categories(args.categories_dir)

        if self.in_context:
            self.support_img_types, self.support_objects, self.support_places, self.support_ppls = get_socratic_categories(args.support_categories_dir)

            for current_example in range(self.n_context):
                train_coco_id=self.support_examples_ids[current_example] 
                grouth_truth = self.support_refence_text[train_coco_id][0]

                img_type=self.support_img_types[train_coco_id]
                ppl_result= self.support_ppls[train_coco_id]
                sorted_places =self.support_places[train_coco_id]
                object_list=self.support_objects[train_coco_id]

                prompt=self.prompt.format(img_type=img_type, ppl_result=ppl_result, sorted_places_0=sorted_places[0],sorted_places_1=sorted_places[1],sorted_places_2=sorted_places[2],object_list=object_list)
                self.prefix += prompt + " " + self.language_prompt[self.language] + self.translate_caption(grouth_truth, self.language) + self.tokenizer.eos_token

    def __getitem__(self, i):
        coco_id= super().__getitem__(i)

        img_type=self.img_types[coco_id]
        ppl_result= self.ppls[coco_id]
        sorted_places =self.places[coco_id]
        object_list=self.objects[coco_id]

        prompt=self.prompt.format(img_type=img_type, ppl_result=ppl_result, sorted_places_0=sorted_places[0],sorted_places_1=sorted_places[1],sorted_places_2=sorted_places[2],object_list=object_list)
        prefix = self.prefix + " " + prompt + " " + self.language_prompt[self.language]
        return prefix, coco_id

    def __len__(self):
        return len(self.split)




class RetrievalDataset(GeneralEvalDataset):
    
    RETRIEVAL_PROMPT = '''I am an intelligent image captioning bot. Similar images have the following captions: '''

    def __init__(self, dataset_splits, tokenizer, args=None):
        super().__init__(dataset_splits, tokenizer, args)

        self.prompt= self.RETRIEVAL_PROMPT
        self.retrieved_text = json.load(open(args.retrieve_filename))

        if self.in_context:
            self.support_retrieved_text = json.load(open(args.support_retrieved_caps))
            self.support_refence_text = json.load(open(args.support_reference_caps))

            for current_example in range(self.n_context):
                train_coco_id=self.support_examples_ids[current_example] 
                grouth_truth = self.support_refence_text[train_coco_id][0]

                self.prefix += self.prompt
                for i in range(self.k):
                    self.prefix+="{}" + self.tokenizer.eos_token + " "
                self.prefix += self.language_prompt[self.language] + self.translate_caption(grouth_truth, self.language) + self.tokenizer.eos_token + " "
                captions=self.support_retrieved_text[train_coco_id][:self.k]
                self.prefix= self.prefix.format(*captions)

    def __getitem__(self, i):

        coco_id=super().__getitem__(i)

        prefix = self.prefix + self.prompt
        for i in range(self.k):
            prefix+="{}" + self.tokenizer.eos_token + " "
        prefix += self.language_prompt[self.language] 
        captions=self.retrieved_text[str(coco_id)][:self.k]
        prefix= prefix.format(*captions)
        return prefix, coco_id

    def __len__(self):
        return len(self.split)