import sys
import logging
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from transformers import XGLMForCausalLM, XGLMTokenizer
import time
from dataset import SocraticDataset,RetrievalDataset
from utils import save_results
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True 
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)

def evaluate(args):

    model = XGLMForCausalLM.from_pretrained("facebook/xglm-"+ args.xglm_type,torch_dtype=torch.float16) #, use_cache=True
    model = XGLMForCausalLM.from_pretrained("facebook/xglm-"+ args.xglm_type) #, use_cache=True

    text_tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-"+args.xglm_type)
    model = model.to(device)
    model.text_tokenizer = text_tokenizer
    model.eval()
    logging.info("Model params: {}".format(vars(model)))

    dataset_name = RetrievalDataset if args.template_type=="retrieval" else SocraticDataset
    data_loader = torch.utils.data.DataLoader(
                    dataset_name(args.dataset_splits, text_tokenizer, args),
                    batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True
    )   

    start = time.time()
    beam_size=args.beam_size
    args.generation_kwargs = {
        'max_new_tokens': args.max_caption_len, 
        'no_repeat_ngram_size': 0, 
        'length_penalty': args.length_penalty,
        'num_beams': 3, 
        'early_stopping': True, 
        'eos_token_id': text_tokenizer.eos_token_id,
        'num_return_sequences':beam_size,
        'min_length':args.min_caption_len,
        'diversity_penalty':args.diversity_penalty,
        'num_beam_groups':args.num_beam_groups
    }
    generated_captions = {}

    for data in tqdm(data_loader, desc="Evaluate with beam size " + str(beam_size)):
        input_with_prefix, cocos_id =data
        dec_input_with_prefix = text_tokenizer(input_with_prefix, add_special_tokens=False, return_tensors='pt', padding=True).to(device)

        output_sequences = model.generate(
            input_ids=dec_input_with_prefix.input_ids, 
            **args.generation_kwargs
        )

        len_prefix=dec_input_with_prefix.input_ids.size()[1]
        output_sequences=output_sequences[:,len_prefix:]  #remove prefix from the generated caps
        caps=text_tokenizer.batch_decode(output_sequences, skip_special_tokens=True)    
        generated_captions.update({cocos_id[i]: caps[beam_size*i: beam_size*i+beam_size] for i in range(args.batch_size)})   

        logging.info("\n prefix {}".format(input_with_prefix))
        logging.info("\n caps {}".format(caps))

    end = time.time()
    logging.info("FINAL cuda memory {}".format(torch.cuda.max_memory_allocated() / 1e9))
    
    save_results(args, generated_captions)



def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset name")
    parser.add_argument("--dataset_splits", help="Json file containing the splits")
    parser.add_argument("--retrieve_filename", help="Json file containing the retrieved captions")                   
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--output_path", help="Folder where to store outputs")
    parser.add_argument("--xglm_type", type=str, default="564M", choices=["564M", "1.7B","2.9B","4.5B","7.5B"])
    parser.add_argument("--language",help="Choose the language to evaluate on", default="en")
    parser.add_argument("--template_type",help="Choose either retrieval prompt or socratic baseline", type=str, default="retrieval", choices=["socratic","retrieval"])
    parser.add_argument("--k", type=int, default=5,help="Number of retrieved caps")
    parser.add_argument("--batch_size", type=int, default=5, help="Size of the batch")
    #generation settings
    parser.add_argument("--max_caption_len", type=int, default=30)
    parser.add_argument("--min_caption_len", type=int, default=0)
    parser.add_argument("--length_penalty", type=float, default=0.0)
    parser.add_argument("--diversity_penalty", type=float, default=0)
    parser.add_argument("--num_beam_groups", type=int, default=1)
    parser.add_argument("--beam_size", type=int, default=3, help="Size of the decoding beam")
    parser.add_argument("--eval_beam_size", type=int, default=3, help="Number of sequences from the beam that should be used for evaluation")    
    #if using in-context with support examples
    parser.add_argument("--in_context", default=False, action="store_true", help="")
    parser.add_argument("--n_context", type=int, default=1, help="Number of context examples")
    parser.add_argument("--ids", type=str,help="Training ids to consider for the in-context learning' support examples" )
    parser.add_argument("--support_reference_caps", type=str, help="File containing the references of the support examples")
    parser.add_argument("--support_retrieved_caps", type=str, help="File containing the retrieved caps of the support examples")
    parser.add_argument("--support_categories_dir", type=str, help="File containing the categories of the support examples")
    #if using socratic template
    parser.add_argument("--categories_dir", help="Json file containing the dir where the categories are stored for the socratic baseline")

    parsed_args = parser.parse_args(args)
    return parsed_args


if __name__ == "__main__":
    args = check_args(sys.argv[1:])
    logging.basicConfig(
            format='%(levelname)s: %(message)s', level=logging.INFO)
    logging.info(args)
    evaluate(args)