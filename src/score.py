import os
import sys
import json
import argparse
from metrics import coco_metrics
import re


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    split = args.results_fn.split("/")[-1].split(".")[0]
   
    output_fn = os.path.join(args.output_dir,
                                "coco" + "." + ".".join(args.results_fn.split("/")[-1].split(".")[1:-1]) + "." + split)
    
    metric2score, each_image_score = coco_metrics(args.results_fn, args.annotations_path, args.language)
    with open(output_fn, "w") as f:
        for m, score in metric2score.items():
            f.write("%s: %f\n" % (m, score))

    with open(output_fn+"individual_scores.json", 'w+') as f:
        json.dump(each_image_score, f, indent=2)


def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_fn", help="Path to JSON file of image -> caption output by the model.")
    parser.add_argument("--output_dir", help="Directory where to store the results.")
    parser.add_argument("--annotations_path", help="Path to the annotations in COCO format")
    parser.add_argument("--language",help="Choose the language to evaluate on", default="en")
    parsed_args = parser.parse_args(args)
    return parsed_args 


if __name__ == "__main__":
    args = check_args(sys.argv[1:])
    main(args)