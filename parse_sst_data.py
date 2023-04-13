from nltk.tree import Tree
from allennlp.common.file_utils import cached_path
import pandas as pd
import argparse
instances = []

# generate command line argument to define file_path
parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, default="train")
args = parser.parse_args()
file_path =  f"https://allennlp.s3.amazonaws.com/datasets/sst/{args.split}.txt"
with open(cached_path(file_path), "r") as data_file:
    for line in data_file.readlines():
        line = line.strip("\n")
        if not line:
            continue
        parsed_line = Tree.fromstring(line)
        tokens = parsed_line.leaves()
        sentiment = parsed_line.label()
        text = " ".join(tokens)
        if int(sentiment) < 2:
            sentiment = 0
        elif int(sentiment) == 2:
            continue
        else:
            sentiment = 1
        instances.append({"label": sentiment, "text": text})

df = pd.DataFrame(instances)
df.to_json(f"../data/sst2/{args.split}.json", orient = "records", lines=True)