from nltk.tree import Tree
from allennlp.common.file_utils import cached_path
import pandas as pd
instances = []
file_path =  "https://allennlp.s3.amazonaws.com/datasets/sst/train.txt"
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
df.to_json("../data/sst2/train.json", orient = "records", lines=True)