"""
Downloads and prepares the dataset used for testing the retrieval model. 
We use out-of-domain dataset developed by ToolRet, which preprocesses many utterances and APIs from other sources.

Mention in the paper: 
We leverage only subsets that focus on APIs and not synthethic function-style data (python-like peusodocodes functions)
"""

from datasets import load_dataset
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import json
import os
from pathlib import Path

dataset_name = "mangopy/ToolRet-Queries"
output_folder = "/home/vitor/Documents/phd/ConstraintAPIBench/data/testing"

subset_names = ["gorilla-huggingface", "gorilla-pytorch", "gorilla-tensor", "metatool", "t-eval-step", "toolalpaca"]      

# initialising variables
dataset_id_map = {}
doc_id_map = {}  
query_id_map = {}

documents = []
test_queries = []
test_labels = []

for subset in tqdm(subset_names):
    ds = load_dataset(dataset_name, subset, split="queries")
    for item in ds:
        utterance = item["query"]
        utterance_id = query_id_map.setdefault(utterance, len(query_id_map) + 1)
        test_queries.append([utterance_id, utterance])

        # iterating over the APIs
        apis = json.loads(item["labels"])
        for api in apis:
            dataset_id = dataset_id_map.setdefault(json.dumps(api), api["id"])
            doc_id = doc_id_map.setdefault(dataset_id, len(doc_id_map) + 1)
            api.pop("id")
            api.pop("relevance")

            test_labels.append([utterance_id, 0, doc_id, 1])
            if doc_id == len(doc_id_map): 
                documents.append([doc_id, json.dumps(api)])
            
print(f"Number of APIs considered: {len(doc_id_map)}")
print(f"Number of test utterances: {len(test_queries)}")

# generating dataframes
documents_df = pd.DataFrame(documents, columns=['docid', 'document_context'])
test_queries_df = pd.DataFrame(test_queries, columns=['qid', 'query_text'])
test_labels_df = pd.DataFrame(test_labels, columns=['qid', 'useless', 'docid', 'label'])

# creating output directories
os.makedirs(output_folder, exist_ok=True)
documents_df.to_csv(Path(output_folder, 'corpus.tsv'), sep='\t', index=False)
test_queries_df.to_csv(Path(output_folder, 'test.query.txt'), sep='\t', index=False, header=False)
test_labels_df.to_csv(Path(output_folder, 'qrels.test.tsv'), sep='\t', index=False, header=False)