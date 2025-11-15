"""
Reads the configuration file and makes the dataset ready for training the retriever model.
"""
import os
import json
import yaml
import sys
import pandas as pd
from sklearn.utils import shuffle
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm

def load_config(path: Path) -> dict:
    """Loads configuration to be used in the generation method."""
    if not path.exists():
        sys.exit(1)
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    required = ["dataset_path", "output_folder", "llm_name", "prompt_design"]
    for key in required:
        if key not in cfg:
            sys.exit(1)
    return cfg

def main():
    # 1 - loading config information
    cfg = load_config(Path(__file__).parent.parent / "config" / "config_retriever_dataset_preprocess.yaml")
    llm_name = cfg["llm_name"]
    prompt_design = cfg["prompt_design"]
    dataset_path = Path(cfg["dataset_path"], llm_name, prompt_design, "utterances")
    output_folder = Path(cfg["output_folder"], llm_name, prompt_design)

    # 2 - initializing variables
    doc_id_map = {}  # Create a mapping from doc to doc_id
    query_id_map = {}  # Create a mapping from query to query_id    
    
    documents = []
    train_pairs = []
    number_of_apis = 0

    # 2 - iterating through all OAS files
    for root, _, files in os.walk(dataset_path):
        for filename in files:
            file_path = os.path.join(root, filename)  # path to the API spec file 
            with open(file_path, 'r') as f:
                data = json.load(f)
            api_name = data.get('name') if data.get('name') else data.get('tool_name', '')
            api_description = data.get('description') if data.get('description') else data.get('tool_description', '')
            print(f"Processing API: {api_name}")

            # organizing apis IDs
            api_methods = data.get('api_methods') if data.get('api_methods') else data.get('api_list', [])
            for api_method in api_methods:
                api_method_name = api_method.get('name', '')
                api_method_description = api_method.get('description', '')
                api_method_parameters = api_method.get('parameters', [])

                document_content = {"api_name": api_name,
                                    "api_description": api_description,
                                    "api_method_name": api_method_name,
                                    "api_method_description": api_method_description,
                                    "api_method_parameters": api_method_parameters}

                doc_id = doc_id_map.setdefault(json.dumps(document_content), len(doc_id_map) + 1)
                documents.append([doc_id, document_content])

                # organizing queries and training pairs
                if isinstance(api_method['utterances'], list):
                    for utterance in api_method.get('utterances', []):
                        utterance_content = utterance.get('utterance', '')
                        utterance_id = query_id_map.setdefault(utterance_content, len(query_id_map) + 1)
                        train_pairs.append(([utterance_id, utterance_content], [utterance_id, 0, doc_id, 1]))

            number_of_apis += 1

    print(f"Total APIs processed: {number_of_apis}")
    train_pairs = shuffle(train_pairs, random_state=42)
    train_queries, train_labels = zip(*train_pairs)

    # generating dataframes
    documents_df = pd.DataFrame(documents, columns=['docid', 'document_context'])
    train_queries_df = pd.DataFrame(train_queries, columns=['qid', 'query_text'])
    train_labels_df = pd.DataFrame(train_labels, columns=['qid', 'useless', 'docid', 'label'])

    # creating output directories
    os.makedirs(output_folder, exist_ok=True)
    documents_df.to_csv(Path(output_folder, 'corpus.tsv'), sep='\t', index=False)
    train_queries_df.to_csv(Path(output_folder, 'train.query.txt'), sep='\t', index=False, header=False)
    train_labels_df.to_csv(Path(output_folder, 'qrels.train.tsv'), sep='\t', index=False, header=False)

if __name__ == '__main__':
    main()
