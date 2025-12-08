""""
Training script for the retrieval model using specified configurations.
"""

import logging
import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from api_evaluator import APIEvaluator
import yaml
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, models, InputExample, losses, LoggingHandler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import sys
import random
import numpy as np
import gc

def load_config(path: Path) -> dict:
    """Loads configuration to be used in the generation method."""
    if not path.exists():
        print(f"Error: Configuration file not found: {path}")
        sys.exit(1)
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    required = ["training_path", "testing_path", "output_folder", "llm_name", "prompt_design"]
    for key in required:
        if key not in cfg:
            sys.exit(1)
    return cfg


def main():
    # 1 - loading config information
    cfg = load_config(Path(__file__).parent.parent.parent / "config" / "config_retriever_training.yaml")
    llm_name = cfg["llm_name"]
    prompt_design = cfg["prompt_design"]
    training_path = Path(cfg["training_path"], llm_name, prompt_design)
    testing_path = Path(cfg["testing_path"])
    output_folder = Path(cfg["output_folder"], llm_name, prompt_design)

    # 2 - training parameters and summary
    embedding_model = cfg["training_params"]["model_name"]
    num_epochs = cfg["training_params"]["epochs"]
    train_batch_size = cfg["training_params"]["train_batch_size"]
    lr = float(cfg["training_params"]["learning_rate"])
    warmup_steps = cfg["training_params"]["warmup_steps"]
    max_seq_length = cfg["training_params"]["max_seq_length"]

    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO, handlers=[LoggingHandler()])
    logger = logging.getLogger(__name__)
    logs_writer = SummaryWriter(os.path.join(output_folder, 'tensorboard', 'name_desc'))

    def log_callback_st(value, epoch, steps):
        logger.info(f"Callback triggered: Epoch {epoch}, Step {steps}, Evaluator Value: {value}")

    # 4 - training dataset loading
    train_samples = []
    train_queries = {}
    ir_corpus = {}

    print("Loading training dataset...")

    corpus_df = pd.read_csv(Path(training_path, 'corpus.tsv'), sep='\t')
    ir_corpus = {row.docid: json.dumps(row.document_context, ensure_ascii=False)
                 if isinstance(row.document_context, dict) else str(row.document_context)
                 for row in corpus_df.itertuples()}

    queries_df = pd.read_csv(Path(training_path, 'train.query.txt'), sep='\t', names=['qid', 'query'])
    for row in queries_df.itertuples():
        train_queries[row.qid] = row.query

    labels_df = pd.read_csv(Path(training_path, 'qrels.train.tsv'), sep='\t', names=['qid', 'useless', 'docid', 'label'])
    for row in labels_df.itertuples():
        if row.qid in train_queries and row.docid in ir_corpus:
            sample = InputExample(texts=[train_queries[row.qid], ir_corpus[row.docid]],
                                  label=float(row.label))
            train_samples.append(sample)

    logger.info(f"Loaded {len(train_samples)} training samples.")

    # 5 - testing dataset loading
    test_corpus_df = pd.read_csv(Path(testing_path, 'corpus.tsv'), sep='\t')
    test_queries_df = pd.read_csv(Path(testing_path, 'test.query.txt'), sep='\t', names=['qid', 'query'])
    test_labels_df = pd.read_csv(Path(testing_path, 'qrels.test.tsv'), sep='\t', names=['qid', 'useless', 'docid', 'label'])

    test_corpus = {}
    for row in test_corpus_df.itertuples():
        doc_text = json.dumps(row.document_context, ensure_ascii=False) if isinstance(row.document_context, dict) else str(row.document_context)
        test_corpus[str(row.docid)] = doc_text

    test_queries = {str(row.qid): row.query for row in test_queries_df.itertuples()}

    test_relevant_docs = {}
    for row in test_labels_df.itertuples():
        qid, docid = str(row.qid), str(row.docid)
        test_relevant_docs.setdefault(qid, set()).add(docid)

    # 6 - evaluator (shared across runs)
    ir_evaluator = APIEvaluator(test_queries, test_corpus, test_relevant_docs)

    # 7 - training loop with multiple seeds
    model_save_path = os.path.join(output_folder, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(model_save_path, exist_ok=True)

    logging.info(f"Training on {len(train_samples)} samples and evaluating on {len(test_queries)} test queries.")

    evaluation_results = {}
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    seeds = random.sample(range(0, 101), 5)
    
    for i, seed in enumerate(seeds):
        logger.info(f"Starting run {i} with seed {seed}")
        
        # Clear previous model from memory
        if i > 0:
            del model
            torch.cuda.empty_cache()
            gc.collect()

        # Set all seeds for reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model = SentenceTransformer(embedding_model, trust_remote_code=True)
        model.max_seq_length = max_seq_length

        # Create fresh dataloader with new seed for shuffling
        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(model)

        model_save_path_i = model_save_path + f"_run_{i+1}"

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=ir_evaluator,
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            optimizer_params={'lr': lr},
            output_path=model_save_path_i,
            callback=log_callback_st
        )

        ndcg_scores = ir_evaluator.compute_metrices(model)
        
        evaluation_results[f'run_{i+1}'] = {
            'seed': seed,
            'NDCG@1': ndcg_scores[0],
            'NDCG@3': ndcg_scores[1],
            'NDCG@5': ndcg_scores[2],
            'NDCG@10': ndcg_scores[3],
        }
        logger.info(f"Final Results for run {i+1} (seed={seeds[i]}): NDCG@1: {ndcg_scores[0]*100:.2f}, NDCG@3: {ndcg_scores[1]*100:.2f}, NDCG@5: {ndcg_scores[2]*100:.2f}, NDCG@10: {ndcg_scores[3]*100:.2f}")
    
        results_file = Path(model_save_path, "evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(evaluation_results, f, indent=4)
    
    logger.info(f"All runs completed. Results saved to {results_file}")

if __name__ == '__main__':
    main()
