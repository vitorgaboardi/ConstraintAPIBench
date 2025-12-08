"""
Quick evaluation of a pretrained (non-finetuned) retriever model using APIEvaluator.
"""
import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from api_evaluator import APIEvaluator
from pathlib import Path

testing_path = "/home/vitor/Documents/phd/ConstraintAPIBench/data/testing"

corpus_df = pd.read_csv(Path(testing_path, 'corpus.tsv'), sep='\t')
queries_df = pd.read_csv(Path(testing_path, 'test.query.txt'), sep='\t', names=['qid', 'query'])
labels_df = pd.read_csv(Path(testing_path, 'qrels.test.tsv'), sep='\t', names=['qid', 'useless', 'docid', 'label'])

corpus = {}
for row in corpus_df.itertuples():
    doc_text = (
        json.dumps(row.document_context, ensure_ascii=False)
        if isinstance(row.document_context, dict)
        else str(row.document_context)
    )
    corpus[str(row.docid)] = doc_text

queries = {str(row.qid): row.query for row in queries_df.itertuples()}

relevant_docs = {}
for row in labels_df.itertuples():
    qid, docid = str(row.qid), str(row.docid)
    relevant_docs.setdefault(qid, set()).add(docid)

print(f"Loaded {len(queries)} queries and {len(corpus)} documents.")
print(f"Loaded relevance mappings for {len(relevant_docs)} queries.")

# model_name = "sentence-transformers/all-mpnet-base-v2"
# model_name = "ToolBench/ToolBench_IR_bert_based_uncased"
# model_name = "Qwen/Qwen3-Embedding-4B"
model_name = "intfloat/multilingual-e5-base"
model = SentenceTransformer(model_name, trust_remote_code=True)

ir_evaluator = APIEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    corpus_chunk_size=4,  # adjust if memory is low
    batch_size=4,
    show_progress_bar=True,
    write_csv=False
)

print(f"\nEvaluating model: {model_name}")
ndcg_scores = ir_evaluator.compute_metrices(model)

print(f"\nResults for {model_name}:")
print(f"NDCG@1:  {ndcg_scores[0]*100:.2f}")
print(f"NDCG@3:  {ndcg_scores[1]*100:.2f}")
print(f"NDCG@5:  {ndcg_scores[2]*100:.2f}")
print(f"NDCG@10:  {ndcg_scores[3]*100:.2f}")
