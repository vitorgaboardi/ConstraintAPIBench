import json
import ast
import copy
import os
import numpy as np
import random
import itertools
import re
from bert_score import score
from typing import Dict, List, Tuple
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from .prompts import NATURALNESS_EVALUATION


def naturalness_evaluation(oas: Dict, api_key: str, base_url: str, model_name: str) -> Dict:
    """Evaluate the naturalness of all utterances related to an API.
    Returns the number of natural and unnatural utterances."""
    # defining LLM client
    openai_client = OpenAI(api_key=api_key, base_url=base_url)

    # defining the amount of natural and unnatural utterances
    natural_count = 0
    unnatural_count = 0
    wrong_count = 0
    results = []

    # iterating through all utterances
    api_name = oas['name']
    print(f"This API method has {len(oas['api_methods'])} methods to evaluate.")
    for api_method_index, api_method in enumerate(oas['api_methods']):
        api_method_name = api_method['name']
        for utterances in api_method.get('utterances', []):
            utterance = utterances["utterance"]

            # defining the prompt
            messages = [
                {"role": "system", "content": NATURALNESS_EVALUATION},
                {"role": "user", "content": f"Evaluate the following utterance for naturalness: '{utterance}'"}]                    

            # calling the LLM
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=500,
                temperature=0)

            # processing the response
            content = response.choices[0].message.content.strip().lower()
            print(f"        - Utterance: '{utterance}' | Evaluation: {content}")
            if content == 'natural':
                natural_count += 1
            elif content == 'unnatural':
                unnatural_count += 1
            else:
                wrong_count += 1
                content = 'invalid response'

            # storing results
            results.append({
                "llm_as_judge": model_name,
                "api": api_name,
                "api_method": api_method_name,
                "utterance": utterance,
                "evaluation": content})
        print()

    return {
        "natural_count": natural_count,
        "unnatural_count": unnatural_count,
        "wrong_count": wrong_count,
        "detailed_results": results}

def bertscore(oas: Dict, embedding_model: str) -> None:
    """BERTScore evaluation method."""
    avg_bertscores = []

    for endpoint in oas['api_methods']:
        # setting reference text of the API method
        reference_text = _api_text_representation(oas, endpoint)

        # defining texts for utterances
        utterances = [utt['utterance'] for utt in endpoint.get('utterances', [])]
        references = [reference_text] * len(utterances)

        # computing BERTScores
        if len(utterances) > 0:
            P, R, F1 = score(utterances, references, model_type=embedding_model, verbose=False)
            avg_f1 = round(F1.mean().item(), 4)
            avg_bertscores.append(avg_f1)

    return round(sum(avg_bertscores) / len(avg_bertscores), 4)

def cosine_similarity(oas: Dict, embedding_model: SentenceTransformer) -> None:
    """Cosine Similarity evaluation method."""
    avg_cosine_scores = []

    for endpoint in oas['api_methods']:
        # setting reference text of the API method
        api_method_text_reference = _api_text_representation(oas, endpoint)
        reference_embedding = embedding_model.encode(api_method_text_reference, convert_to_tensor=True)

        # defining embeddings for utterances
        utterances = [utt['utterance'] for utt in endpoint.get('utterances', [])]
        utterance_embeddings = embedding_model.encode(utterances, convert_to_tensor=True)

        # computing cosine similarities
        cosine_scores = util.cos_sim(reference_embedding, utterance_embeddings)[0]
        avg_score = round(cosine_scores.mean().item(), 4)
        avg_cosine_scores.append(avg_score)

    return round(sum(avg_cosine_scores) / len(avg_cosine_scores), 4)

def parameter_coverage(oas: Dict) -> None:
    """Parameter Coverage evaluation method."""
    # counting total parameters in the API that are not technical
    pc = []
    for endpoint in oas['api_methods']:
        # computing total non-technical parameters
        total_parameters = 0
        for parameter in endpoint.get('parameters', []):
            if parameter.get('constraints', {}) != {}:
                if parameter['constraints'].get('technical', False) == True:
                    continue
            total_parameters += 1

        # computing parameter coverage
        if total_parameters > 0:
            parameters_used = [list(utterances['parameters'].keys()) for utterances in endpoint['utterances']]
            all_parameters_used = list(set([item for sublist in parameters_used for item in sublist])) 
            coverage = round(len(all_parameters_used) / total_parameters, 4)
            pc.append(coverage)
    
    if len(pc) > 0:
        average_pc = round(sum(pc) / len(pc), 4)
        return average_pc
    else:
        return None

def parameter_combination_coverage(oas: Dict) -> None:
    """Placeholder for Parameter Combination Coverage evaluation method."""
    pcc = []
    for endpoint in oas['api_methods']:
        # getting non-technical parameters
        technical_parameters = []
        for parameter in endpoint.get('parameters', []):
            if parameter.get('constraints', {}) != {}:
                if parameter['constraints'].get('technical', False) == True:
                    technical_parameters.append(parameter["name"])

        # computing parameter combination coverage
        parameters_used = [list(utt.get('parameters', {}).keys()) for utt in endpoint.get('utterances', []) if isinstance(utt.get('parameters'), dict)]
        combination_parameters = [param for param in parameters_used if not any(p in technical_parameters for p in param)]
        unique_sets = {frozenset(lst) for lst in combination_parameters if lst != []}

        pcc.append(len(unique_sets))

    if len(pcc) > 0:
        return sum(pcc)
    else:
        return None

def _api_text_representation(oas: Dict, endpoint: Dict) -> str:
    """Generate a text representation of the API method for embedding."""
    method_text = {
        "api name": oas['name'],
        "api description": oas.get('description', ''),
        "method name": endpoint['name'],
        "description": endpoint.get('description', ''),
        "parameters": ' '.join([param['name'] + ' ' + param.get('description', '') for param in endpoint.get('parameters', [])])
    }
    return ' '.join(method_text.values())