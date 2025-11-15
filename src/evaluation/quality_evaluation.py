import json
import ast
import copy
import os
import numpy as np
import random
import itertools
import re
from typing import Dict, List, Tuple
from openai import OpenAI
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
            if content == 'natural':
                natural_count += 1
            elif content == 'unnatural':
                unnatural_count += 1
            else:
                wrong_count += 1
                content = 'invalid response'

            # storing results
            results.append({
                "api": api_name,
                "api_method": api_method_name,
                "utterance": utterance,
                "evaluation": content})

    return {
        "natural_count": natural_count,
        "unnatural_count": unnatural_count,
        "wrong_count": wrong_count,
        "detailed_results": results}

def bertscore(oas: Dict) -> None:
    """Placeholder for BERTScore evaluation method."""
    pass

def cosine_similarity(oas: Dict) -> None:
    """Placeholder for Cosine Similarity evaluation method."""
    pass

def parameter_coverage(oas: Dict) -> None:
    """Placeholder for Parameter Coverage evaluation method."""
    pass

def parameter_combination_coverage(oas: Dict) -> None:
    """Placeholder for Parameter Combination Coverage evaluation method."""
    pass

                    
