import json
import ast
import copy
import os
import numpy as np
import random
import itertools
import re
import torch
import time
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
    api_name = oas.get('name') or oas.get('tool_name')
    api_methods = oas.get('api_methods') or oas.get('api_list')
    print(f"This API method has {len(api_methods)} methods to evaluate.")
    for api_method_index, api_method in enumerate(api_methods):
        api_method_name = api_method['name']

        if isinstance(api_method['utterances'], str):
            continue
        for utterances in api_method.get('utterances', []):
            utterance = utterances["utterance"]

            # defining the prompt
            messages = [
                {"role": "system", "content": NATURALNESS_EVALUATION},
                {"role": "user", "content": f"Evaluate the following utterance for naturalness: '{utterance}'"}]                    

            # calling the LLM
            while True:
                try:
                    response = openai_client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=500,
                        temperature=0)
                except Exception as e:
                    print(f"    - Error occurred: {e}. Retrying in 5 seconds...")
                    time.sleep(5)
                    continue
                break

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

def constraint_adherance(oas: Dict, ground_truth_path: str):
    constraint_violations = [0, 0, 0]  # [value_violations, format_violations, inter_dependency_violations] 

    # reading ground truth reference
    with open(ground_truth_path, 'r') as file:
        reference = json.load(file)

    # iterate over the ground truth:
    for endpoint in reference["api_methods"]: 
        for parameter in endpoint["parameters"]:
            constraints = parameter.get('constraints', [])
            if constraints == []: 
                continue

            # retriving all constraints for this parameter in the oas
            api_methods = oas.get('api_methods') or oas.get('api_list')   
            oas_endpoint = next((ep for ep in api_methods if ep["name"] == endpoint["name"]), None)

            # value constraints
            if constraints.get('values', []) != []:
                # min value
                min_value = constraints['values'].get('min', None)
                if min_value is not None:
                    min_value = float(min_value)
                    for utt in oas_endpoint.get('utterances', []):
                        param_value = utt.get('parameters', {}).get(parameter['name'], None)
                        param_value = float(param_value) if param_value is not None else None
                        if param_value is not None and param_value < min_value:
                            constraint_violations[0] += 1

                # max value
                max_value = constraints['values'].get('max', None)
                if max_value is not None:
                    max_value = float(max_value)
                    for utt in oas_endpoint.get('utterances', []):
                        param_value = utt.get('parameters', {}).get(parameter['name'], None)
                        param_value = float(param_value) if param_value is not None else None
                        if param_value is not None and param_value > max_value:
                            constraint_violations[0] += 1

                # enumerated values
                enum_values = constraints['values'].get('enumerated', [])
                if enum_values != []:
                    for utt in oas_endpoint.get('utterances', []):
                        param_value = utt.get('parameters', {}).get(parameter['name'], None)
                        if param_value is not None and param_value not in enum_values:
                            constraint_violations[0] += 1

            # format constraints
            if constraints.get('format', '') != '':
                format_regex = constraints['format']
                pattern = re.compile(format_regex)
                for utt in oas_endpoint.get('utterances', []):
                    param_value = utt.get('parameters', {}).get(parameter['name'], None)
                    if param_value is not None and not pattern.match(str(param_value)):
                        constraint_violations[1] += 1

            # inter-dependency constraints
            if constraints.get('inter-dependency', {}) != {}:
                inter_dependency = constraints['inter-dependency']
                type_of_constraint = inter_dependency.split(":")[0].strip()

                if type_of_constraint == "AtLeastOne":
                    parameters_list = ast.literal_eval(inter_dependency.split(":")[1].strip())
                    for utt in oas_endpoint.get('utterances', []):
                        params_in_utt = utt.get('parameters', {}).keys()
                        count_present = sum(1 for param in parameters_list if param in params_in_utt)
                        if count_present < 1:
                            constraint_violations[2] += 1
                            print(f"(Endpoint: {endpoint['name']}) Violation found in utterance: {utt['utterance']}: parameters {parameters_list} present count = {count_present}, expected exactly one.")
                    
                elif type_of_constraint == "RequireOtherParameters":
                    parameters_list = ast.literal_eval(inter_dependency.split(":")[1].strip())
                    for utt in oas_endpoint.get('utterances', []):
                        params_in_utt = utt.get('parameters', {}).keys()
                        if parameter['name'] in params_in_utt:
                            for req_param in parameters_list:
                                if req_param not in params_in_utt:
                                    constraint_violations[2] += 1

                elif type_of_constraint == "Arithmetic":
                    expression = inter_dependency.split(":")[1].strip()
                    for utt in oas_endpoint.get('utterances', []):
                        params_in_utt = utt.get('parameters', {}).keys()
                        local_dict = {}
                        for param in re.findall(r'\b\w+\b', expression):
                            if param in params_in_utt:
                                param_value = utt.get('parameters', {}).get(param, None)
                                try:
                                    local_dict[param] = float(param_value)
                                except:
                                    local_dict[param] = None
                        try:
                            if None not in local_dict.values():
                                if not eval(expression, {}, local_dict):
                                    constraint_violations[2] += 1
                        except Exception as e:
                            pass

                elif type_of_constraint == "OnlyOne":
                    parameters_list = ast.literal_eval(inter_dependency.split(":")[1].strip())
                    for utt in oas_endpoint.get('utterances', []):
                        params_in_utt = utt.get('parameters', {}).keys()
                        count_present = 0
                        for param_group in parameters_list:
                            if all(param in params_in_utt for param in param_group):
                                count_present += 1
                        if count_present != 1:
                            constraint_violations[2] += 1

                elif type_of_constraint == "AllOrNone":
                    parameters_list = ast.literal_eval(inter_dependency.split(":")[1].strip())
                    for utt in oas_endpoint.get('utterances', []):
                        params_in_utt = utt.get('parameters', {}).keys()
                        count_present = sum(1 for param in parameters_list if param in params_in_utt)
                        if count_present != 0 and count_present != len(parameters_list):
                            constraint_violations[2] += 1

    return constraint_violations

def bertscore(oas: Dict, embedding_model: str) -> None:
    """BERTScore evaluation method."""
    avg_bertscores = []

    api_methods = oas.get('api_methods') or oas.get('api_list')
    for endpoint in api_methods:
        # setting reference text of the API method
        reference_text = _api_text_representation(oas, endpoint)

        # defining embeddings for utterances
        if isinstance(endpoint['utterances'], str):
            continue
        utterances = [utt['utterance'] for utt in endpoint.get('utterances', [])]
        references = [reference_text] * len(utterances)

        # computing BERTScores
        if len(utterances) > 0:
            P, R, F1 = score(utterances, references, model_type=embedding_model, verbose=False)
            avg_f1 = round(F1.mean().item(), 4)
            avg_bertscores.append(avg_f1)

    return round(sum(avg_bertscores) / len(avg_bertscores), 4)


def cosine_similarity(oas: Dict, embedding_model) -> float:
    """Cosine Similarity evaluation method."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_model.to(device)

    avg_cosine_scores = []
    api_methods = oas.get('api_methods') or oas.get('api_list', [])
    
    for endpoint in api_methods:
        # skip malformed utterances
        if isinstance(endpoint.get('utterances'), str):
            continue

        # prepare reference text
        api_method_text_reference = _api_text_representation(oas, endpoint)
        reference_embedding = embedding_model.encode(api_method_text_reference, convert_to_tensor=True, device=device)

        # encode utterances
        utterances = [utt['utterance'] for utt in endpoint.get('utterances', []) if isinstance(utt, dict)]
        if not utterances:
            continue

        utterance_embeddings = embedding_model.encode(utterances, convert_to_tensor=True, device=device)

        # make sure both tensors are on the same device
        reference_embedding = reference_embedding.to(device)
        utterance_embeddings = utterance_embeddings.to(device)

        # compute cosine similarity
        cosine_scores = util.cos_sim(reference_embedding, utterance_embeddings)[0]
        avg_score = round(cosine_scores.mean().item(), 4)
        avg_cosine_scores.append(avg_score)

    if not avg_cosine_scores:
        return 0.0

    return round(sum(avg_cosine_scores) / len(avg_cosine_scores), 4)


def parameter_coverage(oas: Dict) -> None:
    """Parameter Coverage evaluation method."""
    # counting total parameters in the API that are not technical
    pc = []
    total_parameters = 0
    api_methods = oas.get('api_methods') or oas.get('api_list')
    for endpoint in api_methods:
        # computing total non-technical parameters
        number_parameters = 0
        for parameter in endpoint.get('parameters', []):
            if parameter.get('constraints', {}) != {}:
                if parameter['constraints'].get('technical', False) == True:
                    continue
            number_parameters += 1

        # computing parameter coverage
        if number_parameters > 0:
            if isinstance(endpoint.get('utterances'), list):
                parameters_used = [
                    list(utt.get('parameters', {}).keys())
                    for utt in endpoint['utterances']
                    if isinstance(utt, dict)]
            else:
                parameters_used = []
            all_parameters_used = list(set([item for sublist in parameters_used for item in sublist]))
            total_parameters += number_parameters
            coverage = round(len(all_parameters_used) / number_parameters, 4)
            pc.append(coverage)
    
    if len(pc) > 0:
        average_pc = round(sum(pc) / len(pc), 4)
        return average_pc, total_parameters
    else:
        return None

def parameter_combination_coverage(oas: Dict) -> None:
    """Placeholder for Parameter Combination Coverage evaluation method."""
    pcc = []
    total_parameters = 0
    api_methods = oas.get('api_methods') or oas.get('api_list')
    for endpoint in api_methods:
        # getting non-technical parameters
        technical_parameters = []
        for parameter in endpoint.get('parameters', []):
            if parameter.get('constraints', {}) != {}:
                if parameter['constraints'].get('technical', False) == True:
                    technical_parameters.append(parameter["name"])

        # computing parameter combination coverage
        if isinstance(endpoint.get('utterances'), list):
            parameters_used = [
                list(utt.get('parameters', {}).keys())
                for utt in endpoint['utterances']
                if isinstance(utt, dict)]
        else:
            parameters_used = []
        total_parameters += len([param for param in endpoint.get('parameters', []) if param.get('constraints', {}) == {} or not param['constraints'].get('technical', False)])
        combination_parameters = [param for param in parameters_used if not any(p in technical_parameters for p in param)]
        unique_sets = {frozenset(lst) for lst in combination_parameters if lst != []}

        pcc.append(len(unique_sets))

    if len(pcc) > 0:
        return sum(pcc), total_parameters
    else:
        return None

def _api_text_representation(oas: Dict, endpoint: Dict) -> str:
    """Generate a text representation of the API method for embedding."""
    method_text = {
        "api name": oas.get('name') or oas.get('tool_name'),
        "api description": oas.get('description', '') or oas.get('tool_description'),
        "method name": endpoint['name'],
        "description": endpoint.get('description', ''),
        "parameters": ' '.join([param['name'] + ' ' + param.get('description', '') for param in endpoint.get('parameters', [])])
    }
    return ' '.join(method_text.values())