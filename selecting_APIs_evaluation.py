import os
import json
import random
import pandas as pd
from collections import defaultdict

# This code randomly selects 30 endpoints that have constraints detected by GPT-4o and DeepSeek. 
# It will get only the selected endpoints and paste in the folder manual evaluation/model/constraint-aware/constraints
# Based on the files on this path, we look for the utterances and then evaluate them (using manual evaluation, BS, parameter coverage and parameter combination coverage)

random.seed(87)  # I am building with 370

models = ["deepseek-v3", "gpt-4o"]
constraint_types = ["values", "enumerated", "format", "id", "technical", "inter_dependency"]
constraint_map = {k: [] for k in constraint_types}

# 1. Selecting the endpoints for manually evaluation
for model in models:
    constraint_folder = os.path.join('./data/dataset', model, 'constraint-aware/constraints/') 
    print(constraint_folder)

    for root, _, files in os.walk(constraint_folder):
        for filename in files:
            file_path = os.path.join(constraint_folder, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)

                for api_method in data.get("api_methods", []):
                    method_name = api_method["name"]
                    identifier = f"{filename}+++{method_name}"
                    for param in api_method.get("parameters", []):
                        constraints = param.get("constraints", {})
                        if not constraints:
                            continue

                        if "values" in constraints:
                            if any(k in constraints["values"] for k in ["min", "max"]):
                                if identifier not in constraint_map["values"]:
                                    constraint_map["values"].append(identifier)
                            if "enumerated" in constraints["values"]:
                                if identifier not in constraint_map["enumerated"]:
                                    constraint_map["enumerated"].append(identifier)

                        for c_type in ["format", "id", "technical", "inter-dependency"]:
                            mapped_type = c_type.replace("-", "_")
                            if c_type in constraints and identifier not in constraint_map[mapped_type]:
                                constraint_map[mapped_type].append(identifier)

print("\nNumber of endpoints with the following constraints for both models:")
for k in constraint_types:
    print(f"{k} constraints:", len(constraint_map[k]))

selected_endpoints = []
for constraint, items in constraint_map.items():
    pool = [x for x in items if x not in selected_endpoints]
    selected = random.sample(pool, k=min(5, len(pool)))
    selected_endpoints.extend(selected)

# Count per constraint
constraint_counts = defaultdict(int)
for endpoint in selected_endpoints:
    for constraint, items in constraint_map.items():
        if endpoint in items:
            constraint_counts[constraint] += 1

selected_APIs = []
for endpoints in selected_endpoints:
    API = endpoints.split("+++")[0]
    if API not in selected_APIs:
        selected_APIs.append(API)

print("\nSelected APIs (", len(selected_APIs),'):')
print(selected_APIs)
print("\nSelected endpoints")
print(selected_endpoints)
print("\nSelected Count per Constraint:")
print(dict(constraint_counts))


# Step 2: Copying the constraints to the manual evaluation folder:
for model in models:
    constraint_folder = os.path.join('./data/dataset', model, 'constraint-aware/constraints/') 
    save_folder = os.path.join('./data/manual evaluation', model, 'constraint-aware/constraints/')

    selected_by_api = defaultdict(list)
    for endpoint in selected_endpoints:
        api_name, method_name = endpoint.split("+++")
        selected_by_api[api_name].append(method_name)

    for filename, method_names in selected_by_api.items():
        file_path = os.path.join(constraint_folder, filename)

        with open(file_path, 'r') as f:
            data = json.load(f)
        
            methods_to_save = []
            seen_methods = []
            for endpoint in data['api_methods']:
                if endpoint['name'] in method_names and endpoint['name'] not in seen_methods:
                    methods_to_save.append(endpoint)
                    seen_methods.append(endpoint['name'])

            data['api_methods'] = methods_to_save

            output_file = os.path.join(save_folder, filename)
            with open(output_file, 'w') as out_f:
                json.dump(data, out_f, indent=4)

# Step 3: Copying the utterances of all prompt strategies to the manual evaluation folder:
prompts = ['constraint-aware', 'sheng et al']
#models = ['gpt-4o']
for model in models:
    for prompt in prompts:
        utterance_folder = os.path.join('./data/dataset', model, prompt, 'utterances' if prompt == 'constraint-aware' else '') 
        save_folder = os.path.join('./data/manual evaluation', model, prompt, 'utterances' if prompt == 'constraint-aware' else '')

        selected_by_api = defaultdict(list)
        for endpoint in selected_endpoints:
            api_name, method_name = endpoint.split("+++")
            selected_by_api[api_name].append(method_name)

        for filename, method_names in selected_by_api.items():
            file_path = os.path.join(utterance_folder, filename)

            with open(file_path, 'r') as f:
                data = json.load(f)
            
                methods_to_save = []
                seen_methods = []
                for endpoint in data['api_methods']:
                    if endpoint['name'] in method_names and endpoint['name'] not in seen_methods:
                        methods_to_save.append(endpoint)
                        seen_methods.append(endpoint['name'])

                data['api_methods'] = methods_to_save

                output_file = os.path.join(save_folder, filename)
                with open(output_file, 'w') as out_f:
                    json.dump(data, out_f, indent=4)
