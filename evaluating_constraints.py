## this file is to select endpoints for manual validation and organise the files to perform the manual validation
## the selection of the endpoints is based on the constraints detected: we randomly select 5 endpoints that contain each of the 6 constraints.
## this ensures that the analysis if the constraints are respected is consistent with all the constraints detected. 
## the final analysis will be related to whether the constraints are violated in the utterance or not. 
## (the ID and the technical constraints) are there to improve the naturalness/semantic relevance of the utterance. they may be in the utterance and still be correct, but they are not natural...

import os
import json
import random
from collections import defaultdict

random.seed(29)

# Model selection and path setup
model = "gpt-4.1-mini"
model_name = model.split("/")[1] if "/" in model else model
constraint_folder = f'./dataset/Constraint-based prompt/{model_name}/constraints/'

# Initialize constraint categories
constraint_types = ["values", "specific", "format", "id", "api_related", "inter_dependency"]
constraint_map = {k: [] for k in constraint_types}

# Preload all API files and organize by filename
api_data = {}
categories = sorted(os.listdir(constraint_folder))
for category in categories:
    category_path = os.path.join(constraint_folder, category)
    for root, _, files in os.walk(category_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                api_data[filename] = data

# Step 1: Populate constraints
for filename, data in api_data.items():
    for api_method in data.get("api_list", []):
        method_name = api_method["name"]
        identifier = f"{filename}+++{method_name}"
        for param in api_method.get("parameters", []):
            constraints = param.get("constraints", {})
            if not constraints:
                continue

            # Check all constraint types
            if "values" in constraints:
                if any(k in constraints["values"] for k in ["min", "max"]):
                    if identifier not in constraint_map["values"]:
                        constraint_map["values"].append(identifier)
                if "specific" in constraints["values"]:
                    if identifier not in constraint_map["specific"]:
                        constraint_map["specific"].append(identifier)

            for c_type in ["format", "id", "api_related", "inter-dependency"]:
                mapped_type = c_type.replace("-", "_")
                if c_type in constraints and identifier not in constraint_map[mapped_type]:
                    constraint_map[mapped_type].append(identifier)

# Print constraint counts
print("\nNumber of endpoints with the following constraints:")
for k in constraint_types:
    print(f"{k} constraints:", len(constraint_map[k]))


# Step 2: Random sampling
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

print("\nSelected Endpoints:")
print(selected_endpoints)
print("\nSelected Count per Constraint:")
print(dict(constraint_counts))


# Step 3: Save selected endpoint documentation
prompts = ['Sheng et al prompt']
for prompt in prompts:
    save_path = f'./dataset/{prompt}/{model}/manual evaluation/APIs/'
    os.makedirs(save_path, exist_ok=True)

    # Group selected methods by API file
    selected_by_api = defaultdict(list)
    for endpoint in selected_endpoints:
        api_name, method_name = endpoint.split("+++")
        selected_by_api[api_name].append(method_name)

    for api_name, method_names in selected_by_api.items():
        data = api_data[api_name]
        api_description = data['tool_description']

        methods_to_save = []
        seen_methods = set()
        for api_method in data["api_list"]:
            if api_method["name"] in method_names and api_method["name"] not in seen_methods:
                seen_methods.add(api_method["name"])
                methods_to_save.append({
                    "API method name": api_method["name"],
                    "API method description": api_method["description"],
                    "API method parameters": api_method["parameters"]
                })

        documentation = {
            "API name": api_name,
            "API description": api_description,
            "API method": methods_to_save
        }

        output_file = os.path.join(save_path, filename)
        with open(output_file, 'w') as out_f:
            json.dump(documentation, out_f, indent=4)


# Step 4: Save utterances generated 
