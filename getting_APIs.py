import os
import json
import shutil
import random

# This file reads the dataset from ToolLLM and selects more relevant APIs based on some criteria. 
# The idea is that there are too many tools here and there is no need to request utterances for all of them. 
# Selecting high-quality OAS: 
# * 1. there must be at least one parameter among all the API methods (Make sure we can extract constraints from the OAS)
# * 2. the tool description must have at least 5 words. (We found that some OAS has poor quality and is general, without many details about what it accomplishes)
# * 3. order by most popular APIs since they represent more commonly used APIs. (A way to check for more relevant APIs is to oder by popularity)
# * 4. get the first 20 APIs to limit number of request and evaluate if using less APIs with more quality improves the results. (DEPRECEATED)
# * 4. get only APIs with API methods lower than 30 methods to limit the number of requests (since we create everything per API method, we decided to limit the number of APIs)

# We sort by popularity and select the 'number_tool' tools to be considered. The documentation of each individual is copied in 'new_folder'
# It also saves the following information:
# - "api_knowledge.json", which saves all the selected APIs into one JSON file.
# - "api_knowledge5.json", which randomly selects 5 API methods for each tool/API


# return a list with the tools that respect the given criteria above and sorted by popularity
def get_top_popularity_scores(base_folder, number_APIs, max_number_API_methods=30):
    category_scores = {}

    # Get list of categories (folders) sorted alphabetically
    categories = sorted(os.listdir(base_folder))

    for category in categories:
        category_path = os.path.join(base_folder, category)
        
        if os.path.isdir(category_path):
            print(category_path)
            scores = []
            
            for root, _, files in os.walk(category_path):
                for file in files:
                    if file.endswith('.json'):
                        file_path = os.path.join(root, file)
                        
                        with open(file_path, 'r') as f:
                            try:
                                data = json.load(f)
                                
                                if 'score' in data:
                                    if data['score'] is not None:
                                        popularity_score = data['score']['popularityScore']

                                        # Check if 'api_list' exists and is a list
                                        if 'api_list' in data:
                                            if isinstance(data['api_list'], list):
                                                api_list_length = len(data['api_list'])
                                                
                                                # 1. check if there is any parameter (in general, there were some docs that did not have any parameter)
                                                has_parameters = any(
                                                    'required_parameters' in api_method and api_method['required_parameters'] or
                                                    'optional_parameters' in api_method and api_method['optional_parameters']
                                                    for api_method in data['api_list'])

                                                # 2. check the number of words in the tool description
                                                if isinstance(data['tool_description'], str):
                                                    tool_description = data['tool_description']
                                                    number_of_words_tool_description =  len(tool_description.split(" "))
                                                else:
                                                    number_of_words_tool_description = 0                               
                                                
                                                # only in case both options are considered, the tool is appended
                                                if has_parameters and number_of_words_tool_description >= 5 and api_list_length <= max_number_API_methods: 
                                                    scores.append((file_path, popularity_score, api_list_length))

                            except json.JSONDecodeError:
                                print(f"Error decoding JSON in file: {file_path}")

            if scores:
                # Sort scores by popularityScore in descending order and get top n
                top_scores = sorted(scores, key=lambda x: (-x[1], x[2], x[0]))[:number_APIs]
                category_scores[category] = top_scores

    return category_scores

# it prints the API, popularity score and number of APIs
def print_top_scores(category_scores):
    for category, scores in category_scores.items():
        print(f"Top {len(scores)} popularity scores in category '{category}':")
        for rank, (file_path, score, api_list_length) in enumerate(scores, 1):
            print(f"Rank {rank}: File: {file_path}, Popularity Score: {score}, Number of APIs: {api_list_length}")
        print()

# copy the original files from the dataset to a new folder
def copy_and_modify_files(top_scores, new_folder, number_of_apis_per_category=20):
    # Create new folder if it doesn't exist
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    category_idx = 0
    for category, scores in top_scores.items():
        # Create folder with the Category if it does not exist
        if not os.path.exists(new_folder+'/'+category):
            os.makedirs(new_folder+'/'+category)

        # Ensure we only copy the first two files per category
        copied_count = 0
        for file_path, _, _ in scores[:number_of_apis_per_category]:  # Copying the first two only
            try:
                # Modify the copied file to add 'category' field
                #new_file_name = f"{category_idx}_{os.path.basename(file_path)}"
                new_file_name = f"{category}/{os.path.basename(file_path)}"
                new_file_path = os.path.join(new_folder, new_file_name)
                shutil.copy(file_path, new_file_path)
                copied_count += 1

                with open(new_file_path, 'r+') as new_file:
                    data = json.load(new_file)

                    # Update api_list structure
                    updated_api_list = []
                    for api_method in data['api_list']:
                        required_params = api_method.get('required_parameters', [])
                        optional_params = api_method.get('optional_parameters', [])
                        
                        # Combine required and optional parameters into a single list
                        parameters = []
                        for param in required_params:
                            param['required'] = True
                            parameters.append(param)

                        for param in optional_params:
                            param['required'] = False
                            parameters.append(param)
                        
                        # Update api_method with combined parameters
                        api_method['parameters'] = parameters
                        # Remove old keys
                        api_method.pop('required_parameters', None)
                        api_method.pop('optional_parameters', None)
                        
                        updated_api_list.append(api_method)

                    data['api_list'] = updated_api_list

                    data['category'] = category  # Adding category field
                    new_file.seek(0)
                    json.dump(data, new_file, indent=4)
                    new_file.truncate()

                print(f"Copied and modified: {new_file_path}")

            except (IOError, OSError, json.JSONDecodeError) as e:
                print(f"Error processing file {file_path}: {e}")

        category_idx+=1

# save documentation in a new JSON file
def save_documentation(new_folder):
    # getting the name all files sorted in numerical order (given the first number)
    files = sorted([f for f in os.listdir(new_folder) if os.path.isfile(os.path.join(new_folder, f))])
    files.sort(key=lambda f: int(f.split('_')[0]))

    # reading all files and storing information in a dictionary
    api_knowledge = [] # each item will be one API

    for f in files:
        file_path = os.path.join(new_folder, f)
        with open(file_path, 'r') as json_file:
            api_documentation = json.load(json_file)
            api_knowledge.append(api_documentation)

    return api_knowledge

# save documentation after getting a specific number of API methods
def select_k_methods(new_folder, k=5, random_seed=42):
    # reading file names in order
    files = sorted([f for f in os.listdir(new_folder) if os.path.isfile(os.path.join(new_folder, f))])
    files.sort(key=lambda f: int(f.split('_')[0]))
    random.seed(random_seed)
    
    # reading file
    api_knowledge_k = []
    for file in files:
        # reading file
        file_path = os.path.join(new_folder, file)
        with open(file_path, 'r') as json_file:
            file = json.load(json_file)

        # getting information
        number_of_APIs = len(file["api_list"])

        if number_of_APIs <= k: 
            api_knowledge_k.append(file)
        else: 
            select_API_index = sorted(random.sample(range(0, number_of_APIs), k))

            api_methods = []
            for index in select_API_index:
                api_methods.append(file["api_list"][index])
    
            file["api_list"] = api_methods
            api_knowledge_k.append(file)

    return api_knowledge_k


def number_of_methods(api_knowledge):
    total_methods = 0
    for api in api_knowledge:
        method_number = len(api['api_list'])
        total_methods+=method_number

    return total_methods


def number_of_parameters(api_knowledge):
    total_parameters = 0
    for api in api_knowledge:
        for method in api['api_list']:
            total_parameters+=len(method['parameters'])

    return total_parameters


# variables
base_folder = '/home/vitor/Documents/phd/other works/ToolBench/data/data/toolenv/tools' 
new_folder = '/home/vitor/Documents/phd/api constraints_3/dataset/tools'
number_of_APIs_per_category = 5    # maximum number of APIs per category
max_number_API_methods = 30        # maximum number of API methods per API

# running code
top_scores = get_top_popularity_scores(base_folder, number_of_APIs_per_category, max_number_API_methods)
print_top_scores(top_scores)
copy_and_modify_files(top_scores, new_folder, number_of_APIs_per_category)

# # Saving documentation in a single JSON file
# api_knowledge =  save_documentation(new_folder)
# with open('api_knowledge.json', 'w') as output_json_file:
#     json.dump(api_knowledge, output_json_file, indent=4)

# # checking the number of methods and parameters after selecting a maximum of 5 methods per API. 
# print('total number of methods:', number_of_methods(api_knowledge))
# print('total number of parameters:', number_of_parameters(api_knowledge))