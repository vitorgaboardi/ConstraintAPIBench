"""
This file reads the dataset from ToolLLM and selects more relevant APIs based on some criteria. 

Criteria for selecting high-quality OAS: 
* 1. There must be at least one parameter among all the API methods
* 2. The tool description must have at least 5 words. 
* 3. Order by most popular APIs since they represent more commonly used APIs.
* 4. Get only APIs with API methods lower than 30 methods to limit the number of requests
"""

import os
import json
import shutil
import random
from typing import List


class ToolBenchPreProcessing:
    def __init__(self, 
                 base_folder: str,
                 new_folder: str): 

        # standard parameters                
        self.base_folder = base_folder
        self.new_folder = new_folder

        # filtering 
        self.apis = {}

    def get_top_popularity_scores(self, number_of_APIs_per_category: int = 5, max_number_API_methods: int = 30):
        self.apis = {}

        # Get list of categories (folders) sorted alphabetically
        categories = sorted(os.listdir(self.base_folder))

        for category in categories:
            category_path = os.path.join(self.base_folder, category)
            
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
                    top_scores = sorted(scores, key=lambda x: (-x[1], x[2], x[0]))[:number_of_APIs_per_category]
                    self.apis[category] = top_scores

        # copy the original files from the dataset to a new folder
    def save_apis(self, number_of_apis_per_category: str = 5):
        # Create new folder if it doesn't exist
        if not os.path.exists(self.new_folder):
            os.makedirs(self.new_folder)

        category_idx = 0
        for category, scores in self.apis.items():
            # Create folder with the Category if it does not exist
            if not os.path.exists(self.new_folder+'/'+category):
                os.makedirs(self.new_folder+'/'+category)

            # Ensure we only copy the first two files per category
            copied_count = 0
            for file_path, _, _ in scores[:number_of_apis_per_category]:
                try:
                    # Modify the copied file to add 'category' field
                    new_file_name = f"{category}/{os.path.basename(file_path)}"
                    new_file_path = os.path.join(self.new_folder, new_file_name)
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
                        data['category'] = category
                        new_file.seek(0)
                        json.dump(data, new_file, indent=4)
                        new_file.truncate()

                    print(f"Copied and modified: {new_file_path}")

                except (IOError, OSError, json.JSONDecodeError) as e:
                    print(f"Error processing file {file_path}: {e}")

            category_idx+=1

    def print_top_scores(self):
        for category, scores in self.apis.items():
            print(f"Top {len(scores)} popularity scores in category '{category}':")
            for rank, (file_path, score, api_list_length) in enumerate(scores, 1):
                print(f"Rank {rank}: File: {file_path.split('/')[-1]}, Popularity Score: {score}, Number of APIs: {api_list_length}")
            print()


def main():
    base_folder = '/home/vitor/Documents/phd/other works/ToolBench/data/data/toolenv/tools'         # path from the ToolBench dataset
    new_folder = './dataset/tools'

    processor = ToolBenchPreProcessing(base_folder, new_folder)
    processor.get_top_popularity_scores(number_of_APIs_per_category=5, max_number_API_methods=30)
    processor.print_top_scores()
    # processor.save_apis(number_of_apis_per_category=5)

if __name__ == "__main__":
    main()