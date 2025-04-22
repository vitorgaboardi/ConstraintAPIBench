# the goal of this part is to generate combination of parameters that will be used to create utterances.

# one first thing that I have to do is to check if there are any constraint condition. 
# In this case, the sampled values must be used altogether.

import json
import ast
import copy
import os
import numpy as np
from openai import OpenAI

client = OpenAI(api_key="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")

PROMPT_BASE_INSTRUCTION = """You are an expert in interpreting OpenAPI specification (OAS). 
I will provide you with the API name and description, the API method name and description, and the list of parameters, each with its name, description, and constraints.
Your task is to generate {} utterances that users may ask that must be solved using the given API method information. 

    - All required parameters must be added in all utterances.
    - Create utterances different from each other, making sure they are lexically (rich vocabulary) and syntactically (rich syntax structures) diverse. 
    - Do not generate new parameters. You must only use the parameters defined in the documentation provided.
    - Generate natural utterances that represent how users would normally say when trying to fulfill the task.
    - Do not add the API name or the API method name in the utterance.
    - Do not add any other text explaining the choices that you made.

    - The parameters of the generated utterances must respect to all the constraints defined under the "constraints" key for each parameter. This includes:
        - Values must conform to format constraints (e.g., ISO 8601 date/time, country codes, email).
        - Values must respect range limitations (minimum and maximum values) or respect the specific set of possible values.
        - If the parameter represents an ID or reference (i.e., "id = True" in the constraint option), create values representing what the IDs represent instead of actually creating an ID. For example, for a parameter like "hotelId", generate hotel names (e.g., "Hotel California", "The Grand Hyatt"), and for "signId", generate zodiac signs (e.g., "Gemini", "Pisces").
        - Inter-parameter constraints (i.e. "conditional" key in the constraint option) describes constraints among parameters. The combination of parameters values must respect these inter-parameter constraints.

The final output must be Python list of dictionaries, where each instance have an "utterance" key (representing the generated utterance), and a "parameter" key (representing all the parameters included in the utterance).
"""


# basic variables
OAS_folder = "./dataset/GPT-4.1-mini/constraints"
parameter_folder = './dataset/GPT-4.1-mini/parameter_combination'

# defining variables
number_of_utterances_per_method = 10
number_of_utterances_only_required_parameter = 5
number_of_utterances_both_type_parameter = number_of_utterances_per_method - number_of_utterances_only_required_parameter

# auxiliary variables
API_count = 0
API_methods_count = 0
mistakes = 0
conditional_found = 0

number_of_methods_with_no_optional_parameters = 0
number_of_methods_with_no_parameters = 0
number_of_methods_with_condition_constraint = 0

# iterating over the categories
categories = sorted(os.listdir(OAS_folder))
for category_index, category in enumerate(categories):
    category_path = os.path.join(OAS_folder, category)
    #print(category_path)

    # generate folder to save OAS now enriched with constraints information
    saving_new_OAS = os.path.join(parameter_folder, category)
    if not os.path.exists(saving_new_OAS):
        os.makedirs(saving_new_OAS)

    # iterating over the OAS for each category
    for root, _, files in os.walk(category_path):
        for filename in files:
            # make sure that the file is not there
            if not os.path.exists(saving_new_OAS+'/'+filename):
                # read file for each API
                file_path = os.path.join(category_path, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    #print(category_index, category, data['tool_name'])
                    api_name = data['tool_name']
                    api_description = data['tool_description']

                    # iterating over each API method
                    for api_method_index, api_method in enumerate(data['api_list']):
                        api_method_name = api_method['name']
                        api_method_description = api_method['description']
                        # discard the parameters that are related to the API 
                        api_method_parameters = [
                            param for param in api_method['parameters']
                            if not ('constraints' in param and param['constraints'].get('api_related') is True)]

                        ### ORGANISING PARAMETER INFORMATION
                        # analysing parameter information
                        required_parameters = [param["name"] for param in api_method_parameters if param.get('required', False)]
                        optional_parameters = [param["name"] for param in api_method_parameters if 'required' in param and param['required'] is False] 
                        conditional_constraints = [param['constraints']['conditional'] for param in api_method_parameters if 'constraints' in param and 'conditional' in param['constraints']]

                        #condition_count = sum(1 for param in api_method_parameters if 'constraints' in param and 'conditional' in param['constraints'])

                        # 1) There is no parameter
                        if len(required_parameters) + len(optional_parameters) == 0:
                            sets = []
                            # include a more general instruction
                            number_of_methods_with_no_parameters+=1

                        # 2) There are only required parameters 
                        elif len(optional_parameters) == 0:
                            print('required parameters:', required_parameters)
                            sets = [] 
                            # include instructions that make clear that all the required parameters must be added. 
                            number_of_methods_with_no_optional_parameters+=1

                        # 3) There are only optional parameters
                        elif len(required_parameters) == 0:
                            print('optional parameters:', optional_parameters)

                            # 3.1: check the constraints and group the parameters that have a common constraints as one!
                            # COME BACK HERE!
                            if len(conditional_constraints) > 0:
                                print('conditional constraints:', conditional_constraints)

                            # generate the code here to create the set of parameters
                            sets = []

                        # 4) There are required and optional parameters.
                        else:
                            special_combination = "use only required parameters: "
                            sets = []


                        if len(conditional_constraints) > 0:
                            number_of_methods_with_condition_constraint+=1

                        API_methods_count+=1

                        #print(api_method['name'], required_count, optional_count)

            #break
    #break

print('remember the results are after removing parameters that are API-centered')
print('number of methods with conditional constraints:', number_of_methods_with_condition_constraint)
print('number of methods with no parameters:', number_of_methods_with_no_parameters)
print('number of methods with no optional parameters:', number_of_methods_with_no_optional_parameters)
print('number of methods:', API_methods_count)