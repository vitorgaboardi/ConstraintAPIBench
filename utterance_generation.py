# the goal of this part is to generate combination of parameters that will be used to create utterances.

# one first thing that I have to do is to check if there are any constraint condition. 
# In this case, the sampled values must be used altogether.

import json
import ast
import copy
import os
import numpy as np
import random
import itertools
import re
from openai import OpenAI

PROMPT_BASE_INSTRUCTION = """You are an expert in interpreting OpenAPI Specification (OAS). 
I will provide you with API name and description, API method name and description, and list of parameters, each with its name, description, and constraints.
Your task is to generate {number_of_utterances} utterances that users may ask that must be solved using the given API method. Consider the following guidelines:
    - Generate utterances different from each other, making sure they are lexically (rich vocabulary) and syntactically (sentence structures) diverse. 
    - Do not repeat the same value for a parameter across different utterances. The "default" key is just an example and must not be used in all utterances. 
    - Do not use general placeholders such as "this URL", "this link", "this name", "a candidate" for parameters. Always use actual and realistic parameter values.
    - Generate natural utterances that represent what users would normally say when trying to fulfill the task.
    - Do not add the API name or the API method name in the utterance.
    - If parameters are provided, you must generate values for these parameters in such a way that all constraints defined under the "constraints" key are respected. This includes:
        - Values must conform to format constraints (e.g., ISO 8601 date/time, country codes, email).
        - Values must conform range limitations (minimum and maximum values). 
        - For parameters with "specific" values constraints (i.e., a fixed set of allowed values), cycle through different allowed values in different utterances.  
        - If the "id" constraint is True, do not generate artificial IDs (e.g., "1234" or "abc_123"). Instead, create values that represent what the ID refers to. For example, for "hotelId" parameter, generate hotel names such as "Hotel California" or "The Grand Hyatt" instead of "ACPAR419".
        - Inter-parameter constraints (i.e. "inter-dependency" key in the "constraint" field) describes constraints between parameters. The combination of parameters values must strictly respect these constraints. For instance, there are cases where two parameters must be added simultaneously in an API call or only one parameter must be included between a group of parameters.

    {api_context}

The output must be a Python list of dictionaries, where each dictionary has two keys:
    - "utterance": the natural language request.
    - "parameters": a dictionary containing the name-value pairs for all parameters used in the utterance. 
    
The "parameters" dictionary must have pairs that can be infered or recognised from the generated natural utterance. 
Finally, you must only output the Python list and do not output anything else, such as notes or explanation about the reasoning.
"""

# basic variables
# model name
#model="gpt-4.1-mini"
#model="gpt-4.1"
model='deepseek-ai/DeepSeek-V3'

if model == "gpt-4.1-mini" or model == "gpt-4.1":
  client = OpenAI(api_key="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")
  constraint_folder = '/home/vitor/Documents/phd/ConstraintAPIBench/dataset/'+model+'/constraints/'
  utterance_folder = './dataset/'+model+'/utterances/'
else:
  client = client = OpenAI(api_key="0cTfrP7S4xnxVcQeMnkCvfY7nrgZs41e",
                           base_url="https://api.deepinfra.com/v1/openai",)
  model_name = model.split('/')[1]
  constraint_folder = '/home/vitor/Documents/phd/ConstraintAPIBench/dataset/'+model_name+'/constraints/'
  utterance_folder = './dataset/'+model_name+'/utterances/'


# defining variables
number_of_utterances_per_method = 10

# auxiliary variables
API_count = 0
API_methods_count = 0
mistakes = 0
conditional_found = 0

number_of_methods_with_no_optional_parameters = 0
number_of_methods_with_no_parameters = 0
number_of_methods_with_condition_constraint = 0

## It will be tough to explain. 
# given a number of optional parameters, it keeps sampling different combination of parameters that must be used when generating utterances
# the idea is to make sure we include utterances where we maximize the use of optional parameters, while generating a specific number of utterances 
# we keep selecting combination of parameters using a window approach
def defining_combination_of_parameters(parameters, conditional_constraints, only_required_parameters=False):
    # 1: if there is any conditional constraint. we make sure that all the variables related to the constraint must be sampled together, mainly when related to 
    # the goal is to make sure that the utterances respect the parameters constraints.
    parameter_set_by_constraint = []
    if len(conditional_constraints) > 0:
        for condition in conditional_constraints:
            param = condition.split(':')[0].replace(' ', '').split(',')
            if param not in parameter_set_by_constraint:
                parameter_set_by_constraint.append(param)
    
    parameter_set = []
    for param in parameters:
        is_optional_param_inside_a_conditional_constraint = False
        for set in parameter_set_by_constraint:
            if param in set:
                is_optional_param_inside_a_conditional_constraint = True
                break
        
        if is_optional_param_inside_a_conditional_constraint is False:
            parameter_set.append(param)

    # 2) Sampling combinations of parameters to be used when generating utterances
    # the idea is to sample group of optional parameters that will be used together when creating a utterances
    # we build an algorithm in such a way that all parameters are sampled at least once. 
    # although the generation of the utterances may not include the all the parameters (it depends whether it makes sense to include them.)
    combination_of_parameters = []
    number_of_parameters = len(parameter_set)
    number_of_parameters_to_be_sampled_per_utterance = int(np.floor(number_of_parameters/number_of_utterances_per_method) + 1) # the number of utterances is 10 for experiments

    window_size = number_of_parameters_to_be_sampled_per_utterance
    start_index = 0

    if only_required_parameters is True:
        combination_of_parameters.append(['use only required parameters.'])

    while len(combination_of_parameters) < number_of_utterances_per_method:
        if start_index + window_size > number_of_parameters:
            start_index = 0
            window_size += 1
            if window_size > number_of_parameters:
                window_size = number_of_parameters_to_be_sampled_per_utterance
                if len(combination_of_parameters) < number_of_utterances_per_method:
                    combination_of_parameters.append(['use only required parameters.'])

        window = parameter_set[start_index:start_index+window_size]
        if len(combination_of_parameters) < number_of_utterances_per_method:
            combination_of_parameters.append(window)

        start_index +=1
    
    return combination_of_parameters



## Focusing on generating utterances
# iterating over the categories
categories = sorted(os.listdir(constraint_folder))
for category_index, category in enumerate(categories):
    category_path = os.path.join(constraint_folder, category)
    print(category_path)

    # generate folder to save OAS now enriched with constraints information
    saving_utterances_path = os.path.join(utterance_folder, category)
    if not os.path.exists(saving_utterances_path):
        os.makedirs(saving_utterances_path)

    # iterating over the OAS for each category
    for root, _, files in os.walk(category_path):
        for filename in files:
            # make sure that the file is not there
            if not os.path.exists(saving_utterances_path+'/'+filename):
                # read file for each API
                file_path = os.path.join(category_path, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    print(category_index, category, data['tool_name'])
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
                        # updating the new API method parameters
                        api_method['parameters'] = api_method_parameters
                        print('api_method:', api_method_name)

                        ### ORGANISING PARAMETER INFORMATION
                        # analysing parameter information
                        required_parameters = [param["name"] for param in api_method_parameters if param.get('required', False)]
                        optional_parameters = [param["name"] for param in api_method_parameters if 'required' in param and param['required'] is False] 
                        conditional_constraints = [param['constraints']['inter-dependency'] for param in api_method_parameters if 'constraints' in param and 'inter-dependency' in param['constraints']]

                        ### ORGANISING PROMPT
                        # 1) There is no parameter
                        if len(required_parameters) + len(optional_parameters) == 0:                            
                            prompt = PROMPT_BASE_INSTRUCTION.format(
                                number_of_utterances=number_of_utterances_per_method,
                                api_context='')

                            number_of_methods_with_no_parameters+=1

                        # 2) There are only required parameters 
                        elif len(optional_parameters) == 0:
                            prompt = PROMPT_BASE_INSTRUCTION.format(
                                number_of_utterances=number_of_utterances_per_method,
                                api_context='All utterances must include the required parameters: '+ str(required_parameters))

                            number_of_methods_with_no_optional_parameters+=1

                        # 3) There are only optional parameters
                        elif len(required_parameters) == 0:
                            combination_of_parameters = defining_combination_of_parameters(optional_parameters, conditional_constraints)
                            
                            text = ("Additionally, I will provide a set of parameters for each utterance. You should include them if they can be naturally combined in a realistic user utterance.\n")
                            for index, parameters in enumerate(combination_of_parameters):
                                parameter_names_as_string = [item for group in parameters for item in (group if isinstance(group, list) else [group])]
                                result = ', '.join(parameter_names_as_string)

                                text+= "    - Utterance " + str(index+1) + ": " + result + "\n"
                            
                            prompt = PROMPT_BASE_INSTRUCTION.format(
                                number_of_utterances=number_of_utterances_per_method,
                                api_context=text)    

                        # 4) There are required and optional parameters.
                        else:
                            combination_of_parameters = defining_combination_of_parameters(optional_parameters, conditional_constraints, only_required_parameters=True)
                            
                            text = ("All utterances must include the required parameters: " + str(required_parameters) + "\n"
                                    "Additionally, I will provide a set of parameters for each utterance. You should include them if they can be naturally combined in a realistic user utterance.\n")
                            for index, parameters in enumerate(combination_of_parameters):
                                parameter_names_as_string = [item for group in parameters for item in (group if isinstance(group, list) else [group])]
                                result = ', '.join(parameter_names_as_string)

                                text+= "    - Utterance " + str(index+1) + ": " + result + "\n"
                            
                            prompt = PROMPT_BASE_INSTRUCTION.format(
                                number_of_utterances=number_of_utterances_per_method,
                                api_context=text) 

                        if len(conditional_constraints) > 0:
                            number_of_methods_with_condition_constraint+=1

                        ### MAKING THE CALL AND STORING THE RESULTS
                        ## ORGANISE THE INPUT TO BE included
                        input = {"API Name": api_name,
                                "API Description": api_description if len(api_description) < 4000 else '',
                                "API Method Name": api_method_name,
                                "API Method Description": api_method_description if len(api_method_description) < 4000 else '',
                                "Parameters": api_method_parameters}
                          
                        messages = [{"role": "system", "content": prompt},
                                    {"role": "user", "content": str(input)}]

                        #calling API
                        response = client.chat.completions.create(
                              model=model,
                              messages=messages,
                              max_tokens=3000,
                              temperature=0)

                        try: 
                            #content = re.sub(r"```(json)?", "", response.choices[0].message.content).strip()
                            #content = re.sub(r"//.*", "", content)
                            content = response.choices[0].message.content
                            content = content.replace("```python","").replace("```","")
                            content = content.replace("True", "true").replace("False", "false").replace("None", "null")
                            content = json.loads(content)

                            api_method['utterances'] = content

                        except Exception as e:
                            print("Exception arised!")
                            print(e)
                            print(response.choices[0].message.content)
                            api_method['utterances'] = 'error parsing the information!'
                            mistakes+=1

                        API_methods_count+=1
                
                # saving file with utterances and parameters mapping
                with open(saving_utterances_path+'/'+filename, 'w') as json_file:
                    json.dump(data, json_file, indent=4) 
            
            API_count+=1
            break
        break

print('remember the results are after removing parameters that are API-centered')
print('number of methods with inter-dependency constraints:', number_of_methods_with_condition_constraint)
print('number of methods with no parameters:', number_of_methods_with_no_parameters)
print('number of methods with no optional parameters:', number_of_methods_with_no_optional_parameters)
print('number of methods:', API_methods_count)