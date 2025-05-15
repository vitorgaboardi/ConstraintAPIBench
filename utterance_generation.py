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

# OK

random.seed(10)

PROMPT_BASE_INSTRUCTION = """You are an expert in interpreting OpenAPI Specification (OAS) and generating utterances and API calls related to an OAS. 
I will provide you with API name and description, API method name and description, and list of parameters, each with its name, description, and constraints.
Your task is to generate utterances and API calls that users may ask that must be solved using the given API method. Consider the following guidelines:

    1) Generated utterances must respect the constraints imposed by the OAS and defined under the "constraints" key over each parameter. This includes the following rules:
        - Required parameters: All required parameters must be included in all utterances and API calls.
        - Value constraints: Values must conform to range limitations (minimum and maximum) and respect the enumerated set of values that the parameter can assume.
        - Format constraints: The parameter values in the API call must strictly conform with the format constraint defined in the OAS.
        - Inter-dependency constraints: They describes constraints between parameters. The combination of parameters values must strictly respect these constraints. For instance, there are cases where two parameters must be added simultaneously in an API call or only one parameter must be included between a group of parameters.

    2) Generated utterances must be semantically related to the API method and closely resemble what users would normally say when interacting with chatbots or requesting for tasks. This includes the following rules:
        - Do not add the API name or the API method name in the utterance. Exceptions may be used when the API is related to a known brand or product (e.g., Spotify for music, IMDb for movies, Booking.com for hotels, Google for search).
        - Do not use general placeholders such as "this URL", "this link", "this name", "a candidate" for parameters. Instead, add the parameter infomration, (e.g., URL, link, name) in the utterance instead of only in the API call. 
        - Every parameter value used in the API call must be explicitly inferable from the utterance. For instance, a link, name, destination mapped in the arguments must be mentioned in the utterance.
        - Do not include parameter values in the utterance with a format that the user would not normally say in the utterance. For instance, instead of using precise latitutude and longitude to represent a place, a more natural utterance would reference a city name or landmark. In other words, the "format" constraint must be strictly respect in the API call but not in the utterance.
        - If the "id" constraint is True, do not generate artificial IDs for those parameters (e.g., "1234" or "abc_123"). Instead, create values that represent what the ID refers to. For example, for "hotelId" parameter, generate hotel names such as "The Grand Hyatt" instead of "ACPAR419". Exceptions are parameters where the user actually have access or knowledge about the ID, such as a tracking package ID, flight number, invoice ticket.

    3) Generated utterances must have a diverse combination of optional parameters. This includes the following rules:
        - Parameter coverage: Include all optional parameters at least once in the generated utterances. The only exception is when the number of optional parameters is too large for the number of utterances requested, which would lead to unnatural utterances.
        - Parameter combination diversity: Use different combinations of optional parameters across utterances to reflect the many ways utterances should be related to the given API.
        - Enumerated values: If a parameter includes an "enumerated" constraint (i.e., a fixed set of allowed values), cycle through different allowed values across the utterances to maximize parameter value diversity.
        - Linguistic diversity: Ensure that utterances have diversity in terms of words (rich vocabulary with different verbs and nouns) and sentence structure (declarative, interrogative, imperative, compound, complex sentence, and so on).
        - Parameter value diversity: Avoid repeating the same value for any parameter across different utterances. The "default" key is just an example and must not be used in all utterances. 
         
The output must be a Python list of dictionaries, where each dictionary has two keys:
    - "utterance": the natural language request.
    - "parameters": a dictionary containing the name-value pairs for all parameters used in the utterance. 
    
The "parameters" dictionary must have pairs that can be infered or recognised from the generated natural utterance and following the rules defined in the OAS. 
Finally, you must only output the Python list and do not output anything else, such as notes or explanation about the reasoning.
"""

## variables
model="deepseek-ai/DeepSeek-V3" #model="gpt-4o" model='deepseek-ai/DeepSeek-V3'
API_count = 0
API_methods_count = 0
number_of_utterances_per_method = 10
number_of_methods_with_no_parameters = 0
number_of_methods_with_no_required_parameters = 0
number_of_methods_with_no_optional_parameters = 0
number_of_methods_with_optional_and_required_parameters = 0
number_of_methods_with_condition_constraint = 0
mistakes = 0 

## folders
if not '/' in model:
  client = OpenAI(api_key="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")
  model_name = model
else:
  client = OpenAI(api_key="0cTfrP7S4xnxVcQeMnkCvfY7nrgZs41e", base_url="https://api.deepinfra.com/v1/openai",)
  model_name = model.split('/')[1].lower()

constraint_folder = os.path.join('./data/dataset', model_name, 'constraint-aware/constraints/')
utterance_folder = os.path.join('./data/dataset', model_name, 'constraint-aware/utterances/')

## this is temporary, only to not get all the information
OAS_with_all_constraints = ['working_days.json', 'flight_radar.json', 'horse_racing.json', 'moviesdatabase.json', 'cheapshark_game_deals.json', 'redline_zipcode.json', 'ott_details.json', 'sportspage_feeds.json', 'currencyapi_net.json', 'webcams_travel.json', 'mdblist.json', 'irctc.json', 'foreca_weather.json', 'referential.json', 'real_time_product_search.json', 'workable.json', 'youtube_mp3.json', 'youtube_search_and_download.json', 'solarenergyprediction.json', 'pinnacle_odds.json', 'hotels_com_provider.json', 'subreddit_scraper.json', 'shazam.json', 'indeed_jobs_api_finland.json', 'bayut.json', 'weatherapi_com.json', 'flightera_flight_data.json']

print(model)
## generates utterances using the constraint-augmented information 
for root, _, files in os.walk(constraint_folder):
    for filename in files:
        input_path = os.path.join(constraint_folder, filename)
        output_path = os.path.join(utterance_folder, filename)
        
        if not os.path.exists(output_path) and filename in OAS_with_all_constraints:
            with open(input_path, 'r') as f:
                data = json.load(f)                
                api_name = data['name']
                api_description = data['description']
                print(filename)

                for api_method_index, api_method in enumerate(data['api_methods']):
                    api_method_name = api_method['name']
                    api_method_description = api_method['description'] 
                    api_method_parameters = [param for param in api_method['parameters'] if not ('constraints' in param and param['constraints'].get('technical') is True)]
                    print(API_methods_count, 'api_method:', api_method_name)

                    required_parameters = [param["name"] for param in api_method_parameters if param.get('required', False)]
                    optional_parameters = [param["name"] for param in api_method_parameters if 'required' in param and param['required'] is False] 

                    api_specification = {"API Name": api_name,
                                        "API Description": api_description if len(api_description) < 4000 else '',
                                        "API Method Name": api_method_name,
                                        "API Method Description": api_method_description if len(api_method_description) < 4000 else '',
                                        "Parameters": api_method_parameters}

                    ### ORGANISING PROMPT
                    # 1) There is no parameter
                    if len(required_parameters) + len(optional_parameters) == 0:                            
                        input = (f"API Specification: \n{api_specification}\n"
                                 f"Write {number_of_utterances_per_method} utterances that use the given API.\n")

                        number_of_methods_with_no_parameters+=1

                    # 2) There are only required parameters 
                    elif len(optional_parameters) == 0:
                        input = (f"API Specification: \n{api_specification}\n"
                                 f"Write {number_of_utterances_per_method} utterances that use the given API.\n"
                                 f"Required parameters: {str(required_parameters).strip('[]')}.\n")
                        
                        number_of_methods_with_no_optional_parameters+=1

                    # 3) There are only optional parameters
                    elif len(required_parameters) == 0:
                        input = (f"API Specification: \n{api_specification}\n"
                                 f"Write {number_of_utterances_per_method} utterances that use the given API.\n"
                                 f"Optional parameters: {str(optional_parameters).strip('[]')}.\n")

                        number_of_methods_with_no_required_parameters+=1  

                    # 4) There are required and optional parameters.
                    else:
                        input = (f"API Specification: \n{api_specification}\n"
                                 f"Write {number_of_utterances_per_method} utterances that use the given API.\n"
                                 f"Required parameters: {str(required_parameters).strip('[]')}.\n"
                                 f"Optional parameters: {str(optional_parameters).strip('[]')}.\n")

                        number_of_methods_with_optional_and_required_parameters+=1  
                        
                    messages = [{"role": "system", "content": PROMPT_BASE_INSTRUCTION},
                                {"role": "user", "content": str(input)}]
                    response = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            max_tokens=3000,
                            temperature=0)

                    try: 
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
                    #break
            
            # saving file with utterances and parameters mapping
            with open(output_path, 'w') as json_file:
                json.dump(data, json_file, indent=4) 
        
        API_count+=1
        
    # if API_count > 1:
    #     break

print('number of methods with no parameters:', number_of_methods_with_no_parameters)
print('number of methods with no optional parameters:', number_of_methods_with_no_optional_parameters)
print('number of methods with no required parameters:', number_of_methods_with_no_required_parameters)
print('number of methods with optional AND required parameters:', number_of_methods_with_optional_and_required_parameters)
print('number of methods:', API_methods_count)