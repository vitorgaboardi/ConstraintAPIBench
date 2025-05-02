# recreating the generation of utterance using the prompt from Sheng et Al: Measuring an LLM’s Proficiency at using APIs: A Query Generation Strategy

import json
import ast
import copy
import os
import numpy as np
import random
import itertools
import re
from openai import OpenAI

# instruction taken from their paper. This is a basic one that does not focus on much stuff. 
# I may have to add instructions to make it better and more relevant?
PROMPT_INSTRUCTION = """
Imagine that you are a user who wants to utilize the features provided by various APIs in your daily life. Your task is to come up with realistic scenarios for using these APIs and express them as natural language instructions, as if you were asking a friend or assistant for help.

Please follow these guidelines:
1. The instructions should be 1 to 2 sentences long. Use a mix of interrogative sentences, first-person statements, imperative sentences, and other structures that convey a request. Aim for diversity in your instructions.
2. Do not mention the API's name in your instructions.
3. Your instructions should only involve the features provided by these APIs. The instructions that need multiple times of API call is better.
4. Generate 10 diverse instructions.
5. Use specific nouns and real-world examples from various domains, such as entertainment, sports, or technology. Avoid using any form of placeholder or generic phrases, such as "this xxx", "a xxx" or "a specific xxx", and provide concrete details instead.
6. Try not to repeat the verb for each instruction to maximize diversity.
7. Ensure diversity in language by combining questions with imperative statements and other structures that convey a request.
    
The output must be a Python list of dictionaries, where each dictionary has two keys:
    - "utterance": the natural language request.
    - "parameters": a dictionary containing the name-value pairs for all parameters used in the utterance. 
    
The "parameters" dictionary must have pairs that can be infered or recognised from the generated natural utterance. 
Finally, you must only output the Python list and do not output anything else, such as notes or explanation about the reasoning.
"""

# variables
model="gpt-4.1"  #  gpt-4.1-mini "gpt-4.1" "deepseek-ai/DeepSeek-V3"
OAS_folder = './data/tools'
number_of_utterances_per_method = 10

API_methods_count = 0
API_count = 0
mistakes = 0
OAS_to_create_utterances = ['foreca_weather.json-Daily', 'learn_to_read_and_write_japanese_kanji.json-Kanji grade level', 'getitsms_whatsapp_apis.json-GetIT SMS WHATSAPP API', 'streaming_availability.json-Search Basic (FREE)', 'solarenergyprediction.json-/v2.0/solar/prediction', 'nowpayments.json-3.Getestimatedprice', 'referential.json-Languages', 'covid_19_by_api_ninjas.json-/v1/covid19', 'ott_details.json-Advanced Search', 'dezgo.json-/text2image', 'veriphone.json-verify', 'working_days.json-/1.3/list_non_working_days', 'flightera_flight_data.json-airportDelayDailyStatistics', 'postal_ninja.json-createTrack', 'cricket_live_data.json-Results By Date', 'car_code.json-/obd2/{code}', 'trackingpackage.json-TrackingPackage', 'shazam.json-songs/get-count', 'axesso_amazon_data_service.json-lookupSeller', 'hotels_com_provider.json-Hotel Rooms (offers)', 'netflix_v2.json-Search', 'recipe_food_nutrition.json-Generate Shopping List', 'spotify.json-Artist albums', 'movie_database_alternative.json-By Search', 'tasty.json-tips/list', 'synwave.json-Upload a new file', 'everyearthquake.json-Earthquakes', 'flightera_flight_data.json-airlineStatistics', 'webcams_travel.json-/webcams/list/webcam={webcamid}[,{webcamid}[,...]]', 'working_days.json-/1.3/add_working_days']

if model == "gpt-4.1-mini" or model == "gpt-4.1":
  client = OpenAI(api_key="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")
  model_name = model
else:
  client = client = OpenAI(api_key="0cTfrP7S4xnxVcQeMnkCvfY7nrgZs41e",
                           base_url="https://api.deepinfra.com/v1/openai",)
  model_name = model.split('/')[1].lower()

utterance_folder = os.path.join('./data/dataset', model_name, 'toolalpaca')
base_messages = [{"role": "system", "content": PROMPT_INSTRUCTION}]

# running code
categories = sorted(os.listdir(OAS_folder))
for category_index, category in enumerate(categories):
    category_path = os.path.join(OAS_folder, category)

    for root, _, files in os.walk(category_path):
        for filename in files:
            if not os.path.exists(utterance_folder+'/'+filename): 
                file_path = os.path.join(category_path, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    api_name = data['tool_name']
                    api_description = data['tool_description']
                    home_url = data['home_url']
                    call_made = False

                    API_methods_to_save = []
                    for api_method_index, api_method in enumerate(data['api_list']):
                        api_method_name = api_method['name']
                        api_method_description = api_method['description']
                        api_method_url = api_method['url']
                        api_method_parameters = api_method['parameters']

                        tool_specification = {"API Name": api_name,
                                            "API Description": api_description if len(api_description) < 4000 else '',
                                            "API Method Name": api_method_name,
                                            "API Method Description": api_method_description if len(api_method_description) < 4000 else '',
                                            "Parameters": api_method_parameters}
                        
                        input = (f"<API>: \n{tool_specification}\n </API>\n"
                                 f"Based on the API provided above, generate {number_of_utterances_per_method} natural language instructions with specific examples and diverse language, following the guidelines.")

                        messages = base_messages + [{"role": "user", "content": str(input)}]

                        # this if is temporary: make the call only for the ones that we will evaluate manually.
                        if filename+"-"+api_method_name in OAS_to_create_utterances:
                            print(API_methods_count, filename+"-"+api_method_name)

                            call_made = True
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

                                utterances = content

                            except Exception as e:
                                print(e)
                                print(response.choices[0].message.content)
                                mistakes+=1

                                utterances = 'error parsing the information.'
                                                            
                            API_methods_to_save.append({
                                'name': api_method_name,
                                'description': api_method_description,
                                'url': api_method_url,
                                'parameters': api_method_parameters,
                                'utterances': utterances
                            })

                            API_methods_count+=1
                        
                    documentation = {
                        "name": api_name,
                        "description": api_description,
                        "url": home_url,
                        "api_methods": API_methods_to_save}

                if call_made:
                    output_file = os.path.join(utterance_folder, filename)
                    with open(output_file, 'w') as json_file:
                        json.dump(documentation, json_file, indent=4) 
            
            API_count+=1


print('number of methods:', API_methods_count)
print('mistakes:', mistakes)
print('APIs:', API_count)

print(messages)