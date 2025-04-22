import json
import ast
import tiktoken
import copy
import os
from openai import OpenAI

client = OpenAI(api_key="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")

# use one prompt to generate all parameters from one API method simultaneosly. 
# Update on 19/04: Do not use this part in the pipeline. My focus is not on creating API tests
# or API calls. Therefore, there is no need to first generate the values. Try to do the pipeline without this part!

PROMPT_INSTRUCTION = """You are an expert in understanding OpenAPI Specification (OAS). 
I will provide you with information about an API, including descriptions of API method and its parameters.
Your task is to generate up to 10 diverse examples values for each parameter, according to the provided descriptions and constraints.
The generated values MUST strictly adhere the constraints given. For instance:
    - Values must conform to the expected type and format (e.g., ISO 8601 date/time, country codes, email).
    - Values must respect range limitations (minimum and maximum values).
    - If the parameter represents an ID or reference (i.e., "id = True" in the constraint option), create values representing what the IDs represent instead of actually creating an ID. For example, for a parameter like "hotelId", generate hotel names (e.g., "Hotel California", "The Grand Hyatt"), and for "signId", generate zodiac signs (e.g., "Gemini", "Pisces").

The final output must be a JSON object where each key is the name of a parameter, and the value is a Python list of generated values. Do not return anything other than the JSON.
"""


EXAMPLE_INPUT = """
{
  "API Name": "Flight Search API",
  "API Description": "The Flight Search API retrieves available flights between two locations using IATA codes. It supports one-way and round-trip searches, passenger details, travel class, and paginated results ordered by departure time.",
  "API Method Name": "flightSearch",
  "API Method Description": "Returns a list of flights.",
  "Parameters":{
      {
      "name": "originLocationCode",
      "type": "STRING",
      "description": "city/airport [IATA code] from which the traveler will depart, e.g. BOS for Boston",
      "required": true,
      "constraints": {
        "format": "IATA code"}
    },
    {
      "name": "destinationLocationCode",
      type": "STRING",
      "description": "city/airport [IATA code] to which the traveler is going, e.g. PAR for Paris",
      "required": true,
      "constraints": {
        "format": "IATA code"}
    },
    {
      "name": "departureDate",
      "type": "STRING",
      "description": "the date on which the traveler will depart from the origin to go to the destination. Dates are specified in the [ISO 8601] YYYY-MM-DD format, e.g. 2017-12-25",
      "required": true,
      "constraints": {
        "format": "ISO 8601 date"}
    },
    {
      "name": "adults",
      "description": "the number of adult travelers (age 12 or older on date of departure). The total number of seated travelers (adult and children) can not exceed 9.",
      "type": "INT",
      "required": true,
      "constraints": {
          "values": {
            "max": 9,
            "min": 1,
            }}
    },
    {
      "name": "flightId",
      "type": "STRING",
      "description": "A unique identifier for the flight. This can be used to retrieve specific flight details.",
      "required": false,
      "example": "FL123456"
      "constraints": {
        "id": "True"}
    }
  }
}
"""

EXAMPLE_OUTPUT = """
{
  "originLocationCode": ["BOS", "JFK", "LAX", "ORD", "ATL", "MIA", "DFW", "SEA", "DEN", "SFO"],
  "destinationLocationCode": ["PAR", "LHR", "CDG", "AMS", "FRA", "MAD", "BCN", "ROM", "DUB", "ZRH"],
  "departureDate": ["2024-03-03", "2024-06-18", "2024-07-20", "2024-08-10", "2024-09-05", "2024-10-12", "2024-11-01", "2024-12-24", "2025-01-15", "2025-03-30"],
  "adults": [1, 2, 3, 4, 5, 6, 7, 8, 9],
  "flightId": ["Delta Flight 1001", "United 302", "American Airlines 5678", "Lufthansa 402", "Emirates EK202", "Air France AF123", "British Airways BA2490", "Qatar Airways QR789", "KLM KL345", "Turkish Airlines TK987"]
}
"""

## structuring the prompt
base_messages = [{"role": "system", "content": PROMPT_INSTRUCTION},
                 {"role": "user", "content": EXAMPLE_INPUT},
                 {"role": "assistant", "content": EXAMPLE_OUTPUT}]

# basic variables
model="gpt-4.1-mini"
tokenizer = tiktoken.get_encoding("cl100k_base")
OAS_folder = "./dataset/GPT-4.1-mini/constraints"
parameter_folder = './dataset/GPT-4.1-mini/parameter_generation'

API_count = 0
API_methods_count = 0
total_tokens = 0
mistakes = 0
conditional_found = 0

# iterating over the categories
categories = sorted(os.listdir(OAS_folder))
for category_index, category in enumerate(categories):
    category_path = os.path.join(OAS_folder, category)
    print(category_path)

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
                    print(category_index, data['tool_name'])
                    api_name = data['tool_name']
                    api_description = data['tool_description']

                    # iterating over each API method for the given API
                    # 1) remove the parameters where "api_related" is True
                    # 2) the ones that the type is "Boolean" must be set True or False
                    # 3) the ones with "specific" set already have
                    # 4) the remaining will be asked for 
                    for api_method_index, api_method in enumerate(data['api_list']):
                        print(api_method['name'])
                        api_method_name = api_method['name']
                        api_method_description = api_method['description']
                        api_method_parameters = [
                            param for param in api_method['parameters']
                            if not ('constraints' in param and param['constraints'].get('api_related') is True)]

                        parameters_to_request_for_values = []
                        for parameter in api_method_parameters:
                            if 'constraints' in parameter:
                                if parameter["type"] == "BOOLEAN":
                                    parameter['values'] = [True, False]
                                elif parameter['constraints'].get('values', {}).get('specific') is not None:
                                        parameter['values'] = parameter['constraints']['values']['specific']
                                else:
                                    parameters_to_request_for_values.append(parameter)

                        # defining the input information to request for parameters
                        # organising input
                        input = {"API name": api_name,
                                "API description": api_description if len(api_description) < 4000 else '',
                                "API method name": api_method_name,
                                "API method description": api_method_description if len(api_method_description) < 4000 else '',
                                "Parameters": parameters_to_request_for_values}
                        
                        messages = base_messages + [{"role": "user", "content": str(input)}]

                        # counting the number of input tokens
                        input_tokens = tokenizer.encode(str(messages))
                        total_tokens+=len(input_tokens)

                        try:
                            # calling API
                            response = client.chat.completions.create(
                                    model=model,
                                    messages=messages,
                                    max_tokens=1000,
                                    temperature=0)
                            
                            values = ast.literal_eval(response.choices[0].message.content.replace('false', 'False').replace('true', 'True').replace('null', 'None'))
                            for parameter in api_method_parameters:
                                name = parameter.get("name")
                                if name in values:
                                    parameter['values'] = values[name]
                            
                            # updating the data with the new values!
                            data['api_list'][api_method_index]['parameters'] = api_method_parameters

                        except Exception as e:
                            print("Exception arised!")
                            print(e)
                            print(response.choices[0].message.content)
                            mistakes+=1

                        API_methods_count+=1
                    with open(saving_new_OAS+'/'+filename, 'w') as json_file:
                        json.dump(data, json_file, indent=4) 

            API_count+=1
    if API_count > 9:
        break