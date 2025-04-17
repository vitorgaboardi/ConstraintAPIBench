import json
import ast
import tiktoken
import copy
import os
from openai import OpenAI

client = OpenAI(api_key="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")

### PROMPT
PROMPT_INSTRUCTION = """You are an expert in interpreting OpenAPI specification (OAS). Your task is to extract constraints for each parameter of the API method.
    I will provide you with the API name and description, the API method name and description, and the list of parameters, each with its name and description. For each parameter, identify and extract the following constraints:

    (1) format: Specify any required input format, such as general or standard codes (e.g., ISO 8601 date/time, currency codes, country codes, IATA codes).
    (2) values: Constraints about the values parameters can assume. This includes "max" for maximum numeric value, "min" for minimum numeric value, and "specific" for a closed set of valid values. Only use "specific" if a specific set of values is allowed. Limit the output for a maximum of 40 words.
    (3) id: Boolean value that must be "True" if the parameter likely represents an identifier (e.g., names like "message_id", "gameId", "hotelIds", "locationInternalIDs" or descriptions mentioning that it represents an ID).
    (4) api_related: Boolean value that must be "True" if the parameter relates to the API rather than user-specific data. Examples include pagination controls (e.g., page number, offset, limit, page size), sorting controls (e.g., sort order, fields), authentication tokens (e.g., api key, access token), system management fields (e.g., cache, debug, locale, encoding), and callback URLs.
    (5) conditional: Describe a required dependency between parameters. This includes: the presence of one parameter requires another; at least one parameter must be included given a set of parameters; only one parameter must be included given a set of parameters; either all or no parameters must be included; at most one parameter must be included given a set of parameters; parameters have an arithmetic or relational constraint. Include only required dependencies.

For the conditional constraint, list all related parameters separated by commas, followed by a textual description of the constraint. For instance. "latitude, longitude: both parameters must be included simultaneously".
If any field is not applicable or cannot be inferred for a parameter, omit that field entirely from the output. 
The final output must be a JSON object where each key is the name of a parameter, and the value is a nested object with only the relevant constraint fields. Do not return anything other than the JSON.
"""

EXAMPLE_INPUT = """ 
{
  "API Name": "Flight Search API",
  "API Description": "The Flight Search API retrieves available flights between two locations using IATA codes. It supports one-way and round-trip searches, passenger details, travel class, and paginated results ordered by departure time.",
  "API Method Name": "flightSearch",
  "API Method Description": "Returns a list of flights for a given flight number. Minimum and/or maximum date can optionally be specified to limit the search. Results are ordered by departure date ascending. The next departure time is returned for pagination.",
  "Parameters": [
    {
      "name": "originLocationCode",
      "description": "city/airport [IATA code] from which the traveler will depart, e.g. BOS for Boston",
      "required": true,
    },
    {
      "name": "destinationLocationCode",
      "description": "city/airport [IATA code] to which the traveler is going, e.g. PAR for Paris",
      "required": true,
    },
    {
      "name": "departureDate",
      "description": "the date on which the traveler will depart from the origin to go to the destination. Dates are specified in the [ISO 8601] YYYY-MM-DD format, e.g. 2017-12-25",
      "required": true,
      "format": "date",
      "x-example": "2023-05-02"
    },
    {
      "name": "returnDate",
      "description": "the date on which the traveler will depart from the destination to return to the origin. If this parameter is not specified, only one-way itineraries are found. If this parameter is specified, only round-trip itineraries are found. Dates are specified in the [ISO 8601] YYYY-MM-DD format, e.g. 2018-02-28",
      "required": false,
      "format": "date"
    },
    {
      "name": "adults",
      "description": "the number of adult travelers (age 12 or older on date of departure). The total number of seated travelers (adult and children) can not exceed 9.",
      "required": true,
      "minimum": 1,
      "maximum": 9,
      "default": 1
    },
    {
      "name": "children",
      "description": "the number of child travelers (older than age 2 and younger than age 12 on date of departure) who will each have their own separate seat. If specified, this number should be greater than or equal to 0. The total number of seated travelers (adult and children) can not exceed 9.",
      "required": false,
      "minimum": 0,
      "maximum": 9
    },
    {
      "name": "travelClass",
      "description": "Most of the flight time should be spent in a cabin of this quality or higher. The accepted travel class is economy, premium economy, business or first class. If no travel class is specified, the search considers any travel class.",
      "required": false,
      "enum": ["ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"]
    },
    {
      "name": "page[offset]",
      "description": "start index of the requested page",
      "default": 0
    }
    {
      "name": "flightId",
      "description": "A unique identifier for the flight. This can be used to retrieve specific flight details.",
      "required": false,
      "example": "FL123456"
    }
  ]
}
"""

EXAMPLE_OUTPUT = """
{
  "originLocationCode": {
    "format": "IATA code",
  },
  "destinationLocationCode": {
    "format": "IATA code",
  },
  "departureDate": {
    "format": "ISO 8601 date",
  },
  "returnDate": {
    "format": "ISO 8601 date",
  },
  "adults": {
    "values": {
      "max": 9,
      "min": 1,
    },
    "conditional": "adults, children: The combined number of adults and children must not exceed 9."
  },
  "children": {
    "values": {
      "max": 9,
      "min": 0
    },
    "conditional": "adults, children: The combined number of adults and children must not exceed 9."
  },
  "travelClass": {
    "values": {
      "specific": ["ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"]
    }
  },
  "page[offset]": {
    "values": {
      "min": 0,
    },
    "api_related": true
  }
  "flightId": {
    "id": true
  }
}
"""

## structuring the prompt
base_messages = [{"role": "system", "content": PROMPT_INSTRUCTION},
                 {"role": "user", "content": EXAMPLE_INPUT},
                 {"role": "assistant", "content": EXAMPLE_OUTPUT}]

# basic variables
model="gpt-4.1-mini"
tokenizer = tiktoken.get_encoding("cl100k_base")
OAS_folder = '/home/vitor/Documents/phd/api constraints_3/dataset/tools'
constraint_folder = '/home/vitor/Documents/phd/api constraints_3/dataset/GPT-4.1-mini/constraints/'

API_count = 0
API_methods_count = 0
total_tokens = 0

# iterating over the categories
categories = sorted(os.listdir(OAS_folder))
for category_index, category in enumerate(categories):
    category_path = os.path.join(OAS_folder, category)

    # save OAS now enriched with constraints information
    saving_new_OAS = os.path.join(constraint_folder, category)
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
                    for api_method_index, api_method in enumerate(data['api_list']):
                        print(api_method['name'])
                        api_method_name = api_method['name']
                        api_method_description = api_method['description']
                        api_method_parameters = api_method['parameters']

                        # organising input
                        input = {"API Name": api_name,
                                    "API Description": api_description,
                                    "API Method Name": api_method_name,
                                    "API Method Description": api_method_description if len(api_method_description) < 4000 else '',
                                    "Parameters": api_method_parameters}
                        
                        messages = base_messages + [{"role": "user", "content": str(input)}]

                        # counting the number of input tokens
                        input_tokens = tokenizer.encode(str(messages))
                        total_tokens+=len(input_tokens)

                        # calling API
                        response = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            max_tokens=1000,
                            temperature=0)
                        
                        # updating it in the file
                        try:
                            constraint = ast.literal_eval(response.choices[0].message.content.replace('false', 'False').replace('true', 'True').replace('null', 'None'))
                            for parameter in api_method_parameters:
                                name = parameter.get("name")
                                if name in constraint:
                                    parameter['constraints'] = constraint[name]
                            
                            data['api_list'][api_method_index]['parameters'] = api_method_parameters
                        
                        except:
                            print(response.choices[0].message.content)
                            break

                        API_methods_count+=1

                    with open(saving_new_OAS+'/'+filename, 'w') as json_file:
                        json.dump(data, json_file, indent=4) 

            API_count+=1
                
    if API_count > 10:
        break

print('number of APIs:', API_count)
print('number of API methods:', API_methods_count)