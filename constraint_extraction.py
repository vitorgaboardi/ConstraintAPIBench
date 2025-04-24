import json
import ast
import tiktoken
import copy
import os
import re
from openai import OpenAI

### PROMPT
PROMPT_INSTRUCTION = """You are an expert in interpreting OpenAPI specification (OAS). Your task is to extract constraints for each parameter of the API method.
    I will provide you with the API name and description, the API method name and description, and the list of parameters, each with its name and description. 
    For each parameter, you must extract constraints by analysing the descriptions in the OAS provided. Do not create constraints if they are not defined in the documentation.
    Extract the following constraints:

(1) format: Any required input format, such as general or standard codes (e.g., ISO 8601 date/time, currency codes, country codes, IATA codes). Do not infer formats based on default examples.
(2) values: Constraints about the values parameters can assume. This includes: 
    - "max": for maximum numeric value, 
    - "min": for minimum numeric value, 
    - "specific": for a closed set of only acceptable values. Only use "specific" if a defined, closed list of possible values is allowed (e.g., ["google", "microsoft", "bing"], ["image", "video", "audio"]). Limit the content to a maximum of 40 values.
(3) id: Boolean value that must be "True" if the parameter represents a resource identifier (ID) (e.g., parameter names like "message_id", "gameId", "hotelIds", "locationInternalIDs" or descriptions mentioning the parameter represents an ID).
(4) api_related: Boolean value that must be "True" if the parameter is used to control the API's behavior or response formatting, and is typically used by developers rather than end users. Examples include pagination controls (e.g., page number, offset, limit, page size), authentication tokens (e.g., api key, access token), system management fields (e.g., cache, debug, locale, encoding), and callback URLs.
(5) inter-dependency: Describe a required dependency between parameters. The examples mentioned are illustrative for you to understand better. Only include the constraints if stated in the description of the API method or parameter. Dependencies include: 
    - the presence of one parameter requires the presence of another parameter (e.g., "startDate, endDate: if startDate is provided, endDate must also be included.")
    - given a set of parameters, one or more of them must be included in the API call (e.g., "email, phone: at least one of these contact methods must be provided."), 
    - given a set of parameters, one and only of them must be included in the API call (e.g., "q, name and name_equals: only one of them must be used."), 
    - given a set of parameters, either all of them are provided or none of them (e.g., "latitude and longitude: both parameters must be included together."),
    - given a set of parameters, zero or one parameter can be present in the API call (e.g., "promoCode, discountId: at most one of these can be included in a request."), 
    - given a set of parameters, they are related by means of arithmetic constraints (e.g., "minPrice, maxPrice: minPrice must be less than or equal to maxPrice.").

Additional guidelines:
    * For the inter-dependency constraint, list all related parameters separated by commas, followed by a textual description of the constraint (e.g., "latitude, longitude: both parameters must be included simultaneously".)
    * If a parameter includes values constraints ("minimum", "maximum", "specific"), you must include them under "values".
    * Do not create or use any constraint types that are not listed above (e.g., "require", "mandatory", "optional").
    * If any field is not applicable or cannot be inferred for a parameter, omit that field entirely from the output. 

The final output must be a JSON object where each key is the name of a parameter, and the value is a nested object with only the relevant constraint fields. 
You must strictly follow the output format and do not return anything else other than the JSON object.
"""

EXAMPLE_INPUT = """ 
{
  "API Name": "Flight Search API",
  "API Description": "The Flight Search API retrieves available flights between two locations using IATA codes. It supports one-way and round-trip searches, passenger details, travel class, and paginated results ordered by departure time.",
  "API Method Name": "flightSearch",
  "API Method Description": "Returns a list of flights.",
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
    },
    {
      "name": "returnDate",
      "description": "the date on which the traveler will depart from the destination to return to the origin. If this parameter is not specified, only one-way itineraries are found. If this parameter is specified, only round-trip itineraries are found. Dates are specified in the [ISO 8601] YYYY-MM-DD format, e.g. 2018-02-28",
      "required": false,
    },
    {
      "name": "adults",
      "description": "the number of adult travelers (age 12 or older on date of departure). The total number of seated travelers (adult and children) can not exceed 9.",
      "required": true,
    },
    {
      "name": "children",
      "description": "the number of child travelers (older than age 2 and younger than age 12 on date of departure) who will each have their own separate seat. If specified, this number should be greater than or equal to 0. The total number of seated travelers (adult and children) can not exceed 9.",
      "required": false,
    },
    {
      "name": "travelClass",
      "description": "Most of the flight time should be spent in a cabin of this quality or higher. The accepted travel class is economy, premium economy, business or first class. If no travel class is specified, the search considers any travel class. Available classes are: ECONOMY, PREMIUM_ECONOMY, BUSINESS, FIRST,
      "required": false,
    },
    {
      "name": "page[offset]",
      "description": "start index of the requested page",
      "required": false,
      "default": 0
    },
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
    "format": "IATA code"
  },
  "destinationLocationCode": {
    "format": "IATA code"
  },
  "departureDate": {
    "format": "ISO 8601 YYYY-MM-DD format",
    "inter-dependency": "departureDate, returnDate: departureDate must be before or equal to returnDate."
  },
  "returnDate": {
    "format": "ISO 8601 YYYY-MM-DD format",
    "inter-dependency": "departureDate, returnDate: departureDate must be before or equal to returnDate.
  },
  "adults": {
    "values": {
      "max": 9,
      "min": 1,
    },
    "inter-dependency": "adults, children: The combined number of adults and children must not exceed 9."
  },
  "children": {
    "values": {
      "max": 9,
      "min": 0
    },
    "inter-dependency": "adults, children: The combined number of adults and children must not exceed 9."
  },
  "travelClass": {
    "values": {
      "specific": ["ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"]
    }
  },
  "page[offset]": {
    "api_related": true
  }
  "flightId": {
    "id": true
  }
"""

## structuring the prompt
base_messages = [{"role": "system", "content": PROMPT_INSTRUCTION},
                 {"role": "user", "content": EXAMPLE_INPUT},
                 {"role": "assistant", "content": EXAMPLE_OUTPUT}]

## variables
# model name
model="gpt-4.1-mini"
#model="gpt-4.1"
#model='deepseek-ai/DeepSeek-V3'

OAS_folder = '/home/vitor/Documents/phd/ConstraintAPIBench/dataset/tools'
tokenizer = tiktoken.get_encoding("cl100k_base")
API_count = 0
API_methods_count = 0
total_tokens = 0
constraint_mistakes = 0
conditional_found = 0

if model == "gpt-4.1-mini" or model == "gpt-4.1":
  client = OpenAI(api_key="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")
  constraint_folder = '/home/vitor/Documents/phd/ConstraintAPIBench/dataset/'+model+'/constraints/'
else:
  client = client = OpenAI(api_key="0cTfrP7S4xnxVcQeMnkCvfY7nrgZs41e",
                           base_url="https://api.deepinfra.com/v1/openai",)
  model_name = model.split('/')[1]
  constraint_folder = '/home/vitor/Documents/phd/ConstraintAPIBench/dataset/'+model_name+'/constraints/'

# iterating over the categories
categories = sorted(os.listdir(OAS_folder))
for category_index, category in enumerate(categories):
    category_path = os.path.join(OAS_folder, category)

    # generate folder to save OAS now enriched with constraints information
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
                      print('api_method', api_method['name'])
                      api_method_name = api_method['name']
                      api_method_description = api_method['description']
                      api_method_parameters = api_method['parameters']

                      # request for parameters constraints only if there are any parameter.
                      if len(api_method_parameters) > 0:
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
                            #constraint = ast.literal_eval(response.choices[0].message.content.replace('false', 'False').replace('true', 'True').replace('null', 'None'))
                            # pre-processing
                            constraint = re.sub(r"```(json)?", "", response.choices[0].message.content).strip()
                            constraint = re.sub(r"//.*", "", constraint)
                            constraint = constraint.replace("'", '"')
                            constraint = json.loads(constraint)

                            for parameter in api_method_parameters:
                                name = parameter.get("name")
                                if name in constraint:
                                    parameter['constraints'] = constraint[name]
                            
                            data['api_list'][api_method_index]['parameters'] = api_method_parameters

                            # just checking if there are many conditional found.
                            if 'conditional' in constraint:
                              print(constraint['conditional'])
                              conditional_found+=1
                              
                        
                        except Exception as e:
                            print("Exception arised!")
                            print(e)
                            print(response.choices[0].message.content)
                            constraint_mistakes+=1

                      API_methods_count+=1

                  with open(saving_new_OAS+'/'+filename, 'w') as json_file:
                      json.dump(data, json_file, indent=4) 

          API_count+=1
          break
        break
                
    # if API_count > 200:
    #     break

print('total input tokens:', total_tokens)
print('conditional constraints:', conditional_found)
print('number of constraint mistakes:', constraint_mistakes)
print('number of APIs:', API_count)
print('number of API methods:', API_methods_count)